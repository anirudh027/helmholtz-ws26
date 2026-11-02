"""Few-label correction of HOCT (Bragantini et al. 2026, sec. 4.5).

The HOCT backbone stays frozen; only a logistic-regression head on its
per-link features is refitted from a few annotated links (``hoct.correction``).
Links are proposed by uncertainty (predicted probability closest to 0.5), in
rounds of ~20; the paper reports ~59% fewer AOGM errors after 400 links, at
~2 min per round. This is the cheapest route to high accuracy on a dataset
without ground truth (Helmholtz), and the labels double as ground truth.

    c = Corrector(imgs, masks, HOCT(tta=6))
    res0 = c.result()                          # plain HOCT
    q = c.propose(20)                          # DataFrame of links to check by eye
    q["correct"] = [...]                       # True / False per row (or an oracle)
    c.annotate(q)
    c.refit()
    res1 = c.result()                          # corrected tracks

``scripts/correction_sim.py`` runs the loop with a ground-truth oracle (MuSC,
HeLa) to measure the gain per annotation; ``scripts/correction_sheet.py``
renders the proposed links as image crops for a human annotator.
"""

import dataclasses

import numpy as np
import pandas as pd
import polars as pl

from .ctc import CTCResult
from .hoct import HOCT, load_model, solution_to_ctc

LABEL_MASK, LABEL = "annotated", "annotated_value"


class Corrector:
    """HOCT with a refittable linear probe. Labels live in ``self.labels``
    ({edge_id: bool}) and are written onto the graph only while the probe is
    fitted: HOCT's tracklet solver adds new edges to the graph on every solve
    and rejects graphs that carry edge attributes it does not know."""

    def __init__(self, imgs, masks, hoct: HOCT | None = None):
        # hoct.correction.fit_from_labels cannot iterate tiled datasets (no len()),
        # so predict and fit both run untiled (fine on a 97 GB GPU).
        self.h = dataclasses.replace(hoct or HOCT(), tile="off")
        self.shape = np.shape(masks)
        self.graph = self.h.build_graph(imgs, masks)
        self.model = load_model(self.h.model, self.h._device())
        self.labels: dict[int, bool] = {}
        self.n_labels = 0
        self._solve()

    def _solve(self):
        import tracksdata as td

        self.solution = self.h.predict(self.graph, self.model)
        self.edges = self.graph.edge_attrs(attr_keys=[
            td.DEFAULT_ATTR_KEYS.EDGE_ID, td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
            "similarity", "delta_t", td.DEFAULT_ATTR_KEYS.SOLUTION]).to_pandas()
        # edges the tracklet solver added (links between tracklets, similarity -1) are not candidates
        self.edges = self.edges[self.edges["similarity"] >= 0].reset_index(drop=True)
        self.edges[LABEL_MASK] = self.edges["edge_id"].isin(list(self.labels))

    def result(self) -> CTCResult:
        from .backends import _scratch

        with _scratch("hoct_corr_") as work:
            res = solution_to_ctc(self.solution, self.graph.metadata.get("shape", self.shape), work)
        res.info.update(n_labels=self.n_labels)
        return res

    def nodes(self) -> pd.DataFrame:
        import tracksdata as td

        return self.graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "y", "x"]).to_pandas()

    def propose(self, n=20) -> pd.DataFrame:
        """The ``n`` unlabelled links whose probability is closest to 0.5, with the
        source / target positions an annotator needs."""
        e = self.edges[~self.edges[LABEL_MASK]].copy()
        e["uncertainty"] = -np.abs(e["similarity"] - 0.5)
        e = e.nlargest(n, "uncertainty")
        nd = self.nodes().set_index("node_id")
        for side, col in (("src", "source_id"), ("tgt", "target_id")):
            for k in ("t", "y", "x"):
                e[f"{side}_{k}"] = nd.loc[e[col], k].to_numpy()
        return e.reset_index(drop=True)  # columns edge_id, source_id, target_id, similarity, ...

    def annotate(self, links: pd.DataFrame):
        """Record verdicts (column ``correct``: True = same cell / mother-daughter).

        A correct link also marks every competing link into the same target as wrong
        (mutual exclusion, as ``hoct.correction.label_edge``), so one verdict labels
        several candidates."""
        e = self.edges
        for r in links.itertuples():
            if bool(r.correct):
                for eid in e.loc[e["target_id"] == int(r.target_id), "edge_id"]:
                    self.labels[int(eid)] = False
            self.labels[int(r.edge_id)] = bool(r.correct)
        self.edges[LABEL_MASK] = self.edges["edge_id"].isin(list(self.labels))
        self.n_labels += len(links)

    def refit(self, **kwargs):
        """Fit the linear probe on all labels so far and re-solve."""
        from hoct.correction import fit_from_labels

        ids = list(self.labels)
        self.graph.add_edge_attr_key(LABEL_MASK, pl.Boolean, False)
        self.graph.add_edge_attr_key(LABEL, pl.Boolean, False)
        try:
            self.graph.update_edge_attrs(edge_ids=ids, attrs={LABEL_MASK: [True] * len(ids),
                                                              LABEL: [self.labels[i] for i in ids]})
            self.model = fit_from_labels(self.graph, self.model, LABEL_MASK, LABEL,
                                         tiling_scheme=self.h.tiling(self.graph), window_size=self.h.window,
                                         test_time_augs=self.h.tta, **kwargs)
        finally:
            self.graph.remove_edge_attr_key(LABEL_MASK)
            self.graph.remove_edge_attr_key(LABEL)
        self._solve()


# --------------------------------------------------------------------------- #
# Ground-truth oracle (for simulating annotators on CTC data)
# --------------------------------------------------------------------------- #
class GTOracle:
    """Answers "is this candidate link correct?" from a CTC ``<seq>_GT/TRA`` folder.

    Each segmentation region gets the GT cell whose marker overlaps it most (CTC
    markers can be small dots, so a region's centroid need not fall inside one),
    and each candidate node the GT cell of its region. A link u->v (t_u < t_v) is
    correct if both are the same GT cell (or a gap continuation of it) and no
    candidate node of that cell lies in between, or if v's cell is a daughter of
    u's cell, u at the mother's last frame and v at the daughter's first.
    """

    def __init__(self, gt_dir, nodes: pd.DataFrame, masks):
        import glob
        import os

        import scipy.ndimage as ndi
        import tifffile

        from .evaluate import chain_ids

        files = sorted(glob.glob(os.path.join(gt_dir, "TRA", "man_track*.tif")))
        self.tracks = {}
        for line in open(os.path.join(gt_dir, "TRA", "man_track.txt")):
            if line.strip():
                i, t0, t1, p = map(int, line.split())
                self.tracks[i] = (t0, t1, p)
        self.chain = chain_ids(self.tracks)
        node_gt = np.zeros(len(nodes), dtype=np.int64)
        for t, idx in nodes.groupby("t").indices.items():
            seg = np.asarray(masks[int(t)])
            gt = np.squeeze(tifffile.imread(files[int(t)]))
            fg = (seg > 0) & (gt > 0)
            region_gt = {}
            if fg.any():
                pairs, n = np.unique(np.stack([seg[fg], gt[fg]]), axis=1, return_counts=True)
                for (sl, gl), c in sorted(zip(pairs.T.tolist(), n), key=lambda z: z[1]):
                    region_gt[sl] = gl  # ascending counts: the largest overlap wins
            labels = [lbl for lbl in np.unique(seg) if lbl]
            cents = np.array(ndi.center_of_mass(seg > 0, seg, labels)) if labels else np.zeros((0, 2))
            ny, nx = nodes["y"].to_numpy()[idx], nodes["x"].to_numpy()[idx]
            for j, (y, x) in zip(idx, zip(ny, nx)):
                if len(cents):
                    k = int(np.argmin(np.hypot(cents[:, 0] - y, cents[:, 1] - x)))
                    node_gt[j] = region_gt.get(labels[k], 0)
        self.nodes = nodes.assign(gt=node_gt, chain=[self.chain.get(int(g), 0) for g in node_gt])
        self.by_id = self.nodes.set_index("node_id")
        self.chain_frames = self.nodes[self.nodes.chain > 0].groupby("chain")["t"].apply(set).to_dict()

    def __call__(self, links: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(links), dtype=bool)
        for k, r in enumerate(links.itertuples()):
            u, v = self.by_id.loc[int(r.source_id)], self.by_id.loc[int(r.target_id)]
            gu, gv = int(u["gt"]), int(v["gt"])
            if not gu or not gv:
                continue
            tu, tv = int(u["t"]), int(v["t"])
            if u["chain"] == v["chain"]:
                between = self.chain_frames.get(u["chain"], set())
                out[k] = not any(tu < t < tv for t in between)
            else:
                parent = self.tracks.get(gv, (0, 0, 0))[2]
                out[k] = parent == gu and self.tracks[gu][1] == tu and self.tracks[gv][0] == tv
        return out
