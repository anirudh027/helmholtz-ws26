"""Tracking metrics.

* ``evaluate``: official CTC metrics (TRA, DET, LNK, CT, BC, CCA) via py-ctcmetrics,
  for datasets with a complete CTC ``<seq>_GT`` folder (MuSC, HeLa, HSC, ALFI after conversion).
* ``point_metrics``: for datasets whose ground truth is a set of tracked points
  (C2C12, HematoFate, p53, ALFI boxes, hand-labelled Helmholtz frames). Only the
  annotated cells are scored, so sparse annotations give unbiased numbers.
* ``restrict_to_annotated`` + ``masks_to_ctc_gt``: for datasets with masks of
  only some cells (QPI) - drop the tracks of unannotated cells, then use ``evaluate``.

Point ground truth is a DataFrame with one row per (cell, frame):
``t`` (0-based frame), ``y``, ``x`` (pixels), ``track`` (cell id, any hashable) and
``parent`` (mother's ``track`` id, 0 / "0" / NaN for roots). A mother ends the
frame before its daughters start, as in CTC.
"""

import os

import numpy as np
import pandas as pd

from .ctc import CTCResult, centroids_of, tracks_from_assignment, write_ctc

METRICS = ["TRA", "DET", "LNK", "CT", "BC", "CCA"]


def evaluate(res_dir, gt_dir, metrics=METRICS, threads=8) -> dict:
    """Score a CTC RES folder against a CTC ``<seq>_GT`` folder."""
    from ctc_metrics import evaluate_sequence

    m = evaluate_sequence(str(res_dir), str(gt_dir), metrics=list(metrics), threads=threads)
    out = {k: m.get(k) for k in metrics}
    if "BC" in metrics:
        out["BC"] = m.get("BC(1)", m.get("BC"))
    return out


# --------------------------------------------------------------------------- #
# Point ground truth
# --------------------------------------------------------------------------- #
def _is_root(p):
    return p is None or (isinstance(p, float) and np.isnan(p)) or str(p) in ("0", "", "nan")


def chain_ids(tracks) -> dict[int, int]:
    """Map every CTC tracklet to its *logical cell*: a parent with exactly one
    child is a continuation across a gap (same cell), not a division."""
    children = {}
    for i, (_, _, p) in tracks.items():
        if p:
            children.setdefault(p, []).append(i)
    chain = {}
    for i in sorted(tracks, key=lambda k: tracks[k][0]):
        p = tracks[i][2]
        chain[i] = chain.get(p, p) if p and len(children.get(p, [])) == 1 else i
    return chain


def match_points(res: CTCResult, points: pd.DataFrame, radius=10.0) -> np.ndarray:
    """RES track id under each GT point (0 = missed). A point outside every mask
    takes the nearest RES centroid within ``radius`` px (annotated centroids can
    be a few px off the cell, e.g. HematoFate)."""
    out = np.zeros(len(points), dtype=np.int64)
    T, H, W = res.masks.shape
    for t, idx in points.groupby("t").indices.items():
        t = int(t)
        if not 0 <= t < T:
            continue
        m = res.masks[t]
        yy = np.clip(np.round(points["y"].to_numpy()[idx]).astype(int), 0, H - 1)
        xx = np.clip(np.round(points["x"].to_numpy()[idx]).astype(int), 0, W - 1)
        lab = m[yy, xx].astype(np.int64)
        miss = np.flatnonzero(lab == 0)
        if len(miss) and radius > 0:
            cents = centroids_of(m)
            if cents:
                ids = np.fromiter(cents, dtype=np.int64)
                c = np.array([cents[i] for i in ids])
                for j in miss:
                    d = np.hypot(c[:, 0] - yy[j], c[:, 1] - xx[j])
                    k = int(np.argmin(d))
                    if d[k] <= radius:
                        lab[j] = ids[k]
        out[idx] = lab
    return out


def point_metrics(res: CTCResult, points: pd.DataFrame, radius=10.0, div_tol=2) -> dict:
    """Score a tracking result on the annotated cells only.

    Returns:
        det_recall: GT points covered by a RES detection.
        link_acc: consecutive GT points (same cell, frames t and t+1) assigned to the
            same RES logical cell (``chain_ids``); the point analogue of CTC LNK.
        id_switches: consecutive GT pairs both detected but assigned to different cells.
        completeness: per GT cell, longest stretch followed by one RES cell / its length (mean).
        survival: per GT cell, frames followed from its first point until the first
            break (mean fraction of its length); "how long until we lose it".
        div_recall: GT divisions whose two daughters are RES daughters of the RES cell
            holding the GT mother (within ``div_tol`` frames of the event).
        div_precision_annot: among RES divisions of cells that hold a GT point at the
            mother's last frame, the fraction that are GT divisions there.
    """
    pts = points.sort_values(["track", "t"]).reset_index(drop=True)
    pts["track"] = pts["track"].astype(str)
    pts["parent"] = pts["parent"].map(lambda p: "0" if _is_root(p) else str(p))
    pts["res"] = match_points(res, pts, radius)
    chain = chain_ids(res.tracks)
    pts["cell"] = pts["res"].map(lambda r: chain.get(int(r), 0) if r else 0)
    # duplicated (cell, frame) rows are annotation noise: keep the first
    pts = pts.drop_duplicates(["track", "t"]).reset_index(drop=True)

    pairs = ok = switches = 0
    complete, surv = [], []
    for _, g in pts.groupby("track", sort=False):
        t, cell = g["t"].to_numpy(), g["cell"].to_numpy()
        consec = np.diff(t) == 1
        same = (cell[:-1] == cell[1:]) & (cell[:-1] > 0)
        both = (cell[:-1] > 0) & (cell[1:] > 0)
        pairs += int(consec.sum())
        ok += int((consec & same).sum())
        switches += int((consec & both & ~same).sum())
        # longest run of the same RES cell
        best = run = 1 if cell[0] > 0 else 0
        for k in range(1, len(cell)):
            run = run + 1 if (consec[k - 1] and same[k - 1]) else (1 if cell[k] > 0 else 0)
            best = max(best, run)
        complete.append(best / len(cell))
        breaks = np.flatnonzero(~(consec & same))
        surv.append(((breaks[0] + 1) if len(breaks) else len(cell)) / len(cell) if cell[0] > 0 else 0.0)

    # divisions
    children = {}
    for i, (_, _, p) in res.tracks.items():
        if p:
            children.setdefault(p, []).append(i)
    res_div = {p: set(c) for p, c in children.items() if len(c) >= 2}
    by_track = {k: g for k, g in pts.groupby("track", sort=False)}
    gt_div = {}
    for k, g in by_track.items():
        # first non-root parent: some annotations (ALFI) leave it out of the daughter's first row
        par = next((p for p in g["parent"] if p != "0"), "0")
        if par != "0" and par in by_track:
            gt_div.setdefault(par, []).append(k)
    gt_div = {m: d for m, d in gt_div.items() if len(d) >= 2}

    def mother_tracks(g):
        """RES tracklets under the mother's last ``div_tol`` points."""
        tail = g[g["t"] >= g["t"].max() - div_tol]
        return {int(r) for r in tail["res"] if r}

    def daughter_tracks(g):
        head = g[g["t"] <= g["t"].min() + div_tol]
        return {int(r) for r in head["res"] if r}

    hit = 0
    for m, ds in gt_div.items():
        mt = mother_tracks(by_track[m])
        found = 0
        for d in ds[:2]:
            dt = daughter_tracks(by_track[d])
            if any(res.tracks.get(r, (0, 0, 0))[2] in mt and res.tracks[r][2] in res_div for r in dt):
                found += 1
        hit += found == 2
    # precision among annotated cells: RES divisions whose mother holds a GT point at its last frame
    gt_mother_last = {(int(by_track[m]["t"].max()), m) for m in gt_div}
    annotated_last = {}
    for r in pts.itertuples():
        if r.res:
            annotated_last.setdefault((int(r.t), int(r.res)), r.track)
    n_res_div = n_res_div_ok = 0
    for p in res_div:
        t_end = res.tracks[p][1]
        gt_cell = next((annotated_last[(t, p)] for t in range(t_end, t_end - div_tol - 1, -1)
                        if (t, p) in annotated_last), None)
        if gt_cell is None:
            continue
        n_res_div += 1
        n_res_div_ok += any(m == gt_cell and abs(tm - t_end) <= div_tol for tm, m in gt_mother_last)

    return {
        "gt_cells": int(pts["track"].nunique()),
        "gt_points": int(len(pts)),
        "det_recall": float((pts["res"] > 0).mean()) if len(pts) else np.nan,
        "link_acc": ok / pairs if pairs else np.nan,
        "id_switches": switches,
        "completeness": float(np.mean(complete)) if complete else np.nan,
        "survival": float(np.mean(surv)) if surv else np.nan,
        "gt_divisions": len(gt_div),
        "div_recall": hit / len(gt_div) if gt_div else np.nan,
        "div_precision_annot": n_res_div_ok / n_res_div if n_res_div else np.nan,
    }


def ctc_gt_points(gt_dir) -> pd.DataFrame:
    """Turn a CTC ``<seq>_GT/TRA`` folder into point ground truth (marker centroids);
    used to check ``point_metrics`` against the official metrics."""
    import glob

    import tifffile

    tra = os.path.join(gt_dir, "TRA")
    files = sorted(glob.glob(os.path.join(tra, "man_track*.tif")))
    lineage = {}
    for line in open(os.path.join(tra, "man_track.txt")):
        if line.strip():
            i, _, _, p = map(int, line.split())
            lineage[i] = p
    rows = []
    for f in files:
        t = int("".join(c for c in os.path.basename(f) if c.isdigit()))
        for lbl, (y, x) in centroids_of(np.squeeze(tifffile.imread(f))).items():
            rows.append((t, y, x, lbl, lineage.get(lbl, 0)))
    return pd.DataFrame(rows, columns=["t", "y", "x", "track", "parent"])


# --------------------------------------------------------------------------- #
# Sparse mask ground truth (QPI)
# --------------------------------------------------------------------------- #
def masks_to_ctc_gt(gt_dir, masks, parents=None):
    """Write id-consistent masks (a cell keeps its label across frames, e.g. QPI's
    annotated cells) as a CTC ``<seq>_GT`` folder (TRA + SEG)."""
    frame_map = [{int(i): int(i) for i in np.unique(m) if i} for m in masks]
    gt = tracks_from_assignment(masks, frame_map, parents or {})
    tra = os.path.join(gt_dir, "TRA")
    write_ctc(tra, gt.masks, gt.tracks)
    nd = max(3, len(str(len(masks) - 1)))
    for t in range(len(masks)):
        os.replace(os.path.join(tra, f"mask{t:0{nd}d}.tif"), os.path.join(tra, f"man_track{t:0{nd}d}.tif"))
    os.replace(os.path.join(tra, "res_track.txt"), os.path.join(tra, "man_track.txt"))
    seg = os.path.join(gt_dir, "SEG")
    os.makedirs(seg, exist_ok=True)
    import shutil

    for t in range(len(masks)):
        shutil.copy(os.path.join(tra, f"man_track{t:0{nd}d}.tif"), os.path.join(seg, f"man_seg{t:0{nd}d}.tif"))
    return gt


def restrict_to_annotated(res: CTCResult, gt_masks, min_iou=0.3) -> CTCResult:
    """Keep only the RES tracks that match an annotated cell (IoU >= ``min_iou``
    in at least one frame), so unannotated cells are not counted as false positives."""
    keep = set()
    for t, (r, g) in enumerate(zip(res.masks, gt_masks)):
        fg = (r > 0) & (g > 0)
        if not fg.any():
            continue
        pair, inter = np.unique(np.stack([r[fg], g[fg]]), axis=1, return_counts=True)
        ra = dict(zip(*np.unique(r[r > 0], return_counts=True)))
        ga = dict(zip(*np.unique(g[g > 0], return_counts=True)))
        for (ri, gi), n in zip(pair.T, inter):
            if n / (ra[ri] + ga[gi] - n) >= min_iou:
                keep.add(int(ri))
    masks = np.where(np.isin(res.masks, list(keep)), res.masks, 0).astype(np.uint16)
    tracks = {i: (t0, t1, p if p in keep else 0) for i, (t0, t1, p) in res.tracks.items() if i in keep}
    return CTCResult(masks, tracks, dict(res.info))
