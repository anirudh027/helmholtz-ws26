"""Dataset adapters: every WS26 dataset as time-lapse sequences ``(imgs, masks)`` for ``track()``.

File formats are handled by Florian's frame-level loaders (``dataloaders/``, PR #1);
this module groups their samples into whole sequences of shape (T, Y, X).

    for seq in iter_sequences("alfi", limit=50):
        res = track(seq.imgs, seq.masks, model="hoct")
"""

import glob
import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tifffile

from .ctc import load_sequence

DATA_ROOT = os.environ.get("WS26_DATA", "/mnt/1t/datasets/ws26")


@dataclass
class Sequence:
    dataset: str
    name: str
    imgs: np.ndarray  # (T, Y, X) float32
    masks: np.ndarray | None  # (T, Y, X) instance labels, None = dataset ships no masks
    gt_dir: str | None = None  # CTC <seq>_GT folder, if official metrics are possible
    note: str = ""
    # Partial ground truth, scored with tracking.evaluate.point_metrics:
    # one row per annotated (cell, frame) with columns t, y, x, track, parent.
    gt_points: pd.DataFrame | None = None
    gt_masks: np.ndarray | None = None  # masks of the annotated cells only (QPI), ids consistent in time
    point_radius: float = 10.0  # px tolerance between a GT point and a mask


# Datasets that cannot be tracked as loaded, and why.
NOT_TRACKABLE = {
    "cellx_predict": "64x64 single-cell crops, not full-field time-lapses",
    "celldetective": "single images per sample, no time-lapse in the loader",
    "xenopus": "3 static 3-D stacks (0/10/20 h); the full time-lapses are not fetched",
}


def _from_loader(dataset, ds, key, name_of, limit, masks_from="target", note="", points_of=None, gt_masks_from=None,
                 point_radius=10.0):
    """Group a frame-level loader's samples into sequences (frames in loader order).

    ``points_of(k, samples)`` -> point ground truth (DataFrame t, y, x, track, parent;
    t = index into the sequence) for sequence ``k``."""
    by_seq = {}
    for i, s in enumerate(ds.samples):
        by_seq.setdefault(key(s), []).append(i)
    for k, idxs in by_seq.items():
        idxs = idxs[:limit] if limit else idxs
        samples = [ds[i] for i in idxs]
        imgs = np.stack([np.asarray(x["image"])[0] for x in samples]).astype(np.float32)
        masks = None
        if masks_from and masks_from in samples[0]:
            masks = np.stack([np.asarray(x[masks_from]) for x in samples]).astype(np.int32)
        gt_masks = None
        if gt_masks_from and gt_masks_from in samples[0]:
            gt_masks = np.stack([np.asarray(x[gt_masks_from]) for x in samples]).astype(np.int32)
        points = points_of(k, samples) if points_of else None
        yield Sequence(dataset, name_of(k), imgs, masks, note=note, gt_points=points, gt_masks=gt_masks,
                       point_radius=point_radius)


def _points(rows):
    return pd.DataFrame(rows, columns=["t", "y", "x", "track", "parent"])


def ctc(root, seqs=("01", "02"), dataset="ctc", limit=None, mask_dir=None):
    """Any dataset in CTC layout (``<seq>/t*.tif``, ``<seq>_ST/SEG``, ``<seq>_GT``)."""
    for seq in seqs:
        if not os.path.isdir(os.path.join(root, seq)):
            continue
        imgs, masks = load_sequence(root, seq, limit, mask_dir=mask_dir)
        gt = os.path.join(root, f"{seq}_GT")
        yield Sequence(dataset, seq, imgs.astype(np.float32), masks.astype(np.int32),
                       gt_dir=gt if os.path.isdir(gt) and not limit else None, note="CTC silver-truth masks")


CTC_ROOT = os.environ.get("CTC_ROOT", "/mnt/1t/datasets/ctc")  # <name>/Training/<seq>, as downloaded from the CTC


def musc(limit=None, seqs=("01", "02")):
    root = os.environ.get("MUSC_ROOT", "/mnt/1t/stem_cell_fate_prediction/datasets/BF-C2DL-MuSC/Training")
    yield from ctc(root, seqs, "musc", limit)


def hela(limit=None, seqs=("01", "02")):
    """Fluo-N2DL-HeLa (CTC): fluorescent nuclei, many divisions; second full-GT benchmark."""
    yield from ctc(os.path.join(CTC_ROOT, "Fluo-N2DL-HeLa", "Training"), seqs, "hela", limit)


def hsc(limit=None, seqs=("01", "02")):
    """BF-C2DL-HSC (CTC): hematopoietic stem cells, bright-field, same lab as MuSC."""
    yield from ctc(os.path.join(CTC_ROOT, "BF-C2DL-HSC", "Training"), seqs, "hsc", limit)


def alfi(limit=None, root=None):
    from dataloaders.alfi_dataset import ALFIDataset

    ds = ALFIDataset(root or os.path.join(DATA_ROOT, "alfi"), annotation="dtl")

    def points_of(_, samples):  # box centres; IDs encode the lineage, Parent = mother's ID
        return _points([(t, (b[1] + b[3]) / 2, (b[0] + b[2]) / 2, i, p)
                        for t, x in enumerate(samples)
                        for b, i, p in zip(np.asarray(x["boxes"]), x["ids"], x["parents"])])

    yield from _from_loader("alfi", ds, key=lambda s: s[0], name_of=str, limit=limit, points_of=points_of,
                            note="nucleus masks (MI sequences), connected components; GT = DTL boxes + lineage")


def qpi_cell_death(limit=None, root=None, cell_lines=("LNCaP",)):
    from dataloaders.qpi_cell_death_dataset import QPICellDeathDataset

    ds = QPICellDeathDataset(root or os.path.join(DATA_ROOT, "qpi_cell_death"), cell_lines=cell_lines)
    yield from _from_loader("qpi_cell_death", ds, key=lambda s: s[0],
                            name_of=lambda k: "_".join(map(str, ds.sequences[k])), limit=limit,
                            masks_from=None, gt_masks_from="target",
                            note="no full segmentation: tracked on Cellpose masks, scored on the annotated cells")


def c2c12(limit=None, root=None):
    from dataloaders.c2c12_dataset import C2C12Dataset

    from dataloaders.c2c12_dataset import parse_annotation_xml

    root = root or os.path.join(DATA_ROOT, "c2c12")
    ds = C2C12Dataset(root, annotation="computer", target="instance")

    def points_of(k, samples):
        """Manual (human) tracks of the few annotated lineages, when the field has them."""
        exp, field = ds.sequences[k]
        xml = os.path.join(root, exp, "Annotation_Human", f"Human exp1_{field} Data.xml")
        if not os.path.exists(xml):
            return None
        tr = parse_annotation_xml(xml)
        t_of = {int(x["frame"]): t for t, x in enumerate(samples)}  # 0-based TIFF frame -> sequence index
        keep = np.array([f in t_of for f in tr.frame], dtype=bool)
        tr = tr[keep]
        return _points([(t_of[int(f)], xy[1], xy[0], int(c), int(p))
                        for f, xy, c, p in zip(tr.frame, tr.xy, tr.cell_id, tr.parent_id)])

    yield from _from_loader("c2c12", ds, key=lambda s: s[0],
                            name_of=lambda k: "_".join(ds.sequences[k]), limit=limit, points_of=points_of,
                            note="disk masks around computer-annotated centroids (weak labels); GT = human tracks")


def hemato_fate(limit=None, root=None):
    from dataloaders.hemato_fate_dataset import HematoFateDataset

    ds = HematoFateDataset(root or os.path.join(DATA_ROOT, "hemato_fate"))

    def points_of(_, samples):  # the 2 hand-tracked cells; points can be ~20 px off the centre
        return _points([(t, xy[1], xy[0], c, 0) for t, x in enumerate(samples)
                        for c, xy in zip(ds.cell_ids, np.asarray(x["points"])) if np.isfinite(xy).all()])

    yield from _from_loader("hemato_fate", ds, key=lambda s: s[0], name_of=lambda k: f"pos{k}",
                            limit=limit, masks_from=None, points_of=points_of, point_radius=20.0,
                            note="no masks shipped; GT = 2 hand-tracked cells")


def p53_fate(limit=None, root=None):
    """p53 MNNG1 video: lineage table of every cell (points), no masks.

    The video has the authors' crosses and labels burned in (see the loader), which is
    why zero-shot Cellpose finds only ~6% of the track points: inspect the masks first."""
    from dataloaders.p53_fate_dataset import ANNOTATED, P53FateDataset

    ds = P53FateDataset(root or os.path.join(DATA_ROOT, "p53_fate"), conditions=(ANNOTATED,))
    tab = ds.table  # x, y, time (1-based), event, parent_or_fused_cell, cell, lineage, fused_lineage
    frames = [f for _, f in ds.samples][:limit] if limit else [f for _, f in ds.samples]
    imgs = np.stack([np.asarray(ds[i]["image"])[0] for i in range(len(frames))]).astype(np.float32)
    rows = tab[np.isin(tab[:, 2], frames)]
    # Cell numbers are per lineage, so a cell is (lineage, cell). A daughter's first row
    # (events 31/41/51) carries its mother's cell number in parent_or_fused_cell.
    first = {}
    for r in rows[np.argsort(rows[:, 2], kind="stable")]:
        first.setdefault((int(r[6]), int(r[5])), r)
    parent = {k: (f"{k[0]}:{int(r[4])}" if int(r[3]) in (31, 41, 51) else 0) for k, r in first.items()}
    pts = _points([(int(r[2]) - 1, float(r[1]), float(r[0]), f"{int(r[6])}:{int(r[5])}",
                    parent[(int(r[6]), int(r[5]))]) for r in rows])
    yield Sequence("p53_fate", ANNOTATED, imgs, None, gt_points=pts,
                   note="no masks shipped; overlays burned into the video; GT = lineage table")


def helmholtz(limit=None, root="/mnt/1t/Timelapse/Brightfield_Normalized_TIFF", channel=0, points_csv=None):
    """Our timelapse. ``points_csv``: hand labels (columns t, y, x, track[, parent]),
    e.g. Nikolas' frames 50-51 or the HOCT correction rounds (``tracking.correct``)."""
    files = sorted(glob.glob(os.path.join(root, "t*.tif")))
    files = files[:limit] if limit else files
    imgs = np.stack([tifffile.imread(f)[channel] for f in files]).astype(np.float32)  # plane `channel` of 2
    pts = None
    points_csv = points_csv or os.environ.get("HELMHOLTZ_POINTS")
    if points_csv and os.path.exists(points_csv):
        pts = pd.read_csv(points_csv)
        if "parent" not in pts:
            pts["parent"] = 0
    yield Sequence("helmholtz", "BF" if channel == 0 else f"ch{channel}", imgs, None, gt_points=pts,
                   note="our timelapse; no masks shipped; stage moves between frames (use --register)")


ADAPTERS = {
    "musc": musc,
    "hela": hela,
    "hsc": hsc,
    "alfi": alfi,
    "qpi_cell_death": qpi_cell_death,
    "c2c12": c2c12,
    "hemato_fate": hemato_fate,
    "p53_fate": p53_fate,
    "helmholtz": helmholtz,
}


def iter_sequences(dataset, limit=None, max_seqs=None, **kwargs) -> Iterator[Sequence]:
    """Yield the sequences of ``dataset`` (one of ``ADAPTERS``)."""
    if dataset in NOT_TRACKABLE:
        raise ValueError(f"{dataset} is not trackable: {NOT_TRACKABLE[dataset]}")
    if dataset not in ADAPTERS:
        raise ValueError(f"Unknown dataset {dataset!r}, choose from {list(ADAPTERS)}")
    for i, seq in enumerate(ADAPTERS[dataset](limit=limit, **kwargs)):
        if max_seqs is not None and i >= max_seqs:
            return
        yield seq
