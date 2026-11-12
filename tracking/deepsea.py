"""DeepSea (Zargari et al. 2023) annotated time-lapses -> CTC layout.

DeepSea ships one folder per sequence::

    <set>/images/<frame>.png
    <set>/masks/<frame>_cell_area_masked.png    binary foreground, not instance labels
    <set>/wmaps/<frame>_cell_area_masked.png    optional: edges between touching cells
    <set>/labels/<frame>_cell_pos_labels.txt    header line, then ``label<TAB>x<TAB>y`` per cell

A cell keeps its label over time; at a division the daughters are named ``<mother>_1`` and
``<mother>_2``. ``deepsea_to_ctc`` turns one such folder into ``<seq>/``, ``<seq>_ST/SEG`` and
``<seq>_GT/{TRA,SEG}``, so it runs through ``track()`` and the official CTC metrics like MuSC.

The layout is read from DeepSea's own loader (github.com/abzargar/DeepSea, ``deepsea/data.py``).
"""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
import tifffile

from .ctc import write_ctc_input

MASK_SUFFIX = "_cell_area_masked"
LABEL_SUFFIX = "_cell_pos_labels"
Tracks = dict[int, tuple[int, int, int]]  # track id -> (first frame, last frame, parent id)


def read_labels(path: str | Path) -> list[tuple[str, float, float]]:
    """``[(label, x, y)]`` of one frame; the first line of the file is a header."""
    cells = []
    for line in Path(path).read_text().split("\n")[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            cells.append((parts[0], float(parts[1]), float(parts[2])))
    return cells


def instance_mask(
    foreground: np.ndarray, cells: list[tuple[int, float, float]], tol: float = 5.0
) -> tuple[np.ndarray, list[int], int]:
    """Give every foreground component the id of the annotated cell whose centroid lies in it.

    ``cells`` is ``[(track id, x, y)]``. A centroid that misses the foreground by at most ``tol``
    pixels takes the nearest component. A component holding several centroids (touching cells) is
    split by nearest centroid. Returns the mask, the ids that found no component, and the number of
    components without an annotated cell (these are dropped).
    """
    comps, n_comps = ndi.label(foreground)
    out = np.zeros(foreground.shape, np.uint16)
    if not n_comps:
        return out, [tid for tid, _, _ in cells], 0
    dist, (near_y, near_x) = ndi.distance_transform_edt(comps == 0, return_indices=True)
    members: dict[int, list[tuple[int, float, float]]] = {}
    missing = []
    for tid, x, y in cells:
        r = min(max(int(round(y)), 0), foreground.shape[0] - 1)
        c = min(max(int(round(x)), 0), foreground.shape[1] - 1)
        if dist[r, c] > tol:
            missing.append(tid)
            continue
        members.setdefault(int(comps[near_y[r, c], near_x[r, c]]), []).append((tid, x, y))
    for comp, cells_in in members.items():
        ys, xs = np.nonzero(comps == comp)
        d = [(ys - y) ** 2 + (xs - x) ** 2 for _, x, y in cells_in]
        out[ys, xs] = np.array([tid for tid, _, _ in cells_in], np.uint16)[np.argmin(d, axis=0)]
    present = set(np.unique(out).tolist())
    missing += [tid for tid, _, _ in cells if tid not in present and tid not in missing]
    return out, missing, n_comps - len(members)


def _companion(folder: Path, stem: str, suffix: str) -> Path | None:
    hits = sorted(folder.glob(f"{stem}{suffix}.*")) if folder.is_dir() else []
    return hits[0] if hits else None


def read_deepsea(set_dir: str | Path, limit: int | None = None, tol: float = 5.0):
    """One DeepSea sequence as ``(imgs, masks, tracks, report)``.

    ``imgs`` and ``masks`` are (T, Y, X); ``masks`` holds track ids. ``tracks`` follows CTC
    ``man_track.txt``. ``report`` counts what could not be converted.
    """
    set_dir = Path(set_dir)
    frames = sorted(p for p in (set_dir / "images").iterdir() if not p.name.startswith("."))
    frames = frames[:limit] if limit else frames
    ids: dict[str, int] = {}
    imgs, masks = [], []
    report = {"frames": len(frames), "cells_without_mask": 0, "components_without_label": 0, "links_dropped": 0}
    for frame in frames:
        img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        mask_file = _companion(set_dir / "masks", frame.stem, MASK_SUFFIX)
        label_file = _companion(set_dir / "labels", frame.stem, LABEL_SUFFIX)
        if mask_file is None or label_file is None:
            raise FileNotFoundError(f"{set_dir.name}: no mask or label file for {frame.name}")
        foreground = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
        wmap_file = _companion(set_dir / "wmaps", frame.stem, MASK_SUFFIX)
        if wmap_file is not None:  # DeepSea removes the touching-cell edges before labelling components
            foreground &= cv2.imread(str(wmap_file), cv2.IMREAD_GRAYSCALE) == 0
        sy, sx = foreground.shape[0] / img.shape[0], foreground.shape[1] / img.shape[1]
        cells = [(ids.setdefault(lbl, len(ids) + 1), x * sx, y * sy) for lbl, x, y in read_labels(label_file)]
        mask, missing, unlabelled = instance_mask(foreground, cells, tol)
        report["cells_without_mask"] += len(missing)
        report["components_without_label"] += unlabelled
        if mask.shape != img.shape:
            mask = cv2.resize(mask, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
        imgs.append(img)
        masks.append(mask)

    span: dict[int, list[int]] = {}
    for t, mask in enumerate(masks):
        for tid in np.unique(mask)[1:].tolist():
            span.setdefault(tid, [t, t])[1] = t
    tracks: Tracks = {}
    for label, tid in ids.items():
        if tid not in span:
            continue
        mother = ids.get(re.sub(r"_\d+$", "", label), 0) if re.search(r"_\d+$", label) else 0
        if mother and (mother not in span or span[mother][1] >= span[tid][0]):
            report["links_dropped"] += 1  # CTC needs the mother to end before the daughter starts
            mother = 0
        tracks[tid] = (span[tid][0], span[tid][1], mother)
    report["tracks"] = len(tracks)
    report["divisions"] = sum(1 for m in {p for _, _, p in tracks.values() if p})
    return np.stack(imgs), np.stack(masks), tracks, report


def deepsea_to_ctc(set_dir: str | Path, out_dir: str | Path, seq: str, limit: int | None = None) -> dict:
    """Convert one DeepSea sequence into ``out_dir/<seq>``, ``<seq>_ST/SEG`` and ``<seq>_GT/{TRA,SEG}``."""
    imgs, masks, tracks, report = read_deepsea(set_dir, limit)
    write_ctc_input(str(out_dir), seq, imgs, masks)
    ndigits = max(3, len(str(len(masks) - 1)))
    for sub, prefix in (("TRA", "man_track"), ("SEG", "man_seg")):
        folder = os.path.join(out_dir, f"{seq}_GT", sub)
        os.makedirs(folder, exist_ok=True)
        for t, mask in enumerate(masks):
            tifffile.imwrite(os.path.join(folder, f"{prefix}{t:0{ndigits}d}.tif"), mask, compression="zlib")
    with open(os.path.join(out_dir, f"{seq}_GT", "TRA", "man_track.txt"), "w") as fh:
        for tid in sorted(tracks):
            fh.write("{} {} {} {}\n".format(tid, *tracks[tid]))
    return report
