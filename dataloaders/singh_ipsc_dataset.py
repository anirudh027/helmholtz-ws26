"""PyTorch dataset for the Singh et al. iPSC reprogramming time-lapse (colony instance masks with fate labels).

Source: Singh et al., "Towards Early Prediction of Human iPSC Reprogramming Success", MELBA (2023),
https://arxiv.org/abs/2305.14575. Code: https://github.com/abhineet123/ipsc_prediction. Data: Hugging Face model repo
``abhineet123/ipsc_prediction``, folder ``ipsc_data`` (``roi_images_and_labels.zip``, 5.3 GB, is all this loader needs;
the COCO / YTVIS JSONs there are conversions of the same annotations).

Dataset facts (verified against the files)
------------------------------------------
* One well (``well3``), cropped into 31 ROIs (``all_frames_roi_*`` / ``roi_*``, named after their ``x1_y1_x2_y2`` in
  the 714 MP well image), 1700 x 900 to 4333 x 3833 px (W x H). Each ROI has the same 127 frames ``image146.jpg`` ..
  ``image275.jpg`` (146-275 without 155, 186, 188); frame index 0..126 here is their rank, as in the authors' frame
  ranges (``g2_0_53`` = early, ``g2_54_126`` = late).
* Images are 8-bit JPEG, RGB with R = G = B: read as greyscale.
* ``annotations.csv`` per ROI (23866 rows in all): ``target_id, filename, width, height, class, xmin, ymin, xmax, ymax,
  mask``, one row per colony and frame. ``mask`` is the colony contour as ``"x,y;x,y;..."``. ``target_id`` is a track
  id that is stable across frames (455 tracks, 2-44 per ROI). Classes: ``ipsc`` (6049 rows), ``diff`` (16509) and
  ``cell`` (1308, in 12 ROIs); the class is constant along each track (the colony's fate).
* The authors' COCO JSONs and ``masks/*.png`` (palette: 0 background, 1 ipsc, 2 diff) leave out ``cell``. Filling the
  CSV contours reproduces those PNGs (foreground IoU 1.0 in most checked frames, >= 0.84 elsewhere); colonies overlap
  in < 0.2% of their pixels (later colonies are drawn on top).
* Some ROIs hold older annotation folders (``annotations_1``, ``annotations__*``, ``annotations_tracked_final_*``,
  ``labels``) and per-range ``annotations_<a>_<b>.csv`` subsets; they are not used.

Expected directory layout (what ``download_singh_ipsc.sh`` produces)::

    root/ipsc/well3/all_frames_roi/<roi>/image146.jpg ... image275.jpg
    root/ipsc/well3/all_frames_roi/<roi>/annotations.csv
    root/ipsc/well3/all_frames_roi/<roi>/masks/image146.png ...  (semantic, unused)
"""

from __future__ import annotations

import csv
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ("ipsc", "diff", "cell")
N_FRAMES = 127


def read_annotations(path: str | os.PathLike) -> dict[str, list[dict[str, str]]]:
    """Read an ROI's ``annotations.csv`` into ``{file name: [row dict, ...]}``."""
    csv.field_size_limit(sys.maxsize)  # contours are long
    rows: dict[str, list[dict[str, str]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["filename"], []).append(row)
    return rows


def contour(mask: str) -> np.ndarray:
    """Parse a ``"x,y;x,y;..."`` contour into an int32 [K, 2] (x, y) array."""
    return np.array([p.split(",") for p in mask.strip(";").split(";")], np.float32).round().astype(np.int32)


class SinghIPSCDataset(Dataset):
    """Frame-level dataset over the 31 iPSC reprogramming ROIs, with colony instance masks and fate labels.

    Args:
        root: dataset root (see module docstring for the layout).
        rois: ROI folder names to use (default: all).
        frames: frame indices 0..126 to use (default: all; the authors' early stage is ``range(54)``).
        classes: colony classes to keep (default: ``ipsc`` and ``diff``, like the authors; add ``cell`` for all).
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [1, H, W] in [0, 1], ``"roi"``, ``"frame"`` (index 0..126), ``"target"`` int64
    [H, W] (colony instances 1..N, in the order of the boxes), ``"boxes"`` float32 [N, 4] (xmin, ymin, xmax, ymax),
    ``"labels"`` int64 [N] (index into ``CLASSES``) and ``"ids"`` int64 [N] (track ids, stable across frames). Sizes
    differ per ROI: batch with ``singh_collate`` (one ROI per batch, or crop).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        rois: Sequence[str] | None = None,
        frames: Iterable[int] = range(N_FRAMES),
        classes: Sequence[str] = ("ipsc", "diff"),
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root, self.classes, self.transform = Path(root) / "ipsc" / "well3" / "all_frames_roi", classes, transform
        self.rois = sorted(rois or (p.parent.name for p in self.root.glob("*/annotations.csv")))
        if not self.rois:
            raise RuntimeError(f"No ROIs with annotations.csv under {self.root}")
        by_number = lambda name: int(name[5:-4])  # noqa: E731 - "image146.jpg" -> 146
        self.files = {r: sorted((p.name for p in (self.root / r).glob("*.jpg")), key=by_number) for r in self.rois}
        if bad := [r for r, f in self.files.items() if len(f) != N_FRAMES]:
            raise RuntimeError(f"Expected {N_FRAMES} frames in {bad}")
        self.rows = {r: read_annotations(self.root / r / "annotations.csv") for r in self.rois}
        self.samples = [(r, t) for r in self.rois for t in frames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        roi, frame = self.samples[idx]
        name = self.files[roi][frame]
        image = np.asarray(Image.open(self.root / roi / name).convert("L"), np.float32) / 255
        rows = [r for r in self.rows[roi].get(name, []) if r["class"] in self.classes]
        target = np.zeros(image.shape, np.int32)  # cv2 cannot draw into int64
        for i, r in enumerate(rows, 1):
            cv2.fillPoly(target, [contour(r["mask"])], i)
        boxes = np.array([[float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")] for r in rows], np.float32)
        sample = {
            "image": torch.from_numpy(image)[None],
            "roi": roi,
            "frame": frame,
            "target": torch.from_numpy(target.astype(np.int64)),
            "boxes": torch.from_numpy(boxes.reshape(-1, 4)),
            "labels": torch.tensor([CLASSES.index(r["class"]) for r in rows], dtype=torch.int64),
            "ids": torch.tensor([int(r["target_id"]) for r in rows], dtype=torch.int64),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def singh_collate(batch: list[dict]) -> dict:
    """Stack images and masks (sizes differ per ROI: one ROI per batch, or crop); keep the rest as lists."""
    out: dict[str, Any] = {k: [b[k] for b in batch] for k in batch[0]}
    for k in ("image", "target"):
        out[k] = torch.stack(out[k])
    return out
