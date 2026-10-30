"""PyTorch dataset for the p53 cell-fate tracking time-lapses (p53-expressing / p53 RNAi cells, +- MNNG).

Source: Sato, Sato & Rancourt, "Empirical single-cell tracking and cell-fate simulation reveal dual roles of p53 in
tumor suppression", eLife 11, e72498 (2022). Data: https://doi.org/10.5061/dryad.pk0p2ngp5 (Figure 2 videos and
cell-lineage databases; file downloads need a Dryad API token, see ``download_p53_fate.sh``).

Dataset facts (verified against the files)
------------------------------------------
* 8 conditions (``CONDITIONS``), one ``<condition>.mov`` each: H.264, 401 frames, 8-bit RGB, 10 min/frame (the
  burned-in clock runs 0-4,000 min). Frame sizes differ per condition (1078 x 1318 to 1920 x 1174, H x W). The
  80 um scale bar is 152 px long (~0.53 um/px; measured in 7 of 8 videos).
* The videos are the authors' tracking results: the cells are grey (DIC-like), and coloured crosses, lineage numbers,
  the condition / time label and the scale bar are burned in (~3-10% of pixels, more in later frames). Frames are
  converted to greyscale here; the overlay stays visible as grey marks.
* ``<condition>.xlsx`` (cell-lineage database): one sheet, 265,347 rows x 8 int columns ``COLUMNS``. Time is the
  1-based frame index; one row per tracked cell and frame, 335 cells at T=1 up to 1,113 at T=401; 335 lineages.
  Event types (README.pdf): 1 first entry, 2 no event, 6 mitosis, 7 death, 8 out of frame, 11 impartial division,
  31/32 bipolar division (daughter first entry / mother end), 41/42 tripolar, 51/52 tetrapolar, 91/92 cell fusion
  (fused cell end / fused into). Row counts: 2: 261,859, 31: 1,564, 6: 792, 32: 782, 1: 335, 7: 3, 41: 3, 91: 3,
  92: 3, 11: 2, 42: 1 (no 8, 51 or 52).
* **All 8 xlsx files hold the same table** (only the file metadata differs), a Dryad upload error. Its points and
  lineage numbers match the crosses and labels burned into ``MNNG1.mov``, not ``Control.mov`` or the others (checked
  by point-set matching over scale, flips and offsets, and by reading labels). So only MNNG1 is annotated here:
  table (x, y) - ``OFFSET`` = video pixels (to ~1-2 px; 99.8% of points fall inside the 1166 x 1602 frame).
* There are no segmentation masks; Cellpose-SAM can be used zero-shot (see ``examples/p53_fate_cellpose_sam.py``).

Expected directory layout (what ``download_p53_fate.sh`` produces)::

    root/Control.mov  root/Control.xlsx ...
    root/MNNG1.mov    root/MNNG1.xlsx  (the only table used)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openpyxl  # type: ignore[import-untyped,unused-ignore]
import torch
from torch.utils.data import Dataset

CONDITIONS = (
    "Control",
    "MNNG1",
    "MNNG3",
    "MNNG7",
    "p53_RNAi",
    "p53_RNAi-MNNG1",
    "p53_RNAi-MNNG3",
    "p53_RNAi-MNNG7",
)
ANNOTATED = "MNNG1"  # the condition the (duplicated) lineage table belongs to
OFFSET = (155, 469)  # table (x, y) minus OFFSET = pixel in MNNG1.mov
N_FRAMES = 401
COLUMNS = ("x", "y", "time", "event", "parent_or_fused_cell", "cell", "lineage", "fused_lineage")
EVENTS = {
    1: "first entry",
    2: "no event",
    6: "mitosis",
    7: "death",
    8: "out of frame",
    11: "impartial division",
    31: "bipolar division (daughter)",
    32: "bipolar division (mother end)",
    41: "tripolar division (daughter)",
    42: "tripolar division (mother end)",
    51: "tetrapolar division (daughter)",
    52: "tetrapolar division (mother end)",
    91: "fused cell end",
    92: "fused into",
}


def read_lineage_table(path: str | os.PathLike) -> np.ndarray:
    """Read a cell-lineage ``.xlsx`` into an int64 [N, 8] array (``COLUMNS``); x, y shifted to video pixels."""
    rows = openpyxl.load_workbook(path, read_only=True).worksheets[0].iter_rows(min_row=2, values_only=True)
    table = np.array(list(rows), np.int64)
    table[:, :2] -= OFFSET
    return table


def p53_collate(batch: list[dict]) -> dict:
    """Stack images (sizes differ per condition: one condition per batch, or crop); keep the rest as lists."""
    out: dict[str, Any] = {k: [b[k] for b in batch] for k in batch[0]}
    out["image"] = torch.stack(out["image"])
    return out


class P53FateDataset(Dataset):
    """Frame-level dataset over the tracking videos, with the MNNG1 track points.

    Args:
        root: dataset root (see module docstring for the layout).
        conditions: conditions to use (default: all 8).
        frame_step: use every n-th frame.
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [1, H, W] (greyscale in [0, 1]), ``"condition"`` and ``"frame"`` (1-based, 10 min
    apart). MNNG1 samples add, for the K cells tracked in that frame, ``"points"`` float32 [K, 2] (x, y in pixels),
    ``"cell"``, ``"lineage"`` and ``"event"`` int64 [K] (codes in ``EVENTS``). The full table is ``self.table``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        conditions: Sequence[str] = CONDITIONS,
        frame_step: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root, self.transform = Path(root), transform
        self.samples = [(c, t) for c in conditions for t in range(1, N_FRAMES + 1, frame_step)]
        if missing := [c for c in conditions if not (self.root / f"{c}.mov").exists()]:
            raise RuntimeError(f"No videos under {root} for conditions {missing}")
        self.table = read_lineage_table(self.root / f"{ANNOTATED}.xlsx") if ANNOTATED in conditions else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        condition, frame = self.samples[idx]
        cap = cv2.VideoCapture(str(self.root / f"{condition}.mov"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)  # frame-exact for these files
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame} of {condition}.mov")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sample: dict[str, Any] = {"image": torch.from_numpy(image)[None], "condition": condition, "frame": frame}
        if condition == ANNOTATED and self.table is not None:
            rows = self.table[self.table[:, 2] == frame]
            sample["points"] = torch.from_numpy(rows[:, :2].astype(np.float32))
            sample["cell"], sample["lineage"], sample["event"] = (torch.from_numpy(rows[:, i]) for i in (5, 6, 3))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
