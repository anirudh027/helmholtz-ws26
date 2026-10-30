"""PyTorch dataset for ALFI (Annotations for Label-Free Images): DIC time-lapses of cultured human cells.

Source: Antonelli et al., "ALFI: Cell cycle phenotype annotations of label-free time-lapse imaging data from cultured
human cells", Sci. Data 10, 677 (2023). Data: https://doi.org/10.6084/m9.figshare.23798451 (file 41740227,
``ALFIdatasetFinal.zip``: PNG frames + annotations, the raw ND2 videos and the authors' use examples).

Dataset facts (verified against the files)
------------------------------------------
* 29 sequences, 2366 frames: MI01-MI08 (796 frames), CD01-CD09 (603), TP01-TP12 (967); 16 to 253 frames each.
  ND2 metadata: DIC objective 60x (MI01-05, 0.1766 um/px) or 40x (others, 0.2631 or 0.3707 um/px);
  7 min/frame (MI, TP11-12) or 5 min (CD, TP01-10).
* ``Images/I_<seq>_<frame>.png``: 1024 x 1280 16-bit greyscale, 12-bit data shifted left by 4 (values 0..65520 in
  steps of 16). Frames are numbered 1..T without gaps.
* ``Masks/M_<seq>_<frame>.png`` (MI only, one per frame): 8-bit semantic masks, 0 background, 128 Interphase,
  255 Mitosis, no instance ids. They outline the nucleus of interphase cells and the whole rounded cell in mitosis
  (checked visually), not the full cell body. The connected components of each class give exactly one instance per
  DTL box in 793 of 796 frames (with plain ``mask > 0`` touching interphase / mitotic cells merge: 734 of 796).
* ``<seq>_DTLTruth.csv`` (Task 1, MI only): ``ImNo, ID, Class, xmin, ymin, width, height, Parent``, one row per cell
  and frame; 16627 boxes (15034 Interphase, 1593 Mitosis). IDs are strings that encode the lineage (``7`` divides into
  ``7.1`` / ``7.2``, ``7.2`` into ``7.21`` / ``7.22``); ``Parent`` is the mother's ID or ``0``.
* ``<seq>_PhenoTruth.csv`` (Task 2, all sequences): ``ImNo, ID, Class, xmin, ymin, width, height`` for the cells in an
  annotated phase only: 7518 boxes (5784 EarlyMitosis, 840 LateMitosis, 730 CellDeath, 164 Multipolar).
  Multipolar occurs in TP only. Many frames have no box.

Expected directory layout (what ``download_alfi.sh`` produces)::

    root/Data&Annotations/MI01/Images/I_MI01_0001.png ...
    root/Data&Annotations/MI01/Masks/M_MI01_0001.png ...        (MI only)
    root/Data&Annotations/MI01/MI01_DTLTruth.csv                (MI only)
    root/Data&Annotations/MI01/MI01_PhenoTruth.csv
    root/Videos/MI01.nd2 ...                                    (raw videos, unused)
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped,unused-ignore]
from torch.utils.data import Dataset

SEQUENCES = tuple(f"{p}{i:02d}" for p, n in (("MI", 8), ("CD", 9), ("TP", 12)) for i in range(1, n + 1))
CLASSES = {
    "dtl": ("Interphase", "Mitosis"),
    "pheno": ("EarlyMitosis", "LateMitosis", "CellDeath", "Multipolar"),
}


def read_annotations(path: str | os.PathLike) -> dict[int, list[dict[str, str]]]:
    """Read a ``*_DTLTruth.csv`` / ``*_PhenoTruth.csv`` into ``{frame: [row dict, ...]}``."""
    rows: dict[int, list[dict[str, str]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(int(row["ImNo"]), []).append(row)
    return rows


class ALFIDataset(Dataset):
    """Frame-level dataset over the ALFI sequences, with bounding boxes and (``annotation="dtl"``) instance masks.

    Args:
        root: dataset root (see module docstring for the layout).
        sequences: sequences to use (default: all that have the chosen annotation).
        annotation: ``"pheno"`` (Task 2: boxes of the cells in EarlyMitosis / LateMitosis / CellDeath / Multipolar)
            or ``"dtl"`` (Task 1, MI only: boxes of all cells as Interphase / Mitosis, lineage, instance masks).
        frame_step: use every n-th frame.
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [1, 1024, 1280] in [0, 1], ``"sequence"``, ``"frame"`` (1-based, as in the file
    names), ``"boxes"`` float32 [N, 4] (xmin, ymin, xmax, ymax), ``"labels"`` int64 [N] (index into
    ``CLASSES[annotation]``) and ``"ids"`` (N cell ID strings). ``"dtl"`` adds ``"parents"`` (N ID strings, ``"0"`` =
    none) and ``"target"`` int64 [1024, 1280], the connected components of each mask class as instances. Batch with
    ``alfi_collate``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        sequences: Sequence[str] = SEQUENCES,
        annotation: str = "pheno",
        frame_step: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root, self.annotation, self.transform = Path(root) / "Data&Annotations", annotation, transform
        suffix = {"dtl": "DTLTruth", "pheno": "PhenoTruth"}[annotation]
        self.sequences = [s for s in sequences if (self.root / s / f"{s}_{suffix}.csv").exists()]
        self.rows = {s: read_annotations(self.root / s / f"{s}_{suffix}.csv") for s in self.sequences}
        self.samples = [
            (s, t)
            for s in self.sequences
            for t in range(1, len(list((self.root / s / "Images").glob("*.png"))) + 1, frame_step)
        ]
        if not self.samples:
            raise RuntimeError(f"No {suffix} sequences found under {self.root} for {sequences}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        seq, frame = self.samples[idx]
        image = np.asarray(Image.open(self.root / seq / "Images" / f"I_{seq}_{frame:04d}.png"), np.float32) / 65535
        rows = self.rows[seq].get(frame, [])
        xywh = np.array([[float(r[k]) for k in ("xmin", "ymin", "width", "height")] for r in rows], np.float32)
        xywh = xywh.reshape(-1, 4)
        sample = {
            "image": torch.from_numpy(image)[None],
            "sequence": seq,
            "frame": frame,
            "boxes": torch.from_numpy(np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)),
            "labels": torch.tensor([CLASSES[self.annotation].index(r["Class"]) for r in rows], dtype=torch.int64),
            "ids": [r["ID"] for r in rows],
        }
        if self.annotation == "dtl":
            sample["parents"] = [r["Parent"] for r in rows]
            mask = np.asarray(Image.open(self.root / seq / "Masks" / f"M_{seq}_{frame:04d}.png"))
            interphase, n = ndimage.label(mask == 128)
            mitosis = ndimage.label(mask == 255)[0]
            sample["target"] = torch.from_numpy(np.where(mitosis > 0, mitosis + n, interphase).astype(np.int64))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def alfi_collate(batch: list[dict]) -> dict:
    """Stack images (and masks); keep the per-box tensors and metadata as lists."""
    out: dict[str, Any] = {k: [b[k] for b in batch] for k in batch[0]}
    for k in ("image", "target"):
        if k in out:
            out[k] = torch.stack(out[k])
    return out
