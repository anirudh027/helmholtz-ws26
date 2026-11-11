"""PyTorch dataset for the Leger et al. RPE1 Fucci cell-cycle tracks (bright-field + H2B crops, Fucci targets).

Source: Leger, Leonardi, Salati, Naef & Weigert, "Sequence models for continuous cell cycle stage prediction from
brightfield images", MIDL 2025, PMLR 301 (https://arxiv.org/abs/2502.02182). Data: https://zenodo.org/records/14987478
(``train.zip`` 16.6 GB, ``test.zip`` 1.2 GB, ``test_drugged.zip`` 0.46 GB; ``models.zip`` is not needed).

Dataset facts (verified against the files)
------------------------------------------
* Human RPE1 cells with Fucci reporters, 20x widefield (0.598 um/px), 5 min/frame (from the record / paper). Each
  track follows one nucleus from mitosis to mitosis (M-M), as 64 x 64 crops centred on the nucleus.
* ``<split>/images/<track>.npy``: float16 [T, 2, 64, 64], channels ``CHANNELS`` (0 bright-field, 1 H2B: the bright
  nucleus is in the centre of channel 1). Every crop is z-scored per channel (mean 0, std 1).
* ``<split>/labels/<track>.npy``: float16 [2, T, 1], the Fucci intensities ``FUCCI`` of the nucleus per frame,
  log-transformed by the authors and min-max scaled to [0, 1] per track. Order checked on the test tracks: label 1
  is high in the first half of the cycle and drops at ~60% of it (Cdt1, degraded in S), label 0 rises from then on
  (Geminin).
* Track names are ``<experiment>.<id>``. ``test``: 358 tracks of experiment 0303, 85-482 frames (mean 244, 87,424
  frames). ``test_drugged``: 73 tracks (Palbociclib-treated, out of distribution) of 0304 / 0305, 271-601 frames
  (mean 457, 33,365 frames). ``train`` (5,188 tracks, mean ~230 frames per the paper) was not downloaded here.
* There are no segmentation masks; Cellpose-SAM can be used zero-shot on the H2B channel
  (see ``examples/leger_fucci_cellpose_sam.py``).

Expected directory layout (what ``download_leger_fucci.sh`` produces)::

    root/test/images/<track>.npy  root/test/labels/<track>.npy
    root/test_drugged/images/...  root/test_drugged/labels/...
    root/train/images/...         root/train/labels/...           (only if downloaded)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "test", "test_drugged")
CHANNELS = ("brightfield", "H2B")
FUCCI = ("Geminin", "Cdt1")


class LegerFucciDataset(Dataset):
    """Frames, fixed-length windows or whole tracks of the RPE1 Fucci M-M tracks.

    Args:
        root: dataset root (see module docstring for the layout).
        split: ``"train"``, ``"test"`` or ``"test_drugged"``.
        seq_len: ``None`` for single frames, ``k`` for non-overlapping windows of ``k`` frames (the remainder of each
            track is dropped), ``-1`` for whole tracks (lengths differ: batch with ``leger_collate``).
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [2, 64, 64] (single frames) or [T, 2, 64, 64] (z-scored, ``CHANNELS``),
    ``"fucci"`` float32 [2] or [T, 2] (``FUCCI``, in [0, 1]), ``"track"`` (file stem), ``"frame"`` (first frame) and
    ``"length"`` (frames in the whole track). Tracks are memory-mapped, so ``train`` does not need to fit in RAM.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "test",
        seq_len: int | None = None,
        transform: Callable[[dict], dict] | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        self.root, self.seq_len, self.transform = Path(root) / split, seq_len, transform
        self.tracks = sorted(p.stem for p in (self.root / "images").glob("*.npy"))
        if not self.tracks:
            raise RuntimeError(f"No tracks under {self.root / 'images'}")
        self.lengths = [np.load(self.root / "images" / f"{t}.npy", mmap_mode="r").shape[0] for t in self.tracks]
        if seq_len == -1:
            self.samples = [(i, 0) for i in range(len(self.tracks))]  # (track index, first frame)
        else:
            k = seq_len or 1
            self.samples = [(i, s) for i, n in enumerate(self.lengths) for s in range(0, n - k + 1, k)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        i, start = self.samples[idx]
        track, n = self.tracks[i], self.lengths[i]
        stop = n if self.seq_len == -1 else start + (self.seq_len or 1)
        image = np.load(self.root / "images" / f"{track}.npy", mmap_mode="r")[start:stop].astype(np.float32)
        fucci = np.load(self.root / "labels" / f"{track}.npy", mmap_mode="r")[:, start:stop, 0].T.astype(np.float32)
        if self.seq_len is None:
            image, fucci = image[0], fucci[0]
        sample: dict[str, Any] = {
            "image": torch.from_numpy(image),
            "fucci": torch.from_numpy(fucci),
            "track": track,
            "frame": start,
            "length": n,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def leger_collate(batch: list[dict]) -> dict:
    """Stack equal-length samples; whole tracks (different lengths) stay lists of tensors."""
    out: dict[str, Any] = {k: [b[k] for b in batch] for k in batch[0]}
    for k in ("image", "fucci"):
        if len({x.shape for x in out[k]}) == 1:
            out[k] = torch.stack(out[k])
    return out
