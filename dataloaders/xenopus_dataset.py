"""PyTorch dataset for the segmented Xenopus mucociliary epithelium stacks (Figure 2 source data).

Source: Tolonen et al., "Single-cell morphodynamics predict cell fate decisions during mucociliary epithelial
differentiation", bioRxiv 10.1101/2025.09.17.676780 (2025). Time-lapses: https://doi.org/10.6019/S-BIAD2969,
source data: https://doi.org/10.6019/S-BSST2777

Dataset facts (verified against the files)
------------------------------------------
* Three time points (``0h``, ``10h``, ``20h``) of a live-imaged animal-cap explant (mem-mNeonGreen, H2B-RFP),
  confocal, 0.69 x 0.69 x 2.0 um voxels. Same size as the Exp3 time-lapse of S-BIAD2969.
* ``{tp}.tif``: uint8 [15, 2, 1956, 1956] (Z, C, Y, X), channel 0 membrane (CARE-denoised), channel 1 nuclei. Early
  tilings were canvas-enlarged to the 5 x 5 field, so ``0h`` is mostly zero padding.
* ``{tp}_mem_seg.tif`` / ``{tp}_nuc_seg.tif``: uint16 [15, 1956, 1956] 3D instance labels of all cells (the authors'
  custom Cellpose / StarDist models, not manual): 1676 / 556 (0h), 6468 / 3458 (10h), 8384 / 4335 (20h) objects.

Expected directory layout (what ``download_xenopus.sh`` produces)::

    root/fig2/0h.tif  root/fig2/0h_mem_seg.tif  root/fig2/0h_nuc_seg.tif ...
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

TIMEPOINTS = ("0h", "10h", "20h")
N_SLICES = 15


class XenopusDataset(Dataset):
    """Z-slice dataset over the segmented stacks, with 2D instance masks for Cellpose-SAM.

    Args:
        root: dataset root (see module docstring for the layout).
        timepoints: stacks to use (default: all).
        target: ``"membrane"`` (cells) or ``"nuclei"``.
        transform: callable ``sample -> sample`` applied last (e.g. ``CellposeTransform(RandomCropFlip(256))``).

    Samples hold ``"image"`` float32 [2, 1956, 1956] (membrane, nuclei in [0, 1]), ``"target"`` int64 [1956, 1956]
    (the slice of the 3D labels, ids are shared across slices), ``"timepoint"`` and ``"z"``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        timepoints: Sequence[str] = TIMEPOINTS,
        target: str = "membrane",
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root, self.transform = Path(root) / "fig2", transform
        self.suffix = {"membrane": "mem", "nuclei": "nuc"}[target]
        self.samples = [(tp, z) for tp in timepoints if (self.root / f"{tp}.tif").exists() for z in range(N_SLICES)]
        if not self.samples:
            raise RuntimeError(f"No stacks found under {self.root} for timepoints {timepoints}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        tp, z = self.samples[idx]
        image = tifffile.imread(self.root / f"{tp}.tif", key=[2 * z, 2 * z + 1])  # pages are (z, channel)
        mask = tifffile.imread(self.root / f"{tp}_{self.suffix}_seg.tif", key=z)
        sample = {
            "image": torch.from_numpy(image.astype(np.float32) / 255),
            "target": torch.from_numpy(mask.astype(np.int64)),
            "timepoint": tp,
            "z": z,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
