"""PyTorch dataset for the Celldetective instance-segmentation annotations (immune / cancer cell co-cultures).

Source: Torro et al., "Celldetective, an AI-enhanced image analysis tool for unraveling dynamic cell interactions",
eLife 14, RP105302, https://doi.org/10.7554/eLife.105302. Data: https://doi.org/10.5281/zenodo.17068651
Code: https://github.com/remyeltorro/celldetective

Dataset facts (verified against the files)
------------------------------------------
* Three annotated sets (``SUBSETS``), each a flat folder of ``<name>.tif`` + ``<name>_labelled.tif`` +
  ``<name>.json``; a ``test_set/`` or ``test/`` subfolder holds the authors' test images:
  - ``db_mcf7_nuclei_w_lymphocytes``: ADCC co-culture, MCF-7 nuclei annotated. 114 train + 8 test images,
    5281 + 424 nuclei (3-151 per image).
  - ``db_primary_NK_w_mcf7``: same assay, primary NK cells annotated. 105 images (no test folder), 3798 cells (0-255
    per image; one image has none).
  - ``db_spreading_lymphocytes``: RICM spreading assay, NK / T cells annotated. 125 train + 10 test images,
    1755 + 170 cells (1-57 per image).
* Images: [C, H, W] (adhesion-only RICM images are [H, W]), H and W from 484 to 1628 (mostly 512 x 512 for ADCC,
  1002 x 1004 for RICM). ADCC: uint16 or float32 raw intensities (up to 65535). RICM: float32 with a
  background-normalized adhesion channel (median ~1; the bright-field is raw in 15 images), which has 0.5-2% NaN
  pixels in 78 of 135 images (set to 0 here).
* The json lists the channel names in file order, which varies between images (``brightfield_channel``,
  ``live_nuclei_channel`` Hoechst, ``dead_nuclei_channel`` PI, ``effector_fluo_channel`` CFSE, ``fluo_channel_1``
  a fifth marker, LAMP1 per the record, in 10 MCF-7 / 19 NK images; ``adhesion_channel`` RICM), and
  ``spatial_calibration`` (0.3112 / 0.1568 for ADCC, 0.2 / 0.125 for RICM; um/px in Celldetective). A few ADCC
  jsons add ``cell_type``, ``antibody``, ``concentration``, ``pharmaceutical_agent``.
* ``*_labelled.tif``: int16 [H, W] instance masks (0 = background), same size as the image.

Expected directory layout (what ``download_celldetective.sh`` produces)::

    root/db_mcf7_nuclei_w_lymphocytes/110_snap_0_1_46.tif  .../110_snap_0_1_46_labelled.tif  .../110_snap_0_1_46.json
    root/db_mcf7_nuclei_w_lymphocytes/test_set/...
    root/db_primary_NK_w_mcf7/...
    root/db_spreading_lymphocytes/...  root/db_spreading_lymphocytes/test/...
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

ADCC = ("brightfield_channel", "live_nuclei_channel", "dead_nuclei_channel", "effector_fluo_channel")
SUBSETS = {  # set -> default channel order of the returned image
    "db_mcf7_nuclei_w_lymphocytes": ADCC,
    "db_primary_NK_w_mcf7": ADCC,
    "db_spreading_lymphocytes": ("adhesion_channel", "brightfield_channel"),
}


class CelldetectiveDataset(Dataset):
    """Image-level dataset over one Celldetective segmentation set, with instance masks for Cellpose-SAM.

    Args:
        root: dataset root (see module docstring for the layout).
        subset: one of ``SUBSETS``.
        split: ``"train"`` (top-level images), ``"test"`` (the test subfolder) or ``None`` (both).
        channels: channel names in output order (default ``SUBSETS[subset]``); channels an image lacks are zeros.
        transform: callable ``sample -> sample`` applied last. Image sizes vary: crop (e.g. ``RandomCropFlip(480)``)
            or use ``batch_size=None`` to batch.

    Samples hold ``"image"`` float32 [C, H, W] (raw intensities), ``"target"`` int64 [H, W] instance mask,
    ``"name"`` (path relative to the set folder) and ``"spatial_calibration"`` (um/px).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        subset: str = "db_mcf7_nuclei_w_lymphocytes",
        split: str | None = None,
        channels: Sequence[str] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root = Path(root) / subset
        self.channels = tuple(channels or SUBSETS[subset])
        self.transform = transform
        paths = sorted(p for p in self.root.rglob("*.tif") if not p.stem.endswith("_labelled"))
        is_test = {p: p.parent != self.root for p in paths}
        self.paths = [p for p in paths if split is None or is_test[p] == (split == "test")]
        if not self.paths:
            raise RuntimeError(f"No images found under {self.root} for split {split}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        meta = json.loads(path.with_suffix(".json").read_text())
        image = np.nan_to_num(tifffile.imread(path).astype(np.float32))
        image = image.reshape(-1, *image.shape[-2:])
        out = np.zeros((len(self.channels), *image.shape[1:]), np.float32)
        for i, name in enumerate(meta["channels"]):
            if name in self.channels:
                out[self.channels.index(name)] = image[i]
        mask = tifffile.imread(path.with_name(f"{path.stem}_labelled.tif"))
        sample = {
            "image": torch.from_numpy(out),
            "target": torch.from_numpy(mask.astype(np.int64)),
            "name": path.relative_to(self.root).as_posix(),
            "spatial_calibration": float(meta["spatial_calibration"]),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
