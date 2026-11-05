"""PyTorch dataset for the quantitative phase imaging (QPI) cell-death time-lapses.

Source: Vicar et al., "The Quantitative-Phase Dynamics of Apoptosis and Lytic Cell Death", Sci. Rep. 10, 1566 (2020).
Data: https://doi.org/10.5281/zenodo.2601562

Dataset facts (verified against the files)
------------------------------------------
* 3 prostate cell lines (DU145, LNCaP, PNT1A) x 3 treatments (``bp`` black phosphorus, ``do`` doxorubicin,
  ``st`` staurosporine) x 7 fields of view = 63 time-lapses, 3 min/frame, 1.59 px/um.
* ``QPI_*.tif``: multi-page float32 [T, 600, 600] phase images in pg/um^2 (roughly -4..5).
* ``mask_*.tif``: float32 [T, 600, 600] instance masks of the annotated cells (0 = background); a cell keeps its id
  across frames. Not every cell in the field is annotated.
* ``labels_*.xlsx``: one row per mask id: ``cell_num``, ``death_type`` (1 apoptosis, 2 necrosis / lytic,
  3 ambiguous or surviving) and ``death_frame`` (1-based, empty for type 3).

Expected directory layout (what ``download_qpi_cell_death.sh`` produces)::

    root/QPI_LNCaP_bp_1.tif  root/mask_LNCaP_bp_1.tif  root/labels_LNCaP_bp_1.xlsx ...
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

CELL_LINES = ("DU145", "LNCaP", "PNT1A")
TREATMENTS = ("bp", "do", "st")
DEATH_TYPES = {1: "apoptosis", 2: "necrosis", 3: "ambiguous/surviving"}


def read_labels(path: str | os.PathLike) -> dict[int, tuple[int, int | None]]:
    """Read a ``labels_*.xlsx`` sheet into ``{cell id: (death type, 1-based death frame or None)}``."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    labels = {}
    for row in list(sheet.iter(f"{ns}row"))[1:]:  # skip the header
        cells = {c.get("r", "")[0]: int(float(v.text)) for c in row if (v := c.find(f"{ns}v")) is not None and v.text}
        labels[cells["A"]] = (cells["B"], cells.get("C"))
    return labels


class QPICellDeathDataset(Dataset):
    """Frame-level dataset over the QPI time-lapses, with instance masks for Cellpose-SAM.

    Args:
        root: dataset root (see module docstring for the layout).
        cell_lines, treatments, fovs: selection of time-lapses (default: all). Split by field of view, not by frame:
            consecutive frames are nearly identical.
        frame_step: use every n-th frame.
        transform: callable ``sample -> sample`` applied last (e.g. ``CellposeTransform(RandomCropFlip(256))``).

    Samples hold ``"image"`` float32 [1, 600, 600] (pg/um^2), ``"target"`` int64 [600, 600] instance mask, and
    ``"cell_line"``, ``"treatment"``, ``"fov"``, ``"frame"`` (0-based). Per-cell death labels of time-lapse ``i``
    are in ``self.labels[i]`` (see :func:`read_labels`).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        cell_lines: Sequence[str] = CELL_LINES,
        treatments: Sequence[str] = TREATMENTS,
        fovs: Sequence[int] = range(1, 8),
        frame_step: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ):
        root = Path(root)
        self.transform = transform
        self.sequences: list[tuple[str, str, int]] = []
        self.labels: list[dict[int, tuple[int, int | None]]] = []
        self.samples: list[tuple[int, int]] = []  # (sequence idx, 0-based frame)
        for line in cell_lines:
            for treatment in treatments:
                for fov in fovs:
                    name = f"{line}_{treatment}_{fov}"
                    if not (root / f"QPI_{name}.tif").exists():
                        continue
                    with tifffile.TiffFile(root / f"QPI_{name}.tif") as tif:
                        n_frames = len(tif.pages)
                    self.samples += [(len(self.sequences), t) for t in range(0, n_frames, frame_step)]
                    self.sequences.append((line, treatment, fov))
                    self.labels.append(read_labels(root / f"labels_{name}.xlsx"))
        if not self.samples:
            raise RuntimeError(f"No time-lapses found under {root} for the given selection")
        self.root = root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        seq, frame = self.samples[idx]
        line, treatment, fov = self.sequences[seq]
        name = f"{line}_{treatment}_{fov}.tif"
        image = tifffile.imread(self.root / f"QPI_{name}", key=frame)
        mask = tifffile.imread(self.root / f"mask_{name}", key=frame)
        sample = {
            "image": torch.from_numpy(image.astype(np.float32))[None],
            "target": torch.from_numpy(mask.astype(np.int64)),
            "cell_line": line,
            "treatment": treatment,
            "fov": fov,
            "frame": frame,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
