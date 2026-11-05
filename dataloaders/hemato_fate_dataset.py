"""PyTorch dataset for the HematoFatePrediction example time-lapses (murine hematopoietic progenitors, bright-field).

Source: Buggenthin, Buettner et al., "Prospective identification of hematopoietic lineage choice by deep learning",
Nat. Methods 14, 403-406 (2017). Data: https://doi.org/10.82296/hmgu-nefeli.6qk3j-rsn72
Code: https://github.com/QSCD/HematoFatePrediction

Dataset facts (verified against the files)
------------------------------------------
* Two fields of view (positions 54 and 55) of experiment 3, each 767 consecutive bright-field frames
  (timepoints 2685-3451, ~95 s/frame), 1.0238 um/px. Cells are round, ~15-20 px across.
* ``experiment3_pos00XX_tXXXXX_wl00.png``: 1040 x 1388 8-bit greyscale (palette PNG with an identity grey palette).
* ``backgrounds/.../experiment3_pos00XX_wl00_projbg.png``: one 8-bit background per position; the authors normalize
  frames as ``image / background`` (divided by its mean), see ``flatfield``.
* ``demoTracks.mat``: 810 manual track points (``track.X``, ``track.Y`` in pixels, ``timepoint``, ``positionIndex``,
  ``cellNr``, ``absoluteTime`` in s, marker channels ``wavelength_1..3`` = PU.1 / GATA1 / FcgRIII) of 2 cells:
  cell 82 (pos 54, t 3055-3451, GATA1+ throughout, ``type`` 1 = MEP lineage) and cell 96 (pos 54, t 2685-3097
  with gaps, and pos 55, t 3008-3039 where the fields overlap; ``type`` 0 = no marker, ``invtype_alt`` 2, likely
  the GMP lineage inferred from its progeny). Track points can be ~20 px off the cell centre.
* There are no segmentation masks; Cellpose-SAM can be used zero-shot (see ``examples/hemato_fate_cellpose_sam.py``).

Expected directory layout (what ``download_hemato_fate.sh`` produces)::

    root/demoTracks.mat
    root/experiment3/position54/experiment3_pos0054_t02685_wl00.png ...
    root/experiment3/backgrounds/position54/experiment3_pos0054_wl00_projbg.png
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat  # type: ignore[import-untyped,unused-ignore]
from torch.utils.data import Dataset

POSITIONS = (54, 55)
TIMEPOINTS = range(2685, 3452)


def read_tracks(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Read ``demoTracks.mat`` into ``{field: [N] array}`` (one entry per track point)."""
    return loadmat(path, simplify_cells=True)["track"]


class HematoFateDataset(Dataset):
    """Frame-level dataset over the bright-field time-lapses, with the manual track points.

    Args:
        root: dataset root (see module docstring for the layout).
        positions: fields of view to use (default: both).
        frame_step: use every n-th frame.
        flatfield: divide by the position's background and its mean (the authors' normalization, ~1 = background)
            instead of scaling the raw 8-bit values to [0, 1].
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [1, 1040, 1388], ``"position"``, ``"timepoint"``, and ``"points"`` float32
    [K, 2] (x, y) of the K tracked cells ``self.cell_ids`` in this frame (NaN where a cell is not tracked).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        positions: Sequence[int] = POSITIONS,
        frame_step: int = 1,
        flatfield: bool = False,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.root, self.flatfield, self.transform = Path(root), flatfield, transform
        self.tracks = read_tracks(self.root / "demoTracks.mat")
        self.cell_ids = sorted(set(self.tracks["cellNr"].tolist()))
        self.samples = [(p, t) for p in positions for t in TIMEPOINTS[::frame_step] if self._path(p, t).exists()]
        if not self.samples:
            raise RuntimeError(f"No frames found under {root} for positions {positions}")

    def _path(self, position: int, timepoint: int) -> Path:
        name = f"experiment3_pos{position:04d}_t{timepoint:05d}_wl00.png"
        return self.root / "experiment3" / f"position{position}" / name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        position, timepoint = self.samples[idx]
        image = np.asarray(Image.open(self._path(position, timepoint)).convert("L"), np.float32)
        if self.flatfield:
            bg = self.root / "experiment3" / "backgrounds" / f"position{position}"
            image /= np.asarray(Image.open(bg / f"experiment3_pos{position:04d}_wl00_projbg.png"), np.float32)
            image /= image.mean()
        else:
            image /= 255.0
        tr = self.tracks
        points = np.full((len(self.cell_ids), 2), np.nan, np.float32)
        for i in np.flatnonzero((tr["positionIndex"] == position) & (tr["timepoint"] == timepoint)):
            points[self.cell_ids.index(tr["cellNr"][i])] = tr["X"][i], tr["Y"][i]
        sample = {
            "image": torch.from_numpy(image)[None],
            "position": position,
            "timepoint": timepoint,
            "points": torch.from_numpy(points),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
