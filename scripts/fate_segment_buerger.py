"""Segment Bürger videos with Cellpose-SAM (run with the cellpose environment).

    data/buerger/venv_cellpose/bin/python scripts/fate_segment_buerger.py 001 016 013
Writes data/buerger/out/cp_masks_<video>.npy, shape (186, 1024, 1024) uint16 (frames 49..234).
"""

# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models

root = Path("data/buerger")
model = models.CellposeModel(gpu=True)
for v in sys.argv[1:]:
    imgs = [tifffile.imread(root / "raw" / v / "images" / f"{f:03d}.tif") for f in range(49, 235)]
    masks = model.eval(imgs, batch_size=4)[0]
    out = np.stack(masks).astype(np.uint16)
    np.save(root / "out" / f"cp_masks_{v}.npy", out)
    print(v, out.shape, "cells per frame (mean):", np.mean([len(np.unique(m)) - 1 for m in out]), flush=True)
