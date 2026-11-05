"""Cellpose-SAM zero-shot on C2C12: detection precision/recall against the full human centroid annotation.

python examples/c2c12_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.c2c12_dataset import C2C12Dataset  # noqa: E402

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2

    # C2C12 has centroids only: a predicted mask is correct if it contains an annotated centroid
    ds = C2C12Dataset(root, fields=["F0009"], annotation="human_full", target="points", frame_range=(0, 800))
    tp = n_pred = n_true = 0
    for sample in DataLoader(ds, batch_size=None, num_workers=2):
        masks = model.eval(sample["image"][0].numpy())[0]
        x, y = sample["points"].round().long().T.numpy()
        hits = masks[y.clip(0, masks.shape[0] - 1), x.clip(0, masks.shape[1] - 1)]
        tp, n_pred, n_true = tp + len(np.unique(hits[hits > 0])), n_pred + int(masks.max()), n_true + len(x)
    print(f"zero-shot: precision={tp / n_pred:.3f} recall={tp / n_true:.3f} ({len(ds)} frames, {n_true} cells)")
