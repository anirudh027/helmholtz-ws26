"""Cellpose-SAM on C2C12: zero-shot detection scores, then a short fine-tuning run.

python examples/c2c12_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.c2c12_dataset import C2C12Dataset, RandomCropFlip, c2c12_collate  # noqa: E402
from models.cellpose_sam import CellposeTransform, train_cellpose  # noqa: E402

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2

    # C2C12 has centroids only: a predicted mask is correct if it contains an annotated centroid
    ds = C2C12Dataset(root, fields=["F0009"], annotation="human_full", target="points", frame_range=(0, 800))
    tp = n_pred = n_true = 0
    for i in np.linspace(0, len(ds) - 1, 5).astype(int):
        sample = ds[int(i)]
        masks = model.eval(sample["image"][0].numpy())[0]
        x, y = sample["points"].round().long().T.numpy()
        hits = masks[y.clip(0, masks.shape[0] - 1), x.clip(0, masks.shape[1] - 1)]
        tp, n_pred, n_true = tp + len(np.unique(hits[hits > 0])), n_pred + masks.max(), n_true + len(x)
    print(f"zero-shot: precision={tp / n_pred:.3f} recall={tp / n_true:.3f}")

    ds = C2C12Dataset(root, fields=["F0009"], target="instance", transform=CellposeTransform(RandomCropFlip(256)))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, collate_fn=c2c12_collate)
    train_cellpose(model, loader, n_epochs=12, max_steps_per_epoch=10)
