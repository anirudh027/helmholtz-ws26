"""Cellpose-SAM / SAM2 zero-shot on C2C12: detection precision/recall against the full human centroid annotation.

python examples/c2c12_zero_shot.py <data root> [--model cpsam|sam2] [--every n]
"""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.c2c12_dataset import C2C12Dataset  # noqa: E402
from models.segmenters import every_nth, load_segmenter, parse_options  # noqa: E402

if __name__ == "__main__":
    model_name, every, sys.argv[1:] = parse_options(sys.argv[1:])  # --model cpsam|sam2, --every n
    root = sys.argv[1]
    segment = load_segmenter(model_name)

    # C2C12 has centroids only: a predicted mask is correct if it contains an annotated centroid
    full = C2C12Dataset(root, fields=["F0009"], annotation="human_full", target="points", frame_range=(0, 800))
    ds = every_nth(full, every)
    tp = n_pred = n_true = 0
    for sample in DataLoader(ds, batch_size=None, num_workers=2):
        masks = segment([sample["image"][0].numpy()])[0]
        x, y = sample["points"].round().long().T.numpy()
        hits = masks[y.clip(0, masks.shape[0] - 1), x.clip(0, masks.shape[1] - 1)]
        tp, n_pred, n_true = tp + len(np.unique(hits[hits > 0])), n_pred + int(masks.max()), n_true + len(x)
    print(f"zero-shot: precision={tp / n_pred:.3f} recall={tp / n_true:.3f} ({len(ds)} frames, {n_true} cells)")
