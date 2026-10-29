"""Cellpose-SAM zero-shot on cellX-predict: how often is the centred nucleus of a crop segmented?

python examples/cellx_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.cellx_predict_dataset import CellXPredictDataset  # noqa: E402

PAD = 96  # 64 x 64 crops -> 256 x 256, the Cellpose-SAM tile size; unpadded crops are mostly missed

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2

    for subset in ("patches", "glimpses"):
        ds = CellXPredictDataset(root, subset=subset)
        idx = np.random.default_rng(0).choice(len(ds), 256, replace=False).tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=32, num_workers=2)
        hits = n_masks = 0
        for batch in loader:
            images = np.pad(batch["image"].numpy(), ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
            masks = [m[PAD:-PAD, PAD:-PAD] for m in model.eval(list(images), channel_axis=0)[0]]
            hits += sum(int(m[32, 32] > 0) for m in masks)
            n_masks += sum(len(np.unique(m[m > 0])) for m in masks)
        print(
            f"{subset}: centre nucleus segmented in {hits / len(idx):.1%} of crops, {n_masks / len(idx):.2f} masks/crop"
        )
