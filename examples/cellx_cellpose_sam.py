"""Cellpose-SAM zero-shot on cellX-predict: how often is the centred nucleus of a crop segmented?

Masks are saved per batch of BATCH crops: <output dir>/<subset>/<batch>.tif holds [BATCH, 64, 64], crops
BATCH * batch onwards in dataset order.

python examples/cellx_cellpose_sam.py <data root> <output dir>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.cellx_predict_dataset import CellXPredictDataset  # noqa: E402
from models.cellpose_sam import segment  # noqa: E402

PAD = 96  # 64 x 64 crops -> 256 x 256, the Cellpose-SAM tile size; unpadded crops are mostly missed
BATCH = 64

if __name__ == "__main__":
    root, out = sys.argv[1], Path(sys.argv[2])
    model = models.CellposeModel(gpu=True)  # cpsam_v2

    for subset in ("patches", "glimpses"):
        ds = CellXPredictDataset(root, subset=subset)
        hits = n_masks = 0
        for b, batch in enumerate(tqdm(DataLoader(ds, batch_size=BATCH, num_workers=2), desc=subset)):
            # a [N, C, H, W] stack is segmented image by image; one file per batch keeps the file count low
            path = out / subset / f"{b:05d}.tif"
            masks = segment(model, batch["image"].numpy(), path, pad=PAD, channel_axis=1)
            hits += int((masks[:, 32, 32] > 0).sum())
            n_masks += sum(len(np.unique(m[m > 0])) for m in masks)
        print(
            f"{subset}: centre nucleus segmented in {hits / len(ds):.1%} of crops, {n_masks / len(ds):.2f} masks/crop"
        )
