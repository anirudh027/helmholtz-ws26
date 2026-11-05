"""Cellpose-SAM zero-shot on the HematoFatePrediction time-lapses: is the manually tracked cell segmented?

python examples/hemato_fate_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.hemato_fate_dataset import HematoFateDataset  # noqa: E402

R = 10  # track points can be off the cell centre: count a hit if a mask lies within R px

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    ds = HematoFateDataset(root)
    tr = ds.tracks
    tracked = [i for i, (p, t) in enumerate(ds.samples) if ((tr["positionIndex"] == p) & (tr["timepoint"] == t)).any()]
    hits, n_masks = [], 0
    for batch in DataLoader(Subset(ds, tracked), batch_size=4, num_workers=2):
        masks = model.eval(list(batch["image"][:, 0].numpy()))[0]
        for mask, points in zip(masks, batch["points"].numpy(), strict=True):
            n_masks += len(np.unique(mask[mask > 0]))
            for x, y in points[~np.isnan(points[:, 0])].astype(int):
                hits.append(mask[max(y - R, 0) : y + R + 1, max(x - R, 0) : x + R + 1].any())
    print(f"tracked cell segmented at {np.mean(hits):.1%} of {len(hits)} points in {len(tracked)} frames")
    print(f"{n_masks / len(tracked):.1f} masks/frame")
