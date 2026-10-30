"""Cellpose-SAM zero-shot on the p53 cell-fate videos: are the tracked cells segmented?

Only MNNG1 has a matching lineage table (see ``dataloaders/p53_fate_dataset.py``), so all 401 MNNG1 frames are scored.
python examples/p53_fate_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.p53_fate_dataset import ANNOTATED, P53FateDataset, p53_collate  # noqa: E402

R = 5  # count a hit if a mask lies within R px of the track point (R = 0: the point itself)

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    ds = P53FateDataset(root, conditions=[ANNOTATED])
    hits, exact, n_points, n_masks, n_hit_masks = [], [], 0, 0, 0
    for batch in DataLoader(ds, batch_size=4, num_workers=2, collate_fn=p53_collate):
        masks = model.eval(list(batch["image"][:, 0].numpy()))[0]
        for mask, points in zip(masks, batch["points"], strict=True):
            h, w = mask.shape
            xy = points.numpy().round().astype(int)
            xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)]  # 0.2% lie outside the crop
            n_points += len(xy)
            n_masks += len(np.unique(mask[mask > 0]))
            n_hit_masks += len(np.unique(mask[xy[:, 1], xy[:, 0]][mask[xy[:, 1], xy[:, 0]] > 0]))
            exact += (mask[xy[:, 1], xy[:, 0]] > 0).tolist()
            for x, y in xy:
                hits.append(mask[max(y - R, 0) : y + R + 1, max(x - R, 0) : x + R + 1].any())
    print(f"tracked cells segmented: {np.mean(hits):.1%} within {R} px, {np.mean(exact):.1%} at the point")
    print(f"{n_points} points in {len(ds)} frames, {n_masks / len(ds):.1f} masks/frame")
    print(f"masks hit / points: {n_hit_masks / n_points:.3f} (< 1: several tracked cells fall in one mask)")
