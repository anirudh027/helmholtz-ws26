"""Cellpose-SAM / SAM2 zero-shot on the HematoFatePrediction time-lapses: is the manually tracked cell segmented?

python examples/hemato_fate_zero_shot.py <data root> [--model cpsam|sam2] [--every n]
"""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.hemato_fate_dataset import HematoFateDataset  # noqa: E402
from models.segmenters import load_segmenter, parse_options  # noqa: E402

R = 10  # track points can be off the cell centre: count a hit if a mask lies within R px

if __name__ == "__main__":
    model_name, every, sys.argv[1:] = parse_options(sys.argv[1:])  # --model cpsam|sam2, --every n
    root = sys.argv[1]
    segment = load_segmenter(model_name)
    ds = HematoFateDataset(root)
    tr = ds.tracks
    tracked = [i for i, (p, t) in enumerate(ds.samples) if ((tr["positionIndex"] == p) & (tr["timepoint"] == t)).any()]
    tracked = tracked[::every]
    hits, n_masks = [], 0
    for batch in DataLoader(Subset(ds, tracked), batch_size=4, num_workers=2):
        masks = segment(list(batch["image"][:, 0].numpy()))
        for mask, points in zip(masks, batch["points"].numpy(), strict=True):
            n_masks += len(np.unique(mask[mask > 0]))
            for x, y in points[~np.isnan(points[:, 0])].astype(int):
                hits.append(mask[max(y - R, 0) : y + R + 1, max(x - R, 0) : x + R + 1].any())
    print(f"tracked cell segmented at {np.mean(hits):.1%} of {len(hits)} points in {len(tracked)} frames")
    print(f"{n_masks / len(tracked):.1f} masks/frame")
