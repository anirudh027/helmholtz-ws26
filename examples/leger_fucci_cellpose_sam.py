"""Cellpose-SAM zero-shot on the Leger RPE1 Fucci crops: does it find the tracked nucleus in the crop centre?

There are no masks, but every 64 x 64 crop is centred on the tracked nucleus, so a frame counts as a hit if a predicted
mask lies within 3 px of the centre. Only the H2B channel is the input (nuclei); the bright-field channel alone is
reported as well (whole cells).

python examples/leger_fucci_cellpose_sam.py <data root> [split, default: test] [frame step, default: 10]
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.leger_fucci_dataset import CHANNELS, LegerFucciDataset  # noqa: E402

RADIUS = 3

if __name__ == "__main__":
    root = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    ds = LegerFucciDataset(root, split)
    ds_sub = Subset(ds, range(0, len(ds), step))
    yy, xx = np.mgrid[:64, :64]
    centre = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= RADIUS**2
    for name in ("H2B", "brightfield"):
        channel = CHANNELS.index(name)
        hits, areas = [], []
        for batch in DataLoader(ds_sub, batch_size=64, num_workers=2):
            masks = model.eval(list(batch["image"][:, channel].numpy()))[0]
            for pred in masks:
                ids = np.unique(pred[centre & (pred > 0)])
                hits.append(len(ids) > 0)
                areas += [int((pred == ids[0]).sum())] if len(ids) else []
        print(
            f"{split} {name}: centre hit rate={np.mean(hits):.3f} median mask area={np.median(areas or [0]):.0f} px "
            f"({len(hits)} frames of {len(ds.tracks)} tracks)"
        )
