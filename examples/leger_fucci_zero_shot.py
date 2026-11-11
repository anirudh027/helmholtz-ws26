"""Cellpose-SAM / SAM2 zero-shot on the Leger RPE1 Fucci crops: does it find the tracked nucleus in the crop centre?

There are no masks, but every 64 x 64 crop is centred on the tracked nucleus, so a frame counts as a hit if a predicted
mask lies within 3 px of the centre. Only the H2B channel is the input (nuclei); the bright-field channel alone is
reported as well (whole cells).

python examples/leger_fucci_zero_shot.py <data root> [split, default: test] [--model cpsam|sam2] [--every n]
"""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.leger_fucci_dataset import CHANNELS, LegerFucciDataset  # noqa: E402
from models.segmenters import every_nth, load_segmenter, parse_options  # noqa: E402

RADIUS = 3

if __name__ == "__main__":
    model_name, every, sys.argv[1:] = parse_options(sys.argv[1:])  # --model cpsam|sam2, --every n
    root = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    segment = load_segmenter(model_name)
    ds = LegerFucciDataset(root, split)
    yy, xx = np.mgrid[:64, :64]
    centre = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= RADIUS**2
    for name in ("H2B", "brightfield"):
        channel = CHANNELS.index(name)
        hits, areas = [], []
        for batch in DataLoader(every_nth(ds, every), batch_size=64, num_workers=2):
            masks = segment(list(batch["image"][:, channel].numpy()))
            for pred in masks:
                ids = np.unique(pred[centre & (pred > 0)])
                hits.append(len(ids) > 0)
                areas += [int((pred == ids[0]).sum())] if len(ids) else []
        print(
            f"{split} {name}: centre hit rate={np.mean(hits):.3f} median mask area={np.median(areas or [0]):.0f} px "
            f"({len(hits)} of {len(ds)} frames, {len(ds.tracks)} tracks)"
        )
