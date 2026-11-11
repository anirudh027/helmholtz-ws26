"""Cellpose-SAM / SAM2 zero-shot on the segmented Xenopus stacks: per-slice agreement with the authors' 3D labels.

python examples/xenopus_zero_shot.py <data root> [--model cpsam|sam2] [--every n]
"""

import sys
from pathlib import Path

from cellpose.metrics import average_precision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.xenopus_dataset import XenopusDataset  # noqa: E402
from models.segmenters import every_nth, load_segmenter, parse_options  # noqa: E402

if __name__ == "__main__":
    model_name, every, sys.argv[1:] = parse_options(sys.argv[1:])  # --model cpsam|sam2, --every n
    root = sys.argv[1]
    segment = load_segmenter(model_name)
    for channel, target in enumerate(("membrane", "nuclei")):
        ds = every_nth(XenopusDataset(root, target=target), every)
        tp = fp = fn = 0
        for sample in DataLoader(ds, batch_size=None, num_workers=2):
            masks = segment([sample["image"][channel].numpy()])[0]  # the matching channel alone works best
            counts = average_precision(sample["target"].numpy(), masks, threshold=[0.5])[1:]
            tp, fp, fn = tp + counts[0][0], fp + counts[1][0], fn + counts[2][0]
        print(f"{target}: precision@0.5={tp / (tp + fp):.3f} recall@0.5={tp / (tp + fn):.3f} ({len(ds)} slices)")
