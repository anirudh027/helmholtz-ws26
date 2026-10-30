"""Cellpose-SAM zero-shot on the segmented Xenopus stacks: per-slice agreement with the authors' 3D segmentations.

python examples/xenopus_cellpose_sam.py <data root>
"""

import sys
from pathlib import Path

from cellpose import models
from cellpose.metrics import average_precision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.xenopus_dataset import XenopusDataset  # noqa: E402

if __name__ == "__main__":
    root = sys.argv[1]
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for channel, target in enumerate(("membrane", "nuclei")):
        ds = XenopusDataset(root, target=target)
        tp = fp = fn = 0
        for sample in DataLoader(ds, batch_size=None, num_workers=2):
            masks = model.eval(sample["image"][channel].numpy())[0]  # the matching channel alone works best
            counts = average_precision(sample["target"].numpy(), masks, threshold=[0.5])[1:]
            tp, fp, fn = tp + counts[0][0], fp + counts[1][0], fn + counts[2][0]
        print(f"{target}: precision@0.5={tp / (tp + fp):.3f} recall@0.5={tp / (tp + fn):.3f} ({len(ds)} slices)")
