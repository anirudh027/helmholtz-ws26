"""Cellpose-SAM zero-shot on the segmented Xenopus stacks: per-slice agreement with the authors' 3D segmentations.

Masks are saved to <output dir>/<target>/<timepoint>_<z>.tif.

python examples/xenopus_cellpose_sam.py <data root> <output dir>
"""

import sys
from pathlib import Path

from cellpose import models
from cellpose.metrics import average_precision
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.xenopus_dataset import XenopusDataset  # noqa: E402
from models.cellpose_sam import segment  # noqa: E402

if __name__ == "__main__":
    root, out = sys.argv[1], Path(sys.argv[2])
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for channel, target in enumerate(("membrane", "nuclei")):
        ds = XenopusDataset(root, target=target)
        tp = fp = fn = 0
        loader = tqdm(DataLoader(ds, batch_size=None, num_workers=2), desc=target)
        for (tp_name, z), sample in zip(ds.samples, loader, strict=True):
            # the matching channel alone works best
            masks = segment(model, sample["image"][channel].numpy(), out / target / f"{tp_name}_{z:02d}.tif")
            counts = average_precision(sample["target"].numpy(), masks, threshold=[0.5])[1:]
            tp, fp, fn = tp + counts[0][0], fp + counts[1][0], fn + counts[2][0]
        print(f"{target}: precision@0.5={tp / (tp + fp):.3f} recall@0.5={tp / (tp + fn):.3f} ({len(ds)} slices)")
