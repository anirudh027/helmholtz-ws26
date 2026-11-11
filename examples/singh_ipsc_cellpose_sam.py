"""Cellpose-SAM zero-shot on the Singh iPSC ROIs, per stage, against the colony instance masks.

The annotations outline colonies while Cellpose-SAM segments cells, so instance matching (precision / recall @0.5 IoU)
is expected to be low; the foreground IoU (union of predicted masks vs union of colonies) shows whether the colonies
are found at all. All 3 classes are scored, as Cellpose-SAM does not tell them apart.

python examples/singh_ipsc_cellpose_sam.py <data root> [frame step, default: 1]
"""

import sys
from pathlib import Path

from cellpose import models
from cellpose.metrics import average_precision
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.singh_ipsc_dataset import CLASSES, SinghIPSCDataset  # noqa: E402

STAGES = {"early (0-53)": range(54), "late (54-126)": range(54, 127)}

if __name__ == "__main__":
    root, step = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for stage, frames in STAGES.items():
        ds = SinghIPSCDataset(root, frames=frames[::step], classes=CLASSES)
        tp = fp = fn = inter = union = 0
        for sample in DataLoader(ds, batch_size=None, num_workers=2):
            true = sample["target"].numpy()
            pred = model.eval(sample["image"][0].numpy())[0]
            counts = average_precision(true, pred, threshold=[0.5])[1:]
            tp, fp, fn = tp + counts[0][0], fp + counts[1][0], fn + counts[2][0]
            inter, union = inter + ((true > 0) & (pred > 0)).sum(), union + ((true > 0) | (pred > 0)).sum()
        print(
            f"{stage}: precision@0.5={tp / max(tp + fp, 1):.3f} recall@0.5={tp / max(tp + fn, 1):.3f} "
            f"foreground IoU={inter / union:.3f} ({int(tp + fn)} colonies, {len(ds)} frames)"
        )
