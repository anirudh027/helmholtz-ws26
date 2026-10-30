"""Cellpose-SAM zero-shot on the ALFI MI sequences, per sequence, against the DTL instance masks.

The masks outline nuclei (whole cells only in mitosis) while Cellpose-SAM segments cells, so detection is scored like
C2C12: a predicted mask is correct if it contains the centroid of an annotated cell. Mask IoU is reported as well.

python examples/alfi_cellpose_sam.py <data root> [sequences, default: MI01-MI08]
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from cellpose.metrics import _intersection_over_union
from scipy import ndimage  # type: ignore[import-untyped,unused-ignore]
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.alfi_dataset import ALFIDataset, alfi_collate  # noqa: E402

if __name__ == "__main__":
    root, sequences = sys.argv[1], sys.argv[2:] or [f"MI{i:02d}" for i in range(1, 9)]
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for seq in sequences:
        ds = ALFIDataset(root, [seq], annotation="dtl")
        tp = n_pred = 0
        ious = []  # best IoU of every annotated cell in every frame
        for batch in DataLoader(ds, batch_size=4, num_workers=2, collate_fn=alfi_collate):
            masks = model.eval(list(batch["image"][:, 0].numpy()))[0]
            for true, pred in zip(batch["target"].numpy(), masks, strict=True):
                centroids = ndimage.center_of_mass(true > 0, true, range(1, true.max() + 1))
                y, x = np.round(np.reshape(centroids, (-1, 2))).astype(int).T
                hits = pred[y, x]
                tp, n_pred = tp + len(np.unique(hits[hits > 0])), n_pred + int(pred.max())
                ious += _intersection_over_union(true, pred)[1:, 1:].max(1, initial=0).tolist()
        iou = np.array(ious)
        print(
            f"{seq}: precision={tp / n_pred:.3f} recall={tp / len(iou):.3f} recall@0.5IoU={np.mean(iou >= 0.5):.3f}"
            f" mean IoU={iou.mean():.3f} ({len(iou)} cells, {len(ds)} frames)"
        )
