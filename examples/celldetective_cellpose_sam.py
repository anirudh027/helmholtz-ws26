"""Cellpose-SAM zero-shot on the Celldetective segmentation sets, per set, against the instance masks.

Masks are saved to <output dir>/<set>/<image name>.tif.

python examples/celldetective_cellpose_sam.py <data root> <output dir> [sets, default: all]
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from cellpose.metrics import _intersection_over_union
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.celldetective_dataset import SUBSETS, CelldetectiveDataset  # noqa: E402
from models.cellpose_sam import segment  # noqa: E402

CHANNELS = {  # up to 3 channels that show the annotated population (Cellpose-SAM takes 3)
    "db_mcf7_nuclei_w_lymphocytes": ("live_nuclei_channel", "dead_nuclei_channel"),
    "db_primary_NK_w_mcf7": ("effector_fluo_channel", "live_nuclei_channel", "brightfield_channel"),
    "db_spreading_lymphocytes": ("adhesion_channel", "brightfield_channel"),
}

if __name__ == "__main__":
    root, out, subsets = sys.argv[1], Path(sys.argv[2]), sys.argv[3:] or SUBSETS
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for subset in subsets:
        ds = CelldetectiveDataset(root, subset, channels=CHANNELS[subset])
        ious, n_pred = [], 0  # best IoU of every annotated cell (only one population is annotated: no precision)
        for sample in tqdm(DataLoader(ds, batch_size=None, num_workers=2), desc=subset):
            true = sample["target"].numpy()
            pred = segment(model, sample["image"].numpy(), out / subset / sample["name"], channel_axis=0)
            n_pred += int(pred.max())
            iou = _intersection_over_union(true, pred)[1:, 1:]
            ious += iou.max(1, initial=0)[np.unique(true[true > 0]) - 1].tolist()
        recall = np.mean(np.array(ious) >= 0.5)
        print(
            f"{subset}: recall@0.5={recall:.3f} mean IoU={np.mean(ious):.3f} "
            f"({len(ious)} cells, {n_pred} predicted masks, {len(ds)} images)"
        )
