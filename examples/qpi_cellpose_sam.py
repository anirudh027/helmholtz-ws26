"""Cellpose-SAM zero-shot on the QPI cell-death time-lapses, per cell line, against the (sparse) instance masks.

Masks are saved to <output dir>/<cell line>_<treatment>_<fov>/<frame>.tif.

python examples/qpi_cellpose_sam.py <data root> <output dir> [cell lines, default: all]
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import models
from cellpose.metrics import _intersection_over_union
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.qpi_cell_death_dataset import CELL_LINES, QPICellDeathDataset  # noqa: E402
from models.cellpose_sam import segment  # noqa: E402

if __name__ == "__main__":
    root, out, cell_lines = sys.argv[1], Path(sys.argv[2]), sys.argv[3:] or CELL_LINES
    model = models.CellposeModel(gpu=True)  # cpsam_v2
    for line in cell_lines:
        ds = QPICellDeathDataset(root, [line])
        ious = []  # best IoU of every annotated cell in every frame (not all cells are annotated: no precision)
        for sample in tqdm(DataLoader(ds, batch_size=None, num_workers=2), desc=line):
            true = sample["target"].numpy()
            name = f"{sample['cell_line']}_{sample['treatment']}_{sample['fov']}"
            pred = segment(model, sample["image"][0].numpy(), out / name / f"{sample['frame']:03d}.tif")
            iou = _intersection_over_union(true, pred)[1:, 1:]
            ious += iou.max(1, initial=0)[np.unique(true[true > 0]) - 1].tolist()
        recall = np.mean(np.array(ious) >= 0.5)
        print(f"{line}: recall@0.5={recall:.3f} mean IoU={np.mean(ious):.3f} ({len(ious)} cells, {len(ds)} frames)")
