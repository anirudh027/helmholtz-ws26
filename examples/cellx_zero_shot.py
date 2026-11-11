"""Cellpose-SAM / SAM2 zero-shot on cellX-predict: how often is the centred nucleus of a crop segmented?

python examples/cellx_zero_shot.py <data root> [--model cpsam|sam2] [--every n]
"""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders.cellx_predict_dataset import CellXPredictDataset  # noqa: E402
from models.segmenters import every_nth, load_segmenter, parse_options  # noqa: E402

PAD = 96  # 64 x 64 crops -> 256 x 256, the Cellpose-SAM tile size; unpadded crops are mostly missed

if __name__ == "__main__":
    model_name, every, sys.argv[1:] = parse_options(sys.argv[1:])  # --model cpsam|sam2, --every n
    root = sys.argv[1]
    segment = load_segmenter(model_name)

    for subset in ("patches", "glimpses"):
        ds = every_nth(CellXPredictDataset(root, subset=subset), every)
        loader = DataLoader(ds, batch_size=32, num_workers=2)
        hits = n_masks = 0
        for batch in loader:
            images = np.pad(batch["image"].numpy(), ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect")
            masks = [m[PAD:-PAD, PAD:-PAD] for m in segment(list(images))]
            hits += sum(int(m[32, 32] > 0) for m in masks)
            n_masks += sum(len(np.unique(m[m > 0])) for m in masks)
        print(
            f"{subset}: centre nucleus segmented in {hits / len(ds):.1%} of crops, {n_masks / len(ds):.2f} masks/crop"
        )
