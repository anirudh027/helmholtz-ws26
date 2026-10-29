# helmholtz-ws26

PyTorch dataloaders for cell segmentation, plus adapters for segmentation models.

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate   # Linux/macOS: .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134   # match your CUDA driver
pip install -r requirements.txt
pre-commit install
```

## Datasets

| Dataset | Loader | Download |
| --- | --- | --- |
| [C2C12](https://osf.io/ysaq2/) phase-contrast time-lapse (centroid tracks only) | `dataloaders/c2c12_dataset.py` | `dataloaders/download_c2c12.sh` |

Samples are dicts with `"image"` [C, H, W] and `"target"`; `target="instance"` gives an int64 [H, W] instance mask.

## Models

### Cellpose-SAM

[Cellpose-SAM](https://github.com/MouseLand/cellpose) can be used with every dataset here that provides an instance
mask, zero-shot or fine-tuned through a PyTorch `DataLoader` (`models/cellpose_sam.py`). Install it from a clone,
which is gitignored:

```bash
git clone https://github.com/MouseLand/cellpose.git
pip install -e ./cellpose --config-settings editable_mode=compat   # compat: the folder must not shadow the package
```

```python
model = models.CellposeModel(gpu=True)  # cpsam_v2
masks = model.eval(image)[0]  # zero-shot

ds = C2C12Dataset(root, target="instance", transform=CellposeTransform(RandomCropFlip(256)))
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=c2c12_collate)
train_cellpose(model, loader, n_epochs=100, save_path="runs/cpsam_c2c12")
```

`examples/c2c12_cellpose_sam.py` runs both steps. Results on 5 frames of C2C12 090303/F0009 (RTX 3070, 8 GB):

- Zero-shot: precision 0.97 and recall 0.86 against the human centroid annotation.
- Fine-tuning at batch size 4 uses 6.6 GB of GPU memory, thanks to gradient checkpointing.
- C2C12 masks are disks drawn around centroids, so fine-tuning on them teaches round cells. Use them as weak labels.
