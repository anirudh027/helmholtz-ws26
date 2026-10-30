# helmholtz-ws26

PyTorch dataloaders for cell segmentation, plus adapters for segmentation models.

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate   # Linux/macOS: .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134   # match your CUDA driver
pip install -r requirements.txt
pre-commit install
```

`requirements-lock.txt` pins the exact versions behind [RESULTS.md](RESULTS.md) (use it instead of
`requirements.txt` to reproduce them).

## Datasets

| Dataset | Loader | Download |
| --- | --- | --- |
| [C2C12](https://osf.io/ysaq2/) phase-contrast time-lapse (centroid tracks only) | `dataloaders/c2c12_dataset.py` | `dataloaders/download_c2c12.sh` |
| [cellX-predict](https://doi.org/10.5522/04/16578959) MDCK 64x64 nucleus crops, GFP/RFP (no masks) | `dataloaders/cellx_predict_dataset.py` | `dataloaders/download_cellx_predict.sh` |
| [QPI cell death](https://doi.org/10.5281/zenodo.2601562) quantitative phase time-lapses, sparse instance masks | `dataloaders/qpi_cell_death_dataset.py` | `dataloaders/download_qpi_cell_death.sh` (~64 GB) |
| [HematoFatePrediction](https://doi.org/10.82296/hmgu-nefeli.6qk3j-rsn72) bright-field time-lapses of hematopoietic progenitors, 2 manually tracked cells (no masks) | `dataloaders/hemato_fate_dataset.py` | `dataloaders/download_hemato_fate.sh` |
| [ALFI](https://doi.org/10.6084/m9.figshare.23798451) DIC time-lapses of cultured human cells, boxes with cell cycle phase / death labels, nucleus masks for 8 of 29 sequences | `dataloaders/alfi_dataset.py` | `dataloaders/download_alfi.sh` (8.4 GB) |
| [Celldetective](https://doi.org/10.5281/zenodo.17068651) fluorescence / bright-field / RICM images of NK cells and MCF-7 co-cultures, 3 instance-mask sets | `dataloaders/celldetective_dataset.py` | `dataloaders/download_celldetective.sh` (1.5 GB) |
| [Xenopus mucociliary epithelium](https://doi.org/10.6019/S-BSST2777) 3 confocal stacks (membrane / nuclei), 3D segmentations of all cells | `dataloaders/xenopus_dataset.py` | `dataloaders/download_xenopus.sh` (3.8 GB zip) |
| [p53 cell fate](https://doi.org/10.5061/dryad.pk0p2ngp5) 8 DIC-like tracking videos (p53 / p53 RNAi, +- MNNG) with burned-in track overlays, cell-lineage tracks for MNNG1 only (no masks) | `dataloaders/p53_fate_dataset.py` | `dataloaders/download_p53_fate.sh` (330 MB, needs `DRYAD_TOKEN`) |

Samples are dicts with `"image"` [C, H, W]; C2C12 adds `"target"` (`target="instance"` gives an int64 [H, W]
instance mask). cellX-predict has `subset="patches"` (96k unlabelled crops) and `subset="glimpses"` (frames of 20
time-lapses ending in apoptosis or mitosis, with `"event"` labels). QPI cell death gives 600 x 600 phase frames
(pg/um^2) with an instance `"target"` of the annotated cells; per-cell death type and frame are in `ds.labels`.
HematoFatePrediction gives 1040 x 1388 frames of 2 fields of view (`flatfield=True` divides by the background) and
`"points"` [2, 2], the (x, y) of the two tracked cells (`ds.cell_ids`), NaN where a cell is not tracked.
ALFI gives 1024 x 1280 frames with `"boxes"` [N, 4], `"labels"` and lineage `"ids"`: `annotation="pheno"` (all 29
sequences) boxes only cells in EarlyMitosis / LateMitosis / CellDeath / Multipolar, `annotation="dtl"` (MI01-MI08)
boxes every cell as Interphase / Mitosis and adds an instance `"target"`; batch with `alfi_collate`.
Celldetective gives one image per sample (484-1628 px, sizes vary: crop to batch) with an instance `"target"`;
`subset` picks MCF-7 nuclei, NK cells (both ADCC) or spreading lymphocytes (RICM), `channels` the channel order
(missing channels are zeros), `split="train"/"test"` the authors' split.
Xenopus gives 1956 x 1956 z-slices (membrane, nuclei) of the 0 h / 10 h / 20 h stacks with the slice of the
authors' 3D `target="membrane"` or `"nuclei"` labels; the full time-lapses (S-BIAD2969, 35-64 GB each) are not
fetched.
p53 cell fate gives greyscale frames of 8 conditions x 401 frames (10 min apart, sizes differ per condition: one
condition per batch, `p53_collate`). All 8 Dryad lineage tables are the same file, and it matches the MNNG1 video,
so only MNNG1 samples add `"points"` [K, 2] (x, y) with `"cell"`, `"lineage"` and `"event"` codes (`EVENTS`).

`notebooks/visualize_batches.ipynb` plots 2 batches of every loader (set the data roots in its first cell).
Register the venv as a kernel once (`python -m ipykernel install --user --name helmholtz-ws26`), then pick it in
VS Code or Jupyter.

## Models

### Cellpose-SAM

[Cellpose-SAM](https://github.com/MouseLand/cellpose) can be used with every dataset here that provides an instance
mask, zero-shot or fine-tuned through a PyTorch `DataLoader` (`models/cellpose_sam.py`). Install it from a clone,
which is gitignored:

```bash
git clone https://github.com/MouseLand/cellpose.git
git -C cellpose checkout a54cb48   # the commit behind RESULTS.md
pip install -e ./cellpose --config-settings editable_mode=compat   # compat: the folder must not shadow the package
```

```python
model = models.CellposeModel(gpu=True)  # cpsam_v2
masks = model.eval(image)[0]  # zero-shot

ds = C2C12Dataset(root, target="instance", transform=CellposeTransform(RandomCropFlip(256)))
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, collate_fn=c2c12_collate)
train_cellpose(model, loader, n_epochs=100, save_path="runs/cpsam_c2c12")
```

Fine-tuning at batch size 4 uses 6.6 GB of GPU memory on an RTX 3070 (8 GB), thanks to gradient checkpointing.
C2C12 masks are disks drawn around centroids, so fine-tuning on them teaches round cells: use them as weak labels.

Zero-shot results on every dataset, with the commands to reproduce them: [RESULTS.md](RESULTS.md).
