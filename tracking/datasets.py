"""Dataset adapters: every WS26 dataset as time-lapse sequences ``(imgs, masks)`` for ``track()``.

File formats are handled by Florian's frame-level loaders (``dataloaders/``, PR #1);
this module groups their samples into whole sequences of shape (T, Y, X).

    for seq in iter_sequences("alfi", limit=50):
        res = track(seq.imgs, seq.masks, model="hoct")
"""

import glob
import os
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import tifffile

from .ctc import load_sequence

DATA_ROOT = os.environ.get("WS26_DATA", "/mnt/1t/datasets/ws26")


@dataclass
class Sequence:
    dataset: str
    name: str
    imgs: np.ndarray  # (T, Y, X) float32
    masks: np.ndarray | None  # (T, Y, X) instance labels, None = dataset ships no masks
    gt_dir: str | None = None  # CTC <seq>_GT folder, if official metrics are possible
    note: str = ""


# Datasets that cannot be tracked as loaded, and why.
NOT_TRACKABLE = {
    "cellx_predict": "64x64 single-cell crops, not full-field time-lapses",
    "celldetective": "single images per sample, no time-lapse in the loader",
    "xenopus": "3 static 3-D stacks (0/10/20 h); the full time-lapses are not fetched",
}


def _from_loader(dataset, ds, key, name_of, limit, masks_from="target", note=""):
    """Group a frame-level loader's samples into sequences (frames in loader order)."""
    by_seq = {}
    for i, s in enumerate(ds.samples):
        by_seq.setdefault(key(s), []).append(i)
    for k, idxs in by_seq.items():
        idxs = idxs[:limit] if limit else idxs
        samples = [ds[i] for i in idxs]
        imgs = np.stack([np.asarray(x["image"])[0] for x in samples]).astype(np.float32)
        masks = None
        if masks_from and masks_from in samples[0]:
            masks = np.stack([np.asarray(x[masks_from]) for x in samples]).astype(np.int32)
        yield Sequence(dataset, name_of(k), imgs, masks, note=note)


def musc(limit=None, seqs=("01", "02")):
    root = "/mnt/1t/stem_cell_fate_prediction/datasets/BF-C2DL-MuSC/Training"
    for seq in seqs:
        imgs, masks = load_sequence(root, seq, limit)
        gt = os.path.join(root, f"{seq}_GT")
        yield Sequence("musc", seq, imgs.astype(np.float32), masks.astype(np.int32),
                       gt_dir=None if limit else gt, note="CTC silver-truth masks")


def alfi(limit=None, root=None):
    from dataloaders.alfi_dataset import ALFIDataset

    ds = ALFIDataset(root or os.path.join(DATA_ROOT, "alfi"), annotation="dtl")
    yield from _from_loader("alfi", ds, key=lambda s: s[0], name_of=str, limit=limit,
                            note="nucleus masks (MI sequences), connected components")


def qpi_cell_death(limit=None, root=None, cell_lines=("LNCaP",)):
    from dataloaders.qpi_cell_death_dataset import QPICellDeathDataset

    ds = QPICellDeathDataset(root or os.path.join(DATA_ROOT, "qpi_cell_death"), cell_lines=cell_lines)
    yield from _from_loader("qpi_cell_death", ds, key=lambda s: s[0],
                            name_of=lambda k: "_".join(map(str, ds.sequences[k])), limit=limit,
                            note="sparse masks: only annotated cells are tracked")


def c2c12(limit=None, root=None):
    from dataloaders.c2c12_dataset import C2C12Dataset

    ds = C2C12Dataset(root or os.path.join(DATA_ROOT, "c2c12"), annotation="computer", target="instance")
    yield from _from_loader("c2c12", ds, key=lambda s: s[0],
                            name_of=lambda k: "_".join(ds.sequences[k]), limit=limit,
                            note="disk masks around annotated centroids (weak labels)")


def hemato_fate(limit=None, root=None):
    from dataloaders.hemato_fate_dataset import HematoFateDataset

    ds = HematoFateDataset(root or os.path.join(DATA_ROOT, "hemato_fate"))
    yield from _from_loader("hemato_fate", ds, key=lambda s: s[0], name_of=lambda k: f"pos{k}",
                            limit=limit, masks_from=None, note="no masks shipped")


def helmholtz(limit=None, root="/mnt/1t/Timelapse/Brightfield_Normalized_TIFF"):
    files = sorted(glob.glob(os.path.join(root, "t*.tif")))
    files = files[:limit] if limit else files
    imgs = np.stack([tifffile.imread(f)[0] for f in files]).astype(np.float32)  # plane 0 of 2
    yield Sequence("helmholtz", "BF", imgs, None, note="our timelapse, brightfield; no masks shipped")


ADAPTERS = {
    "musc": musc,
    "alfi": alfi,
    "qpi_cell_death": qpi_cell_death,
    "c2c12": c2c12,
    "hemato_fate": hemato_fate,
    "helmholtz": helmholtz,
}


def iter_sequences(dataset, limit=None, max_seqs=None, **kwargs) -> Iterator[Sequence]:
    """Yield the sequences of ``dataset`` (one of ``ADAPTERS``)."""
    if dataset in NOT_TRACKABLE:
        raise ValueError(f"{dataset} is not trackable: {NOT_TRACKABLE[dataset]}")
    if dataset == "p53_fate":
        raise ValueError("p53_fate needs a Dryad token to download (DRYAD_TOKEN); not fetched yet")
    if dataset not in ADAPTERS:
        raise ValueError(f"Unknown dataset {dataset!r}, choose from {list(ADAPTERS)}")
    for i, seq in enumerate(ADAPTERS[dataset](limit=limit, **kwargs)):
        if max_seqs is not None and i >= max_seqs:
            return
        yield seq
