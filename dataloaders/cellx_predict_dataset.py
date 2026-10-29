"""PyTorch dataset for the cellX-predict MDCK nucleus crops.

Source: Soelistyo et al., "Learning biophysical determinants of cell fate with deep neural networks",
Nat. Mach. Intell. 4, 636-644 (2022). Data: https://doi.org/10.5522/04/16578959

Dataset facts (verified against the files)
------------------------------------------
* Every image is a 64 x 64 uint8 crop centred on one nucleus, with 2 channels:
  0 = GFP (H2B-GFP, wild-type MDCK), 1 = RFP (H2B-RFP, Scribble-knockdown MDCK).
* ``patches``: 96,175 single crops in one TFRecord (``train/image`` is raw HWC bytes), used to train the
  paper's VAE. No labels.
* ``glimpses``: 20 time-lapse crops [T, 2, 64, 64] following one RFP cell to an ``apoptosis`` (10) or
  ``mitosis`` (10) event. Each ``.npz`` holds ``t`` (movie frame per glimpse frame) and ``cutoff`` (a glimpse
  frame index before the event).
* There are no segmentation masks; Cellpose-SAM can be used zero-shot (see ``examples/cellx_cellpose_sam.py``).

Expected directory layout (what ``download_cellx_predict.sh`` produces)::

    root/training/
      patches/image_patches_0.tfrecord
      glimpses/data/{apoptosis,mitosis}/GV0794_Pos4_6_81_RFP_apop.tif ...
      glimpses/metadata/{apoptosis,mitosis}/GV0794_Pos4_6_81_RFP_apop.npz ...
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

EVENTS = ("apoptosis", "mitosis")
PATCH_SHAPE = (64, 64, 2)  # HWC, as stored in the TFRecord


def _proto_fields(buf: bytes) -> Iterator[tuple[int, Any]]:
    """Yield (field number, value) of a protobuf message (varint and length-delimited fields only)."""

    def varint(i: int) -> tuple[int, int]:
        value = shift = 0
        while True:
            byte = buf[i]
            value, shift, i = value | (byte & 0x7F) << shift, shift + 7, i + 1
            if byte < 0x80:
                return value, i

    i = 0
    while i < len(buf):
        key, i = varint(i)
        if key & 7 == 0:
            value, i = varint(i)
            yield key >> 3, value
        else:  # wire type 2 (tf.train.Example only uses packed/bytes fields besides varints)
            n, i = varint(i)
            yield key >> 3, buf[i : i + n]
            i += n


def read_tfrecord_image(record: bytes, key: bytes = b"train/image") -> np.ndarray:
    """Decode the raw ``key`` bytes feature of a serialized ``tf.train.Example`` into a [H, W, C] uint8 array."""
    for _, features in _proto_fields(record):  # Example.features
        for _, entry in _proto_fields(features):  # map<string, Feature> entry
            kv = dict(_proto_fields(entry))
            if kv[1] == key:
                bytes_list = next(_proto_fields(kv[2]))[1]  # Feature.bytes_list
                return np.frombuffer(next(_proto_fields(bytes_list))[1], np.uint8).reshape(PATCH_SHAPE)
    raise KeyError(key)


class CellXPredictDataset(Dataset):
    """Single 64 x 64 nucleus crops from cellX-predict.

    Args:
        root: dataset root (see module docstring for the layout).
        subset: ``"patches"`` (unlabelled TFRecord crops) or ``"glimpses"`` (every frame of the event time-lapses;
            split by glimpse, not by frame, as consecutive frames are nearly identical).
        events: glimpse events to use (default: both).
        transform: callable ``sample -> sample`` applied last.

    Samples hold ``"image"`` float32 [2, 64, 64] in [0, 1]; glimpse samples add ``"event"`` (index into
    ``EVENTS``), ``"glimpse"`` (file stem), ``"frame"`` (glimpse frame), ``"t"`` (movie frame) and ``"cutoff"``.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        subset: str = "patches",
        events: tuple[str, ...] = EVENTS,
        transform: Callable[[dict], dict] | None = None,
    ):
        root = Path(root) / "training"
        self.subset, self.transform = subset, transform
        if subset == "patches":
            self.tfrecord = root / "patches" / "image_patches_0.tfrecord"
            self.offsets: list[tuple[int, int]] = []  # (byte offset, length) of each record
            with open(self.tfrecord, "rb") as f:
                while header := f.read(12):  # uint64 length + uint32 crc, then data + uint32 crc
                    n = struct.unpack("<Q", header[:8])[0]
                    self.offsets.append((f.tell(), n))
                    f.seek(n + 4, os.SEEK_CUR)
        elif subset == "glimpses":
            self.glimpses = []  # (event idx, stem, [T, 2, H, W] uint8, metadata)
            for event in events:
                for tif in sorted((root / "glimpses" / "data" / event).glob("*.tif")):
                    meta = np.load(root / "glimpses" / "metadata" / event / f"{tif.stem}.npz")
                    self.glimpses.append((EVENTS.index(event), tif.stem, tifffile.imread(tif), dict(meta)))
            self.samples = [(g, i) for g, (*_, img, _) in enumerate(self.glimpses) for i in range(len(img))]
        else:
            raise ValueError(f"unknown subset {subset!r}")

    def __len__(self) -> int:
        return len(self.offsets) if self.subset == "patches" else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.subset == "patches":
            offset, n = self.offsets[idx]
            with open(self.tfrecord, "rb") as f:
                f.seek(offset)
                image = read_tfrecord_image(f.read(n)).transpose(2, 0, 1)
            sample: dict = {}
        else:
            g, i = self.samples[idx]
            event, stem, frames, meta = self.glimpses[g]
            image = frames[i]
            sample = {
                "event": event,
                "glimpse": stem,
                "frame": i,
                "t": int(meta["t"][i]),
                "cutoff": int(meta["cutoff"]),
            }
        sample["image"] = torch.from_numpy(image.astype(np.float32) / 255.0)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
