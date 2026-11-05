"""PyTorch dataset for the C2C12 myoblast phase-contrast time-lapse dataset.

Source: Ker et al., "Phase contrast time-lapse microscopy datasets with automated
and manual cell tracking annotations", Sci. Data 5, 180237 (2018).
OSF project: https://osf.io/ysaq2/

Dataset facts (verified against the OSF files)
----------------------------------------------
* 3 experiments (OSF sub-components), each with ~16 fields of view ("exp1_F00xx Data"),
  each a time-lapse of ~1013 frames: 16-bit greyscale TIFF, 1040 x 1392 (H x W),
  12-bit dynamic range (max 4095). Files are named ``exp1_F0001-00001.tif`` (1-based).
* Annotations are XML **cell tracks**, not pixel masks. Every cell (<a id=..>) has one
  centroid (<s i=frame x= y= s=status>) per frame, with 0-based frame indices
  (XML frame i <-> TIFF number i+1). Daughter cells are nested under their mother.
    - ``Annotation_Computer``: automated tracking of *all* cells in the field.
    - ``Annotation_Human``: manual tracking of only a few lineages per field (sparse!),
      where ``f="I"`` marks interpolated points. Exception: 090303 / F0009 also has a
      ``Human Full Annotation`` file covering every cell, but only up to ~frame 800
      (it drops from ~275 to ~50 cells by frame 900); use ``frame_range=(0, 800)``.
* Status codes (inferred from the data, not officially documented): 2 = newborn
  daughter, 4 = normal, 7 = dividing (last frame before daughters),
  10/11/12/18 = track start/end events (cell entering/leaving the field, lost track).
  Computer track ends (11/18) are stored with NaN coordinates; they are dropped.

Because there are no ground-truth masks, segmentation targets are derived from the
centroids: instance/semantic disk masks (bounded Voronoi), Gaussian heatmaps, or the
raw points. ``target="image"`` returns the image itself as target (identity /
reconstruction / self-supervised setups) and needs no annotations.

Expected directory layout (what ``download_c2c12.sh`` produces)::

    root/
      090303-C2C12P15-FGF2,BMP2/
        exp1_F0001 Data/exp1_F0001-00001.tif ...
        Annotation_Computer/Computer exp1_F0001 Data.xml ...
        Annotation_Human/Human exp1_F0001 Data.xml ...
      090318-C2C12P7-FGF2,BMP2/
      090325-C2C12P12-FGF2,BMP2/
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

IMAGE_HEIGHT, IMAGE_WIDTH = 1040, 1392
MAX_INTENSITY = 4095.0  # 12-bit camera stored as uint16

STATUS_NEWBORN, STATUS_NORMAL, STATUS_DIVIDING = 2, 4, 7

_FIELD_RE = re.compile(r"(F\d{4})")
_FRAME_RE = re.compile(r"-(\d+)\.tif$", re.IGNORECASE)

# Annotation sources: (sub-folder, filename pattern for a field)
_ANNOTATION_FILES = {
    "computer": ("Annotation_Computer", "Computer exp1_{field} Data.xml"),
    "human": ("Annotation_Human", "Human exp1_{field} Data.xml"),
    "human_full": ("Annotation_Human", "Human Full Annotation_exp1_{field} Data.xml"),
}


def _field_of(name: str) -> str | None:
    m = _FIELD_RE.search(name)
    return m.group(1) if m else None


def _frame_of(name: str) -> int | None:
    m = _FRAME_RE.search(name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Annotation parsing
# --------------------------------------------------------------------------- #
@dataclass
class Tracks:
    """All annotated centroids of one field of view, sorted by frame (0-based)."""

    frame: np.ndarray  # int32 [N]
    xy: np.ndarray  # float32 [N, 2], pixel coordinates (x = column, y = row)
    cell_id: np.ndarray  # int32 [N]
    parent_id: np.ndarray  # int32 [N], 0 = no parent (root of a lineage)
    status: np.ndarray  # int16 [N]
    interpolated: np.ndarray  # bool [N]

    def __post_init__(self):
        frames = np.arange(self.frame.max() + 2 if len(self.frame) else 1)
        self._offsets = np.searchsorted(self.frame, frames)

    def __getitem__(self, mask) -> Tracks:
        return Tracks(
            self.frame[mask],
            self.xy[mask],
            self.cell_id[mask],
            self.parent_id[mask],
            self.status[mask],
            self.interpolated[mask],
        )

    def at(self, frame: int) -> Tracks:
        """Centroids in a single 0-based frame."""
        if frame < 0 or frame + 1 >= len(self._offsets):
            return self[slice(0, 0)]
        return self[slice(self._offsets[frame], self._offsets[frame + 1])]

    def annotated_frames(self) -> np.ndarray:
        return np.unique(self.frame)


def parse_annotation_xml(path: str | os.PathLike) -> Tracks:
    """Parse a C2C12 tracking XML (AnnotationDocument v2.1) into flat arrays."""
    rows = []

    def visit(a: ET.Element, parent: int) -> None:
        cid = int(a.attrib["id"])
        ss = a.find("ss")
        if ss is not None:
            for s in ss.iter("s"):
                x, y = float(s.get("x", "nan")), float(s.get("y", "nan"))
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue  # track-end markers (status 11/18) carry no position
                rows.append((int(s.attrib["i"]), x, y, cid, parent, int(s.get("s", 0)), s.get("f") == "I"))
        children = a.find("as")
        if children is not None:
            for child in children.findall("a"):
                visit(child, cid)

    root = ET.parse(path).getroot()
    for f in root.iter("f"):  # annotation folders (normally just "Cells")
        top = f.find("as")
        if top is not None:
            for a in top.findall("a"):
                visit(a, 0)

    if not rows:
        empty = np.zeros(0)
        return Tracks(
            empty.astype(np.int32),
            np.zeros((0, 2), np.float32),
            empty.astype(np.int32),
            empty.astype(np.int32),
            empty.astype(np.int16),
            empty.astype(bool),
        )
    frame, x, y, cid, parent, status, interp = map(np.asarray, zip(*rows, strict=True))
    order = np.argsort(frame, kind="stable")
    return Tracks(
        frame=frame[order].astype(np.int32),
        xy=np.stack([x, y], 1)[order].astype(np.float32),
        cell_id=cid[order].astype(np.int32),
        parent_id=parent[order].astype(np.int32),
        status=status[order].astype(np.int16),
        interpolated=interp[order].astype(bool),
    )


# --------------------------------------------------------------------------- #
# Point -> dense target rendering
# --------------------------------------------------------------------------- #
def points_to_instance_mask(xy: np.ndarray, labels: np.ndarray, shape: tuple[int, int], radius: float) -> np.ndarray:
    """Label every pixel within ``radius`` of a centroid with that cell's label.

    Overlapping disks are split by nearest centroid (a Voronoi partition clipped to
    the disks), so touching cells stay separate instances. 0 = background.
    """
    h, w = shape
    out = np.zeros(shape, np.int32)
    best = np.full(shape, np.inf, np.float32)
    r = int(np.ceil(radius))
    for (x, y), lab in zip(xy, labels, strict=True):
        cx, cy = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(cx - r, 0), min(cx + r + 1, w), max(cy - r, 0), min(cy + r + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        win = (d2 <= radius**2) & (d2 < best[y0:y1, x0:x1])
        best[y0:y1, x0:x1][win] = d2[win]
        out[y0:y1, x0:x1][win] = lab
    return out


def points_to_heatmap(xy: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Max-of-Gaussians heatmap (peak 1.0 at each centroid), for detection/counting."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    r = int(np.ceil(3 * sigma))
    for x, y in xy:
        cx, cy = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(cx - r, 0), min(cx + r + 1, w), max(cy - r, 0), min(cy + r + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)).astype(np.float32)
        np.maximum(out[y0:y1, x0:x1], g, out=out[y0:y1, x0:x1])
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def read_tiff(path: str | os.PathLike) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


class C2C12Dataset(Dataset):
    """Frame-level dataset over the C2C12 time-lapse sequences.

    Args:
        root: dataset root (see module docstring for the layout).
        experiments: experiment folder names to use (default: all present under root).
        fields: field-of-view ids such as ``["F0001", "F0002"]`` (default: all present).
            Split train/val/test by experiment or field, not by frame: consecutive
            frames are nearly identical.
        annotation: ``"computer"`` (all cells, default), ``"human"`` (sparse lineages,
            background is NOT reliable), ``"human_full"`` (090303/F0009 only), or
            ``None`` (images only).
        target: what ``sample["target"]`` holds:
            ``"instance"`` int64 [H, W] mask whose labels are the track cell ids
            (consistent across frames), ``"semantic"`` int64 [H, W] {0 bg, 1 cell}
            (``class_by_status=True`` adds 2 = dividing/newborn), ``"heatmap"`` float
            [1, H, W] Gaussians, ``"points"`` float [N, 2] (x, y), ``"image"`` the
            input image itself (identity target), or ``None``.
        radius: disk radius in pixels for instance/semantic masks.
        sigma: Gaussian sigma in pixels for heatmaps.
        include_interpolated: keep human points marked as interpolated (``f="I"``).
        skip_unannotated: drop frames with no annotated cell (e.g. frames past the
            end of a track file). Ignored when no annotation is loaded.
        frame_step: use every n-th frame (subsampling the highly redundant sequence).
        frame_range: optional ``(start, stop)`` of 0-based frames to use (stop exclusive).
        normalize: ``"fixed"`` (/4095 -> [0, 1]), ``"percentile"`` (per-image 1-99.8%
            min-max, clipped), or ``None`` (raw float values).
        transform: callable ``sample -> sample`` applied last (see :class:`RandomCropFlip`).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        experiments: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
        annotation: str | None = "computer",
        target: str | None = "instance",
        radius: float = 12.0,
        sigma: float = 3.0,
        class_by_status: bool = False,
        include_interpolated: bool = True,
        skip_unannotated: bool = True,
        frame_step: int = 1,
        frame_range: tuple[int, int] | None = None,
        normalize: str | None = "fixed",
        transform: Callable[[dict], dict] | None = None,
    ):
        if annotation not in (*_ANNOTATION_FILES, None):
            raise ValueError(f"annotation must be one of {list(_ANNOTATION_FILES)} or None")
        if target not in ("instance", "semantic", "heatmap", "points", "image", None):
            raise ValueError(f"unknown target {target!r}")
        if target in ("instance", "semantic", "heatmap", "points") and annotation is None:
            raise ValueError(f"target={target!r} requires an annotation source")
        if normalize not in ("fixed", "percentile", None):
            raise ValueError(f"unknown normalize {normalize!r}")

        self.root = Path(root)
        self.annotation, self.target = annotation, target
        self.radius, self.sigma, self.class_by_status = radius, sigma, class_by_status
        self.normalize, self.transform = normalize, transform

        exp_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        if experiments is not None:
            exp_dirs = [d for d in exp_dirs if d.name in experiments]

        self.sequences: list[tuple[str, str]] = []  # (experiment, field)
        self.tracks: list[Tracks | None] = []
        self.samples: list[tuple[int, int, Path]] = []  # (sequence idx, 0-based frame, path)
        for exp_dir in exp_dirs:
            for seq_dir in sorted(exp_dir.glob("exp*_F* Data")):
                field = _field_of(seq_dir.name)
                if field is None or (fields is not None and field not in fields):
                    continue
                tracks = None
                if annotation is not None:
                    sub, pattern = _ANNOTATION_FILES[annotation]
                    xml = exp_dir / sub / pattern.format(field=field)
                    if not xml.exists():
                        continue  # no annotation of this kind for this field
                    tracks = parse_annotation_xml(xml)
                    if not include_interpolated:
                        tracks = tracks[~tracks.interpolated]
                # (0-based frame, path); TIFF numbers are 1-based
                tifs = sorted((n - 1, p) for p in seq_dir.glob("*.tif") if (n := _frame_of(p.name)) is not None)
                if frame_range is not None:
                    tifs = [(f, p) for f, p in tifs if frame_range[0] <= f < frame_range[1]]
                tifs = tifs[::frame_step]
                if tracks is not None and skip_unannotated:
                    annotated = set(tracks.annotated_frames().tolist())
                    tifs = [(f, p) for f, p in tifs if f in annotated]
                if not tifs:
                    continue
                seq_idx = len(self.sequences)
                self.sequences.append((exp_dir.name, field))
                self.tracks.append(tracks)
                self.samples += [(seq_idx, f, p) for f, p in tifs]

        if not self.samples:
            raise RuntimeError(f"No frames found under {self.root} for the given selection")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = read_tiff(path).astype(np.float32)
        if self.normalize == "fixed":
            img /= MAX_INTENSITY
        elif self.normalize == "percentile":
            lo, hi = np.percentile(img, (1.0, 99.8))
            img = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return torch.from_numpy(img)[None]  # [1, H, W]

    def __getitem__(self, idx: int) -> dict:
        seq_idx, frame, path = self.samples[idx]
        image = self._load_image(path)
        shape = (image.shape[-2], image.shape[-1])
        sample = {
            "image": image,
            "experiment": self.sequences[seq_idx][0],
            "field": self.sequences[seq_idx][1],
            "frame": frame,  # 0-based, as in the XML
        }

        tracks = self.tracks[seq_idx]
        pts = tracks.at(frame) if tracks is not None else None
        if pts is not None:
            sample.update(
                points=torch.from_numpy(pts.xy.copy()),
                cell_ids=torch.from_numpy(pts.cell_id.astype(np.int64)),
                parent_ids=torch.from_numpy(pts.parent_id.astype(np.int64)),
                status=torch.from_numpy(pts.status.astype(np.int64)),
            )

        if self.target == "image":
            sample["target"] = image.clone()
        elif self.target is not None:
            assert pts is not None  # __init__ requires an annotation source for the other targets
            if self.target == "instance":
                m = points_to_instance_mask(pts.xy, pts.cell_id, shape, self.radius)
                sample["target"] = torch.from_numpy(m.astype(np.int64))
            elif self.target == "semantic":
                labels = np.ones(len(pts.xy), np.int32)
                if self.class_by_status:
                    labels[np.isin(pts.status, (STATUS_NEWBORN, STATUS_DIVIDING))] = 2
                m = points_to_instance_mask(pts.xy, labels, shape, self.radius)
                sample["target"] = torch.from_numpy(m.astype(np.int64))
            elif self.target == "heatmap":
                sample["target"] = torch.from_numpy(points_to_heatmap(pts.xy, shape, self.sigma))[None]
            elif self.target == "points":
                sample["target"] = sample["points"]

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# --------------------------------------------------------------------------- #
# Transforms / collation
# --------------------------------------------------------------------------- #
class RandomCropFlip:
    """Joint random crop + flips + 90-degree rotations of image, dense target and points.

    Points that fall outside the crop are dropped (together with their ids/status).
    """

    _DENSE = ("image", "target")
    _PER_POINT = ("points", "cell_ids", "parent_ids", "status")

    def __init__(
        self,
        size: int | tuple[int, int] | None = 256,
        flip: bool = True,
        rot90: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.size = (size, size) if isinstance(size, int) else size
        self.flip, self.rot90, self.g = flip, rot90, generator

    def _randint(self, n: int) -> int:
        return int(torch.randint(n, (1,), generator=self.g))

    def __call__(self, s: dict) -> dict:
        s = dict(s)
        h, w = s["image"].shape[-2:]
        dense = [
            k for k in self._DENSE if k in s and torch.is_tensor(s[k]) and s[k].dim() >= 2 and s[k].shape[-2:] == (h, w)
        ]
        has_pts = "points" in s
        target_is_points = has_pts and s.get("target") is s["points"]

        if self.size is not None:
            ch, cw = min(self.size[0], h), min(self.size[1], w)
            y0, x0 = self._randint(h - ch + 1), self._randint(w - cw + 1)
            for k in dense:
                s[k] = s[k][..., y0 : y0 + ch, x0 : x0 + cw]
            if has_pts:
                p = s["points"] - torch.tensor([x0, y0], dtype=s["points"].dtype)
                keep = (p[:, 0] >= 0) & (p[:, 0] < cw) & (p[:, 1] >= 0) & (p[:, 1] < ch)
                s["points"] = p
                for k in self._PER_POINT:
                    if k in s:
                        s[k] = s[k][keep]
            h, w = ch, cw

        if self.flip and self._randint(2):  # horizontal
            for k in dense:
                s[k] = s[k].flip(-1)
            if has_pts:
                s["points"] = torch.stack([w - 1 - s["points"][:, 0], s["points"][:, 1]], 1)
        if self.flip and self._randint(2):  # vertical
            for k in dense:
                s[k] = s[k].flip(-2)
            if has_pts:
                s["points"] = torch.stack([s["points"][:, 0], h - 1 - s["points"][:, 1]], 1)
        if self.rot90 and h == w and self._randint(2):  # 90 deg counter-clockwise
            for k in dense:
                s[k] = s[k].rot90(1, (-2, -1))
            if has_pts:
                s["points"] = torch.stack([s["points"][:, 1], w - 1 - s["points"][:, 0]], 1)

        for k in dense:
            s[k] = s[k].contiguous()
        if target_is_points:
            s["target"] = s["points"]
        return s


def c2c12_collate(batch: list[dict]) -> dict:
    """Stack fixed-size tensors; keep variable-length ones (points, ids) as lists."""
    out: dict[str, Any] = {}
    per_point = set(RandomCropFlip._PER_POINT)
    if batch[0].get("target") is not None and batch[0].get("target") is batch[0].get("points"):
        per_point.add("target")
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]) and all(v.shape == vals[0].shape for v in vals) and k not in per_point:
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out
