"""Loader for the Bürger et al. cancer-cell dataset (DOI 10.4126/FRL01-006526608).

Builds one `CellSequence` per cell that dies or divides, with `patch`x`patch` crops
around the cell in every alive frame. Expects the layout of the release:

    <root>/annotations.json
    <root>/raw/<video>/images/<frame>.tif      (1024x1024x3, uint16)

Only videos that are already extracted under raw/ are used. `load_video` / `write_gt_ctc` give the
raw video, the GT masks and the GT tracks in CTC layout (frames renumbered from 0) for tracking.
"""

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from . import CellSequence

T0, T1 = 49, 234  # annotated frame range
LO, HI = 150.0, 30000.0  # log intensity window mapped to uint8


def to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.maximum(img.astype(np.float32), LO)
    return np.clip(255 * np.log(x / LO) / np.log(HI / LO), 0, 255).astype(np.uint8)


def cell_table(root: Path, videos: list[str]) -> list[dict]:
    """Per-cell frames, boxes and fate from annotations.json (cached)."""
    cache = root / f"cells_{videos[0]}_{videos[-1]}_{len(videos)}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    d = json.load(open(root / "annotations.json"))
    vid_name = {v["id"]: v["name"] for v in d["videos"]}
    keep = {i for i, n in vid_name.items() if n in set(videos)}
    img_file = {im["id"]: im["file_name"] for im in d["images"]}
    cells: dict = {}
    for a in d["annotations"]:
        if a["video_id"] not in keep or a["category_id"] != 1:
            continue
        c = cells.setdefault(
            (a["video_id"], a["cell_id"]),
            dict(video=vid_name[a["video_id"]], cell=a["cell_id"], frames=[], boxes=[],
                 files=[], death=None, division=None),
        )
        c["frames"].append(a["time_step"])
        c["boxes"].append(a["bbox"][:2])
        c["files"].append(img_file[a["image_id"]])
        c["death"] = c["death"] if c["death"] is not None else a["time_of_death"]
        c["division"] = c["division"] if c["division"] is not None else a["time_of_division"]
    out = []
    for c in cells.values():
        if c["death"] is None and c["division"] is None:
            continue  # neither dies nor divides within the video -> no fate label
        o = np.argsort(c["frames"])
        c["frames"] = [c["frames"][i] for i in o]
        c["boxes"] = [c["boxes"][i] for i in o]
        c["files"] = [c["files"][i] for i in o]
        c["fate"] = 0 if c["death"] is not None else 1
        out.append(c)
    pickle.dump(out, open(cache, "wb"))
    return out


def load_sequences(root: str | Path, videos: list[str], patch: int = 112,
                   min_len: int = 10, max_len: int = 48) -> list[CellSequence]:
    """Crops of the last `max_len` alive frames of every dying/dividing cell."""
    root = Path(root)
    table = [c for c in cell_table(root, videos) if len(c["frames"]) >= min_len]
    h = patch // 2
    by_file: dict[str, list] = {}
    for ci, c in enumerate(table):
        for k in range(max(0, len(c["frames"]) - max_len), len(c["frames"])):
            by_file.setdefault(c["files"][k], []).append((ci, k))
    crops = {ci: {} for ci in range(len(table))}
    for f, items in tqdm(sorted(by_file.items()), desc="Cropping", disable=not sys.stderr.isatty()):
        img = to_uint8(tifffile.imread(root / "raw" / f))
        img = np.pad(img, ((h, h), (h, h), (0, 0)))
        for ci, k in items:
            x, y = (np.array(table[ci]["boxes"][k]) * 1024).astype(int)
            crops[ci][k] = img[y:y + patch, x:x + patch].transpose(2, 0, 1)  # padded coords = y-h+h
    seqs = []
    for ci, c in enumerate(table):
        ks = sorted(crops[ci])
        seqs.append(CellSequence(c["video"], c["cell"], np.stack([crops[ci][k] for k in ks]),
                                 c["fate"], c["frames"][ks[-1]]))
    return seqs



def load_video(root: Path, video: str, channel: str = "mean"):
    """imgs (T, Y, X) float32 from the 3-channel images, GT instance masks (T, Y, X)."""
    imgs, masks = [], []
    for f in range(T0, T1 + 1):
        im = tifffile.imread(root / "raw" / video / "images" / f"{f:03d}.tif").astype(np.float32)
        imgs.append(im.mean(-1) if channel == "mean" else im[..., int(channel)])
        masks.append(tifffile.imread(root / "raw" / video / "ctc_files" / f"man_track{f:03d}.tif"))
    return np.stack(imgs), np.stack(masks)


def write_gt_ctc(root: Path, video: str, outdir: Path):
    """GT in CTC layout with frames renumbered from 0 (as in Trackastra's output)."""
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    for f in range(T0, T1 + 1):
        tifffile.imwrite(outdir / f"man_track{f - T0:03d}.tif",
                         tifffile.imread(root / "raw" / video / "ctc_files" / f"man_track{f:03d}.tif"))
    t = pd.read_csv(root / "raw" / video / "ctc_files" / "man_track.txt", sep=" ", header=None,
                    names=["label", "t1", "t2", "parent"])
    t["t1"] -= T0
    t["t2"] -= T0
    t.to_csv(outdir / "man_track.txt", sep=" ", header=False, index=False)
