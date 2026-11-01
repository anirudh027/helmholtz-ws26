"""Tracking backends. Each takes ``imgs`` (T, Y, X) and instance ``masks``
(T, Y, X, labels unique per frame, 0 = background) and returns a CTCResult.

Settings mirror Damià's Cell-project benchmark (scripts/run_*.py) so numbers
are comparable with his results tables.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tifffile

from .ctc import CTCResult, centroids_of, read_ctc, tracks_from_assignment

# Scratch space for solver DBs / CLI hand-offs; keep it off the small root disk.
TMP_ROOT = os.environ.get("TRACKING_TMP", "/mnt/1t/tracks/.tmp")


@contextmanager
def _scratch(prefix):
    root = TMP_ROOT if os.path.isdir(os.path.dirname(TMP_ROOT)) else None
    if root:
        os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix=prefix, dir=root)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _device(device):
    if device != "automatic":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_TRACKASTRA_MODELS = {}


def track_trackastra(imgs, masks, mode="greedy", pretrained="general_2d", device="automatic", **kwargs):
    """Trackastra (transformer association scores). Adapted from Nikolas Rieger's tracking.py."""
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc

    key = (pretrained, device)
    if key not in _TRACKASTRA_MODELS:
        _TRACKASTRA_MODELS[key] = Trackastra.from_pretrained(pretrained, device=_device(device))
    model = _TRACKASTRA_MODELS[key]
    graph, _ = model.track(imgs.astype(np.float32), masks, mode=mode, **kwargs)
    ctc_df, masks_tracked = graph_to_ctc(graph, masks)
    tracks = {int(r[0]): (int(r[1]), int(r[2]), int(r[3])) for r in ctc_df.itertuples(index=False)}
    return CTCResult(np.asarray(masks_tracked, dtype=np.uint16), tracks)


def track_laptrack(imgs, masks, max_distance=40.0):
    """LapTrack: TrackMate-style two-pass LAP on centroids, with divisions."""
    from laptrack import LapTrack

    rows = [
        {"frame": t, "label": lbl, "y": cy, "x": cx}
        for t, m in enumerate(masks)
        for lbl, (cy, cx) in centroids_of(m).items()
    ]
    spots = pd.DataFrame(rows)
    d = max_distance
    lt = LapTrack(
        track_cost_cutoff=d**2,
        splitting_cost_cutoff=(1.2 * d) ** 2,
        gap_closing_cost_cutoff=(1.5 * d) ** 2,
        gap_closing_max_frame_count=2,
        merging_cost_cutoff=False,
    )
    track_df, split_df, _ = lt.predict_dataframe(
        spots, coordinate_cols=["y", "x"], frame_col="frame", only_coordinate_cols=False)
    track_df = track_df.reset_index()

    frame_map = [dict() for _ in masks]
    for r in track_df.itertuples(index=False):
        frame_map[int(r.frame)][int(r.label)] = int(r.track_id) + 1
    edges = {int(r.child_track_id) + 1: int(r.parent_track_id) + 1 for r in split_df.itertuples(index=False)}
    return tracks_from_assignment(masks, frame_map, edges)


def track_ultrack(imgs, masks, max_distance=40.0):
    """Ultrack on the whole sequence (no chunking), labels as the only hypothesis."""
    from ultrack import MainConfig, to_ctc
    from ultrack import track as ultrack_track
    from ultrack.utils import labels_to_contours

    detection, edges = labels_to_contours(masks, sigma=2.0)
    areas = [int((m > 0).sum() / max(1, len(np.unique(m)) - 1)) for m in masks if m.max() > 0]
    mean_area = int(np.mean(areas))

    with _scratch("ultrack_") as work:
        cfg = MainConfig()
        cfg.data_config.working_dir = work
        cfg.segmentation_config.min_area = max(20, mean_area // 6)
        cfg.segmentation_config.max_area = mean_area * 6
        cfg.segmentation_config.n_workers = 1
        cfg.linking_config.max_distance = max_distance
        cfg.linking_config.n_workers = 1
        cfg.tracking_config.division_weight = -0.05
        cfg.tracking_config.appear_weight = -0.25
        cfg.tracking_config.disappear_weight = -0.25
        ultrack_track(cfg, detection=detection, edges=edges, overwrite=True)
        res_dir = os.path.join(work, "RES")
        os.makedirs(res_dir)
        to_ctc(res_dir, cfg, overwrite=True)
        return read_ctc(res_dir)


def track_hoct(imgs, masks, model="general_v1", config=None, device="automatic"):
    """HOCT (higher-order candidate-link transformer + ILP) via its CLI."""
    hoct = shutil.which("hoct") or os.path.join(os.path.dirname(sys.executable), "hoct")
    with _scratch("hoct_") as work:
        img_t, seg_t = os.path.join(work, "imgs.tif"), os.path.join(work, "segm.tif")
        tifffile.imwrite(img_t, imgs)
        tifffile.imwrite(seg_t, masks)
        res_dir = os.path.join(work, "RES")
        cmd = [hoct, "track", img_t, seg_t, "-m", model, "-o", res_dir, "-f", "ctc", "--device", _device(device)]
        if config:
            cmd += ["-c", config]
        subprocess.run(cmd, check=True)
        return read_ctc(res_dir)


def track_hoct_koft(imgs, masks, **kwargs):
    """HOCT + KOFT-style optical-flow gap filling (Damià's hybrid, #1 in his benchmark)."""
    raise NotImplementedError(
        "hoct_koft is pending: waiting for Damià to push his hybrid into helmholtz-ws26 "
        "(Cell-project/scripts/run_hybrid.py). Use model='hoct' meanwhile.")
