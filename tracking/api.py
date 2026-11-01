"""Single entry point for cell tracking: the tracker is just an argument.

    from tracking import track
    res = track(imgs, masks, model="hoct")        # imgs/masks: (T, Y, X)
    graph, masks_tracked = res                    # Nikolas' original return shape
    res.tracks                                    # {track_id: (t_start, t_end, parent_id)}
    write_ctc("out/01_RES", res.masks, res.tracks)

Registry design from Nikolas Rieger's tracking.py (feat/tracking-trackastra).
"""

import time
from functools import partial

import numpy as np

from . import backends
from .ctc import CTCResult

# name -> function(imgs, masks, **kwargs) -> CTCResult
TRACKERS = {
    "trackastra": partial(backends.track_trackastra, mode="greedy"),
    "trackastra_ilp": partial(backends.track_trackastra, mode="ilp"),
    "laptrack": backends.track_laptrack,
    "ultrack": backends.track_ultrack,
    "hoct": backends.track_hoct,
    "hoct_koft": backends.track_hoct_koft,
}


def track(imgs, masks, model="trackastra", **kwargs) -> CTCResult:
    """Track a 2-D time-lapse given its instance segmentation.

    Args:
        imgs: raw video, (T, Y, X).
        masks: instance segmentation, same shape; labels unique per frame,
            0 = background. Trackers do not segment, so supply these.
        model: one of ``TRACKERS``.
        **kwargs: passed to the backend (e.g. ``max_distance``).
    """
    if model not in TRACKERS:
        raise ValueError(f"Unknown model {model!r}, choose from {list(TRACKERS)}")
    imgs, masks = np.asarray(imgs), np.asarray(masks)
    if imgs.shape != masks.shape:
        raise ValueError(f"imgs {imgs.shape} and masks {masks.shape} must match")
    t0 = time.time()
    res = TRACKERS[model](imgs, masks, **kwargs)
    res.info.update(model=model, seconds=time.time() - t0, frames=int(imgs.shape[0]))
    return res
