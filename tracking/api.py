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
from .ctc import CTCResult, tracks_from_assignment

# name -> function(imgs, masks, **kwargs) -> CTCResult
TRACKERS = {
    "trackastra": partial(backends.track_trackastra, mode="greedy"),
    "trackastra_ilp": partial(backends.track_trackastra, mode="ilp"),
    "laptrack": backends.track_laptrack,
    "ultrack": backends.track_ultrack,
    "hoct": backends.track_hoct,  # == `hoct track` CLI
    "hoct_tta": partial(backends.track_hoct, tta=6),  # paper's best configuration
    "hoct_koft": backends.track_hoct_koft,
    "hoct_koft_tta": partial(backends.track_hoct_koft, tta=6),
    # + gap filling across up to 8 missing frames (HOCT links stay <= 3 frames, so its
    # division decisions are unchanged): for flickering masks, e.g. the Helmholtz reporter
    "hoct_koft_tta_long": partial(backends.track_hoct_koft, tta=6, max_gap=8),
}


def track(imgs, masks, model="trackastra", register=False, **kwargs) -> CTCResult:
    """Track a 2-D time-lapse given its instance segmentation.

    Args:
        imgs: raw video, (T, Y, X).
        masks: instance segmentation, same shape; labels unique per frame,
            0 = background. Trackers do not segment, so supply these.
        model: one of ``TRACKERS``.
        register: compensate global stage drift first (``tracking.register``), track
            on the aligned canvas and map the result back to the original frames.
            Needed for the Helmholtz timelapse; harmless on static-stage data.
            ``True`` estimates the drift on ``imgs``; pass a (T, Y, X) array to estimate
            it on another channel (e.g. bright-field when tracking a sparse reporter).
        **kwargs: passed to the backend (e.g. ``max_distance``).
    """
    if model not in TRACKERS:
        raise ValueError(f"Unknown model {model!r}, choose from {list(TRACKERS)}")
    imgs, masks = np.asarray(imgs), np.asarray(masks)
    if imgs.shape != masks.shape:
        raise ValueError(f"imgs {imgs.shape} and masks {masks.shape} must match")
    t0 = time.time()
    if register is False or register is None:
        res = TRACKERS[model](imgs, masks, **kwargs)
    else:
        from .register import estimate_drift, from_canvas, to_canvas

        off = estimate_drift(imgs if register is True else register)
        res = TRACKERS[model](to_canvas(imgs, off), to_canvas(masks, off), **kwargs)
        info = res.info
        res = _relabel(from_canvas(res.masks, off, imgs.shape), res.tracks)
        res.info.update(info)
        res.info.update(drift_px=float(np.median(np.hypot(*np.diff(off, axis=0).T))) if len(off) > 1 else 0.0)
    res.info.update(model=model, seconds=time.time() - t0, frames=int(imgs.shape[0]),
                    registered=register is not False and register is not None)
    return res


def _relabel(masks, tracks) -> CTCResult:
    """Rebuild a CTC-legal lineage after masks were edited (e.g. cropped back from
    the canvas, which can empty a track's frame and split it)."""
    frame_map = [{int(i): int(i) for i in np.unique(m) if i} for m in masks]
    return tracks_from_assignment(masks, frame_map, {i: p for i, (_, _, p) in tracks.items() if p})
