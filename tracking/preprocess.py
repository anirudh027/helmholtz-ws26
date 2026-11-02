"""Per-frame contrast enhancement before segmentation / tracking.

    imgs = enhance(imgs, "clahe")          # (T, Y, X) float32 in [0, 1]

Methods:
    none        as is
    percentile  per-frame 1-99.8 percentile stretch
    hist_eq     global histogram equalisation per frame
    clahe       contrast-limited adaptive histogram equalisation (local contrast;
                lifts dim cells next to bright ones and flattens uneven illumination)
"""

import numpy as np

METHODS = ("none", "percentile", "hist_eq", "clahe")


def _stretch(f, lo=1, hi=99.8):
    a, b = np.percentile(f, [lo, hi])
    return np.clip((f - a) / max(b - a, 1e-6), 0, 1)


def enhance(imgs, method="clahe", clip_limit=0.01, kernel=None):
    """Return enhanced frames (float32, [0, 1]). ``kernel``: CLAHE tile size in px
    (default 1/8 of the image side)."""
    from skimage import exposure

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, choose from {METHODS}")
    imgs = np.asarray(imgs, dtype=np.float32)
    if method == "none":
        return imgs
    out = np.empty_like(imgs)
    for t, f in enumerate(imgs):
        f = _stretch(f)
        if method == "hist_eq":
            f = exposure.equalize_hist(f)
        elif method == "clahe":
            f = exposure.equalize_adapthist(f, kernel_size=kernel, clip_limit=clip_limit)
        out[t] = f
    return out
