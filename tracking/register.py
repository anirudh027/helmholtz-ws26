"""Global stage-drift compensation (rigid translation per frame).

In the Helmholtz timelapse the stage returns to the well with a ~10 px error
every frame (median 9.5 px, jumps up to 30 px, one ~660 px jump), which
breaks linking: in Damià's audit ~80% of the apparent cell motion was the
stage, and registering first raised the linked fraction from 0.87 to 0.95.

    off = estimate_drift(imgs)                       # (T, 2) int, canvas positions
    imgs_c, masks_c = to_canvas(imgs, off), to_canvas(masks, off)
    ... track on the canvas ...
    masks = from_canvas(masks_c, off, imgs.shape[1:])

Ported from Cell-project ``scripts/stabilize_helmholtz.py``.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

MARGIN = 8


def _highpass(img, down=2, sigma=20):
    x = np.asarray(img, dtype=np.float32)
    x = x - gaussian_filter(x, sigma)
    if down != 1:
        x = zoom(x, 1 / down, order=1)
    return (x - x.mean()) / (x.std() + 1e-6)


def _overlap_ncc(a, b, d, min_overlap=150):
    """NCC of the high-passed overlap when ``b`` is displaced by ``d`` w.r.t. ``a``."""
    H, W = a.shape
    d = np.round(d).astype(int)
    y0, y1 = max(0, d[0]), min(H, H + d[0])
    x0, x1 = max(0, d[1]), min(W, W + d[1])
    if y1 - y0 < min_overlap or x1 - x0 < min_overlap:
        return None
    pa = _highpass(a[y0:y1, x0:x1], down=1)
    pb = _highpass(b[y0 - d[0]:y1 - d[0], x0 - d[1]:x1 - d[1]], down=1)
    return float((pa * pb).mean())


def _resolve_alias(a, b, d):
    """Phase correlation is circular: shift s is indistinguishable from s +/- the
    image size. Test the aliases on the real overlap and keep the best."""
    H, W = a.shape
    cands = [(v, c) for ay in (0, H, -H) for ax in (0, W, -W)
             if (v := _overlap_ncc(a, b, (c := d + np.array([ay, ax])))) is not None]
    return max(cands, key=lambda z: z[0])[1] if cands else d


def estimate_drift(imgs, down=2, blank_frac=0.3, min_ncc=0.03, alias_px=100):
    """Per-frame canvas offsets (T, 2) int so that ``to_canvas`` aligns the stage.

    Shifts are estimated by phase correlation on high-passed, downscaled images
    against the last two usable frames, keeping the estimate whose overlap
    correlates best (normalised cross-correlation, NCC); shifts above
    ``alias_px`` are checked against the circular aliases. A frame keeps the
    previous offset (and is not used as a reference) if it is blank (std below
    ``blank_frac`` x the median std; checking the NCC alone is not enough, since
    dust on the optics correlates at zero shift) or its best NCC is below
    ``min_ncc`` (keep it low: a real 176-px jump at frame 37 of the Helmholtz
    video correlates at only 0.08, because cells move and the overlap shrinks). On the Helmholtz video: blank frames <= 0.29 x
    median std, defocused 0.28-0.36, good >= 0.38.
    """
    from skimage.registration import phase_cross_correlation

    imgs = np.asarray(imgs)
    stds = np.array([float(np.std(f)) for f in imgs])
    off = np.zeros((len(imgs), 2))
    refs = []  # the last two usable frames: (t, high-passed image)
    for t in range(len(imgs)):
        if not stds[t] > blank_frac * np.median(stds):
            off[t] = off[t - 1] if t else 0
            continue
        hp = _highpass(imgs[t], down)
        best = (-np.inf, None)
        for r, r_hp in refs:
            s, _, _ = phase_cross_correlation(r_hp, hp, upsample_factor=20)
            d = s * down
            if np.abs(d).max() > alias_px:
                d = _resolve_alias(imgs[r], imgs[t], d)
            ncc = _overlap_ncc(imgs[r], imgs[t], d)
            if ncc is not None and ncc > best[0]:
                best = (ncc, off[r] + d)
        if not refs:
            refs = [(t, hp)]
        elif best[0] >= min_ncc:
            off[t] = best[1]
            refs = (refs + [(t, hp)])[-2:]
        else:
            off[t] = off[t - 1]
    off = np.round(off).astype(int)
    return off - off.min(0)


def canvas_shape(shape, offsets, margin=MARGIN):
    H, W = shape[-2:]
    return H + int(offsets[:, 0].max()) + 2 * margin, W + int(offsets[:, 1].max()) + 2 * margin


def to_canvas(stack, offsets, margin=MARGIN, fill=None):
    """Place every frame at its offset on a common canvas. ``fill`` defaults to 0
    for integer stacks (masks) and to each frame's median for images."""
    stack = np.asarray(stack)
    H, W = stack.shape[-2:]
    CH, CW = canvas_shape(stack.shape, offsets, margin)
    out = np.zeros((len(stack), CH, CW), dtype=stack.dtype)
    for t, (f, (dy, dx)) in enumerate(zip(stack, offsets)):
        if fill is None and not np.issubdtype(stack.dtype, np.integer):
            out[t] = np.median(f)
        elif fill is not None:
            out[t] = fill
        out[t, dy + margin:dy + margin + H, dx + margin:dx + margin + W] = f
    return out


def from_canvas(stack, offsets, shape, margin=MARGIN):
    """Inverse of ``to_canvas``: crop each frame back to the original field of view."""
    H, W = shape[-2:]
    return np.stack([f[dy + margin:dy + margin + H, dx + margin:dx + margin + W]
                     for f, (dy, dx) in zip(stack, offsets)])
