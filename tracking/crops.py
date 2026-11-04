"""Per-cell crop sequences from tracks: the input of fate prediction.

One sequence per *logical cell* (a CTC tracklet plus its gap continuations,
``evaluate.chain_ids``), from birth (or first frame) to division, death or the
end of the video, as in bozeklab's temporal-attention fate model
(arXiv 2603.16562), which classifies sequences of single-cell crops.

    cells, table = cell_crops(imgs, res, size=64)
    cells[cell_id]           # (n_frames, size, size) float32, centred on the cell
    table                    # DataFrame: cell, parent, t_start, t_end, divides, n_children,
                             #            ends_at_border, area_mean
    np.savez_compressed("crops.npz", **{str(k): v for k, v in cells.items()})
"""

import numpy as np
import pandas as pd

from .ctc import CTCResult, centroids_of
from .evaluate import chain_ids


def cell_crops(imgs, res: CTCResult, size=64, mask_out=False, border=10):
    """Crop every logical cell in every frame it is tracked.

    Args:
        imgs: raw video (T, Y, X).
        res: tracking result.
        size: crop side (px); crops are zero-padded at the image border.
        mask_out: zero the pixels outside the cell's own mask (removes neighbours).
        border: a cell whose last centroid is within this many px of the image
            border is flagged ``ends_at_border`` (it left the field: censored, not dead).
    """
    imgs = np.asarray(imgs)
    T, H, W = imgs.shape
    chain = chain_ids(res.tracks)
    h = size // 2
    pad = np.pad(imgs, ((0, 0), (h, h), (h, h)))
    crops, frames, areas, last_c = {}, {}, {}, {}
    for t in range(T):
        m = res.masks[t]
        for tid, (y, x) in centroids_of(m).items():
            cid = chain.get(tid, tid)
            yi, xi = int(round(y)), int(round(x))
            c = pad[t, yi:yi + size, xi:xi + size].astype(np.float32)
            if mask_out:
                mm = np.pad(m == tid, h)[yi:yi + size, xi:xi + size]
                c = c * mm
            crops.setdefault(cid, []).append(c)
            frames.setdefault(cid, []).append(t)
            areas.setdefault(cid, []).append(int((m == tid).sum()))
            last_c[cid] = (y, x)

    # lineage between logical cells: a division is a tracklet with >= 2 children
    children = {}
    for i, (_, _, p) in res.tracks.items():
        if p:
            children.setdefault(chain.get(p, p), set()).add(chain.get(i, i))
    parent_of = {}
    for i, (_, _, p) in res.tracks.items():
        ci, cp = chain.get(i, i), chain.get(p, p) if p else 0
        if cp and cp != ci:
            parent_of[ci] = cp
    rows = []
    for cid, ts in frames.items():
        y, x = last_c[cid]
        n_kids = len(children.get(cid, set()) - {cid})
        rows.append(dict(cell=cid, parent=parent_of.get(cid, 0), t_start=min(ts), t_end=max(ts),
                         n_frames=len(ts), divides=n_kids >= 2, n_children=n_kids,
                         ends_at_border=bool(min(y, x, H - 1 - y, W - 1 - x) < border),
                         ends_at_last_frame=max(ts) == T - 1, area_mean=float(np.mean(areas[cid]))))
    cells = {cid: np.stack(v) for cid, v in crops.items()}
    return cells, pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)
