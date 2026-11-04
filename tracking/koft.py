"""KOFT-style gap filling on top of any tracker's output (the ``hoct_koft`` hybrid).

Ported from Damià's Cell-project ``scripts/run_hybrid.py``. The two mechanisms
come from KOFT (Reme et al., "Particle tracking in biological images with
optical-flow enhanced Kalman filtering", ISBI 2024):

1. Gap filling: HOCT may link a parent to a single child across 1-3 missing
   frames. CTC counts those frames as missed detections (FN weight 10 in AOGM),
   so the child's first mask is advected backwards through the gap along
   Farneback optical flow and painted into the empty frames.
2. Motion-predicted gap closing: remaining track ends are matched to later
   track births (Hungarian, constant-velocity prediction), then filled the same way.

Each fill is appearance-gated: the advected position must still show cell
texture (patch contrast relative to the cell and to random background patches),
so we bridge segmentation dropouts without inventing cells. The child is
extended backwards rather than merged into the parent, so the parent's and the
child's original endpoints survive (CTC's CT metric needs exact endpoints).
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .ctc import CTCResult


class _GapFiller:
    def __init__(self, imgs, masks, tracks, max_gap, max_dist, flow_downscale, appearance_thresh):
        self.imgs = imgs
        self.masks = [np.array(m, dtype=np.uint16) for m in masks]
        self.tracks = {i: list(v) for i, v in tracks.items()}
        self.T = len(self.masks)
        self.max_gap, self.max_dist = max_gap, max_dist
        self.ds, self.thresh = flow_downscale, appearance_thresh
        self._small = {}
        self._rng = np.random.default_rng(0)
        self.cent = {}
        for t, m in enumerate(self.masks):
            ids = np.unique(m)
            ids = ids[ids > 0]
            if len(ids):
                for i, c in zip(ids, ndi.center_of_mass(m > 0, m, ids)):
                    self.cent[(int(i), t)] = np.array(c)

    # -- image helpers -------------------------------------------------------
    def small(self, t):
        import cv2

        if t not in self._small:
            img = np.asarray(self.imgs[t], dtype=np.float32)
            lo, hi = np.percentile(img, [1, 99.5])
            img = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            self._small[t] = cv2.resize(img, (img.shape[1] // self.ds, img.shape[0] // self.ds))
        return self._small[t]

    def patch(self, t, c, h, w):
        img = self.imgs[t]
        y0 = int(np.clip(round(c[0] - h / 2), 0, img.shape[0] - h))
        x0 = int(np.clip(round(c[1] - w / 2), 0, img.shape[1] - w))
        return np.asarray(img[y0:y0 + h, x0:x0 + w], dtype=np.float32)

    def velocity(self, tid, t_end, k=3):
        pts = [self.cent.get((tid, t)) for t in range(t_end - k, t_end + 1)]
        pts = [p for p in pts if p is not None]
        return (pts[-1] - pts[0]) / (len(pts) - 1) if len(pts) >= 2 else np.zeros(2)

    def flow_path(self, c0, c1, ta, tb):
        """Advect c0 from frame ta with Farneback flow, then spread the drift
        linearly so the path ends exactly at c1 (frame tb)."""
        import cv2

        pts, p = [c0.copy()], c0.copy()
        for t in range(ta, tb - 1):
            flow = cv2.calcOpticalFlowFarneback(self.small(t), self.small(t + 1), None, 0.5, 3, 15, 3, 5, 1.2, 0)
            yi = int(np.clip(p[0] / self.ds, 0, flow.shape[0] - 1))
            xi = int(np.clip(p[1] / self.ds, 0, flow.shape[1] - 1))
            p = p + np.array([flow[yi, xi, 1], flow[yi, xi, 0]]) * self.ds
            pts.append(p.copy())
        n = tb - ta
        pred_end = pts[-1] + (pts[-1] - pts[-2] if len(pts) > 1 else 0)
        resid = c1 - pred_end
        return [pts[k] + resid * (k / n) for k in range(1, len(pts))]

    def appearance_ok(self, cid, tb, targets):
        """Contrast = (sd_patch - sd_bg) / (sd_cell - sd_bg): a real dropout still
        shows cell texture at the advected position, a false gap shows background."""
        ys, xs = np.nonzero(self.masks[tb] == cid)
        h = max(12, int(ys.max()) - int(ys.min()) + 1)
        w = max(12, int(xs.max()) - int(xs.min()) + 1)
        sd_cell = self.patch(tb, self.cent[(cid, tb)], h, w).std()
        scores = []
        for t, c in targets:
            H, W = self.imgs[t].shape
            bg = np.median([self.patch(t, (self._rng.uniform(0, H), self._rng.uniform(0, W)), h, w).std()
                            for _ in range(12)])
            scores.append((self.patch(t, c, h, w).std() - bg) / max(sd_cell - bg, 1e-6))
        return float(np.mean(scores)) >= self.thresh

    # -- gap painting --------------------------------------------------------
    def paint_child_back(self, cid, ta, tb):
        """Paint frames ta+1..tb-1 with the child's first mask (dilated), shifted
        along the flow path; background pixels only. All-or-nothing."""
        src = self.masks[tb] == cid
        if not src.any():
            return False
        r_eq = np.sqrt(src.sum() / np.pi)
        src = ndi.binary_dilation(src, iterations=max(1, int(r_eq * 0.4)))
        c0, c1 = self.cent[(cid, ta)], self.cent[(cid, tb)]
        path = self.flow_path(c0, c1, ta, tb)
        targets = [(t, path[t - ta - 1] if t - ta - 1 < len(path) else c0 + (c1 - c0) * (t - ta) / (tb - ta))
                   for t in range(ta + 1, tb)]
        if targets and not self.appearance_ok(cid, tb, targets):
            return False
        ys, xs = np.nonzero(src)
        planned = []
        for t, target in targets:
            dy, dx = np.round(target - c1).astype(int)
            yy, xx = ys + dy, xs + dx
            m = self.masks[t]
            ok = (yy >= 0) & (yy < m.shape[0]) & (xx >= 0) & (xx < m.shape[1])
            yy, xx = yy[ok], xx[ok]
            free = m[yy, xx] == 0
            if not free.any():
                return False
            planned.append((t, yy[free], xx[free]))
        for t, yy, xx in planned:
            self.masks[t][yy, xx] = cid
            self.cent[(cid, t)] = np.array([yy.mean(), xx.mean()])
        return True

    def extend_child_back(self, par, cid):
        """Move the child's birth to parent_end+1 by painting the gap with its id."""
        p_t1, c_t0 = self.tracks[par][1], self.tracks[cid][0]
        self.cent[(cid, p_t1)] = self.cent[(par, p_t1)]
        ok = self.paint_child_back(cid, p_t1, c_t0)
        if ok:
            self.tracks[cid][0] = p_t1 + 1
        del self.cent[(cid, p_t1)]
        return ok

    # -- the two passes ------------------------------------------------------
    def fill_linked_gaps(self):
        children = {}
        for i, (_, _, par) in self.tracks.items():
            if par > 0:
                children.setdefault(par, []).append(i)
        n = 0
        for par, chs in sorted(children.items()):
            if len(chs) != 1 or par not in self.tracks or (par, self.tracks[par][1]) not in self.cent:
                continue
            gap = self.tracks[chs[0]][0] - self.tracks[par][1] - 1
            if 1 <= gap <= self.max_gap and self.extend_child_back(par, chs[0]):
                n += 1
        return n

    def close_gaps(self):
        tracks, cent = self.tracks, self.cent
        mothers = {v[2] for v in tracks.values() if v[2] > 0}
        ends = [(i, t1) for i, (_, t1, _) in tracks.items()
                if t1 < self.T - 1 and i not in mothers and (i, t1) in cent]
        starts = [(i, t0) for i, (t0, _, par) in tracks.items() if t0 > 0 and par == 0 and (i, t0) in cent]
        if not ends or not starts:
            return 0
        cost = np.full((len(ends), len(starts)), 1e9)
        for a, (ei, et) in enumerate(ends):
            v = self.velocity(ei, et)
            for b, (si, st) in enumerate(starts):
                dt = st - et
                if not (1 <= dt <= self.max_gap) or si == ei:
                    continue
                d = np.linalg.norm(cent[(ei, et)] + v * dt - cent[(si, st)])
                if d <= self.max_dist:
                    cost[a, b] = d + 5.0 * (dt - 1)
        n = 0
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] >= 1e9:
                continue
            ei, si = ends[a][0], starts[b][0]
            if tracks[si][0] - tracks[ei][1] >= 1:
                tracks[si][2] = ei  # new lineage link (kept even if the gap cannot be painted)
                if tracks[si][0] - tracks[ei][1] > 1:
                    self.extend_child_back(ei, si)
                n += 1
        return n


def fill_gaps(imgs, res: CTCResult, max_gap=3, max_dist=60.0, flow_downscale=4,
              appearance_thresh=0.25) -> CTCResult:
    """Apply KOFT-style gap filling + motion gap closing to a tracking result.

    Args:
        imgs: raw video (T, Y, X), used for optical flow and the appearance gate.
        res: tracking result (e.g. from HOCT).
        max_gap: longest gap (missing frames) to bridge.
        max_dist: max distance (px) between the motion prediction and a birth for gap closing.
        flow_downscale: Farneback runs on images downscaled by this factor.
        appearance_thresh: min relative patch contrast for a fill to be accepted.
    """
    g = _GapFiller(np.asarray(imgs), res.masks, res.tracks, max_gap, max_dist, flow_downscale, appearance_thresh)
    n_filled = g.fill_linked_gaps()
    n_closed = g.close_gaps()
    out = CTCResult(np.stack(g.masks), {i: tuple(v) for i, v in g.tracks.items()}, dict(res.info))
    out.info.update(koft_gaps_filled=n_filled, koft_links_closed=n_closed)
    return out
