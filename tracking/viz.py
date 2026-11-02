"""Small tracking videos with a live lineage tree, to check divisions and lost tracks by eye.

Left: the raw frames with each cell's mask filled in its colour (as Nikolas' GIFs);
cells of one family share a hue, each generation is a different shade. Right: the
lineage tree of the families with divisions, growing with the video.

Markers (image and tree):
    white ring       division (on the two daughters, for a few frames)
    red x            track lost: the cell ends without dividing, away from the border and
                     before the last frame (a dropout, a death, or a tracking error)
    green +          cell appears out of nowhere (away from the border, after frame 0):
                     often a lost track that restarts under a new id
  with ground truth (CTC GT or tracked points):
    yellow diamond   missed division (GT mother divides, the tracks don't)
    magenta ring     false division (tracks divide, GT doesn't)

    from tracking.viz import render_lineage
    render_lineage(imgs, res, "musc01.gif", gt_points=pts, start=600, n_frames=150)
"""

import colorsys

import numpy as np

from .ctc import CTCResult, centroids_of
from .evaluate import chain_ids, division_events

GOLDEN = 0.618033988749895


def _lineage(res: CTCResult):
    """Logical cells (gap continuations merged), their spans, parents and children."""
    chain = chain_ids(res.tracks)
    span, parent = {}, {}
    for i, (t0, t1, p) in res.tracks.items():
        c = chain.get(i, i)
        a, b = span.get(c, (t0, t1))
        span[c] = (min(a, t0), max(b, t1))
        pc = chain.get(p, p) if p else 0
        if pc and pc != c:
            parent[c] = pc
    children = {}
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    for v in children.values():
        v.sort(key=lambda c: span[c][0])
    root = {}
    for c in sorted(span, key=lambda c: span[c][0]):
        root[c] = root.get(parent[c], parent[c]) if c in parent else c
    gen = {}
    for c in sorted(span, key=lambda c: span[c][0]):
        gen[c] = gen.get(parent.get(c, 0), -1) + 1 if c in parent else 0
    return chain, span, parent, children, root, gen


def _cell_colours(span):
    """A distinct colour per logical cell (golden-angle hue, alternating shade)."""
    col = {}
    for k, c in enumerate(sorted(span, key=lambda c: (span[c][0], c))):
        col[c] = np.array(colorsys.hsv_to_rgb((k * GOLDEN) % 1.0, 0.8, 1.0 if k % 2 else 0.8))
    return col


def _colours(span, parent, children, root, gen):
    """Family hue (golden angle), shade by generation, small hue offset per sibling."""
    roots = sorted({root[c] for c in span})
    hue = {r: (k * GOLDEN) % 1.0 for k, r in enumerate(roots)}
    col = {}
    for c in span:
        h = hue[root[c]]
        if c in parent:
            sib = children[parent[c]].index(c)
            h = (h + (0.05 if sib % 2 else -0.05) * (1 + gen[c] // 2)) % 1.0
        v = max(0.55, 1.0 - 0.12 * gen[c])
        col[c] = np.array(colorsys.hsv_to_rgb(h, 0.85, v))
    return col


def render_lineage(imgs, res: CTCResult, out_path, gt_points=None, start=0, n_frames=None, stride=1,
                   width=520, fps=8, max_families=14, border=15, title="", alpha=0.6, radius=10.0,
                   color_by="auto"):
    """Write a GIF (``.gif``) or MP4 (``.mp4``) of the tracks with a live lineage tree.

    Args:
        imgs, res: raw video (T, Y, X) and its tracking result.
        gt_points: optional point ground truth (``evaluate.ctc_gt_points`` or a dataset's
            ``gt_points``) to mark missed / false divisions.
        start, n_frames, stride: which frames to show (keep GIFs to ~100-200 frames).
        width: pixel width of the image panel (frames are subsampled to fit).
        max_families: families shown in the tree (the ones with most divisions in the window).
        border: px from the image edge within which an ending / appearing cell counts as
            leaving / entering the field (not lost / not appearing).
        color_by: "family" (one hue per family, shade per generation), "cell" (distinct colour
            per cell) or "auto" (family when the window has >= 5 dividing families, else cell:
            a single colony would otherwise be all one hue).
    Returns a dict of event counts in the window.
    """
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs = np.asarray(imgs)
    T, H, W = imgs.shape
    t_end = T if n_frames is None else min(T, start + n_frames * stride)
    frames = list(range(start, t_end, stride))
    chain, span, parent, children, root, gen = _lineage(res)
    n_fam = len({root[c] for c, ch in children.items() if len(ch) >= 2 and start <= span[c][1] + 1 < t_end})
    family = color_by == "family" or (color_by == "auto" and n_fam >= 5)
    col = _colours(span, parent, children, root, gen) if family else _cell_colours(span)

    # positions of every logical cell per frame (only what we need: first/last)
    first_pos, last_pos = {}, {}
    for t in range(T):
        for tid, c in centroids_of(res.masks[t]).items():
            cid = chain.get(tid, tid)
            first_pos.setdefault(cid, (t, c))
            last_pos[cid] = (t, c)

    def near_border(c):
        return min(c[0], c[1], H - 1 - c[0], W - 1 - c[1]) < border

    divisions = [(span[c][1] + 1, [first_pos[d][1] for d in ch if d in first_pos])
                 for c, ch in children.items() if len(ch) >= 2]
    lost = [(last_pos[c][0], last_pos[c][1], c) for c in span
            if span[c][1] < T - 1 and len(children.get(c, [])) < 2 and c in last_pos
            and not near_border(last_pos[c][1]) and not children.get(c)]
    appeared = [(first_pos[c][0], first_pos[c][1], c) for c in span
                if span[c][0] > 0 and c not in parent and c in first_pos and not near_border(first_pos[c][1])]
    missed, false = [], []
    if gt_points is not None and len(gt_points):
        dv = division_events(res, gt_points, radius=radius)
        missed = [(t, (y, x)) for t, y, x in dv["missed"]]
        false = [(last_pos[chain.get(p, p)][0], last_pos[chain.get(p, p)][1]) for p in dv["res_false"]
                 if chain.get(p, p) in last_pos]

    # tree: families with the most divisions inside the window
    n_div = {}
    for c, ch in children.items():
        if len(ch) >= 2 and start <= span[c][1] + 1 < t_end:
            n_div[root[c]] = n_div.get(root[c], 0) + 1
    fams = sorted(n_div, key=lambda r: -n_div[r])[:max_families]
    fams.sort(key=lambda r: span[r][0])
    ypos, slot = {}, [0]

    def layout(c):
        kids = children.get(c, [])
        if not kids:
            ypos[c] = slot[0]
            slot[0] += 1
        else:
            for k in kids:
                layout(k)
            ypos[c] = float(np.mean([ypos[k] for k in kids]))

    for r in fams:
        layout(r)
        slot[0] += 0.6
    in_tree = set(ypos)
    lost_cells = {c for _, _, c in lost}
    appeared_cells = {c for _, _, c in appeared}

    f = max(1, int(np.ceil(W / width)))
    lo, hi = np.percentile(imgs[frames[0]], [1, 99.7])
    fig_w = 2 * (W // f) / 100 + 0.4
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(fig_w, max(3.6, (H // f) / 100 + 0.9)),
                                  gridspec_kw=dict(width_ratios=[1, 0.95]), dpi=100)
    is_gif = str(out_path).endswith(".gif")
    writer = None if is_gif else imageio.get_writer(out_path, fps=fps, codec="libx264", quality=7,
                                                    macro_block_size=8)
    gif_frames = []
    counts = dict(divisions=0, lost=0, appeared=0, missed_div=0, false_div=0)
    try:
        for t in frames:
            img = (np.clip((imgs[t].astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1))[::f, ::f]
            rgb = np.repeat(img[..., None], 3, axis=2)
            m = res.masks[t][::f, ::f]
            for tid in np.unique(m):
                if tid:
                    c = chain.get(int(tid), int(tid))
                    a = alpha if c in in_tree or not in_tree else alpha * 0.55
                    sel = m == tid
                    rgb[sel] = rgb[sel] * (1 - a) + col.get(c, np.ones(3)) * a
            ax.clear()
            ax.imshow(rgb, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            def mark(events, recent, **kw):
                pts = [(c[1] / f, c[0] / f) for te, c in events if 0 <= t - te < recent]
                if pts:
                    ax.scatter(*zip(*pts), **kw)

            mark([(te, d) for te, ds in divisions for d in ds], 6, s=160, facecolors="none",
                 edgecolors="white", linewidths=1.6)
            mark([(te, c) for te, c, _ in lost], 10, s=90, marker="x", c="#ff3030", linewidths=2.2)
            mark([(te, c) for te, c, _ in appeared], 6, s=90, marker="+", c="#30ff60", linewidths=2.0)
            mark(missed, 12, s=150, marker="D", facecolors="none", edgecolors="#ffd400", linewidths=2.0)
            mark(false, 12, s=200, facecolors="none", edgecolors="#ff40ff", linewidths=2.0)
            counts = dict(divisions=sum(start <= te <= t for te, _ in divisions),
                          lost=sum(start <= te <= t for te, _, _ in lost),
                          appeared=sum(start <= te <= t for te, _, _ in appeared),
                          missed_div=sum(start <= te <= t for te, _ in missed),
                          false_div=sum(start <= te <= t for te, _ in false))
            txt = f"t={t}   divisions {counts['divisions']}   lost {counts['lost']}   appeared {counts['appeared']}"
            if gt_points is not None:
                txt += f"   | GT: missed div {counts['missed_div']}   false div {counts['false_div']}"
            if title:
                fig.suptitle(title, fontsize=9, x=0.01, ha="left")
            ax.set_title(txt, fontsize=7.5, loc="left")
            legend = "○ division   × lost   + appeared" + ("   ◇ missed div (GT)   ○ false div (GT)"
                                                         if gt_points is not None else "")
            ax.set_xlabel(legend, fontsize=7)

            axt.clear()
            # Every division happens at x = mother's last frame + 0.5: the mother's line ends there,
            # the vertical connector sits there and the daughters' lines start there, so the
            # three always meet (also when a daughter's first mask comes after a gap).
            def split_x(c):
                return span[c][1] + 0.5

            for c in in_tree:
                b0, b1 = span[c]
                if b0 > t:
                    continue
                y = ypos[c]
                x0 = split_x(parent[c]) if parent.get(c) in ypos else b0
                x1 = min(b1 + 0.5, t + 0.5)
                axt.plot([max(x0, start), x1], [y, y], color=col[c], lw=2.0, solid_capstyle="round", zorder=2)
                kids = [k for k in children.get(c, []) if k in ypos and span[k][0] <= t]
                if len(kids) >= 2:
                    ys = [ypos[k] for k in kids] + [y]
                    axt.plot([split_x(c)] * 2, [min(ys), max(ys)], color=col[c], lw=2.0,
                             solid_capstyle="round", zorder=2)
                if c in lost_cells and b1 <= t:
                    axt.scatter([b1 + 0.5], [y], marker="x", c="#e01010", s=28, zorder=3)
                if c in appeared_cells and b0 >= start:
                    axt.scatter([b0], [y], marker="+", c="#10a040", s=28, zorder=3)
            for te, (yy, xx) in missed:
                if te <= t:
                    near = [c for c in in_tree if span[c][1] in range(te - 2, te + 3)]
                    if near:
                        axt.scatter([te], [ypos[near[0]]], marker="D", facecolors="none", edgecolors="#d4a800", s=30)
            axt.axvline(t, color="#888888", lw=0.8)
            axt.set_xlim(start, frames[-1] + 1)
            axt.set_ylim(max(slot[0], 1), -1)
            axt.set_yticks([])
            axt.tick_params(labelsize=7)
            axt.set_xlabel("frame", fontsize=8)
            axt.set_title(f"lineage tree: {len(fams)} families with divisions", fontsize=8)
            for sp in ("top", "right", "left"):
                axt.spines[sp].set_visible(False)
            fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94 if title else 1))
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            if is_gif:
                gif_frames.append(frame)
            else:
                writer.append_data(frame)
    finally:
        if writer is not None:
            writer.close()
        plt.close(fig)
    if is_gif and gif_frames:
        from PIL import Image

        # one shared 48-colour palette keeps GIFs small
        ims = [Image.fromarray(f) for f in gif_frames]
        pal = ims[len(ims) // 2].quantize(colors=48, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        q[0].save(out_path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return counts
