"""One round of HOCT few-label correction with a human annotator (e.g. Helmholtz).

    # round 1: proposes 20 links -> <out>/labels.csv (column `correct` empty) + <out>/sheet_r01.png
    python scripts/correction_round.py --dataset helmholtz --register --masks-dir /mnt/1t/masks/{dataset}
    # look at the sheet, fill `correct` with 1 (same cell, or mother -> daughter) / 0 (wrong) / skip blank
    # round 2..N: same command; applies every verdict so far, refits, writes tracks, proposes 20 more
    python scripts/correction_round.py --dataset helmholtz --register --masks-dir /mnt/1t/masks/{dataset}

Each round writes ``<out>/round_XX/<seq>_RES`` (CTC tracks, original frame
coordinates) and keeps all verdicts in ``labels.csv``, which also serves as
point ground truth for ``evaluate.point_metrics`` (both ends of every correct link).
The graph is rebuilt every round, so links are stored by node position (t, y, x)
and matched back to the nearest node.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_all import masks_for  # noqa: E402

from tracking import write_ctc  # noqa: E402
from tracking.api import _relabel  # noqa: E402
from tracking.correct import Corrector  # noqa: E402
from tracking.datasets import iter_sequences  # noqa: E402
from tracking.hoct import HOCT  # noqa: E402

KEY = ["src_t", "src_y", "src_x", "tgt_t", "tgt_y", "tgt_x"]


def match_nodes(nodes, t, y, x):
    """node_id of the node at frame t nearest to (y, x)."""
    out = []
    for tt, yy, xx in zip(t, y, x):
        n = nodes[nodes["t"] == tt]
        out.append(int(n["node_id"].iloc[int(np.argmin(np.hypot(n["y"] - yy, n["x"] - xx)))]) if len(n) else -1)
    return out


def render_sheet(imgs, links, path, crop=96):
    """Numbered crop pairs: source frame | target frame, cross on each node."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(links)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.6, rows * 1.8))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(links.itertuples()):
        cy, cx = (r.src_y + r.tgt_y) / 2, (r.src_x + r.tgt_x) / 2  # same window in both frames
        for j, (t, y, x) in enumerate(((r.src_t, r.src_y, r.src_x), (r.tgt_t, r.tgt_y, r.tgt_x))):
            img = imgs[int(t)]
            H, W = img.shape
            y0 = int(np.clip(cy - crop, 0, max(0, H - 2 * crop)))
            x0 = int(np.clip(cx - crop, 0, max(0, W - 2 * crop)))
            patch = img[y0:y0 + 2 * crop, x0:x0 + 2 * crop]
            lo, hi = np.percentile(patch, [1, 99.5])
            ax = axes[k // cols, 2 * (k % cols) + j]
            ax.imshow(patch, cmap="gray", vmin=lo, vmax=hi)
            ax.plot(x - x0, y - y0, "+", color="red" if j == 0 else "cyan", ms=12, mew=1.5)
            ax.set_title(f"#{r.Index} t={int(t)}" + (f"  p={r.similarity:.2f}" if j == 0 else ""), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="helmholtz")
    ap.add_argument("--seq", default=None, help="sequence name (default: the first)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--masks", choices=["provided", "cellpose"], default="provided")
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--tta", type=int, default=6)
    ap.add_argument("--n", type=int, default=20, help="links to propose this round")
    ap.add_argument("--out", default="/mnt/1t/tracks/correction")
    args = ap.parse_args()

    seq = next(s for s in iter_sequences(args.dataset, limit=args.limit) if args.seq in (None, s.name))
    out = os.path.join(args.out, f"{args.dataset}_{seq.name}")
    os.makedirs(out, exist_ok=True)
    masks, _ = masks_for(seq, os.path.join(args.out, args.dataset), args.masks, args.masks_dir)
    imgs = seq.imgs
    if args.register:
        from tracking.register import estimate_drift, from_canvas, to_canvas

        off = estimate_drift(imgs)
        imgs, masks = to_canvas(imgs, off), to_canvas(masks, off)

    c = Corrector(imgs, masks, HOCT(tta=args.tta))
    nodes = c.nodes()
    lab_path = os.path.join(out, "labels.csv")
    labels = pd.read_csv(lab_path, index_col="link") if os.path.exists(lab_path) else pd.DataFrame(columns=KEY + ["round", "correct"])
    done = labels[labels["correct"].notna() & (labels["correct"].astype(str).str.strip() != "")].copy()
    rnd = int(labels["round"].max()) + 1 if len(labels) else 1
    if len(done):
        done["source_id"] = match_nodes(nodes, done.src_t, done.src_y, done.src_x)
        done["target_id"] = match_nodes(nodes, done.tgt_t, done.tgt_y, done.tgt_x)
        e = c.edges.set_index(["source_id", "target_id"])["edge_id"]
        done["edge_id"] = [e.get((s, t), -1) for s, t in zip(done.source_id, done.target_id)]
        done = done[done.edge_id >= 0]
        done["correct"] = done["correct"].astype(float).astype(bool)
        c.annotate(done)
        c.refit()
        print(f"refitted on {len(done)} verdicts", flush=True)

    res = c.result()
    if args.register:
        res = _relabel(from_canvas(res.masks, off, seq.imgs.shape), res.tracks)
    res_dir = os.path.join(out, f"round_{rnd:02d}", f"{seq.name}_RES")
    write_ctc(res_dir, res.masks, res.tracks)

    known = set(map(tuple, labels[KEY].round(1).to_numpy())) if len(labels) else set()
    q = c.propose(args.n * 3)
    q = q[[tuple(np.round(v, 1)) not in known for v in q[KEY].to_numpy()]].head(args.n).reset_index(drop=True)
    q["round"] = rnd
    q["correct"] = ""
    start = len(labels)
    q.index = range(start, start + len(q))
    render_sheet(imgs, q, os.path.join(out, f"sheet_r{rnd:02d}.png"))
    pd.concat([labels, q[KEY + ["similarity", "round", "correct"]]]).to_csv(lab_path, index_label="link")
    print(f"round {rnd}: tracks -> {res_dir}; {len(q)} new links -> {lab_path}, sheet_r{rnd:02d}.png", flush=True)


if __name__ == "__main__":
    main()
