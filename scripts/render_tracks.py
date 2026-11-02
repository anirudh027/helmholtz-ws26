"""Render a tracking result as a small GIF/MP4 with a live lineage tree (``tracking.viz``).

    python scripts/render_tracks.py --dataset musc --seq 01 --res /mnt/1t/tracks/musc/hoct_koft_tta/01_RES \
        --start 1100 --frames 150 --out musc01_hoct_koft_tta.gif
    python scripts/render_tracks.py --dataset helmholtz --res .../BF_RES --frames 120 --stride 2 --out helm.mp4

Ground truth (CTC GT or the dataset's tracked points) is used automatically to mark
missed / false divisions.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.ctc import read_ctc  # noqa: E402
from tracking.datasets import iter_sequences  # noqa: E402
from tracking.evaluate import ctc_gt_points  # noqa: E402
from tracking.viz import render_lineage  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--res", required=True, help="CTC RES folder of the tracks")
    ap.add_argument("--out", required=True, help=".gif or .mp4")
    ap.add_argument("--start", type=int, default=-1, help="first frame; -1 = the window with most divisions")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--families", type=int, default=14)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    seq = next(s for s in iter_sequences(args.dataset) if args.seq in (None, s.name))
    res = read_ctc(args.res)
    gt = seq.gt_points
    if gt is None and seq.gt_dir and os.path.isdir(seq.gt_dir):
        gt = ctc_gt_points(seq.gt_dir)
    if args.start < 0:
        span = args.frames * args.stride
        div_t = [res.tracks[p][1] + 1 for p in {v[2] for v in res.tracks.values() if v[2]}
                 if sum(v[2] == p for v in res.tracks.values()) >= 2]
        T = len(res.masks)
        cands = range(0, max(1, T - span + 1), max(1, span // 10))
        args.start = max(cands, key=lambda s0: sum(s0 <= t < s0 + span for t in div_t)) if div_t else 0
    title = args.title or f"{args.dataset} {seq.name} - {os.path.basename(os.path.dirname(args.res.rstrip('/')))}"
    counts = render_lineage(seq.imgs, res, args.out, gt_points=gt, start=args.start, n_frames=args.frames,
                            stride=args.stride, width=args.width, fps=args.fps, max_families=args.families,
                            title=title, radius=seq.point_radius)
    print(json.dumps({"out": args.out, **counts}))


if __name__ == "__main__":
    main()
