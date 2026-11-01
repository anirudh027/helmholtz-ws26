"""Run tracking backends over datasets in CTC layout and score them.

    python scripts/track_all.py --datasets BF-C2DL-MuSC --backends laptrack trackastra hoct
    python scripts/track_all.py --datasets BF-C2DL-MuSC --backends laptrack --limit 20   # smoke test

Writes ``<out>/<dataset>/<backend>/<seq>_RES/`` (CTC) + ``runtime.json`` and,
where ground truth exists, ``<out>/<dataset>/metrics_<seq>.csv``.
"""

import argparse
import json
import os
import sys
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import TRACKERS, load_sequence, track, write_ctc  # noqa: E402

# name -> CTC dataset folder (contains <seq>/, <seq>_ST/SEG, optionally <seq>_GT)
DATASETS = {
    "BF-C2DL-MuSC": "/mnt/1t/stem_cell_fate_prediction/datasets/BF-C2DL-MuSC/Training",
}

# Per-dataset backend kwargs (Damià's benchmark settings).
BACKEND_KWARGS = {
    "BF-C2DL-MuSC": {"laptrack": {"max_distance": 40.0}, "ultrack": {"max_distance": 40.0}},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--backends", nargs="+", default=["laptrack", "trackastra", "hoct"])
    ap.add_argument("--seqs", nargs="+", default=["01"])
    ap.add_argument("--out", default="/mnt/1t/tracks")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    unknown = set(args.backends) - set(TRACKERS)
    if unknown:
        ap.error(f"unknown backends {sorted(unknown)}; choose from {list(TRACKERS)}")

    for ds in args.datasets:
        ds_dir = DATASETS.get(ds, ds)
        name = ds if ds in DATASETS else os.path.basename(os.path.normpath(ds))
        out_ds = os.path.join(args.out, name + (f"_first{args.limit}" if args.limit else ""))
        for seq in args.seqs:
            imgs, masks = load_sequence(ds_dir, seq, args.limit)
            print(f"[{name}/{seq}] loaded {imgs.shape}", flush=True)
            gt_dir = os.path.join(ds_dir, f"{seq}_GT")
            rows = []
            for backend in args.backends:
                res_dir = os.path.join(out_ds, backend, f"{seq}_RES")
                try:
                    kw = BACKEND_KWARGS.get(name, {}).get(backend, {})
                    res = track(imgs, masks, model=backend, **kw)
                except Exception:
                    print(f"[{name}/{seq}] {backend} FAILED", flush=True)
                    traceback.print_exc()
                    continue
                write_ctc(res_dir, res.masks, res.tracks)
                with open(os.path.join(out_ds, backend, "runtime.json"), "w") as f:
                    json.dump(res.info, f)
                row = {"method": backend, "tracks": len(res.tracks), "runtime_s": round(res.info["seconds"], 1)}
                if os.path.isdir(gt_dir) and not args.no_eval and not args.limit:
                    from tracking.evaluate import evaluate

                    row.update(evaluate(res_dir, gt_dir))
                rows.append(row)
                print(f"[{name}/{seq}] {row}", flush=True)
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(out_ds, f"metrics_{seq}.csv"), index=False)
                print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
