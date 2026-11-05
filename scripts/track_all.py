"""Run tracking backends over the WS26 datasets and score them where ground truth exists.

    python scripts/track_all.py --datasets musc --backends laptrack trackastra hoct
    python scripts/track_all.py --datasets all --backends laptrack trackastra --limit 50 --max-seqs 1  # smoke test

Writes ``<out>/<dataset>/<backend>/<seq>_RES/`` (CTC) and ``<out>/<dataset>/results.csv``.
Datasets without masks get provisional Cellpose-SAM masks (cached under ``_provisional_masks/``).
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import TRACKERS, track, write_ctc  # noqa: E402
from tracking.datasets import ADAPTERS, NOT_TRACKABLE, iter_sequences  # noqa: E402


def masks_for(seq, out_ds):
    if seq.masks is not None:
        return seq.masks, False
    cache = os.path.join(out_ds, "_provisional_masks", f"{seq.name}_T{len(seq.imgs)}.npy")
    if os.path.exists(cache):
        return np.load(cache), True
    from tracking.segment import segment_cellpose_sam

    masks = segment_cellpose_sam(seq.imgs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, masks)
    return masks, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["musc"], help=f"{list(ADAPTERS)} or 'all'")
    ap.add_argument("--backends", nargs="+", default=["laptrack", "trackastra", "hoct"])
    ap.add_argument("--out", default="/mnt/1t/tracks")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames of each sequence")
    ap.add_argument("--max-seqs", type=int, default=None, help="only the first N sequences per dataset")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    datasets = list(ADAPTERS) if args.datasets == ["all"] else args.datasets
    unknown = set(args.backends) - set(TRACKERS)
    if unknown:
        ap.error(f"unknown backends {sorted(unknown)}; choose from {list(TRACKERS)}")
    for name, why in NOT_TRACKABLE.items():
        print(f"[{name}] skipped: {why}", flush=True)

    for ds_name in datasets:
        out_ds = os.path.join(args.out, ds_name + (f"_first{args.limit}" if args.limit else ""))
        rows = []
        try:
            seqs = iter_sequences(ds_name, limit=args.limit, max_seqs=args.max_seqs)
            for seq in seqs:
                masks, provisional = masks_for(seq, out_ds)
                print(f"[{ds_name}/{seq.name}] imgs {seq.imgs.shape}, masks {'provisional ' if provisional else ''}"
                      f"{masks.shape}, {seq.note}", flush=True)
                for backend in args.backends:
                    row = {"dataset": ds_name, "seq": seq.name, "method": backend, "frames": len(seq.imgs),
                           "provisional_masks": provisional}
                    try:
                        res = track(seq.imgs, masks, model=backend)
                    except Exception as e:
                        print(f"[{ds_name}/{seq.name}] {backend} FAILED", flush=True)
                        traceback.print_exc()
                        rows.append({**row, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
                        continue
                    res_dir = os.path.join(out_ds, backend, f"{seq.name}_RES")
                    write_ctc(res_dir, res.masks, res.tracks)
                    with open(os.path.join(res_dir, "runtime.json"), "w") as f:
                        json.dump(res.info, f)
                    row.update(ok=True, tracks=len(res.tracks), runtime_s=round(res.info["seconds"], 1))
                    if seq.gt_dir and os.path.isdir(seq.gt_dir) and not args.no_eval:
                        from tracking.evaluate import evaluate

                        row.update(evaluate(res_dir, seq.gt_dir))
                    rows.append(row)
                    print(f"[{ds_name}/{seq.name}] {row}", flush=True)
        except Exception as e:
            print(f"[{ds_name}] FAILED to load", flush=True)
            traceback.print_exc()
            rows.append({"dataset": ds_name, "ok": False, "error": f"load: {type(e).__name__}: {e}"[:200]})
        if rows:
            os.makedirs(out_ds, exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_ds, "results.csv"), index=False)
            print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
