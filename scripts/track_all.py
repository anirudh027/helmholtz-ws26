"""Run tracking backends over the WS26 datasets and score them where ground truth exists.

    python scripts/track_all.py --datasets musc --backends laptrack trackastra hoct
    python scripts/track_all.py --datasets all --backends laptrack trackastra --limit 50 --max-seqs 1  # smoke test

Writes ``<out>/<dataset>/<backend>/<seq>_RES/`` (CTC) and ``<out>/<dataset>/results.csv``.
Datasets without masks get provisional Cellpose-SAM masks (cached under ``_provisional_masks/``).

Scoring, whatever ground truth the dataset has:
  * complete CTC GT (MuSC, HeLa, HSC): TRA, DET, LNK, CT, BC, CCA;
  * tracked points (ALFI, C2C12 human tracks, HematoFate, p53, Helmholtz labels):
    det_recall, link_acc, id_switches, completeness, survival, div_recall (``evaluate.point_metrics``);
  * masks of some cells (QPI): CTC metrics on the tracks of the annotated cells.

    python scripts/track_all.py --datasets helmholtz --backends hoct_koft --register
    python scripts/track_all.py --datasets alfi --backends hoct hoct_koft --masks cellpose
    python scripts/track_all.py --datasets musc --backends hoct --kw '{"max_distance": 150}' --tag d150
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


def load_masks_dir(masks_dir, seq):
    """Team 2's masks for ``seq``: ``<dir>/<seq>.npy`` / ``.tif`` (T, Y, X) or a folder
    ``<dir>/<seq>/`` of per-frame TIFFs (sorted by name). None if absent."""
    import glob

    import tifffile

    for ext, load in ((".npy", np.load), (".tif", tifffile.imread), (".tiff", tifffile.imread)):
        f = os.path.join(masks_dir, seq.name + ext)
        if os.path.exists(f):
            return np.asarray(load(f))
    files = sorted(glob.glob(os.path.join(masks_dir, seq.name, "*.tif*")))
    return np.stack([np.squeeze(tifffile.imread(f)) for f in files]) if files else None


def masks_for(seq, out_ds, source="provided", masks_dir=None, seq_tag=""):
    if masks_dir:
        masks_dir = masks_dir.format(dataset=seq.dataset)
        m = load_masks_dir(masks_dir, seq)
        if m is not None:
            m = m[:len(seq.imgs)]
            if m.shape != seq.imgs.shape:
                raise ValueError(f"masks in {masks_dir} for {seq.name}: {m.shape} != images {seq.imgs.shape}")
            return m.astype(np.int32), False
        print(f"[{seq.dataset}/{seq.name}] no masks in {masks_dir}, falling back to '{source}'", flush=True)
    if seq.masks is not None and source == "provided":
        return seq.masks, False
    cache = os.path.join(out_ds, "_provisional_masks", f"{seq.name}_T{len(seq.imgs)}{seq_tag}.npy")
    if os.path.exists(cache):
        return np.load(cache), True
    from tracking.segment import segment_cellpose_sam

    masks = segment_cellpose_sam(seq.imgs)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, masks)
    return masks, True


def score(seq, res, res_dir, out_ds, method, point_radius):
    """Every metric the sequence's ground truth allows."""
    from tracking.evaluate import evaluate, masks_to_ctc_gt, point_metrics, restrict_to_annotated

    out = {}
    if seq.gt_dir and os.path.isdir(seq.gt_dir):
        out.update(evaluate(res_dir, seq.gt_dir))
    if seq.gt_points is not None and len(seq.gt_points):
        out.update({f"pt_{k}": v for k, v in point_metrics(res, seq.gt_points, radius=point_radius or seq.point_radius).items()})
    if seq.gt_masks is not None:
        gt_dir = os.path.join(out_ds, "_gt", f"{seq.name}_GT")
        if not os.path.isdir(gt_dir):
            masks_to_ctc_gt(gt_dir, seq.gt_masks)
        sub = restrict_to_annotated(res, seq.gt_masks)
        sub_dir = os.path.join(out_ds, method, f"{seq.name}_RES_annotated")
        write_ctc(sub_dir, sub.masks, sub.tracks)
        out.update({f"annot_{k}": v for k, v in evaluate(sub_dir, gt_dir).items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["musc"], help=f"{list(ADAPTERS)}, 'all', or ctc:<path> (any CTC-layout folder)")
    ap.add_argument("--backends", nargs="+", default=["laptrack", "trackastra", "hoct"])
    ap.add_argument("--out", default="/mnt/1t/tracks")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames of each sequence")
    ap.add_argument("--max-seqs", type=int, default=None, help="only the first N sequences per dataset")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--register", action="store_true", help="compensate stage drift before tracking")
    ap.add_argument("--masks", choices=["provided", "cellpose"], default="provided",
                    help="'cellpose': ignore shipped masks (e.g. ALFI's GT masks) and segment with Cellpose-SAM")
    ap.add_argument("--masks-dir", default=None,
                    help="external masks (e.g. Team 2's Cellpose-SAM), per sequence: <dir>/<seq>.npy|.tif or <dir>/<seq>/*.tif; '{dataset}' in the path is filled in")
    ap.add_argument("--preprocess", choices=["none", "percentile", "hist_eq", "clahe"], default="none",
                    help="contrast enhancement of the frames before segmentation (Cellpose) and tracking")
    ap.add_argument("--kw", type=json.loads, default={}, help="JSON kwargs for every backend")
    ap.add_argument("--tag", default="", help="suffix for the backend folder / method name")
    ap.add_argument("--point-radius", type=float, default=None,
                    help="px tolerance for point ground truth (default: per dataset, 10; HematoFate 20)")
    args = ap.parse_args()

    datasets = list(ADAPTERS) if args.datasets == ["all"] else args.datasets
    unknown = set(args.backends) - set(TRACKERS)
    if unknown:
        ap.error(f"unknown backends {sorted(unknown)}; choose from {list(TRACKERS)}")
    for name, why in NOT_TRACKABLE.items():
        print(f"[{name}] skipped: {why}", flush=True)

    for ds_name in datasets:
        ds_dir = os.path.basename(ds_name[4:].rstrip("/")) if ds_name.startswith("ctc:") else ds_name
        out_ds = os.path.join(args.out, ds_dir + (f"_first{args.limit}" if args.limit else ""))
        rows = []
        try:
            seqs = iter_sequences(ds_name, limit=args.limit, max_seqs=args.max_seqs)
            for seq in seqs:
                if args.preprocess != "none":
                    from tracking.preprocess import enhance

                    seq.imgs = enhance(seq.imgs, args.preprocess)
                ptag = "" if args.preprocess == "none" else f"_{args.preprocess}"
                masks, provisional = masks_for(seq, out_ds, args.masks, args.masks_dir, ptag)
                print(f"[{ds_name}/{seq.name}] imgs {seq.imgs.shape}, masks {'provisional ' if provisional else ''}"
                      f"{masks.shape}, {seq.note}", flush=True)
                for backend in args.backends:
                    method = backend + (f"_{args.tag}" if args.tag else "") + ptag
                    row = {"dataset": ds_dir, "seq": seq.name, "method": method, "frames": len(seq.imgs),
                           "provisional_masks": provisional, "registered": args.register,
                           "preprocess": args.preprocess}
                    try:
                        res = track(seq.imgs, masks, model=backend, register=args.register, **args.kw)
                    except Exception as e:
                        print(f"[{ds_name}/{seq.name}] {backend} FAILED", flush=True)
                        traceback.print_exc()
                        rows.append({**row, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
                        continue
                    res_dir = os.path.join(out_ds, method, f"{seq.name}_RES")
                    write_ctc(res_dir, res.masks, res.tracks)
                    with open(os.path.join(res_dir, "runtime.json"), "w") as f:
                        json.dump(res.info, f, default=str)
                    row.update(ok=True, tracks=len(res.tracks), runtime_s=round(res.info["seconds"], 1))
                    if not args.no_eval:
                        row.update(score(seq, res, res_dir, out_ds, method, args.point_radius))
                    rows.append(row)
                    print(f"[{ds_name}/{seq.name}] {row}", flush=True)
        except Exception as e:
            print(f"[{ds_name}] FAILED to load", flush=True)
            traceback.print_exc()
            rows.append({"dataset": ds_dir, "ok": False, "error": f"load: {type(e).__name__}: {e}"[:200]})
        if rows:
            os.makedirs(out_ds, exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out_ds, "results.csv"), index=False)
            print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
