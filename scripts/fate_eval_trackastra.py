"""Fate prediction from Trackastra's tracks instead of ground-truth tracks.

For every dying/dividing GT cell of the test videos: find the detection that overlaps the
cell at its last alive frame (IoU >= --min_iou), find the Trackastra track that contains it,
crop the cell along that *predicted* track and let the fate model(s) classify it. Compared
against the same models on crops along the GT annotation, and against Trackastra's own call
(track ends with daughters or not). Detections are either the GT masks or Cellpose-SAM masks.

    python scripts/fate_eval_trackastra.py --masks cellpose --weights base=a.pt trunc=b.pt
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics

import fate
from fate import CellSequence
from fate import buerger as bd
from fate.crop_transformer import CropTransformer
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/buerger")
ap.add_argument("--weights", nargs="+", required=True, help="name=path.pt ...")
ap.add_argument("--videos", nargs="+", default=["001", "016", "013"])
ap.add_argument("--masks", choices=["gt", "cellpose"], default="gt", help="detections given to Trackastra")
ap.add_argument("--min_iou", type=float, default=0.3)
ap.add_argument("--channel", default="mean", help="channel given to Trackastra")
ap.add_argument("--mode", default="ilp")
ap.add_argument("--patch", type=int, default=112)
ap.add_argument("--max_len", type=int, default=48)
ap.add_argument("--out", default=None)
args = ap.parse_args()
root = Path(args.root)
out_path = args.out or str(root / "out" / f"trackastra_fate_{args.masks}.json")
h = args.patch // 2


def majority(labels):
    """Most frequent non-zero label, and its pixel count (None, 0 if there is none)."""
    ids, cnt = np.unique(labels[labels > 0], return_counts=True)
    return (int(ids[cnt.argmax()]), int(cnt.max())) if len(ids) else (None, 0)


models = {n: CropTransformer.load(p) for n, p in (a.split("=") for a in args.weights)}
trackastra = Trackastra.from_pretrained("general_2d", device="automatic")
gt_cells = {(c["video"], c["cell"]): c for c in bd.cell_table(root, [f"{i:03d}" for i in range(1, 21)])
            if len(c["frames"]) >= 10}
gt_seqs = {(s.video, s.cell): s for s in bd.load_sequences(root, args.videos)}

rows, track_metrics = [], {}
for v in args.videos:
    imgs, gt_masks = bd.load_video(root, v, args.channel)
    det_masks = gt_masks if args.masks == "gt" else np.load(root / "out" / f"cp_masks_{v}.npy")
    graph, masks_tr = trackastra.track(imgs, det_masks, mode=args.mode)
    tracks, pred = graph_to_ctc(graph, masks_tr, outdir=str(root / "out" / f"pred_{args.masks}_{v}"))
    pred = np.asarray(pred)
    if args.masks == "gt":  # CTC metrics only make sense when detections are the GT masks
        bd.write_gt_ctc(root, v, root / "out" / f"gt_{v}")
        res, _ = run_metrics(gt_data=load_ctc_data(str(root / "out" / f"gt_{v}"), name="GT"),
                             pred_data=load_ctc_data(str(root / "out" / f"pred_{args.masks}_{v}"), name="P"),
                             matcher=CTCMatcher(), metrics=[CTCMetrics()])
        track_metrics[v] = {k: round(float(x), 4) for k, x in res[0]["results"].items() if k in ("TRA", "DET", "LNK")}
        print(v, "tracking", track_metrics[v], flush=True)
    n_det = float(np.mean([len(np.unique(m)) - 1 for m in det_masks]))
    print(v, f"detections per frame {n_det:.1f} vs GT {np.mean([len(np.unique(m)) - 1 for m in gt_masks]):.1f}", flush=True)

    full = {f: np.pad(bd.to_uint8(tifffile.imread(root / "raw" / v / "images" / f"{f + bd.T0:03d}.tif")),
                      ((h, h), (h, h), (0, 0))) for f in range(pred.shape[0])}
    children = tracks[tracks.parent > 0].groupby("parent").size().to_dict()
    tr = tracks.set_index("label")

    for (vv, cell), c in gt_cells.items():
        if vv != v or (vv, cell) not in gt_seqs:
            continue
        k = c["frames"][-1] - bd.T0  # last alive frame (0-based)
        m_gt = gt_masks[k] == cell
        row = dict(video=v, cell=cell, fate=c["fate"], pred_len=0, iou=0.0)
        d, _ = majority(det_masks[k][m_gt])
        if d is None:
            rows.append(row | dict(status="missed"))
            continue
        m_det = det_masks[k] == d
        row["iou"] = float((m_det & m_gt).sum() / (m_det | m_gt).sum())
        if row["iou"] < args.min_iou:
            rows.append(row | dict(status="missed"))
            continue
        p, _ = majority(pred[k][m_det])
        fr = [f for f in range(max(int(tr.loc[p, "t1"]), k - args.max_len + 1), k + 1) if (pred[f] == p).any()]
        crops, pure = [], 0
        for f in fr:
            mp = pred[f] == p
            y, x = (int(round(a)) for a in regionprops(mp.astype(np.uint8))[0].centroid)
            crops.append(full[f][y:y + args.patch, x:x + args.patch].transpose(2, 0, 1))
            g, _ = majority(gt_masks[f][mp])
            pure += int(g == cell)
        rows.append(row | dict(pred_len=len(fr), purity=pure / len(fr), status="ok",
                               own_call=int(children.get(p, 0) >= 2),
                               _seq=CellSequence(v, cell, np.stack(crops), c["fate"], k + bd.T0)))

missed = sum(r["status"] == "missed" for r in rows)
ok = [r for r in rows if r["status"] == "ok" and r["pred_len"] >= 3]
print(f"\n{len(rows)} GT dying/dividing cells in test videos [{args.masks} detections]: "
      f"{missed} without a detection of IoU >= {args.min_iou}, {len(rows) - missed - len(ok)} with <3 predicted frames")
ious = np.array([r["iou"] for r in rows])
print(f"IoU of matched detection: median {np.median(ious[ious > 0]):.3f}, share >= 0.5: {(ious >= 0.5).mean():.3f}")
print("mean predicted history purity:", round(float(np.mean([r["purity"] for r in ok])), 3),
      "| mean predicted length (frames):", round(float(np.mean([r["pred_len"] for r in ok])), 1),
      "vs GT:", round(float(np.mean([len(gt_seqs[(r['video'], r['cell'])].crops) for r in ok])), 1))

y = np.array([r["fate"] for r in ok])
own = np.array([r["own_call"] for r in ok])
own_m = fate.metrics(y, np.stack([1 - own, own], 1).astype(float))
print("Trackastra own call (daughters => division):", {k: round(x, 3) for k, x in own_m.items() if k != "n"})

out = dict(masks=args.masks, tracking=track_metrics, n_cells=len(rows), n_missed=missed, n_evaluated=len(ok),
           own_call=own_m, fate_model={})
for name, model in models.items():
    out["fate_model"][name] = {}
    for t in (0, 5, 10, 20):
        res = {}
        for src, seqs in (("gt_tracks", [gt_seqs[(r["video"], r["cell"])] for r in ok]),
                          ("trackastra_tracks", [r["_seq"] for r in ok])):
            s = fate.truncate(seqs, t)
            keep = [i for i, q in enumerate(s) if len(q.crops) >= 3]
            res[src] = fate.metrics(y[keep], model.predict_proba([s[i] for i in keep]))
        out["fate_model"][name][t] = res
        print(f"[{name}] discard last {t:2d}: " + " | ".join(
            f"{n}: auroc {m['auroc']:.3f} bal.acc {m['balanced_acc']:.3f} (n={m['n']})" for n, m in res.items()), flush=True)
json.dump(out, open(out_path, "w"), indent=1)
