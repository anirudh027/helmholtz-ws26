"""Train/evaluate a fate model on the Bürger dataset, split by video.

    python scripts/fate_train_buerger.py --videos 1-20 --model crop_transformer --save weights.pt
"""

# ruff: noqa: E402

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import fate  # noqa: E402
from fate.buerger import load_sequences  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/buerger")
ap.add_argument("--videos", default="1-20", help="range of video numbers, e.g. 1-20")
ap.add_argument("--model", default="crop_transformer", choices=list(fate.FATE_MODELS))
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--truncations", type=int, nargs="+", default=[0, 5, 10, 20])
ap.add_argument("--train_trunc", type=int, default=0, help="train on sequences with up to N final frames dropped")
ap.add_argument("--out", default="fate_results.json")
ap.add_argument("--save", default=None, help="save trained weights here (crop_transformer only)")
args = ap.parse_args()

a, b = map(int, args.videos.split("-"))
names = [f"{i:03d}" for i in range(a, b + 1)]
seqs = load_sequences(args.root, names)
print(f"{len(seqs)} sequences from {len(names)} videos, division fraction "
      f"{np.mean([s.fate for s in seqs]):.2f}", flush=True)

# split by video (not by cell) so that cells of one video never appear in both sets
rng = np.random.RandomState(0)
vids = rng.permutation(sorted({s.video for s in seqs}))
n = len(vids)
split = {v: "train" for v in vids[: int(0.7 * n)]}
split.update({v: "val" for v in vids[int(0.7 * n): int(0.85 * n)]})
split.update({v: "test" for v in vids[int(0.85 * n):]})
part = {k: [s for s in seqs if split[s.video] == k] for k in ("train", "val", "test")}
print({k: (len(v), round(float(np.mean([s.fate for s in v])), 2)) for k, v in part.items()}, flush=True)

kw = dict(epochs=args.epochs, train_trunc=args.train_trunc) if args.model == "crop_transformer" else {}
model = fate.FATE_MODELS[args.model](**kw).fit(part["train"], part["val"])
if args.save:
    model.save(args.save)
results = {}
for t in args.truncations:
    test = fate.truncate(part["test"], t)
    test = [s for s in test if len(s.crops) >= 3]
    m = fate.metrics(np.array([s.fate for s in test]), model.predict_proba(test))
    results[t] = m
    print(f"discard last {t:2d} frames: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()), flush=True)
json.dump(results, open(args.out, "w"), indent=1)
