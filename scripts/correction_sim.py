"""Simulate HOCT's few-label correction with a ground-truth oracle.

Rounds of ``--per-round`` uncertain links are answered from the CTC GT, the
linear probe is refitted, and the corrected tracks are scored. Tells us how
many links a human would need to annotate on Helmholtz for a given gain.

    python scripts/correction_sim.py --dataset musc --seq 02 --rounds 20 --per-round 20
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import write_ctc  # noqa: E402
from tracking.correct import Corrector, GTOracle  # noqa: E402
from tracking.datasets import iter_sequences  # noqa: E402
from tracking.evaluate import evaluate  # noqa: E402
from tracking.hoct import HOCT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="musc")
    ap.add_argument("--seq", default="02")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2, help="score every N rounds")
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="/mnt/1t/tracks/correction_sim")
    args = ap.parse_args()

    seq = next(s for s in iter_sequences(args.dataset, limit=args.limit) if s.name == args.seq)
    if not seq.gt_dir:
        ap.error(f"{args.dataset}/{args.seq} has no CTC ground truth (use the full sequence, no --limit)")
    gt_dir = seq.gt_dir
    out = os.path.join(args.out, f"{args.dataset}_{args.seq}_tta{args.tta}")
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    c = Corrector(seq.imgs, seq.masks, HOCT(tta=args.tta))
    oracle = GTOracle(gt_dir, c.nodes(), seq.masks)
    rows = []

    def score(r):
        res_dir = os.path.join(out, f"round{r:02d}", f"{args.seq}_RES")
        res = c.result()
        write_ctc(res_dir, res.masks, res.tracks)
        m = evaluate(res_dir, gt_dir, metrics=["TRA", "DET", "LNK", "BC"])
        m.update(round=r, n_labels=c.n_labels, minutes=(time.time() - t0) / 60,
                 solution_link_precision=float(oracle(c.edges[c.edges["solution"]]).mean()))
        rows.append(m)
        print(json.dumps(m), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "curve.csv"), index=False)

    score(0)
    for r in range(1, args.rounds + 1):
        q = c.propose(args.per_round)
        q["correct"] = oracle(q)
        c.annotate(q)
        c.refit()
        print(f"round {r}: {c.n_labels} labels, {q['correct'].mean():.2f} of this round correct, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if r % args.eval_every == 0 or r == args.rounds:
            score(r)


if __name__ == "__main__":
    main()
