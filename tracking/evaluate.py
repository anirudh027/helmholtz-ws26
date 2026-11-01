"""Official CTC metrics (TRA, DET, LNK, CT, BC, CCA) via py-ctcmetrics."""

from ctc_metrics import evaluate_sequence

METRICS = ["TRA", "DET", "LNK", "CT", "BC", "CCA"]


def evaluate(res_dir, gt_dir, metrics=METRICS, threads=8) -> dict:
    """Score a CTC RES folder against a CTC ``<seq>_GT`` folder."""
    m = evaluate_sequence(str(res_dir), str(gt_dir), metrics=list(metrics), threads=threads)
    out = {k: m.get(k) for k in metrics}
    if "BC" in metrics:
        out["BC"] = m.get("BC(1)", m.get("BC"))
    return out
