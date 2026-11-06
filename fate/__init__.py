"""Tracker-independent cell fate prediction.

A fate model only sees `CellSequence`s: the image crops of one tracked cell over
time plus (for training) its eventual fate. Where the tracks come from (Trackastra,
another tracker, or ground truth) and which model predicts the fate is swappable:

    sequences = ...                       # list[CellSequence], from any tracker
    model = FATE_MODELS["length_hazard"]()   # or "crop_transformer" (needs torch)
    model.fit(train_sequences)
    proba = model.predict_proba(test_sequences)   # (N, len(FATES))

Fates: 0 = death, 1 = division (class order matches Bürger et al., 2026).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

FATES = ("death", "division")


@dataclass
class CellSequence:
    video: str
    cell: int
    crops: np.ndarray  # (T, C, H, W) uint8, last frame = last observed frame
    fate: int | None = None  # index into FATES, None if unknown
    t_end: int | None = None  # frame index of the last crop


def truncate(seqs: list[CellSequence], n_frames: int) -> list[CellSequence]:
    """Drop the last `n_frames` frames, i.e. predict the fate `n_frames` ahead."""
    if n_frames == 0:
        return seqs
    return [
        CellSequence(s.video, s.cell, s.crops[:-n_frames], s.fate,
                     None if s.t_end is None else s.t_end - n_frames)
        for s in seqs
    ]


class FatePredictor(Protocol):
    """Interface of every fate model. Models may ignore the image content."""

    def fit(self, train: list[CellSequence], val: list[CellSequence] | None = None) -> "FatePredictor": ...

    def predict_proba(self, seqs: list[CellSequence]) -> np.ndarray:
        """(N, len(FATES)) probabilities."""
        ...


class LengthHazard:
    """Baseline that ignores the images: fate frequency per sequence-length bin.

    Tells how much of the task is explained by how long a cell was observed alone,
    which any real model has to beat.
    """

    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins

    def _len(self, seqs):
        return np.array([len(s.crops) for s in seqs])

    def _bin(self, n):
        return np.digitize(n, self.edges_)

    def fit(self, train, val=None):
        n = self._len(train)
        y = np.array([s.fate for s in train])
        self.edges_ = np.unique(np.quantile(n, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        b = self._bin(n)
        counts = np.ones((len(self.edges_) + 1, len(FATES)))  # +1 smoothing
        np.add.at(counts, (b, y), 1)
        self.proba_ = counts / counts.sum(1, keepdims=True)
        return self

    def predict_proba(self, seqs):
        return self.proba_[self._bin(self._len(seqs))]


def _crop_transformer(**kw):
    from .crop_transformer import CropTransformer  # imported lazily so that torch is only needed when used

    return CropTransformer(**kw)


FATE_MODELS = {"length_hazard": LengthHazard, "crop_transformer": _crop_transformer}


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC of `score` for the positive class y == 1 (rank formulation)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # average ranks of ties
        m = score == v
        ranks[m] = ranks[m].mean()
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    rec = [float((pred[y == k] == k).mean()) for k in range(len(FATES))]
    f1 = []
    for k in range(len(FATES)):
        tp = ((pred == k) & (y == k)).sum()
        p = tp / max((pred == k).sum(), 1)
        r = tp / max((y == k).sum(), 1)
        f1.append(2 * p * r / max(p + r, 1e-9))
    return dict(
        balanced_acc=float(np.mean(rec)),
        macro_f1=float(np.mean(f1)),
        auroc=auroc(y, proba[:, 1]),
        n=len(y),
        frac_division=float(y.mean()),
    )
