import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fate  # noqa: E402
from fate import CellSequence  # noqa: E402


def make(n, length, fate_, rng):
    return [CellSequence("v", i, rng.integers(0, 255, (length, 3, 8, 8), dtype=np.uint8), fate_, length) for i in range(n)]


def test_truncate_drops_last_frames():
    s = make(2, 10, 1, np.random.default_rng(0))
    t = fate.truncate(s, 4)
    assert all(len(q.crops) == 6 and q.t_end == 6 for q in t)
    assert fate.truncate(s, 0) is s


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    score = np.round(rng.random(60), 1)  # ties on purpose
    pos, neg = score[y == 1], score[y == 0]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert fate.auroc(y, score) == pytest.approx(expected)


def test_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    perfect = fate.metrics(y, np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float))
    assert perfect["balanced_acc"] == 1 and perfect["auroc"] == 1 and perfect["macro_f1"] == 1
    constant = fate.metrics(y, np.array([[0, 1]] * 4, dtype=float))
    assert constant["balanced_acc"] == 0.5


def test_length_hazard_learns_length_signal():
    rng = np.random.default_rng(2)
    train = make(30, 12, 1, rng) + make(30, 40, 0, rng)   # divisions short, deaths long
    model = fate.FATE_MODELS["length_hazard"]().fit(train)
    p = model.predict_proba(make(1, 12, 1, rng) + make(1, 40, 0, rng))
    assert p[0, 1] > 0.5 > p[1, 1]
    assert np.allclose(p.sum(1), 1)


def test_registry_has_both_models():
    assert {"length_hazard", "crop_transformer"} <= set(fate.FATE_MODELS)


def test_crop_transformer_forward_and_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from fate.crop_transformer import CropTransformerNet, collate

    seqs = make(2, 9, 0, np.random.default_rng(3)) + make(1, 5, 1, np.random.default_rng(4))
    x, pad = collate(seqs, max_len=8, train=False, device="cpu")
    assert x.shape == (3, 8, 3, 8, 8) and pad.shape == (3, 8)      # longest sequence cut to max_len
    assert bool(pad[2, 5:].all()) and not bool(pad[2, :5].any())   # short sequence is padded at the end
    net = CropTransformerNet(pretrained=False, max_len=8).eval()
    with torch.no_grad():
        assert net(x, pad).shape == (3, len(fate.FATES))
