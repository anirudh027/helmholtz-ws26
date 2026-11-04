"""HOCT through its Python API instead of the CLI.

The CLI (``hoct track``) already uses HOCT's best solver settings: candidate
links up to 3 frames, the two-pass tracklet ILP and an appearance cost scaled
by the predicted orphan probability (``ILPSolverConfig.default()``). What it
does not expose is test-time augmentation (TTA: the paper's best configuration
averages 6 randomly flipped/rotated copies of the graph) or the candidate
graph itself, which the few-label correction (``tracking.correct``) needs.

    from tracking.hoct import HOCT
    h = HOCT(tta=6)
    res = h.track(imgs, masks)                 # CTCResult
    g = h.last_graph                           # candidate graph with 'similarity' / 'solution'
"""

import os
import random
from dataclasses import dataclass, field

import numpy as np

from .ctc import CTCResult, read_ctc

_MODELS = {}


def load_model(name="general_v1", device="cuda"):
    """Load (and cache) a HOCT TorchScript checkpoint by registered name or path."""
    import torch
    from hoct._models import resolve_model

    key = (name, device)
    if key not in _MODELS:
        m = torch.jit.load(resolve_model(name), map_location=device).to(device)
        m.eval()
        _MODELS[key] = m
    return _MODELS[key]


def _seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def solution_to_ctc(graph, shape, work_dir) -> CTCResult:
    """Rasterise a solved tracksdata graph into a CTCResult (via its CTC writer)."""
    graph.assign_tracklet_ids()
    out = os.path.join(work_dir, "RES")
    graph.to_ctc(output_dir=out, shape=shape, overwrite=True)
    res = read_ctc(out)
    res.masks = res.masks.reshape((res.masks.shape[0],) + tuple(shape[-2:]))
    return res


@dataclass
class HOCT:
    """HOCT tracker with every knob of the paper's configuration exposed.

    Args:
        model: registered checkpoint name or path (``general_v1`` = trained on 16 CTC datasets).
        tta: number of test-time augmentations (0 = CLI behaviour, 6 = paper).
        max_distance: candidate-link radius in px (CLI default 300).
        n_neighbors: candidates per node.
        max_dt: longest temporal gap a link may bridge (frames).
        window: temporal window of the transformer.
        solver: ``ILPSolverConfig`` field overrides, e.g. ``{"division_weight": 0.5}``.
        tile: ``"auto"`` / ``"on"`` / ``"off"`` spatial tiling (as the CLI).
    """

    model: str = "general_v1"
    tta: int = 6
    max_distance: float = 300.0
    n_neighbors: int = 5
    max_dt: int = 3
    window: int = 5
    solver: dict = field(default_factory=dict)
    tile: str = "auto"
    device: str = "automatic"
    seed: int = 0
    last_graph: object = field(default=None, repr=False)

    def _device(self):
        if self.device != "automatic":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def solver_config(self):
        from hoct.tracking import ILPSolverConfig

        return ILPSolverConfig(**{**ILPSolverConfig.default().model_dump(), **self.solver})

    def tiling(self, graph):
        from hoct.cli import TileMode, _maybe_build_tiling_scheme

        return _maybe_build_tiling_scheme(graph, TileMode(self.tile))

    def build_graph(self, imgs, masks):
        import dask.array as da
        from hoct.features import create_graph

        # One dask chunk per frame: with dask's automatic multi-frame chunks, create_graph's
        # per-frame rechunk makes every frame walk the whole task graph (superlinear in T).
        def per_frame(x):
            x = np.asarray(x)
            return da.from_array(x, chunks=(1, *x.shape[1:]))

        return create_graph(labels=per_frame(masks), images=per_frame(imgs),
                            distance_threshold=self.max_distance, n_neighbors=self.n_neighbors,
                            delta_t=self.max_dt)

    def predict(self, graph, model=None):
        """Score every candidate link and solve the ILP; ``graph`` gets 'similarity',
        'orphan_prob' and 'solution' attributes in place. Returns the solution graph."""
        from hoct._api import predict

        _seed(self.seed)
        model = model if model is not None else load_model(self.model, self._device())
        return predict(model, graph=graph, solver_config=self.solver_config(), window_size=self.window,
                       tiling_scheme=self.tiling(graph), test_time_augs=self.tta)

    def track(self, imgs, masks, model=None) -> CTCResult:
        from .backends import _scratch

        graph = self.build_graph(imgs, masks)
        solution = self.predict(graph, model)
        self.last_graph = graph
        with _scratch("hoct_") as work:
            res = solution_to_ctc(solution, graph.metadata.get("shape", np.shape(masks)), work)
        res.info.update(hoct={k: v for k, v in self.__dict__.items() if k != "last_graph"})
        return res
