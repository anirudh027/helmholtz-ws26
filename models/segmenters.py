"""Zero-shot instance segmenters behind one interface, so every ``examples/*_zero_shot.py`` script can compare models.

``load_segmenter(name)`` returns ``segment(images) -> masks``: ``images`` is a list of [H, W] or [C, H, W] (C <= 3)
float arrays, ``masks`` a list of int [H, W] instance masks (0 = background, 1..N).

* ``cpsam``: Cellpose-SAM (``cpsam_v2``), called exactly as before (``model.eval`` on the list, ``channel_axis=0`` for
  multi-channel images).
* ``sam2``: SAM2.1 Hiera-L automatic mask generator (``models/sam2.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
from torch.utils.data import Dataset, Subset

MODELS = ("cpsam", "sam2")
Segmenter = Callable[[list[np.ndarray]], list[np.ndarray]]
D = TypeVar("D", bound=Dataset)


def load_segmenter(name: str) -> Segmenter:
    if name == "cpsam":
        from cellpose import models

        model = models.CellposeModel(gpu=True)  # cpsam_v2

        def segment(images: list[np.ndarray]) -> list[np.ndarray]:
            return list(model.eval(list(images), channel_axis=0 if images[0].ndim == 3 else None)[0])

        return segment
    if name == "sam2":
        from models.sam2 import SAM2Segmenter

        return SAM2Segmenter()
    raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")


def parse_options(argv: list[str]) -> tuple[str, int, list[str]]:
    """Split ``--model <name>`` (default ``cpsam``) and ``--every <n>`` (default 1) off a script's arguments.

    ``--every n`` scores every n-th sample only: SAM2's automatic mask generator takes seconds per image.
    """
    options = {"--model": "cpsam", "--every": "1"}
    rest = list(argv)
    for flag in options:
        if flag in rest:
            i = rest.index(flag)
            if i + 1 >= len(rest):
                raise SystemExit(f"{flag} needs a value")
            options[flag] = rest.pop(i + 1)
            rest.pop(i)
    if options["--model"] not in MODELS:
        raise SystemExit(f"--model needs one of {MODELS}")
    return options["--model"], int(options["--every"]), rest


def every_nth(ds: D, n: int) -> D | Subset:
    """Every n-th sample of ``ds`` (``ds`` itself for n = 1)."""
    return ds if n == 1 else Subset(ds, range(0, len(ds), n))  # type: ignore[arg-type]  # map-style: has len
