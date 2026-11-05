"""Provisional instance masks with Cellpose-SAM, for datasets that ship none.

Used to test the tracking pipeline end-to-end; final tracks should use the
team's agreed segmentation once it is available.
"""

import numpy as np

_MODEL = None


def segment_cellpose_sam(imgs, batch_size=8, **eval_kwargs) -> np.ndarray:
    """Segment every frame of ``imgs`` (T, Y, X) -> instance masks (T, Y, X) uint16."""
    global _MODEL
    from cellpose import models

    if _MODEL is None:
        _MODEL = models.CellposeModel(gpu=True)  # cpsam
    masks = _MODEL.eval(list(imgs), batch_size=batch_size, **eval_kwargs)[0]
    return np.stack(masks).astype(np.uint16)
