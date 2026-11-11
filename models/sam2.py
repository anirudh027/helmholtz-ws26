"""Zero-shot instance segmentation with SAM2's automatic mask generator (https://github.com/facebookresearch/sam2).

SAM2 is not trained on microscopy and takes 8-bit RGB: images are stretched per channel (1-99.5 percentile), 1 channel
is repeated to 3, 2 channels go to R and G. The automatic mask generator prompts a grid of points; its masks overlap,
so they are painted largest first (smaller masks on top) into one instance mask, and masks covering more than
``max_area`` of the image (background, whole field of view) are dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# By default the generator runs with SAM2's own settings (32 x 32 point grid, no crops). DENSE_KWARGS are the settings
# of SAM2's example notebook, for SAM2Segmenter(**DENSE_KWARGS): more, smaller masks, but ~14x slower, and they split
# 64 x 64 nucleus crops into many masks.
DENSE_KWARGS = dict(
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25,
    use_m2m=True,
)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_sam2_image(image: np.ndarray) -> np.ndarray:
    """[H, W] or [C, H, W] (C <= 3) float image -> [H, W, 3] uint8, stretched per channel to the 1-99.5 percentile."""
    x = np.asarray(image, np.float32).reshape(-1, *np.shape(image)[-2:])
    lo, hi = np.percentile(x, (1, 99.5), axis=(1, 2), keepdims=True)
    x = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    x = np.repeat(x, 3, 0) if len(x) == 1 else np.concatenate([x, np.zeros((3 - len(x), *x.shape[1:]), x.dtype)])
    return (x.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def masks_to_instances(masks: list[dict], shape: tuple[int, int], max_area: float = 0.5) -> np.ndarray:
    """SAM2 mask records -> int32 [H, W] instance mask with ids 1..N (largest painted first, so smaller win)."""
    out = np.zeros(shape, np.int32)
    for i, m in enumerate(sorted(masks, key=lambda m: -m["area"]), 1):
        if m["area"] <= max_area * out.size:
            out[m["segmentation"]] = i
    return np.unique(out, return_inverse=True)[1].reshape(shape).astype(np.int32)  # renumber 1..N, 0 stays 0


class SAM2Segmenter:
    """Callable ``images -> instance masks`` around ``SAM2AutomaticMaskGenerator`` (checkpoint: sam2.1_hiera_large).

    ``generator_kwargs`` go to the generator (default: SAM2's own defaults; see ``DENSE_KWARGS``).
    """

    def __init__(
        self,
        checkpoint: str | Path = CHECKPOINT,
        config: str = CONFIG,
        device: torch.device | None = None,
        max_area: float = 0.5,
        **generator_kwargs,
    ):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        self.device, self.max_area = device or default_device(), max_area
        model = build_sam2(config, str(checkpoint), device=self.device, apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(model, **generator_kwargs)

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad()
        out = []
        with torch.inference_mode(), autocast:
            for image in images:
                masks = self.generator.generate(to_sam2_image(image))
                out.append(masks_to_instances(masks, np.shape(image)[-2:], self.max_area))
        return out
