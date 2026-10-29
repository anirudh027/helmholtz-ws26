"""Train Cellpose-SAM (https://github.com/MouseLand/cellpose) from a PyTorch DataLoader.

Works with any dataset whose samples hold ``"image"`` [C, H, W] and an instance mask ``"target"`` [H, W].
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import fastremap
import numpy as np
import torch
from cellpose import dynamics, transforms
from cellpose.train import _loss_fn_seg
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader


class CellposeTransform:
    """Dataset transform: cellpose normalization, ``augment``, then flow targets ``sample["flows"]`` [3, H, W].

    Flows are computed after augmentation so they match the cropped/rotated mask. Cellpose-SAM trains on 256 x 256
    crops, e.g. ``augment=RandomCropFlip(256)``.
    """

    def __init__(self, augment: Callable[[dict], dict] | None = None):
        self.augment = augment

    def __call__(self, sample: dict) -> dict:
        img = sample["image"].float().numpy()[:3]
        img = transforms.normalize_img(img, axis=0)  # per-channel 1-99 percentile
        img = np.concatenate([img, np.zeros((3 - len(img), *img.shape[1:]), img.dtype)])  # pad to 3 channels
        sample = {**sample, "image": torch.from_numpy(img)}
        if self.augment is not None:
            sample = self.augment(sample)
        mask = fastremap.renumber(sample["target"].numpy(), in_place=False)[0]  # flows need ids 1..N
        flows = dynamics.masks_to_flows_gpu(mask, device=torch.device("cpu"))[0]
        sample["flows"] = torch.from_numpy(np.concatenate([(mask > 0)[None], flows]).astype(np.float32))
        return sample


def train_cellpose(
    model,
    loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    max_steps_per_epoch: int | None = None,
    save_path: str | Path | None = None,
) -> list[float]:
    """Fine-tune a ``CellposeModel`` with the loss and AdamW settings of ``cellpose.train.train_seg``.

    Uses bfloat16 autocast and gradient checkpointing so that Cellpose-SAM fits on an 8 GB GPU. The learning rate
    warms up linearly over 10 epochs. Returns the mean loss per epoch; saved weights load with
    ``CellposeModel(pretrained_model=save_path)``.
    """
    net, device = model.net, model.device
    original_dtype = net.dtype
    net.dtype = torch.float32  # train in fp32 weights
    blocks = list(net.encoder.blocks)
    forwards = [blk.forward for blk in blocks]
    for blk, fwd in zip(blocks, forwards, strict=True):
        blk.forward = lambda x, fwd=fwd: checkpoint(fwd, x, use_reentrant=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses = []
    try:
        net.train()
        for epoch in range(n_epochs):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, epoch / 10)
            total, n = 0.0, 0
            for step, batch in enumerate(loader):
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break
                x, lbl = batch["image"].to(device), batch["flows"].to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    y = net(x)[0]
                loss = _loss_fn_seg(lbl, y.float(), device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, n = total + loss.item() * len(x), n + len(x)
            losses.append(total / max(n, 1))
            print(f"epoch {epoch}: loss={losses[-1]:.4f}")
        if save_path is not None:
            net.save_model(save_path)
    finally:
        for blk, fwd in zip(blocks, forwards, strict=True):
            blk.forward = fwd
        net.dtype = original_dtype
        net.eval()
    return losses
