"""Track a video given its instance segmentation masks.

Usage:
    python tracking.py --imgs path/to/imgs --masks path/to/masks --out tracked_ctc
    python tracking.py --imgs video.tif --masks masks.tif --model trackastra --mode ilp

`--imgs` / `--masks` are either a folder of per-frame .tif files or a single
multi-frame .tif stack with shape (T, (Z), Y, X). Output is written in Cell
Tracking Challenge (CTC) format.
"""

import argparse
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import tifffile

from trackastra.data import load_tiff_timeseries
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

logger = logging.getLogger(__name__)


def load_stack(path: str | Path) -> np.ndarray:
    """Load a folder of per-frame tifs or a multi-frame tif as (T, (Z), Y, X)."""
    path = Path(path)
    if path.is_dir():
        return load_tiff_timeseries(path)
    return tifffile.imread(path)


def track_trackastra(
    imgs: np.ndarray,
    masks: np.ndarray,
    mode: str = "ilp",
    pretrained: str = "general_2d",
    device: str = "automatic",
    **kwargs,
) -> tuple[nx.DiGraph, np.ndarray]:
    model = Trackastra.from_pretrained(pretrained, device=device)
    return model.track(imgs, masks, mode=mode, **kwargs)


# Registry of tracking models: name -> function(imgs, masks, **kwargs) -> (graph, masks)
TRACKERS = {"trackastra": track_trackastra}


def track(
    imgs: np.ndarray,
    masks: np.ndarray,
    model: str = "trackastra",
    **kwargs,
) -> tuple[nx.DiGraph, np.ndarray]:
    """Run the selected tracking model on a video and its masks.

    Args:
        imgs: Video of shape (T, (Z), Y, X).
        masks: Instance segmentation of the same shape.
        model: Name of the tracking model, one of `TRACKERS`.
        **kwargs: Passed to the model's tracking function (e.g. `mode`).

    Returns:
        Track graph (nodes = detections, edges = links) and the filtered masks.
    """
    if model not in TRACKERS:
        raise ValueError(f"Unknown model {model!r}, choose from {list(TRACKERS)}")
    if imgs.shape != masks.shape:
        raise ValueError(f"imgs {imgs.shape} and masks {masks.shape} must match")
    return TRACKERS[model](imgs, masks, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imgs", required=True, help="Folder of tifs or a tif stack")
    ap.add_argument("--masks", required=True, help="Folder of tifs or a tif stack")
    ap.add_argument("--out", default="tracked_ctc", help="Output folder (CTC format)")
    ap.add_argument("--model", default="trackastra", choices=list(TRACKERS))
    ap.add_argument("--mode", default="ilp", choices=["ilp", "greedy", "greedy_nodiv"])
    ap.add_argument("--pretrained", default="general_2d", help="Pretrained model name")
    ap.add_argument("--device", default="automatic")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    imgs, masks = load_stack(args.imgs), load_stack(args.masks)
    logger.info(f"imgs {imgs.shape}, masks {masks.shape}")

    extra = (
        dict(mode=args.mode, pretrained=args.pretrained, device=args.device)
        if args.model == "trackastra"
        else {}
    )
    graph, masks_tracked = track(imgs, masks, model=args.model, **extra)

    # graph_to_ctc relabels masks so each track has one consistent ID
    graph_to_ctc(graph, masks_tracked, outdir=args.out)
    logger.info(f"{graph.number_of_nodes()} detections, {graph.number_of_edges()} links -> {args.out}")


if __name__ == "__main__":
    main()
