"""Convert the DeepSea tracking dataset into CTC layout, one folder per sequence.

    python scripts/deepsea_to_ctc.py /mnt/1t/.../deepsea_datasets/tracking_dataset/test /mnt/1t/datasets/ws26/deepsea

Every subfolder of the input with ``images/``, ``masks/`` and ``labels/`` becomes
``<out>/<set>/01``, ``01_ST/SEG`` and ``01_GT/{TRA,SEG}``. Prints what could not be converted.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.deepsea import deepsea_to_ctc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", type=Path, help="folder that holds the DeepSea set_* folders")
    parser.add_argument("out", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames of each sequence")
    args = parser.parse_args()

    sets = sorted(p for p in args.src.iterdir() if (p / "images").is_dir() and (p / "labels").is_dir())
    if not sets:
        sys.exit(f"No DeepSea sequence folders (images/ + labels/) in {args.src}")
    for set_dir in sets:
        report = deepsea_to_ctc(set_dir, args.out / set_dir.name, "01", args.limit)
        print(set_dir.name, json.dumps(report))


if __name__ == "__main__":
    main()
