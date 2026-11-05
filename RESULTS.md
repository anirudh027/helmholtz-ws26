# Zero-shot Cellpose-SAM results

Pre-trained Cellpose-SAM (`cpsam_v2`, no fine-tuning) on the full annotated data of every dataset, one
`examples/<dataset>_cellpose_sam.py` script per dataset. Metrics differ because the annotations do (masks, centroids,
track points); each section says what is scored.

## Reproduce

1. Environment (see [README](README.md#setup)): `pip install -r requirements-lock.txt` for the exact versions used
   here, after installing torch from the CUDA index; Cellpose from the pinned clone.
2. Data: `bash dataloaders/download_<dataset>.sh <root>` (sizes in the README table).
3. Run: `python examples/<dataset>_cellpose_sam.py <root>`; the command per dataset is listed below. Results are
   printed to stdout.

Environment of the runs below: Windows 11, Python 3.11.4, torch 2.15.0+cu134, NVIDIA RTX 3070 (8 GB),
Cellpose 4.2.1.1 at commit `a54cb48`, model `cpsam_v2`. Runtimes are wall-clock; runs marked "shared GPU" overlapped
with other evaluations and are slower than a run on an idle GPU.

## Summary

| Dataset | Evaluated data | Metric | Result | Runtime |
| --- | --- | --- | --- | --- |
| HematoFatePrediction | all 767 tracked frames, 810 track points | track point hit rate (mask within 10 px) | 88.6% | 33 min (shared GPU) |
| C2C12 | 090303/F0009, first 800 frames, 98195 cells | centroid precision / recall | 0.965 / 0.874 | 61 min (shared GPU) |
| cellX-predict | | | pending | |
| QPI cell death | | | pending | |
| p53 cell fate | all 401 MNNG1 frames, 264747 track points | track point hit rate (mask within 5 px) | 5.7% | 39 min (shared GPU) |
| Xenopus | 3 stacks, 45 z-slices per target | precision / recall @0.5 IoU | cells 0.80 / 0.81, nuclei 0.66 / 0.58 | 8.4 min |
| ALFI | 796 MI frames, 16629 cells | centroid precision / recall | 0.46-0.85 / 0.24-0.80 per sequence | ~2 h (shared GPU) |
| Celldetective | all 362 images of 3 sets | recall @0.5 IoU / mean IoU | 0.76 / 0.65, 0.49 / 0.42, 0.49 / 0.41 | 49 min (shared GPU) |

Not evaluated: the Xenopus time-lapses (S-BIAD2969, 35-64 GB each, not downloaded, no masks).

## Xenopus

`python examples/xenopus_cellpose_sam.py D:/datasets/xenopus`. Each z-slice of the 0h / 10h / 20h stacks against the
slice of the authors' 3D labels; the matching channel alone is the input (membrane for cells, nuclei for nuclei).
The reference labels come from the authors' Cellpose / StarDist models, not from manual annotation.

| Target | precision @0.5 | recall @0.5 |
| --- | --- | --- |
| membrane (cells) | 0.796 | 0.810 |
| nuclei | 0.664 | 0.578 |

## ALFI

`python examples/alfi_cellpose_sam.py D:/datasets/alfi`. The masks outline nuclei (whole cells only in mitosis) while
Cellpose-SAM segments whole cells, so detection is scored like C2C12: a predicted mask is correct if it contains the
centroid of an annotated cell. IoU against the nucleus masks is reported as well. Precision is understated a little:
cells cut by the image border are often not annotated.

| Sequence | precision | recall | recall @0.5 IoU | mean IoU |
| --- | --- | --- | --- | --- |
| MI01 | 0.560 | 0.477 | 0.177 | 0.226 |
| MI02 | 0.464 | 0.484 | 0.057 | 0.173 |
| MI03 | 0.591 | 0.487 | 0.147 | 0.217 |
| MI04 | 0.642 | 0.532 | 0.468 | 0.420 |
| MI05 | 0.760 | 0.491 | 0.275 | 0.285 |
| MI06 | 0.845 | 0.795 | 0.153 | 0.319 |
| MI07 | 0.730 | 0.353 | 0.031 | 0.102 |
| MI08 | 0.625 | 0.241 | 0.050 | 0.088 |

## Celldetective

`python examples/celldetective_cellpose_sam.py D:/datasets/celldetective`. Best IoU of every annotated cell; only one
population is annotated per set, so precision is not meaningful. Input channels per set are in the script.

| Set | recall @0.5 | mean IoU | annotated cells | predicted masks | images |
| --- | --- | --- | --- | --- | --- |
| MCF-7 nuclei | 0.756 | 0.648 | 5705 | 10643 | 122 |
| NK cells | 0.493 | 0.417 | 3798 | 7832 | 105 |
| RICM lymphocytes | 0.487 | 0.408 | 1925 | 1507 | 135 |

## HematoFatePrediction

`python examples/hemato_fate_cellpose_sam.py D:/datasets/hemato_fate`. There are no masks, only the manual track
points of 2 cells, which can be ~20 px off the cell centre. A point counts as a hit if a predicted mask lies within
10 px of it. Every frame with a track point is evaluated (bright-field, raw 8-bit scaled to [0, 1]).

| Track points | Frames | hit rate | masks / frame |
| --- | --- | --- | --- |
| 810 | 767 | 0.886 | 33.2 |

## p53 cell fate

`python examples/p53_fate_cellpose_sam.py D:/datasets/p53_fate` (the download needs a Dryad API token:
`DRYAD_TOKEN=<token> bash dataloaders/download_p53_fate.sh D:/datasets/p53_fate`, token from
https://datadryad.org/account, valid 10 h). All 8 lineage tables on Dryad are the same file, and its points and lineage
numbers match the MNNG1 video only, so only MNNG1 is scored: a track point is a hit if a mask lies within 5 px of it
(points outside the video crop, 0.2%, are skipped). The videos are the authors' tracking output with crosses and
lineage numbers burned in (converted to greyscale). Zero-shot Cellpose-SAM fails on them: masks are mostly small
(median 45 px) and 64% of them lie on the burned-in overlay (checked on frame 201), while the large, flat cells are
rarely segmented.

| Track points | Frames | hit rate (5 px) | hit rate (at point) | masks / frame |
| --- | --- | --- | --- | --- |
| 264747 | 401 | 0.057 | 0.021 | 353.1 |

## C2C12

`python examples/c2c12_cellpose_sam.py D:/datasets/c2c12`. C2C12 has centroids only: a predicted mask is correct if it
contains an annotated centroid (the human "full" annotation of field F0009, the first 800 frames of 090303).

| Frames | Annotated cells | precision | recall |
| --- | --- | --- | --- |
| 800 | 98195 | 0.965 | 0.874 |
