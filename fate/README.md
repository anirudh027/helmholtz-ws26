# fate: tracker-independent cell fate prediction

Predicts from a tracked cell's history whether it will **die** or **divide**. The input is the
cell's image crops over time (`CellSequence`), so it does not depend on which tracker produced
the track, and the model can be swapped by name.

```python
import fate
model = fate.FATE_MODELS["crop_transformer"](epochs=12)   # or "length_hazard" (baseline, numpy only)
model.fit(train_sequences, val_sequences)                  # list[fate.CellSequence]
proba = model.predict_proba(test_sequences)                # (N, 2): [death, division]
```

* `fate/__init__.py`: `CellSequence`, the `FatePredictor` protocol, `truncate` (drop the last N frames to
  predict N frames ahead), the `LengthHazard` baseline, metrics, and the model registry `FATE_MODELS`.
* `fate/crop_transformer.py`: `CropTransformer`, after Bürger et al. 2026 ("Understanding cell fate
  decisions with temporal attention"): ResNet-18 per frame, then a 2-layer Transformer over time with a
  summary token. Smaller than the paper's ResNet-50 / 8-layer model. This is our own reimplementation, not
  the authors' code or weights. Needs torch and torchvision.
* `fate/buerger.py`: loader for the Bürger dataset (CC BY 4.0, DOI 10.4126/FRL01-006526608).
* `scripts/fate_train_buerger.py`, `scripts/fate_eval_trackastra.py`, `scripts/fate_segment_buerger.py`:
  train and evaluate, evaluate on Trackastra's tracks, segment with Cellpose-SAM (run this one in a separate
  Cellpose environment).
* `fate/results/`: JSON results of the runs below.
* `tests/test_fate.py`: unit tests (`pytest tests/test_fate.py`).

## Data
Download `annotations.json` and the video archives from the dataset page and extract the videos to
`data/buerger/raw/<video>/`. The archives are about 19 GB per 20 videos, so we used videos 1-20 and
streamed them out of `video_1-20.tar.gz`. A cell is labelled division if it has an annotated division
time, death if it has an annotated death time; cells with neither are dropped (857 cells in videos 1-20,
38% divisions). Sequences are the last 48 alive frames, 112x112 crops, 3 fluorescence channels.

## Results
Split by video: 14 train, 3 validation, 3 test (`001`, `016`, `013`; 141 test cells). One seed, single run.
AUROC for division vs death when the last N frames are cut off (one frame = 30 min):

| model / input | N=0 | 5 | 10 | 20 |
|---|---|---|---|---|
| `LengthHazard` (ignores images) | 0.777 | 0.764 | 0.709 | 0.756 |
| `CropTransformer`, annotation tracks | 0.997 | 0.961 | 0.940 | 0.919 |
| `CropTransformer`, trained with truncation | 0.933 | 0.925 | 0.913 | 0.874 |
| `CropTransformer`, Trackastra tracks on GT masks | 0.998 | 0.964 | 0.940 | 0.908 |
| `CropTransformer`, Trackastra tracks on Cellpose-SAM masks (108 of 141 cells) | 0.999 | 0.958 | 0.932 | 0.903 |

Balanced accuracy at N=0 / 5 / 10 / 20 for `CropTransformer` on annotation tracks: 0.974 / 0.834 / 0.788 / 0.772.
Training with truncated sequences did not help. Replacing annotation tracks with Trackastra tracks changed
AUROC by at most about 0.013.

## Caveats
* **Cells are lost, not mispredicted.** With Cellpose-SAM masks, 26 of the 141 dying or dividing cells (18%)
  have no detection with IoU >= 0.3 at their last frame; 23 of the 26 are dying cells, many faded to background
  brightness. The Cellpose row only covers the 108 cells that were found and tracked for at least 3 frames.
* **GT masks are mostly boxes.** About 90% of the dataset's `ctc_files` masks are exact rectangles, so tracking
  scores against them (TRA about 0.999) are weak evidence, and IoU against them understates real overlap.
* **Sequences end at the annotated event.** This classifies cells about to die or divide; it does not detect
  when an event happens.
* **Small test set:** 3 videos, 141 cells, one seed, no confidence intervals.
* **Channel order is unverified.** The three channels are presumably circadian NR1D1::VNP, p53-mKate and
  hGeminin-CFP, in that order. That comes from the authors' analysis code, not from the dataset documentation,
  and a quick intensity check on the data did not clearly confirm it. Do not rely on channel semantics.
* Death versus division only; no differentiation labels exist in this dataset.
* The trained weights are not in the repo.

## Using it with `tracking.track()`
`tracking.track()` returns track-ID masks and a track table (`CTCResult`). Cropping each cell along its track
into `CellSequence`s is done inline in `scripts/fate_eval_trackastra.py`; it has not been turned into a
reusable function yet.
