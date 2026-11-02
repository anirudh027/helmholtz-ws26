# tracking: one function, any tracker

```python
from tracking import track, write_ctc

res = track(imgs, masks, model="hoct_koft_tta")   # imgs, masks: (T, Y, X); masks = instance segmentation
graph, masks_tracked = res                         # detection-level nx.DiGraph + track-id masks
res.tracks                                         # {track_id: (t_start, t_end, parent_id)}  (CTC lineage)
write_ctc("out/01_RES", res.masks, res.tracks)

res = track(imgs, masks, model="hoct_koft_tta", register=True)   # moving stage (Helmholtz)
res = track(red, masks, model="hoct_koft_tta", register=bf)      # estimate the drift on another channel
```

Swap the tracker by changing `model`:

| `model` | Method | Notes |
|---|---|---|
| `trackastra` / `trackastra_ilp` | Trackastra `general_2d`, greedy / ILP linking | GPU |
| `laptrack` | LapTrack (TrackMate-style LAP + divisions) | CPU, seconds |
| `ultrack` | Ultrack, whole sequence | CPU; built for several segmentation hypotheses, weak with a single one |
| `hoct` | HOCT `general_v1` (Royer lab 2026), = `hoct track` CLI | GPU, ~1 min for 1376 MuSC frames |
| `hoct_tta` | HOCT + 6 test-time augmentations (the paper's best configuration) | ~3x `hoct` |
| `hoct_koft` | HOCT + KOFT-style optical-flow gap filling (Damià's hybrid) | +~1.5 min |
| `hoct_koft_tta` | both | **best overall** |

Trackers don't segment. Pass masks (e.g. CTC `ST/SEG`, Cellpose-SAM).
Backend options go through `track(..., **kwargs)`, e.g. `max_distance=150`, `max_dt=3`,
`solver={"division_weight": 0.5}` for HOCT, `max_gap=3`, `appearance_thresh=0.25` for the gap filling.

### What each HOCT piece does

* `tracking/hoct.py` calls HOCT through its Python API. The CLI already uses the paper's solver
  (links up to 3 frames, two-pass tracklet ILP, appearance cost scaled by the predicted orphan
  probability) but no test-time augmentation; `tta=6` adds it. Arrays are handed to HOCT as one
  dask chunk per frame (with automatic chunks the graph construction is superlinear in T: 15 min
  instead of ~1 min on MuSC).
* `tracking/koft.py` is the gap filling of the hybrid: when HOCT links a parent to a single child
  across missing frames, the child's first mask is advected backwards along Farneback optical flow
  into the gap (appearance-gated, so no cells are invented), and remaining track ends are joined to
  later births by constant-velocity prediction (KOFT, Reme et al. ISBI 2024).
* `tracking/register.py` compensates the stage drift (phase correlation against the last two usable
  frames, alias check for jumps larger than 100 px, blank/defocused frames skipped). On the Helmholtz
  timelapse it reproduces the hand-verified offsets of Damià's stabiliser within 3 px on all 309 kept frames.

## Batch runs + scoring

```bash
/mnt/1t/conda_envs/tracking/bin/python scripts/track_all.py \
    --datasets musc hela hsc --backends hoct hoct_koft hoct_tta hoct_koft_tta
python scripts/track_all.py --datasets helmholtz --backends hoct_koft_tta --register
python scripts/track_all.py --datasets alfi c2c12 --backends hoct_koft_tta --masks-dir /mnt/1t/masks/{dataset}
```

Results go to `/mnt/1t/tracks/<dataset>/<backend>/<seq>_RES/` (CTC format) plus `results.csv`.
Every metric the dataset's ground truth allows is computed:

| Ground truth | Datasets | Metrics |
|---|---|---|
| complete CTC `<seq>_GT` | MuSC, HeLa, HSC (`scripts/download_ctc.sh`) | TRA, DET, LNK, CT, BC, CCA (py-ctcmetrics) |
| tracked points | ALFI (boxes + lineage), C2C12 (human tracks), HematoFate (2 cells), p53 (lineage table), Helmholtz (`HELMHOLTZ_POINTS=labels.csv`) | `pt_det_recall`, `pt_link_acc` (point LNK), `pt_id_switches`, `pt_completeness`, `pt_survival`, `pt_div_recall`, `pt_div_precision_annot` |
| masks of some cells | QPI | `annot_*`: CTC metrics on the tracks of the annotated cells only |

`point_metrics` scores only the annotated cells, so sparse annotations are fine. On MuSC (GT
centroids as points) it ranks hoct > laptrack > trackastra exactly like LNK and BC.

Options: `--masks-dir` reads Team 2's masks (`<dir>/<seq>.npy|.tif` or `<dir>/<seq>/*.tif`,
`{dataset}` is filled in); `--masks cellpose` ignores shipped masks (ALFI's are ground truth);
`--register`; `--kw '{"max_distance": 150}' --tag d150` for parameter sweeps.

## Few-label correction (HOCT paper, sec. 4.5)

The HOCT backbone stays frozen and only a logistic-regression head on its link features is refitted
from a few verified links, proposed by uncertainty in rounds of 20 (`tracking/correct.py`).

```bash
# simulate an annotator with the GT (how much does each round buy?)
python scripts/correction_sim.py --dataset hela --seq 02 --rounds 20
# real annotation on our data: each call applies all verdicts in labels.csv, refits,
# writes tracks and proposes 20 more links as an image sheet
python scripts/correction_round.py --dataset helmholtz --register --masks-dir /mnt/1t/masks/{dataset}
```

Fill the `correct` column of `labels.csv` with 1 (same cell, or mother → daughter) or 0, looking at
`sheet_rXX.png` (red cross = cell in the earlier frame, cyan = proposed match). The verdicts double
as point ground truth for the Helmholtz video.

Credits: the registry API is from Nikolas Rieger, the unified module and dataset adapters from
Giuseppe Giglio, the dataloaders from Florian Donhauser. CTC I/O, backend settings, the HOCT+KOFT
hybrid and the stage registration are from Damià Vicens Ramis' Cell-project benchmark.
