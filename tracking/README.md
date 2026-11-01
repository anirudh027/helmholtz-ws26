# tracking: one function, any tracker

```python
from tracking import track, write_ctc

res = track(imgs, masks, model="hoct")   # imgs, masks: (T, Y, X); masks = instance segmentation
graph, masks_tracked = res               # detection-level nx.DiGraph + track-id masks
res.tracks                               # {track_id: (t_start, t_end, parent_id)}  (CTC lineage)
write_ctc("out/01_RES", res.masks, res.tracks)
```

Swap the tracker by changing `model`:

| `model` | Method | Notes |
|---|---|---|
| `trackastra` / `trackastra_ilp` | Trackastra `general_2d`, greedy / ILP linking | GPU |
| `laptrack` | LapTrack (TrackMate-style LAP + divisions) | CPU, seconds |
| `ultrack` | Ultrack, whole sequence | CPU |
| `hoct` | HOCT `general_v1` (Royer lab 2026) | GPU, best single model in Damià's benchmark |
| `hoct_koft` | HOCT + KOFT optical-flow gap filling (Damià) | pending |

Trackers don't segment. Pass masks (e.g. CTC `ST/SEG`, Cellpose-SAM).

## Batch runs + official CTC metrics

```bash
/mnt/1t/conda_envs/tracking/bin/python scripts/track_all.py \
    --datasets BF-C2DL-MuSC --backends laptrack trackastra trackastra_ilp ultrack hoct
```

Results go to `/mnt/1t/tracks/<dataset>/<backend>/<seq>_RES/` (CTC format), plus
`metrics_<seq>.csv` (TRA/DET/LNK/CT/BC/CCA via `py-ctcmetrics`) where `<seq>_GT` exists.
Any dataset in CTC layout can be passed as a path. Use `tracking.write_ctc_input` to
convert other datasets into that layout.

Credits: the registry API is from Nikolas Rieger. CTC I/O and backend settings are from
Damià Vicens Ramis' Cell-project benchmark.
