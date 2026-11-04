# Tracking results and videos (Team 1, Damià)

* `tracking/summary.md`: tables for every dataset, method and mask source (mean over sequences);
  `tracking/all_runs.csv`: one row per sequence. Regenerate with
  `python scripts/collect_results.py "<log dir>/*.log" --out results/tracking`.
* `videos/`: lineage videos made with `scripts/render_tracks.py` (`tracking/viz.py`).

## How to read the videos

Left: the frames with each cell's mask filled in its colour (one colour per cell; per family
when there are many families). Right: the lineage tree of the families that divide in the
window, growing with the video; a mother's line, the vertical connector and her two daughters
meet at the division.

| marker | meaning |
|---|---|
| white ring | division (on the two daughters) |
| red × | track lost: ends without dividing, away from the border, before the last frame |
| green + | cell appears from nothing (away from the border, after frame 0); next to a red × it is usually the same cell re-found under a new id |
| yellow ◇ | missed division (ground truth divides, the tracks don't) |
| magenta ring | false division (tracks divide, ground truth doesn't) |

## Videos

| file | data | what to look at |
|---|---|---|
| `musc01_hoct.mp4` vs `musc01_hoct_koft_tta.mp4` | BF-C2DL-MuSC 01, last frames (densest divisions), CTC silver-truth masks | the hybrid + TTA misses fewer divisions (window: 4 → 2 missed) |
| `musc02_hoct_koft_tta.mp4` | MuSC 02 | |
| `hela01_hoct_koft_tta.mp4`, `hela02_hoct_koft_tta.mp4` | Fluo-N2DL-HeLa, all 92 frames | many families, coloured per family |
| `hsc02_hoct_koft.mp4` | BF-C2DL-HSC 02 | dense proliferation |
| `alfi-dic03_hoct.mp4` vs `alfi-dic03_hoct_koft_tta.mp4`, same for `alfi-dic04_*` | ALFI-DIC 03 / 04 (not in HOCT's training data), GT masks, all 69 frames | TTA finds the divisions plain HOCT misses: 03 goes from 3 found / 2 missed to 5 / 0 (BC .75 → 1.0), 04 from 0 / 1 to 1 / 0 (BC 0 → 1.0) |
| `hela01_hoct_koft_tta_short.gif` | HeLa 01, 60 frames | small GIF for Slack |
| `helmholtz_red_256-329_*.mp4`, `helmholtz_red_66-109_*.mp4` | our timelapse, red reporter, Cellpose masks (recipe v2), stage-registered | most red × have a green + next to them a few frames later: the red mask of dim cells vanishes for > 3 frames (segmentation flicker, not linking). Frame 283 of the raw video (index 27 of the first window) is the 663-px stage jump: many cells leave the field there. `hoct_koft_tta_long` (gap filling across 8 frames) cuts track ends by 25% on 284-329 without changing divisions |
