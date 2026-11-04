#!/usr/bin/env bash
# Download Cell Tracking Challenge training sets (full CTC ground truth) into $CTC_ROOT,
# laid out as tracking/datasets.py expects: $CTC_ROOT/<name>/Training/{01,01_GT,01_ST,...}.
#   bash scripts/download_ctc.sh                       # HeLa + HSC (+ MuSC if missing)
#   CTC_ROOT=/mnt/1t/datasets/ctc bash scripts/download_ctc.sh Fluo-N2DL-HeLa
# URL pattern of the CTC training-data server (could not be checked from the Cambridge box).
set -euo pipefail
ROOT=${CTC_ROOT:-/mnt/1t/datasets/ctc}
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(Fluo-N2DL-HeLa BF-C2DL-HSC BF-C2DL-MuSC)
for name in "${NAMES[@]}"; do
  dst="$ROOT/$name/Training"
  if [ -d "$dst/01_GT" ]; then echo "$name: already in $dst"; continue; fi
  mkdir -p "$dst"
  zip="$ROOT/$name.zip"
  wget -q --show-progress -O "$zip" "https://data.celltrackingchallenge.net/training-datasets/$name.zip"
  unzip -q "$zip" -d "$ROOT/.unzip_$name"
  mv "$ROOT/.unzip_$name/$name/"* "$dst/"
  rm -rf "$ROOT/.unzip_$name" "$zip"
  echo "$name -> $dst"
done
