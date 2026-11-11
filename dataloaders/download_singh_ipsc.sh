#!/usr/bin/env bash
# Download + extract the Singh et al. iPSC reprogramming ROIs (https://github.com/abhineet123/ipsc_prediction), one
# 5.3 GB zip from the Hugging Face repo abhineet123/ipsc_prediction holding the JPEG frames + per-ROI annotation CSVs
# (5.7 GB extracted). The raw well images (7 GB) and the COCO / YTVIS JSON conversions are not downloaded.
# Usage: ./download_singh_ipsc.sh [ROOT=data/singh_ipsc]
set -euo pipefail
ROOT=${1:-data/singh_ipsc}
SHA256=de085f85b54cc16e63b85e9adb9abde9d70315959729c620089ae3b282be31e5
[[ -e "$ROOT/.done" ]] && exit 0
mkdir -p "$ROOT"
ZIP="$ROOT/roi_images_and_labels.zip"
curl -fL# -C - --retry 5 -o "$ZIP" \
  https://huggingface.co/abhineet123/ipsc_prediction/resolve/main/ipsc_data/roi_images_and_labels.zip
echo "$SHA256  $ZIP" | shasum -a 256 -c -
unzip -oq "$ZIP" -d "$ROOT" && rm "$ZIP" && touch "$ROOT/.done"
