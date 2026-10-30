#!/usr/bin/env bash
# Download + extract the HematoFatePrediction example data (https://doi.org/10.82296/hmgu-nefeli.6qk3j-rsn72), ~1.1 GB.
# Usage: ./download_hemato_fate.sh [ROOT=data/hemato_fate]
set -euo pipefail
ROOT=${1:-data/hemato_fate}
[[ -e "$ROOT/.done" ]] && exit 0
mkdir -p "$ROOT"
curl -fL# -C - --retry 5 -o "$ROOT.zip" \
  https://nefeli.helmholtz-munich.de/api/records/6qk3j-rsn72/files/Rawdata_buggenthin_buettner_naturemethods2016.zip/content
unzip -oq "$ROOT.zip" -d "$ROOT" && rm "$ROOT.zip" && touch "$ROOT/.done"
