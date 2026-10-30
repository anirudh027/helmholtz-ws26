#!/usr/bin/env bash
# Download + extract the cellX-predict dataset (https://doi.org/10.5522/04/16578959), ~470 MB.
# Usage: ./download_cellx_predict.sh [ROOT=data/cellx_predict]
set -euo pipefail
ROOT=${1:-data/cellx_predict}
[[ -e "$ROOT/.done" ]] && exit 0
mkdir -p "$ROOT"
curl -fL# --retry 5 -o "$ROOT.tar.gz" https://ndownloader.figshare.com/files/34925232
tar -xzf "$ROOT.tar.gz" -C "$ROOT" --strip-components=1 --exclude="._*" && rm "$ROOT.tar.gz" && touch "$ROOT/.done"
