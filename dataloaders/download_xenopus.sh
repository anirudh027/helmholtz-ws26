#!/usr/bin/env bash
# Download the Xenopus mucociliary epithelium Figure 2 stacks + 3D segmentations (https://doi.org/10.6019/S-BSST2777),
# 3.8 GB zip, ~1 GB kept. The full time-lapses (https://doi.org/10.6019/S-BIAD2969, 35-64 GB each) are not fetched.
# Usage: ./download_xenopus.sh [ROOT=data/xenopus]
set -euo pipefail
ROOT=${1:-data/xenopus}
[[ -e "$ROOT/.done_fig2" ]] && exit 0
mkdir -p "$ROOT"
curl -fL# -C - --retry 5 -o "$ROOT/fig2.zip" "https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/777/S-BSST2777/Files/Figure%202.zip"
unzip -ojq "$ROOT/fig2.zip" "Figure 2/2A/*.tif" "Figure 2/2C/*.tif" -d "$ROOT/fig2"
rm "$ROOT/fig2.zip" && touch "$ROOT/.done_fig2"
