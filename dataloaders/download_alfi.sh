#!/usr/bin/env bash
# Download + extract the ALFI dataset (https://doi.org/10.6084/m9.figshare.23798451), one 8.4 GB zip holding the
# PNG frames + masks + CSV annotations ("Data&Annotations"), the raw ND2 videos and the authors' use examples.
# (springernature.figshare.com/ndownloader serves a bot challenge to curl; ndownloader.figshare.com does not.)
# Usage: ./download_alfi.sh [ROOT=data/alfi]
set -euo pipefail
ROOT=${1:-data/alfi}
[[ -e "$ROOT/.done" ]] && exit 0
mkdir -p "$ROOT"
curl -fL# -C - --retry 5 -o "$ROOT.zip" https://ndownloader.figshare.com/files/41740227
unzip -oq "$ROOT.zip" -d "$ROOT" && rm -r "$ROOT.zip" "$ROOT/__MACOSX" && touch "$ROOT/.done"
