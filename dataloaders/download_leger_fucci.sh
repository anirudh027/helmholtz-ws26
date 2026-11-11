#!/usr/bin/env bash
# Download + extract the Leger et al. RPE1 Fucci tracks (https://zenodo.org/records/14987478): test (1.2 GB) and
# test_drugged (0.46 GB) by default, train (16.6 GB) on request. Zenodo often closes the connection early without an
# error, so each zip is resumed until its MD5 matches.
# Usage: ./download_leger_fucci.sh [ROOT=data/leger] [SPLITS="test test_drugged"]   (SPLITS="train test test_drugged")
set -euo pipefail
ROOT=${1:-data/leger}
read -ra SPLITS <<< "${2:-test test_drugged}"
md5of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
mkdir -p "$ROOT"
for SPLIT in "${SPLITS[@]}"; do
  [[ -e "$ROOT/.done_$SPLIT" ]] && continue
  case $SPLIT in  # no associative arrays: macOS ships bash 3.2
    train) MD5=c9f44b10794bb796644838c27a35d820 ;;
    test) MD5=734352f2df732e3efce1b967110d506b ;;
    test_drugged) MD5=b64128abbdb155fadf362b68b2e0da5d ;;
    *) echo "unknown split $SPLIT" >&2; exit 1 ;;
  esac
  ZIP="$ROOT/$SPLIT.zip"
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [[ -e "$ZIP" && "$(md5of "$ZIP")" == "$MD5" ]] && break
    curl -fL# -C - --retry 5 -o "$ZIP" "https://zenodo.org/records/14987478/files/$SPLIT.zip?download=1" || true
  done
  [[ "$(md5of "$ZIP")" == "$MD5" ]] || { echo "MD5 mismatch for $ZIP" >&2; exit 1; }
  unzip -oq "$ZIP" -d "$ROOT" -x '__MACOSX/*' '*.DS_Store' && rm "$ZIP" && touch "$ROOT/.done_$SPLIT"
done
