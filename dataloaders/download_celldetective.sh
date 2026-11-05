#!/usr/bin/env bash
# Download + extract the Celldetective segmentation datasets (https://doi.org/10.5281/zenodo.17068651), ~1.5 GB.
# The record's demos, models and single-cell signal sets (db-si-*) are not downloaded.
# Usage: ./download_celldetective.sh [ROOT=data/celldetective]
#        [DBS="db_mcf7_nuclei_w_lymphocytes db_primary_NK_w_mcf7 db_spreading_lymphocytes"]
set -euo pipefail
ROOT=${1:-data/celldetective}
read -ra DBS <<< "${2:-db_mcf7_nuclei_w_lymphocytes db_primary_NK_w_mcf7 db_spreading_lymphocytes}"
mkdir -p "$ROOT"
for DB in "${DBS[@]}"; do
  [[ -e "$ROOT/.done_$DB" ]] && continue
  ZIP="$ROOT/$DB.zip"
  curl -fL# -C - --retry 5 -o "$ZIP" "https://zenodo.org/records/17068651/files/$DB.zip?download=1"
  unzip -oq "$ZIP" -d "$ROOT" && rm "$ZIP" && touch "$ROOT/.done_$DB"
done
