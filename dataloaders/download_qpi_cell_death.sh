#!/usr/bin/env bash
# Download + extract the QPI cell-death time-lapses (https://doi.org/10.5281/zenodo.2601562), ~64 GB (+ one zip).
# Usage: ./download_qpi_cell_death.sh [ROOT=data/qpi_cell_death] [CELL_LINES="LNCaP PNT1A DU145"]
set -euo pipefail
ROOT=${1:-data/qpi_cell_death}
read -ra LINES <<< "${2:-LNCaP PNT1A DU145}"
mkdir -p "$ROOT"
for LINE in "${LINES[@]}"; do
  [[ -e "$ROOT/.done_$LINE" ]] && continue
  ZIP="$ROOT/$LINE.zip"
  curl -fL# -C - --retry 5 -o "$ZIP" "https://zenodo.org/records/2601562/files/QPI_annotated_timelapse_$LINE.zip?download=1"
  unzip -oq "$ZIP" -d "$ROOT" && rm "$ZIP" && touch "$ROOT/.done_$LINE"
done
