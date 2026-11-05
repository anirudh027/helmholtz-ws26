#!/usr/bin/env bash
# Download + unzip the C2C12 dataset (https://osf.io/ysaq2/). Full image set is ~140 GB.
# Usage: ./download_c2c12.sh [ROOT=data] [REGEX=.]   (REGEX filters "<experiment>/<folder>")
#   ./download_c2c12.sh data 'F0001|Annotation'      # field F0001 of each experiment + annotations
set -euo pipefail
ROOT=${1:-data}; FILTER=${2:-.}
declare -A EXPS=([090303-C2C12P15-FGF2,BMP2]=acfj9 [090318-C2C12P7-FGF2,BMP2]=rd6h3 [090325-C2C12P12-FGF2,BMP2]=br6cw)

for exp in "${!EXPS[@]}"; do
  curl -fsSLg "https://api.osf.io/v2/nodes/${EXPS[$exp]}/children/?page[size]=100" |
    python3 -c 'import json,sys; [print(c["id"], c["attributes"]["title"]) for c in json.load(sys.stdin)["data"]]' |
    while read -r id title; do
      dest="$ROOT/$exp/$title"
      [[ "$exp/$title" =~ $FILTER && ! -e "$dest/.done" ]] || continue
      echo "Downloading $exp/$title"
      mkdir -p "$dest"
      curl -fL# --retry 5 -o "$dest.zip" "https://files.osf.io/v1/resources/$id/providers/osfstorage/?zip="
      unzip -qo "$dest.zip" -d "$dest" && rm "$dest.zip" && touch "$dest/.done"
    done
done
