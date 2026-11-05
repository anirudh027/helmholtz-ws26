#!/usr/bin/env bash
# Download the p53 cell-fate tracking Figure 2 data (https://doi.org/10.5061/dryad.pk0p2ngp5): per condition one
# single-cell-tracking video (.mov) + its cell-lineage table (.xlsx), ~330 MB for all 8 (dataset total 451 MB).
# Dryad file downloads need an API token (https://datadryad.org/account, valid 10 h); anonymous requests get a
# bot challenge. Usage: DRYAD_TOKEN=... ./download_p53_fate.sh [ROOT=data/p53_fate] [CONDITIONS="Control MNNG1 ..."]
set -euo pipefail
ROOT=${1:-data/p53_fate}
read -ra CONDS <<< "${2:-Control MNNG1 MNNG3 MNNG7 p53_RNAi p53_RNAi-MNNG1 p53_RNAi-MNNG3 p53_RNAi-MNNG7}"
: "${DRYAD_TOKEN:?set DRYAD_TOKEN (Dryad API token from https://datadryad.org/account)}"
# Dryad file ids per condition: "<video id> <lineage table id>"
declare -A IDS=([Control]="1819661 1819668" [MNNG1]="1819662 1819670" [MNNG3]="1819664 1819672"
  [MNNG7]="1819663 1819673" [p53_RNAi]="1819665 1819674" [p53_RNAi-MNNG1]="1819666 1819675"
  [p53_RNAi-MNNG3]="1819667 1819676" [p53_RNAi-MNNG7]="1819669 1819682")
mkdir -p "$ROOT"
for COND in "${CONDS[@]}"; do
  [[ -e "$ROOT/.done_$COND" ]] && continue
  read -r VID TAB <<< "${IDS[$COND]}"
  for X in "$VID mov" "$TAB xlsx"; do
    read -r ID EXT <<< "$X"
    curl -fL# -C - --retry 5 -H "Authorization: Bearer $DRYAD_TOKEN" -o "$ROOT/$COND.$EXT" \
      "https://datadryad.org/api/v2/files/$ID/download"
  done
  touch "$ROOT/.done_$COND"
done
