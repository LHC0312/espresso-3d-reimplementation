#!/bin/bash
# Download the datasets used here into ${DATA_DIR:-data}:
#   - Affogato-750K : huggingface.co/datasets/project-affogato/affogato (16 tar.gz parts, ~30 GB compressed,
#                     ~39 GB extracted, 150,101 objects). Streamed and extracted in parallel.
#   - LASO          : official Google Drive release of LASO (CVPR 2024), ~250 MB.
# Usage: bash scripts/download_data.sh [affogato|laso|all]
set -uo pipefail
DATA_DIR=${DATA_DIR:-data}
WHAT=${1:-all}
mkdir -p "$DATA_DIR/affogato"

if [ "$WHAT" = all ] || [ "$WHAT" = laso ]; then
  pip -q install gdown >/dev/null 2>&1 || true
  [ -s "$DATA_DIR/LASO_dataset.tar.gz" ] || gdown -q 1P4CwQeSALtUOgzhg1ILCiKovE-tcXlSu -O "$DATA_DIR/LASO_dataset.tar.gz"
  [ -d "$DATA_DIR/LASO_dataset" ] || tar xzf "$DATA_DIR/LASO_dataset.tar.gz" -C "$DATA_DIR"   # the ".zip" is a tar.gz
  echo "[laso] $(ls "$DATA_DIR/LASO_dataset")"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = affogato ]; then
  BASE=https://huggingface.co/datasets/project-affogato/affogato/resolve/main
  fetch() {
    p=$1
    [ -f "$DATA_DIR/affogato/.done_$p" ] && return 0
    for t in 1 2 3 4 5 6; do
      if curl -sSL --retry 10 --retry-all-errors "$BASE/affogato_all_part$p.tar.gz" | tar xz -C "$DATA_DIR/affogato"; then
        touch "$DATA_DIR/affogato/.done_$p"; echo "[affogato] part $p ok"; return 0
      fi
      echo "[affogato] part $p retry $t"; sleep 5
    done
    echo "[affogato] part $p FAILED"; return 1
  }
  export -f fetch; export BASE DATA_DIR
  seq -f "%03g" 0 15 | xargs -P 8 -I{} bash -c 'fetch {}'
  # object index (directories only; each holds xyzc.npy (16384 x 8 float16: xyz + 5 heatmaps) + queries.json)
  ls "$DATA_DIR/affogato" | grep -v -e '^\.' -e '\.txt$' > /tmp/affogato_index.txt && mv /tmp/affogato_index.txt "$DATA_DIR/affogato/index.txt"
  echo "[affogato] objects: $(wc -l < "$DATA_DIR/affogato/index.txt")"
fi
