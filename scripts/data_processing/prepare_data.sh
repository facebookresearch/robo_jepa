#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Stage a dataset's tar packs onto the node-local SSD and put dataset_index.csv at its root.
#
#   DATA_ROOT=/data bash scripts/data_processing/prepare_data.sh <lerobot|droid|agibot>
#
# Runs once per node, e.g. from a config's `slurm_setup`:
#   srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash scripts/data_processing/prepare_data.sh droid
# $DATA_ROOT/<name>/ holds the packs (droid_NN.tar, part_NN.tar, agibot_NN.tar) and the dataset_index.csv written
# by the converter. Packs extract to $SLURM_TMPDIR/<name>/ (point the config's data.s3_prefix there). Each pack
# leaves a marker when it has extracted fully, so a requeued job only redoes the missing ones.
set -euo pipefail

NAME="${1:?usage: prepare_data.sh <lerobot|droid|agibot>}"
: "${DATA_ROOT:?set DATA_ROOT to the directory holding <name>/ with the tar packs}"
: "${SLURM_TMPDIR:?run inside a Slurm job (or set SLURM_TMPDIR)}"

case "$NAME" in
    lerobot) PACKS=("$DATA_ROOT/lerobot"/part_*.tar) ;;
    droid) PACKS=("$DATA_ROOT/droid"/droid_*.tar) ;;
    agibot) PACKS=("$DATA_ROOT/agibot"/agibot_*.tar) ;;
    *) echo "unknown dataset: $NAME" >&2; exit 1 ;;
esac
INDEX="$DATA_ROOT/$NAME/dataset_index.csv"
[[ -f "$INDEX" ]] || { echo "[prepare_data] missing $INDEX" >&2; exit 1; }

DEST="$SLURM_TMPDIR/$NAME"
mkdir -p "$DEST"
echo "[prepare_data] $(date +%T) $(hostname): ${#PACKS[@]} packs of $NAME -> $DEST"

pids=()
for pack in "${PACKS[@]}"; do
    marker="$DEST/.extracted_$(basename "$pack" .tar)"
    [[ -e "$marker" ]] && continue
    ( tar -xf "$pack" -C "$DEST" && touch "$marker" ) &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then
    echo "[prepare_data] extraction failed" >&2
    exit 1
fi

cp "$INDEX" "$DEST/dataset_index.csv"
echo "[prepare_data] $(date +%T) done: $(($(wc -l < "$DEST/dataset_index.csv") - 1)) episodes, $(du -sh "$DEST" | cut -f1)"
