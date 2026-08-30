#!/bin/bash
# SLURM array over the training grid. One array index == one (dataset, model, arm, seed)
# cell, resolved by `tfmdm cells --index`.
#
# The stage skips any cell whose prediction file already exists, so a killed or
# partially-failed array can simply be resubmitted: it reruns only the gaps.
#
#SBATCH --job-name=tfmdm-sweep
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Compute nodes are assumed to have no outbound network; runs are pushed later
# with `task wandb:sync` from a login node.
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONHASHSEED=0

CELL=$(uv run tfmdm cells --index "${SLURM_ARRAY_TASK_ID}")
echo "Cell: ${CELL}"

DATASET=$(echo "${CELL}" | python -c 'import json,sys; print(json.load(sys.stdin)["dataset"])')
MODEL=$(echo "${CELL}"   | python -c 'import json,sys; print(json.load(sys.stdin)["model"])')
ARM=$(echo "${CELL}"     | python -c 'import json,sys; print(json.load(sys.stdin)["arm"])')
SEED=$(echo "${CELL}"    | python -c 'import json,sys; print(json.load(sys.stdin)["seed"])')

uv run tfmdm train --dataset "${DATASET}" --model "${MODEL}" --arm "${ARM}" --seed "${SEED}"
