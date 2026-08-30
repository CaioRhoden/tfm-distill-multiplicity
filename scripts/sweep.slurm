#!/bin/bash
# SLURM array over the training grid. One array index == one GROUP, and a group is
# every run seed of one (dataset, model, arm, split_seed).
#
# Groups rather than cells because the expensive part of a cell is not the fit -- it is
# loading the feature view, the split and the soft labels, which all 30 run seeds
# share. One job pays that once instead of thirty times. Set CHUNK to split a group's
# seeds across several jobs when the queue has room for more parallelism.
#
# The stage skips any run seed whose prediction file already exists, so a killed or
# partially-failed array can simply be resubmitted: it resumes at the missing seeds.
#
#SBATCH --job-name=tfmdm-sweep
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Compute nodes are assumed to have no outbound network; runs are pushed later
# with `task wandb:sync` from a login node.
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONHASHSEED=0

# The grid definition travels from `task slurm` through these, so the submitting
# shell and the worker enumerate exactly the same groups in the same order.
DATASETS="${TFMDM_DATASETS:-adult taiwan}"
MODELS="${TFMDM_MODELS:-ebm nam}"
ARMS="${TFMDM_ARMS:-hard distilled}"
SPLIT_SEEDS="${TFMDM_SPLIT_SEEDS:-0}"
CHUNK_ARG=""
if [[ -n "${TFMDM_CHUNK:-}" ]]; then
  CHUNK_ARG="--chunk ${TFMDM_CHUNK}"
fi

# shellcheck disable=SC2086
GROUP=$(uv run tfmdm groups \
  --datasets ${DATASETS} --models ${MODELS} --arms ${ARMS} \
  --split-seeds ${SPLIT_SEEDS} ${CHUNK_ARG} \
  --index "${SLURM_ARRAY_TASK_ID}")

echo "Group: ${GROUP}"

read -r DATASET MODEL ARM SPLIT_SEED SEEDS <<<"$(python - "${GROUP}" <<'PY'
import json, sys
g = json.loads(sys.argv[1])
print(g["dataset"], g["model"], g["arm"], g["split_seed"], " ".join(str(s) for s in g["seeds"]))
PY
)"

# shellcheck disable=SC2086
uv run tfmdm train-group \
  --dataset "${DATASET}" --model "${MODEL}" --arm "${ARM}" \
  --split-seed "${SPLIT_SEED}" --seeds ${SEEDS}
