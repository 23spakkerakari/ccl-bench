#!/usr/bin/env bash
# Submit the four Perlmutter settings as separate jobs (run from the repo root).
#   bash scripts/submit_all_gpu.sh <slurm-account> [extra sbatch args, e.g. -C "gpu&hbm80g"]
set -euo pipefail
ACCOUNT="${1:?usage: submit_all_gpu.sh <account> [sbatch args]}"; shift
for s in gpu-ep4-allgather gpu-ep4-alltoall; do
  sbatch -A "$ACCOUNT" -N 1 "$@" slurm/perlmutter_ep.sbatch "$s"
done
for s in gpu-ep8-allgather gpu-ep8-alltoall; do
  sbatch -A "$ACCOUNT" -N 2 "$@" slurm/perlmutter_ep.sbatch "$s"
done
