#!/usr/bin/env bash
# Run the TPU settings back to back on a TPU VM (each starts/stops its own server).
#   bash scripts/run_all_tpu.sh            # all four (8-chip settings need a v6e-8)
#   bash scripts/run_all_tpu.sh tpu-ep4-allgather tpu-ep4-alltoall
set -eo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ./activate_env.sh
SETTINGS=("$@")
(( ${#SETTINGS[@]} )) || SETTINGS=(tpu-ep4-allgather tpu-ep4-alltoall tpu-ep8-allgather tpu-ep8-alltoall)
for s in "${SETTINGS[@]}"; do
  python bench/run_benchmark.py --setting "$s" --launch-server
done
