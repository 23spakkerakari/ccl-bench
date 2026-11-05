#!/bin/bash
# =============================================================================
# Perlmutter batch job for one GPU setting (server on every node + client on node 0).
#
#   4 GPUs / 1 node : sbatch -A <account> -N 1 slurm/perlmutter_ep.sbatch gpu-ep4-allgather
#   8 GPUs / 2 nodes: sbatch -A <account> -N 2 slurm/perlmutter_ep.sbatch gpu-ep8-allgather
#   80 GB A100s     : add  -C "gpu&hbm80g"
#
# Extra args after the setting are passed to run_benchmark.py, e.g. --concurrency 1 16 64
# Submit from the repo root (paths below are relative to $SLURM_SUBMIT_DIR).
# =============================================================================
#SBATCH -J ep-bench
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -o slurm-%x-%j.out

set -eo pipefail
SETTING="${1:?usage: sbatch ... perlmutter_ep.sbatch <setting> [run_benchmark args]}"; shift
cd "${SLURM_SUBMIT_DIR}"
# shellcheck disable=SC1091
source ./activate_env.sh

RUN_ID="${EPB_RUN_ID:-$(date +%Y%m%d-%H%M%S)-j${SLURM_JOB_ID}}"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD="${NODES[0]}"
HEAD_IP="$(getent ahostsv4 "${HEAD}" | awk 'NR==1{print $1}')"
echo "[sbatch] setting=${SETTING} run_id=${RUN_ID} nodes=${NODES[*]} head=${HEAD} (${HEAD_IP})"

if (( ${#NODES[@]} > 1 )); then
  # Keep vLLM's DP control plane and NCCL bootstrap on the Slingshot HSN interfaces.
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-hsn0}"
fi

SERVER_PIDS=()
for i in "${!NODES[@]}"; do
  srun --overlap -N1 -n1 -w "${NODES[$i]}" -c 120 --cpu-bind=none \
    python bench/launch_server.py --setting "${SETTING}" --run-id "${RUN_ID}" \
      --node-rank "${i}" --head-addr "${HEAD_IP}" &
  SERVER_PIDS+=($!)
done
cleanup() { for p in "${SERVER_PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT

srun --overlap -N1 -n1 -w "${HEAD}" -c 8 --cpu-bind=none \
  python bench/run_benchmark.py --setting "${SETTING}" --run-id "${RUN_ID}" --host localhost "$@"
