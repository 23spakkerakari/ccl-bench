#!/usr/bin/env bash
# =============================================================================
# Grab one Perlmutter A100 node and get it ready for the DeepEP all2all backend.
#
# Put this in <repo>/slurm/ and run from the repo root (next to activate_env.sh).
#
#   ACCOUNT=m1234 bash slurm/perlmutter_deepep_node.sh                    # interactive shell
#   ACCOUNT=m1234 RUN_SETTING=gpu-ep4-alltoall \
#       bash slurm/perlmutter_deepep_node.sh                              # run the benchmark
#   ACCOUNT=m1234 bash slurm/perlmutter_deepep_node.sh -- python -c "import deep_ep"
#
# What it does:
#   login node : (optional) pre-clone DeepEP, since compute nodes may lack internet,
#                then `srun --pty` onto 1 node / 4 A100s and re-run itself there.
#   compute node: source activate_env.sh -> check GPUs are SM 8.x -> make sure a
#                pre-Hopper deep_ep (intranode only, no NVSHMEM) is installed in the
#                env's python -> verify -> shell / benchmark / your command.
#
# Limits (by construction of the pre-Hopper wheel):
#   * single node only (EP <= 4). Multi-node needs NVSHMEM + Hopper+ kernels.
#   * only the intranode / high-throughput DeepEP path; low-latency needs NVSHMEM.
#
# Env knobs:
#   ACCOUNT, QOS(interactive), CONSTRAINT(gpu; use "gpu&hbm80g" for 80GB), TIME(02:00:00)
#   DEEPEP_WHEEL=/path/to.whl     use this wheel instead of searching wheels/deepep/
#   DEEPEP_BUILD_DIR              source/build dir (default $SCRATCH/deepep_build)
#   DEEPEP_COMMIT                 default 7febc6e25660af0f54d95dd781ecdcd62265ecca
#   DEEPEP_PATCH=/path/deepep.patch  optional patch (NVIDIA recipe uses one from the
#                                 Automodel image: /opt/Automodel/docker/common/deepep.patch)
#   DEEPEP_UNINSTALL_NVSHMEM=1    pip-uninstall nvidia-nvshmem-* before a source build
#   FORCE_REINSTALL=1             reinstall even if the current deep_ep looks fine
# =============================================================================
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${SCRIPT}")/.." && pwd)"
WHEEL_DIR="${ROOT}/wheels/deepep"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-7febc6e25660af0f54d95dd781ecdcd62265ecca}"
BUILD_DIR="${DEEPEP_BUILD_DIR:-${SCRATCH:-/tmp}/deepep_build}"

log() { echo "[deepep-node $(date +%H:%M:%S)] $*" >&2; }
die() { echo "[deepep-node ERROR] $*" >&2; exit 1; }

# Split "-- cmd args" off the end.
USER_CMD=()
if [[ "${1:-}" == "--" ]]; then shift; USER_CMD=("$@"); fi

find_wheel() {
  if [[ -n "${DEEPEP_WHEEL:-}" ]]; then echo "${DEEPEP_WHEEL}"; return; fi
  ls -t "${WHEEL_DIR}"/deep_ep-*prehopper8089*.whl 2>/dev/null | head -1 || true
}

clone_source() {
  [[ -d "${BUILD_DIR}/DeepEP/.git" ]] && return 0
  log "cloning DeepEP (hybrid-ep) -> ${BUILD_DIR}/DeepEP"
  mkdir -p "${BUILD_DIR}"
  git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git "${BUILD_DIR}/DeepEP"
  (
    cd "${BUILD_DIR}/DeepEP"
    git fetch origin "${DEEPEP_COMMIT}"
    git checkout FETCH_HEAD
    if [[ -n "${DEEPEP_PATCH:-}" && -f "${DEEPEP_PATCH}" ]]; then
      log "applying patch ${DEEPEP_PATCH}"
      patch -p1 < "${DEEPEP_PATCH}"
    fi
  )
}

# -----------------------------------------------------------------------------
# Login-node mode: allocate and re-exec on the compute node
# -----------------------------------------------------------------------------
if [[ "${EPB_ON_NODE:-0}" != "1" ]] && ! { [[ -n "${SLURM_JOB_ID:-}" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
  ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
  [[ -n "${ACCOUNT}" ]] || die "set ACCOUNT=<your NERSC project> (e.g. ACCOUNT=m1234)"
  [[ "${ACCOUNT}" == *_g ]] || ACCOUNT="${ACCOUNT}_g"   # GPU allocations use the _g suffix

  if [[ -z "$(find_wheel)" ]]; then
    clone_source || log "warning: pre-clone failed; the compute-node build may not have internet"
  fi

  log "requesting 1 node / 4 A100 (account=${ACCOUNT} qos=${QOS:-interactive} constraint=${CONSTRAINT:-gpu})"
  exec srun -A "${ACCOUNT}" -C "${CONSTRAINT:-gpu}" -q "${QOS:-interactive}" \
    -N 1 --ntasks=1 --gpus-per-node=4 -c 128 --cpu-bind=none \
    -t "${TIME:-02:00:00}" --export=ALL,EPB_ON_NODE=1 \
    --pty bash "${SCRIPT}" ${USER_CMD:+--} "${USER_CMD[@]}"
fi

# -----------------------------------------------------------------------------
# Compute-node mode
# -----------------------------------------------------------------------------
cd "${ROOT}"
[[ -f ./activate_env.sh ]] || die "activate_env.sh not found in ${ROOT}; run from the repo root checkout"
# shellcheck disable=SC1091
source ./activate_env.sh
PYTHON="$(command -v python)"
log "python=${PYTHON}"

NGPU="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
SM="$("${PYTHON}" -c 'import torch; print(torch.cuda.get_device_capability(0)[0])')"
[[ "${NGPU}" == "4" ]] || die "expected 4 GPUs on this node, found ${NGPU}"
[[ "${SM}" -lt 9 ]] || die "GPU is SM ${SM}.x (Hopper+); use the post-Hopper wheel instead of this script"
log "node ok: ${NGPU} GPUs, SM ${SM}.x ($(hostname))"

deepep_ok() {
  # check_deepep_wheel.py needs `strings` (binutils) to read -arch sm_* from the .so.
  if [[ -f "${WHEEL_DIR}/check_deepep_wheel.py" ]] && command -v strings >/dev/null 2>&1; then
    "${PYTHON}" "${WHEEL_DIR}/check_deepep_wheel.py" >/dev/null 2>&1
  else
    "${PYTHON}" -c "import deep_ep" >/dev/null 2>&1
  fi
}

build_from_source() {
  command -v nvcc >/dev/null 2>&1 || { command -v module >/dev/null 2>&1 && module load cudatoolkit; }
  command -v nvcc >/dev/null 2>&1 || die "nvcc not found (module load cudatoolkit)"
  clone_source
  if "${PYTHON}" -c "import nvidia.nvshmem" >/dev/null 2>&1; then
    if [[ "${DEEPEP_UNINSTALL_NVSHMEM:-0}" == "1" ]]; then
      log "uninstalling NVSHMEM pip packages for the pre-Hopper build"
      "${PYTHON}" -m pip uninstall -y nvidia-nvshmem-cu12 nvidia-nvshmem-cu13 2>/dev/null || true
    else
      log "warning: NVSHMEM is installed; DeepEP's setup.py may refuse DISABLE_SM90_FEATURES with it."
      log "         If the build fails on that, re-run with DEEPEP_UNINSTALL_NVSHMEM=1."
    fi
  fi
  mkdir -p "${BUILD_DIR}/out"
  log "building pre-Hopper deep_ep for SM 8.0 (10-30+ min)"
  (
    cd "${BUILD_DIR}/DeepEP"
    export TORCH_CUDA_ARCH_LIST="8.0"
    export DISABLE_SM90_FEATURES=1 DISABLE_AGGRESSIVE_PTX_INSTRS=1 HYBRID_EP_MULTINODE=0
    export MAX_JOBS="${MAX_JOBS:-16}"
    unset RDMA_CORE_HOME || true
    "${PYTHON}" -m pip wheel . --no-build-isolation --no-deps -w "${BUILD_DIR}/out"
  )
  local built
  built="$(ls -t "${BUILD_DIR}"/out/deep_ep-*.whl 2>/dev/null | head -1 || true)"
  [[ -n "${built}" ]] || die "build produced no deep_ep wheel"
  "${PYTHON}" -m pip install --force-reinstall --no-deps "${built}"
}

install_deepep() {
  local wheel
  wheel="$(find_wheel)"
  if [[ -n "${wheel}" ]]; then
    log "installing wheel ${wheel}"
    if [[ -f "${WHEEL_DIR}/install_deepep_wheel.sh" ]]; then
      PYTHON="${PYTHON}" DEEPEP_WHEEL="${wheel}" DEEPEP_WHEEL_PROFILE=pre-hopper-sm80-sm89 \
        DEEPEP_FORCE_INSTALL=1 bash "${WHEEL_DIR}/install_deepep_wheel.sh"
    else
      "${PYTHON}" -m pip install --force-reinstall --no-deps "${wheel}"
    fi
  else
    log "no pre-Hopper wheel in ${WHEEL_DIR}; building from source"
    build_from_source
  fi
}

if [[ "${FORCE_REINSTALL:-0}" == "1" ]] || ! deepep_ok; then
  install_deepep
fi
deepep_ok || die "deep_ep failed verification after install (torch ABI mismatch is the usual cause: rebuild inside this env)"
"${PYTHON}" -c "import deep_ep; print('deep_ep import OK:', deep_ep.__file__)"

log "vLLM all2all backend choices in this build:"
vllm serve --help=all 2>/dev/null | grep -i -A8 'all2all' | head -20 || log "(could not read vllm help)"

cat >&2 <<'EOF'

Next: set ep_impl.gpu.alltoall.all2all_backend in configs/experiments.yaml to the DeepEP
high-throughput value listed above (low-latency needs NVSHMEM and will not work here).
After a run, confirm DeepEP kernels appear in the trace; vLLM silently falls back to
allgather if the backend is unavailable.
EOF

# -----------------------------------------------------------------------------
# Hand off
# -----------------------------------------------------------------------------
if [[ ${#USER_CMD[@]} -gt 0 ]]; then
  exec "${USER_CMD[@]}"
elif [[ -n "${RUN_SETTING:-}" ]]; then
  [[ "${RUN_SETTING}" == gpu-ep4-* ]] || die "RUN_SETTING must be a single-node setting (gpu-ep4-*); got ${RUN_SETTING}"
  exec "${PYTHON}" bench/run_benchmark.py --setting "${RUN_SETTING}" --launch-server
else
  log "dropping into an interactive shell (env is active)"
  exec bash -i
fi