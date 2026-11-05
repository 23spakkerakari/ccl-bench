# shellcheck shell=bash
# =============================================================================
# Team 15 — activate the vLLM conda env created by setup_env.sh.
#
#   source activate_env.sh            # must be *sourced*, not executed
#
# Optional knobs (export before sourcing):
#   EPB_ONLINE=1        allow HF Hub network access (default: offline, since
#                       Perlmutter compute nodes can't download weights)
#   EPB_NCCL_DEBUG=1    NCCL_DEBUG=INFO — use it once on 2-node runs to confirm
#                       NCCL picked the Slingshot/OFI transport, not sockets
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "activate_env.sh must be sourced:  source ${0}" >&2
  exit 1
fi

_epb_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ ! -f "${_epb_dir}/.ep_bench_env" ]]; then
  echo "[activate] ${_epb_dir}/.ep_bench_env not found — run 'bash setup_env.sh' first." >&2
  return 1
fi
# shellcheck disable=SC1091
source "${_epb_dir}/.ep_bench_env"

# --- conda ------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  if [[ -n "${MODULESHOME:-}" ]] || type module >/dev/null 2>&1; then
    module load conda 2>/dev/null
  fi
fi
if ! command -v conda >/dev/null 2>&1 && [[ -n "${EPB_CONDA_ROOT_HINT:-}" ]]; then
  export PATH="${EPB_CONDA_ROOT_HINT}/bin:${PATH}"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[activate] conda not found (tried 'module load conda' and ${EPB_CONDA_ROOT_HINT:-Miniforge})." >&2
  return 1
fi
eval "$(conda shell.bash hook)"
conda activate "${EPB_ENV_PREFIX}" || { echo "[activate] could not activate ${EPB_ENV_PREFIX}" >&2; return 1; }

# --- project paths ----------------------------------------------------------
export EPB_ROOT="${_epb_dir}"
export EPB_PLATFORM EPB_DATA_ROOT EPB_DATA_DIR EPB_RESULTS_DIR EPB_CCL_BENCH_DIR
export HF_HOME="${EPB_HF_HOME}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${EPB_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${EPB_ONLINE:-0}" == "1" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

if [[ "${EPB_PLATFORM}" == "gpu" ]]; then
  # torch.compile / Triton caches on scratch instead of a tiny $HOME quota
  export TRITON_CACHE_DIR="${EPB_DATA_ROOT}/cache/triton"
  export VLLM_CACHE_ROOT="${EPB_DATA_ROOT}/cache/vllm"
  export TORCHINDUCTOR_CACHE_DIR="${EPB_DATA_ROOT}/cache/inductor"
  mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
  [[ "${EPB_NCCL_DEBUG:-0}" == "1" ]] && export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
else
  # XLA compilation cache survives server restarts -> much faster TPU warm starts
  export VLLM_XLA_CACHE_PATH="${EPB_DATA_ROOT}/cache/xla"
  mkdir -p "$VLLM_XLA_CACHE_PATH"
fi

echo "[activate] ${EPB_PLATFORM} env ${EPB_ENV_PREFIX} | HF_HOME=${HF_HOME} | results -> ${EPB_RESULTS_DIR}"
unset _epb_dir
