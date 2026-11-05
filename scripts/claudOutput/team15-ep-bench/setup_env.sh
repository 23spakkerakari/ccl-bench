#!/usr/bin/env bash
# =============================================================================
# Team 15 — Expert-parallel inference on GPUs vs TPUs
# One-time environment setup: creates a conda env that can run vLLM.
#
#   Perlmutter (run on a LOGIN node — compute nodes cannot download weights):
#       bash setup_env.sh            # auto-detects GPU
#   Cloud TPU VM (v6e):
#       bash setup_env.sh            # auto-detects TPU
#
#   Options:
#       bash setup_env.sh gpu|tpu    # force a platform
#       --force                      # delete and recreate an existing env
#       --skip-download              # don't download model weights / ShareGPT
#
#   Overridable environment variables (defaults in brackets):
#       EPB_ENV_PREFIX     conda env location   [$PSCRATCH/envs/ep-bench or ~/envs/ep-bench]
#       EPB_DATA_ROOT      HF cache + datasets  [$PSCRATCH/ep-bench or ~/ep-bench]
#       EPB_VLLM_VERSION   vLLM for GPU         [0.32.0]
#       EPB_VLLM_TPU_VERSION vllm-tpu for TPU   [0.32.0]   (kept equal to GPU on purpose)
#       EPB_CUDA_VARIANT   cu129|cu130|...      [auto from driver]
#       EPB_PYTHON         python version       [3.12]
#
# Afterwards, in every new shell:   source activate_env.sh
# =============================================================================
set -eo pipefail   # (no -u: conda activate scripts reference unset vars)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
STATE_FILE="${SCRIPT_DIR}/.ep_bench_env"

PLATFORM="auto"
FORCE=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    gpu|tpu) PLATFORM="$arg" ;;
    --force) FORCE=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    -h|--help) sed -n 2,26p "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] WARNING:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. Detect platform
# ----------------------------------------------------------------------------
if [[ "$PLATFORM" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PLATFORM="gpu"
  elif compgen -G "/dev/vfio/[0-9]*" >/dev/null || compgen -G "/dev/accel*" >/dev/null; then
    PLATFORM="tpu"
  else
    die "Could not detect a GPU (nvidia-smi) or TPU (/dev/vfio, /dev/accel*). Pass 'gpu' or 'tpu' explicitly."
  fi
fi
log "Platform: ${PLATFORM}"

PY_VER="${EPB_PYTHON:-3.12}"
VLLM_VERSION="${EPB_VLLM_VERSION:-0.32.0}"
VLLM_TPU_VERSION="${EPB_VLLM_TPU_VERSION:-0.32.0}"

if [[ -n "${PSCRATCH:-}" ]]; then BASE="${PSCRATCH}"; else BASE="${HOME}"; fi
ENV_PREFIX="${EPB_ENV_PREFIX:-${BASE}/envs/ep-bench}"
DATA_ROOT="${EPB_DATA_ROOT:-${BASE}/ep-bench}"
HF_HOME_DIR="${DATA_ROOT}/huggingface"
DATA_DIR="${DATA_ROOT}/data"
RESULTS_DIR="${DATA_ROOT}/results"
CCL_BENCH_DIR="${EPB_CCL_BENCH_DIR:-${SCRIPT_DIR}/third_party/ccl-bench}"
mkdir -p "$HF_HOME_DIR" "$DATA_DIR" "$RESULTS_DIR" "$(dirname "$ENV_PREFIX")"

# ----------------------------------------------------------------------------
# 2. Get conda
# ----------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1 || [[ -n "${MODULESHOME:-}" ]]; then
    # Perlmutter / HPC: conda comes from the module system
    # shellcheck disable=SC1091
    [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]] && source "${MODULESHOME}/init/bash"
    module load conda 2>/dev/null || true
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  # Cloud TPU VMs ship without conda -> install Miniforge into $HOME
  MF_DIR="${HOME}/miniforge3"
  if [[ ! -x "${MF_DIR}/bin/conda" ]]; then
    log "conda not found; installing Miniforge to ${MF_DIR}"
    installer="/tmp/miniforge-$$.sh"
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o "$installer"
    bash "$installer" -b -p "$MF_DIR"
    rm -f "$installer"
  fi
  export PATH="${MF_DIR}/bin:${PATH}"
  CONDA_ROOT_HINT="$MF_DIR"
fi
eval "$(conda shell.bash hook)"

# ----------------------------------------------------------------------------
# 3. Create the env (pip-only torch: conda's torch statically links NCCL,
#    which breaks vLLM — see vLLM install docs / issue #8420)
# ----------------------------------------------------------------------------
if [[ -d "$ENV_PREFIX" && "$FORCE" == 1 ]]; then
  log "Removing existing env ${ENV_PREFIX} (--force)"
  conda env remove -y -p "$ENV_PREFIX" >/dev/null || rm -rf "$ENV_PREFIX"
fi
if [[ ! -d "$ENV_PREFIX" ]]; then
  log "Creating conda env at ${ENV_PREFIX} (python ${PY_VER})"
  conda create -y -p "$ENV_PREFIX" "python=${PY_VER}" pip
else
  log "Reusing existing env at ${ENV_PREFIX} (use --force to recreate)"
fi
conda activate "$ENV_PREFIX"

# Ignore packages in ~/.local and point HF at scratch (same trick as HW2).
conda env config vars set -p "$ENV_PREFIX" PYTHONNOUSERSITE=1 HF_HOME="$HF_HOME_DIR" >/dev/null
conda deactivate && conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip wheel setuptools

# ----------------------------------------------------------------------------
# 4. Install vLLM
# ----------------------------------------------------------------------------
if [[ "$PLATFORM" == "gpu" ]]; then
  # PyPI torch for vLLM >= 0.2x is built for CUDA 13, which needs driver >= 580.
  # Older drivers get the +cu129 vLLM wheel and cu129 torch instead.
  CUDA_VARIANT="${EPB_CUDA_VARIANT:-}"
  if [[ -z "$CUDA_VARIANT" ]]; then
    DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' || true)"
    if [[ -n "$DRV_CUDA" && "${DRV_CUDA%%.*}" -ge 13 ]]; then CUDA_VARIANT="cu130"; else CUDA_VARIANT="cu129"; fi
    log "Driver supports CUDA ${DRV_CUDA:-unknown} -> using ${CUDA_VARIANT} build"
  fi

  if [[ "$CUDA_VARIANT" == "cu130" ]]; then
    python -m pip install "vllm==${VLLM_VERSION}"
  else
    CUDA_NUM="${CUDA_VARIANT#cu}"
    WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_VARIANT}-cp38-abi3-manylinux_2_28_$(uname -m).whl"
    log "Installing ${WHEEL}"
    python -m pip install "$WHEEL" --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_NUM}" \
      || die "Could not install vLLM ${VLLM_VERSION} for ${CUDA_VARIANT}. Check the release page for available +cuXXX wheels, then re-run with EPB_CUDA_VARIANT=<variant>."
  fi
  # FlashInfer precompiled kernels (avoids JIT at server start); non-fatal.
  vllm download-kernels || warn "'vllm download-kernels' failed — vLLM will JIT FlashInfer kernels at startup instead."
  python -m pip install nvtx
else
  python -m pip install "vllm-tpu==${VLLM_TPU_VERSION}"
  # xprof converts the JAX profiler's xplane.pb into Chrome-trace JSON for CCL-Bench.
  python -m pip install xprof || warn "xprof failed to install; TPU traces will stay as xplane.pb until converted manually."
fi

# Client / analysis deps (CCL-Bench requirements minus the agent SDK)
python -m pip install pandas numpy matplotlib duckdb PyYAML requests aiohttp "huggingface_hub[hf_xet]" prometheus_client

# ----------------------------------------------------------------------------
# 5. CCL-Bench checkout (metric tools + workload-card template)
# ----------------------------------------------------------------------------
if [[ ! -d "${CCL_BENCH_DIR}/.git" ]]; then
  log "Cloning CCL-Bench into ${CCL_BENCH_DIR}"
  mkdir -p "$(dirname "$CCL_BENCH_DIR")"
  git clone https://github.com/cornell-sysphotonics/ccl-bench.git "$CCL_BENCH_DIR"
else
  log "CCL-Bench already present at ${CCL_BENCH_DIR}"
fi

# ----------------------------------------------------------------------------
# 6. Remember where everything lives (read by activate_env.sh)
# ----------------------------------------------------------------------------
cat > "$STATE_FILE" <<EOF
# Written by setup_env.sh on $(date -Iseconds). Sourced by activate_env.sh.
EPB_PLATFORM="${PLATFORM}"
EPB_ENV_PREFIX="${ENV_PREFIX}"
EPB_DATA_ROOT="${DATA_ROOT}"
EPB_HF_HOME="${HF_HOME_DIR}"
EPB_DATA_DIR="${DATA_DIR}"
EPB_RESULTS_DIR="${RESULTS_DIR}"
EPB_CCL_BENCH_DIR="${CCL_BENCH_DIR}"
EPB_CONDA_ROOT_HINT="${CONDA_ROOT_HINT:-}"
EOF
log "Wrote ${STATE_FILE}"

# ----------------------------------------------------------------------------
# 7. Sanity check
# ----------------------------------------------------------------------------
if [[ "$PLATFORM" == "gpu" ]]; then
  python - <<'PY'
import torch, vllm
print(f"vllm {vllm.__version__} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"cuda available={torch.cuda.is_available()} | devices={torch.cuda.device_count()}")
PY
else
  python - <<'PY'
import importlib.metadata as md, jax, vllm
print(f"vllm {vllm.__version__} | tpu_inference {md.version('tpu_inference')} | jax devices: {jax.devices()}")
PY
fi

# ----------------------------------------------------------------------------
# 8. Model weights + ShareGPT (login node has internet; compute nodes don't)
# ----------------------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" == 0 ]]; then
  log "Downloading Qwen/Qwen3-30B-A3B (~61 GB) and ShareGPT V3 into ${DATA_ROOT}"
  HF_HOME="$HF_HOME_DIR" python "${SCRIPT_DIR}/bench/prepare_data.py" --data-dir "$DATA_DIR"
fi

log "Done. In every new shell run:   source ${SCRIPT_DIR}/activate_env.sh"
