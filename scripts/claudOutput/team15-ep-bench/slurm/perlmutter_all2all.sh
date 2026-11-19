#!/usr/bin/env bash
# =============================================================================
# Grab one Perlmutter A100 node and get it ready for vLLM's DeepEP all2all
# backend (--all2all-backend deepep_high_throughput).
#
# Lives in <repo>/slurm/; run it from the repo root (next to activate_env.sh).
#
#   ACCOUNT=m1234 bash slurm/perlmutter_all2all.sh                        # interactive shell
#   ACCOUNT=m1234 RUN_SETTING=gpu-ep4-alltoall \
#       bash slurm/perlmutter_all2all.sh                                  # run the benchmark
#   ACCOUNT=m1234 bash slurm/perlmutter_all2all.sh -- python -c "import deep_ep"
#
# What it does:
#   login node  : pre-clone DeepEP (compute nodes may lack internet), then
#                 `srun --pty` onto 1 node / 4 A100s and re-run itself there.
#   compute node: source activate_env.sh -> check GPUs are SM 8.x -> (if
#                 RUN_SETTING is set) check the config can actually run ->
#                 make sure an A100 build of deep_ep (intranode only, no NVSHMEM)
#                 is installed -> 4-GPU dispatch/combine smoke test ->
#                 shell / benchmark / your command.
#
# Why this DeepEP commit: vLLM 0.32 pins DeepEP main, whose setup.py no longer
# builds for A100 (DISABLE_SM90_FEATURES=1 hits `assert False, 'Not implemented'`).
# The pinned hybrid-ep commit below still has the A100 intranode path, and its
# deep_ep.Buffer API matches what vLLM 0.32's DeepEP high-throughput code calls.
# The build patches that checkout: NVSHMEM forced off (torch's pip deps install it),
# and the Hopper-only hybrid-EP extension is skipped. It builds with the first
# GCC >= 10 it finds (torch needs C++20; Perlmutter's plain `c++` is GCC 7).
#
# Limits (by construction of the A100 build):
#   * single node only (EP <= 4). Multi-node needs NVSHMEM + Hopper+ kernels.
#   * only the high-throughput DeepEP path; low-latency needs NVSHMEM.
#   * vLLM turns CUDA graphs OFF for deepep_high_throughput with DP > 1. Set
#     serving.enforce_eager: true in configs/experiments.yaml for the allgather
#     runs too, or the two backends are not compared like-for-like.
#
# Env knobs:
#   ACCOUNT, QOS(interactive), CONSTRAINT(gpu; use "gpu&hbm80g" for 80GB), TIME(02:00:00)
#   DEEPEP_WHEEL=/path/to.whl     install this wheel instead of the cached/locally built one
#   DEEPEP_BUILD_DIR              source/build dir (default $SCRATCH/deepep_build);
#                                 built wheels are cached in $DEEPEP_BUILD_DIR/out
#   DEEPEP_COMMIT                 default 7febc6e25660af0f54d95dd781ecdcd62265ecca
#   DEEPEP_PATCH=/path/deepep.patch  optional extra patch applied before building
#   FORCE_REINSTALL=1             rebuild/reinstall even if the current deep_ep looks fine
#   DEEPEP_SMOKE_TEST=0           skip the 4-GPU dispatch/combine test (~30 s)
# =============================================================================
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${SCRIPT}")/.." && pwd)"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-7febc6e25660af0f54d95dd781ecdcd62265ecca}"
BUILD_DIR="${DEEPEP_BUILD_DIR:-${SCRATCH:-/tmp}/deepep_build}"
SRC_DIR="${BUILD_DIR}/DeepEP"
OUT_DIR="${BUILD_DIR}/out"

log() { echo "[deepep-node $(date +%H:%M:%S)] $*" >&2; }
die() { echo "[deepep-node ERROR] $*" >&2; exit 1; }

# Split "-- cmd args" off the end.
USER_CMD=()
if [[ "${1:-}" == "--" ]]; then shift; USER_CMD=("$@"); fi

# An explicit DEEPEP_WHEEL wins; otherwise the newest wheel this script built before.
find_wheel() {
  if [[ -n "${DEEPEP_WHEEL:-}" ]]; then
    [[ -f "${DEEPEP_WHEEL}" ]] || die "DEEPEP_WHEEL=${DEEPEP_WHEEL} does not exist"
    echo "${DEEPEP_WHEEL}"; return
  fi
  ls -t "${OUT_DIR}"/deep_ep-*.whl 2>/dev/null | head -1 || true
}

# Network step: make sure the repo and the pinned commit are present locally.
clone_source() {
  if [[ ! -d "${SRC_DIR}/.git" ]]; then
    log "cloning DeepEP (hybrid-ep) -> ${SRC_DIR}"
    mkdir -p "${BUILD_DIR}"
    git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git "${SRC_DIR}"
  fi
  if ! git -C "${SRC_DIR}" cat-file -e "${DEEPEP_COMMIT}^{commit}" 2>/dev/null; then
    log "fetching DeepEP commit ${DEEPEP_COMMIT}"
    git -C "${SRC_DIR}" fetch origin "${DEEPEP_COMMIT}"
  fi
}

# -----------------------------------------------------------------------------
# Login-node mode: allocate and re-exec on the compute node
# -----------------------------------------------------------------------------
if [[ "${EPB_ON_NODE:-0}" != "1" ]] && ! { [[ -n "${SLURM_JOB_ID:-}" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
  ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
  [[ -n "${ACCOUNT}" ]] || die "set ACCOUNT=<your NERSC project> (e.g. ACCOUNT=m1234)"
  [[ "${ACCOUNT}" == *_g ]] || ACCOUNT="${ACCOUNT}_g"   # GPU allocations use the _g suffix
  [[ -z "${DEEPEP_WHEEL:-}" || -f "${DEEPEP_WHEEL}" ]] || die "DEEPEP_WHEEL=${DEEPEP_WHEEL} does not exist"

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
[[ "${SM}" -lt 9 ]] || die "GPU is SM ${SM}.x (Hopper+); use vLLM's own DeepEP install (tools/ep_kernels) instead of this script"
log "node ok: ${NGPU} GPUs, SM ${SM}.x ($(hostname))"

# -----------------------------------------------------------------------------
# Fail fast (before a 10-30 min build) if RUN_SETTING can't run as configured.
# Prints: "<ep_impl> <backend> <enforce_eager>"
# -----------------------------------------------------------------------------
check_run_setting() {
  "${PYTHON}" - "$1" <<'PY'
import sys, yaml
name = sys.argv[1]
cfg = yaml.safe_load(open("configs/experiments.yaml"))
s = (cfg.get("settings") or {}).get(name)
if s is None:
    sys.exit(f"unknown setting '{name}'; choose one of: {', '.join(cfg.get('settings', {}))}")
if s.get("platform") != "gpu" or int(s.get("nodes", 1)) != 1:
    sys.exit(f"'{name}' is not a single-node GPU setting; this script only supports 1 node / EP4")
impl = s["ep_impl"]
backend = (cfg["ep_impl"]["gpu"].get(impl) or {}).get("all2all_backend")
eager = bool(cfg.get("serving", {}).get("enforce_eager", False))
print(impl, backend or "unset", int(eager))
PY
}

if [[ ${#USER_CMD[@]} -eq 0 && -n "${RUN_SETTING:-}" ]]; then
  RUN_INFO="$(check_run_setting "${RUN_SETTING}")" || die "cannot run setting ${RUN_SETTING} (see message above)"
  read -r RUN_IMPL RUN_BACKEND RUN_EAGER <<<"${RUN_INFO}"
  if [[ "${RUN_IMPL}" == "alltoall" && "${RUN_BACKEND}" != "deepep_high_throughput" ]]; then
    die "ep_impl.gpu.alltoall.all2all_backend in configs/experiments.yaml is '${RUN_BACKEND}'.
       Set it to deepep_high_throughput (the only all-to-all backend this A100 build supports)."
  fi
  if [[ "${RUN_BACKEND}" == "deepep_high_throughput" && "${RUN_EAGER}" == "0" ]]; then
    log "note: vLLM disables CUDA graphs for deepep_high_throughput; set serving.enforce_eager: true"
    log "      for the allgather runs as well so the comparison is like-for-like."
  fi
fi

# -----------------------------------------------------------------------------
# DeepEP install / build
# -----------------------------------------------------------------------------
deepep_ok() {
  "${PYTHON}" -c "import deep_ep, deep_ep_cpp" >/dev/null 2>&1
}

# torch's extension builder refuses to compile if nvcc's CUDA major != torch's.
ensure_matching_nvcc() {
  local want have
  want="$("${PYTHON}" -c 'import torch; print(torch.version.cuda or "")')"
  [[ -n "${want}" ]] || die "this torch build has no CUDA support"
  nvcc_release() { nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p'; }
  have="$(command -v nvcc >/dev/null 2>&1 && nvcc_release || true)"
  if [[ "${have%%.*}" != "${want%%.*}" ]] && type module >/dev/null 2>&1; then
    log "nvcc is ${have:-missing} but torch was built with CUDA ${want}; trying module load cudatoolkit/${want}"
    # Lmod is not written for `set -eu`; relax it just for the module call.
    set +eu
    module load "cudatoolkit/${want}" >/dev/null 2>&1 || module load cudatoolkit >/dev/null 2>&1
    set -eu
    have="$(command -v nvcc >/dev/null 2>&1 && nvcc_release || true)"
  fi
  [[ "${have%%.*}" == "${want%%.*}" ]] || die "need nvcc for CUDA ${want%%.*}.x to match torch (found: ${have:-none}).
       Run 'module avail cudatoolkit' and load a matching one before running this script."
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
  log "nvcc ${have} (CUDA_HOME=${CUDA_HOME}) matches torch CUDA ${want}"
}

# torch >= 2.10 compiles extensions with -std=c++20 and takes the host compiler from
# $CXX (default `c++`) and nvcc's -ccbin from $CC. On Perlmutter `c++` is the SLES
# system GCC 7, which rejects -std=c++20; newer GCCs are installed as g++-13 etc.
# (module gcc-native). Pick the first GCC >= 10 and export CC/CXX for the build.
ensure_host_compiler() {
  gxx_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }
  find_gxx() {
    local c v
    for c in ${CXX:-} g++-13 g++-12 g++-14 g++-11 g++-15 g++-10 g++ c++; do
      command -v "${c}" >/dev/null 2>&1 || continue
      v="$(gxx_major "${c}")"
      [[ "${v}" =~ ^[0-9]+$ && "${v}" -ge 10 ]] && { command -v "${c}"; return 0; }
    done
    return 1
  }
  local gxx gcc
  gxx="$(find_gxx || true)"
  if [[ -z "${gxx}" ]] && type module >/dev/null 2>&1; then
    log "no GCC >= 10 on PATH (c++ is GCC $(gxx_major c++ || echo '?')); trying module load gcc-native"
    set +eu
    module load gcc-native >/dev/null 2>&1
    set -eu
    gxx="$(find_gxx || true)"
  fi
  [[ -n "${gxx}" ]] || die "need GCC >= 10 for torch's -std=c++20 (found GCC $(gxx_major c++ || echo none) as c++).
       Try 'module load gcc-native' (then check 'ls /usr/bin/g++-*'), or install one into the env:
       conda install -c conda-forge gxx_linux-64=13   and re-run with CXX=x86_64-conda-linux-gnu-g++"
  # Matching C compiler: g++-13 -> gcc-13, x86_64-conda-linux-gnu-g++ -> ...-gcc, g++ -> gcc
  gcc="$(dirname "${gxx}")/$(basename "${gxx}" | sed -e 's/g++/gcc/' -e 's/^c++$/cc/')"
  [[ -x "${gcc}" ]] || gcc="${gxx}"
  export CXX="${gxx}" CC="${gcc}" NVCC_CCBIN="${gcc}"
  log "host compiler: CXX=${CXX} (GCC $(gxx_major "${CXX}")), CC=${CC}"
}

# Three edits to the pinned DeepEP checkout. Idempotent; refuses if the files look different.
#  1. setup.py turns NVSHMEM off only when `import nvidia.nvshmem` fails, and the A100
#     build asserts it is off. torch >= 2.8 wheels depend on nvidia-nvshmem-cu1x, so in a
#     vLLM env that import always works. Force it off instead of uninstalling a torch dep.
#  2. Don't build hybrid_ep_cpp: its kernels use TMA (Hopper+), vLLM's deepep_high_throughput
#     doesn't use it, and it hard-codes -std=c++17 against torch headers built for C++20.
#  3. Make deep_ep/__init__.py import fine without hybrid_ep_cpp.
patch_deepep_source() {
  "${PYTHON}" - "${SRC_DIR}" <<'PY'
import pathlib, sys
src = pathlib.Path(sys.argv[1])
TAG = "patched by perlmutter_all2all.sh"
edits = [
    ("setup.py",
     "    disable_nvshmem = False\n    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)\n",
     f"    disable_nvshmem = True  # {TAG}: A100 build must not use NVSHMEM\n"
     "    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)\n"),
    ("setup.py",
     "            get_extension_deep_ep_cpp(),\n            get_extension_hybrid_ep_cpp()\n",
     f"            get_extension_deep_ep_cpp(),\n            # hybrid_ep_cpp not built ({TAG}): Hopper+ only\n"),
    ("deep_ep/__init__.py",
     "from .hybrid_ep_buffer import HybridEPBuffer\n",
     f"try:  # {TAG}: hybrid-EP is not built for A100\n"
     "    from .hybrid_ep_buffer import HybridEPBuffer\n"
     "except ImportError:\n    HybridEPBuffer = None\n"),
    ("deep_ep/__init__.py",
     "from hybrid_ep_cpp import HybridEpConfigInstance\n",
     "try:\n    from hybrid_ep_cpp import HybridEpConfigInstance\n"
     "except ImportError:\n    HybridEpConfigInstance = None\n"),
]
for rel, old, new in edits:
    p = src / rel
    s = p.read_text()
    if new in s:
        continue
    if s.count(old) != 1:
        sys.exit(f"DeepEP {rel} does not look like the expected commit; cannot apply A100 patch:\n{old}")
    p.write_text(s.replace(old, new))
print("patched DeepEP: NVSHMEM off, hybrid-EP extension skipped")
PY
}

prepare_source() {
  clone_source || die "could not clone/fetch DeepEP (no internet on this node? run once from a login node first)"
  log "checking out DeepEP ${DEEPEP_COMMIT}"
  git -C "${SRC_DIR}" checkout -q -f --detach "${DEEPEP_COMMIT}"
  rm -rf "${SRC_DIR}/build" "${SRC_DIR}"/*.egg-info    # never reuse objects built with other flags
  if [[ -n "${DEEPEP_PATCH:-}" ]]; then
    [[ -f "${DEEPEP_PATCH}" ]] || die "DEEPEP_PATCH=${DEEPEP_PATCH} does not exist"
    log "applying patch ${DEEPEP_PATCH}"
    (cd "${SRC_DIR}" && patch -p1 < "${DEEPEP_PATCH}")
  fi
  patch_deepep_source
}

build_from_source() {
  ensure_matching_nvcc
  ensure_host_compiler
  prepare_source
  if ! "${PYTHON}" -c "import ninja" >/dev/null 2>&1; then
    "${PYTHON}" -m pip install -q ninja >/dev/null 2>&1 \
      || log "warning: ninja not available; the build will be serial and much slower"
  fi
  mkdir -p "${OUT_DIR}"
  rm -f "${OUT_DIR}"/deep_ep-*.whl
  log "building A100 (SM 8.0) deep_ep, intranode only (10-30+ min)"
  (
    cd "${SRC_DIR}"
    unset NVSHMEM_DIR RDMA_CORE_HOME
    export TORCH_CUDA_ARCH_LIST="8.0"
    export DISABLE_SM90_FEATURES=1 DISABLE_AGGRESSIVE_PTX_INSTRS=1 HYBRID_EP_MULTINODE=0
    export MAX_JOBS="${MAX_JOBS:-16}"
    "${PYTHON}" -m pip wheel . --no-build-isolation --no-deps -w "${OUT_DIR}"
  )
  local built
  built="$(ls -t "${OUT_DIR}"/deep_ep-*.whl 2>/dev/null | head -1 || true)"
  [[ -n "${built}" ]] || die "build produced no deep_ep wheel"
  log "installing ${built}"
  "${PYTHON}" -m pip install --force-reinstall --no-deps "${built}"
}

install_deepep() {
  local wheel
  wheel="$(find_wheel)"
  # FORCE_REINSTALL skips the cached wheel and rebuilds, but an explicit DEEPEP_WHEEL is always used.
  if [[ -n "${wheel}" && ( -n "${DEEPEP_WHEEL:-}" || "${FORCE_REINSTALL:-0}" != "1" ) ]]; then
    log "installing wheel ${wheel}"
    "${PYTHON}" -m pip install --force-reinstall --no-deps "${wheel}"
    deepep_ok && return 0
    [[ -z "${DEEPEP_WHEEL:-}" ]] || die "DEEPEP_WHEEL=${DEEPEP_WHEEL} does not import in this env"
    log "cached wheel does not import in this env (torch changed?); rebuilding"
  fi
  build_from_source
}

if [[ "${FORCE_REINSTALL:-0}" == "1" ]] || ! deepep_ok; then
  install_deepep
fi
if ! deepep_ok; then
  "${PYTHON}" -c "import deep_ep, deep_ep_cpp" 2>&1 | tail -3 >&2 || true
  die "deep_ep failed to import after install. 'GLIBCXX_... not found' means the runtime libstdc++ is older
       than the GCC used to build: run 'conda install -c conda-forge libstdcxx-ng' in the env. Otherwise a
       torch ABI mismatch is the usual cause: FORCE_REINSTALL=1 to rebuild in this env."
fi
"${PYTHON}" -c "import deep_ep; print('deep_ep import OK:', deep_ep.__file__, '| SM90 features compiled:', deep_ep.Buffer.is_sm90_compiled())"

# -----------------------------------------------------------------------------
# 4-GPU smoke test: bf16 dispatch -> combine through deep_ep.Buffer, the same
# intranode high-throughput path vLLM uses (num_rdma_bytes=0). Qwen3-30B-A3B shapes.
# -----------------------------------------------------------------------------
run_smoke_test() {
  local test_py="${BUILD_DIR}/deepep_smoke_test.py"
  mkdir -p "${BUILD_DIR}"
  cat > "${test_py}" <<'PY'
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import deep_ep

NUM_TOKENS, HIDDEN, TOPK, EXPERTS = 512, 2048, 8, 128


def worker(rank, world, port):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank)
    torch.manual_seed(rank)
    buf = deep_ep.Buffer(dist.group.WORLD, int(1e9), 0, low_latency_mode=False,
                         num_qps_per_rank=1, explicitly_destroy=True)
    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    topk_idx = torch.topk(torch.randn(NUM_TOKENS, EXPERTS, device="cuda"), TOPK, dim=-1, sorted=False)[1]
    topk_w = torch.rand(NUM_TOKENS, TOPK, dtype=torch.float32, device="cuda")

    per_rank, per_rdma_rank, per_expert, in_rank, _ = buf.get_dispatch_layout(topk_idx, EXPERTS)
    recv_x, _, _, _, handle, _ = buf.dispatch(
        x=x, num_tokens_per_rank=per_rank, num_tokens_per_rdma_rank=per_rdma_rank,
        is_token_in_rank=in_rank, num_tokens_per_expert=per_expert,
        topk_idx=topk_idx, topk_weights=topk_w,
        config=deep_ep.Buffer.get_dispatch_config(world))

    expected = per_rank.clone()
    dist.all_reduce(expected)
    assert recv_x.shape[0] == int(expected[rank]), \
        f"rank {rank}: received {recv_x.shape[0]} tokens, expected {int(expected[rank])}"

    combined, _, _ = buf.combine(recv_x, handle, config=deep_ep.Buffer.get_combine_config(world))
    copies = in_rank.sum(dim=1, keepdim=True).float()          # combine sums the copies
    err = ((combined.float() / copies) - x.float()).abs().max().item()
    assert err < 5e-2, f"rank {rank}: dispatch->combine max error {err}"
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"DeepEP smoke test OK: {world} GPUs, max round-trip error {err:.3g}", flush=True)
    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(int(sys.argv[1]), int(sys.argv[2])), nprocs=int(sys.argv[1]))
PY
  log "running 4-GPU DeepEP dispatch/combine smoke test"
  "${PYTHON}" "${test_py}" "${NGPU}" "$((20000 + RANDOM % 20000))" \
    || die "DeepEP smoke test failed: deep_ep imports but its kernels do not work on this node"
}

if [[ "${DEEPEP_SMOKE_TEST:-1}" == "1" ]]; then
  run_smoke_test
fi

if "${PYTHON}" -c 'import typing, sys; from vllm.config.parallel import All2AllBackend as B; sys.exit(0 if "deepep_high_throughput" in typing.get_args(B) else 1)' >/dev/null 2>&1; then
  log "vLLM accepts --all2all-backend deepep_high_throughput"
else
  log "warning: could not confirm that this vLLM accepts --all2all-backend deepep_high_throughput"
fi

cat >&2 <<'EOF'

Next: ep_impl.gpu.alltoall.all2all_backend in configs/experiments.yaml must be
deepep_high_throughput (deepep_low_latency needs NVSHMEM and will not work here).
After a run, check the server log for "Using DeepEPHTAll2AllManager" and look for
DeepEP dispatch/combine kernels in the trace.
EOF

# -----------------------------------------------------------------------------
# Hand off
# -----------------------------------------------------------------------------
if [[ ${#USER_CMD[@]} -gt 0 ]]; then
  exec "${USER_CMD[@]}"
elif [[ -n "${RUN_SETTING:-}" ]]; then
  exec "${PYTHON}" bench/run_benchmark.py --setting "${RUN_SETTING}" --launch-server
else
  log "dropping into an interactive shell (env is active)"
  exec bash -i
fi