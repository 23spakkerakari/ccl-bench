#!/usr/bin/env bash
# One-time setup: download Qwen3-30B-A3B to local/shared disk so servers never touch the network.
#
# Run on a machine WITH internet (TPU VM, or a Perlmutter login node):
#   bash scripts/setup_model.sh
#
# Overridable env vars:
#   MODEL_ID    HF repo id                      (default Qwen/Qwen3-30B-A3B)
#   REVISION    commit SHA or branch to pin     (default: resolve current main to a SHA and record it)
#   MODEL_DIR   where weights are stored        (default /mnt/models/Qwen3-30B-A3B)
#   HF_HOME     HF cache dir (kept off $HOME)   (default /mnt/models/hf)
#   GCS_URI     optional gs://bucket/path       (restore from it if present, back up to it after download)
#   MIN_FREE_GB required free space             (default 80)
#
# Examples:
#   REVISION=<sha> bash scripts/setup_model.sh
#   MODEL_DIR=$SCRATCH/models/Qwen3-30B-A3B HF_HOME=$SCRATCH/hf bash scripts/setup_model.sh   # Perlmutter
#   GCS_URI=gs://my-bucket/models/Qwen3-30B-A3B bash scripts/setup_model.sh                    # restore/backup
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B}"
REVISION="${REVISION:-}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/Qwen3-30B-A3B}"
export HF_HOME="${HF_HOME:-/mnt/models/hf}"
GCS_URI="${GCS_URI:-}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

# Reuse the project env if present (same as run_all_tpu.sh)
# shellcheck disable=SC1091
[[ -f ./activate_env.sh ]] && source ./activate_env.sh
# activate_env.sh may force offline mode for benchmark runs; this script is the one place that needs the network.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

log() { echo "[setup_model] $*"; }

# ---- 0. Preflight: disk space -------------------------------------------------------------
mkdir -p "$MODEL_DIR" "$HF_HOME"
free_gb=$(df -Pk "$MODEL_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')
log "target: $MODEL_DIR  (free: ${free_gb} GB, need >= ${MIN_FREE_GB} GB)"
if (( free_gb < MIN_FREE_GB )); then
  echo "ERROR: not enough free space on $(df -P "$MODEL_DIR" | awk 'NR==2{print $1}')." >&2
  echo "       Attach a bigger/persistent disk or set MODEL_DIR to another filesystem." >&2
  exit 1
fi

# ---- 1. Restore from GCS if a backup already exists --------------------------------------
if [[ -n "$GCS_URI" ]] && ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  if gcloud storage ls "$GCS_URI/config.json" >/dev/null 2>&1; then
    log "restoring from $GCS_URI"
    gcloud storage cp -r "$GCS_URI/*" "$MODEL_DIR/"
  fi
fi

# ---- 2. Download from Hugging Face (resumable; skipped files are verified, not re-fetched) -
python - <<PY
import os, sys
from huggingface_hub import HfApi, snapshot_download

repo, rev, dest = "${MODEL_ID}", "${REVISION}" or None, "${MODEL_DIR}"
sha = HfApi().model_info(repo, revision=rev).sha   # resolve branch/None -> immutable commit SHA
print(f"[setup_model] {repo} @ {sha}")
snapshot_download(
    repo_id=repo,
    revision=sha,
    local_dir=dest,
    allow_patterns=["*.safetensors", "*.json", "merges.txt"],   # weights + config + tokenizer
    max_workers=8,
)
open(os.path.join(dest, ".revision"), "w").write(sha)
PY

# ---- 3. Verify every shard in the index exists and is non-empty ----------------------------
python - <<PY
import json, os, sys
d = "${MODEL_DIR}"
need = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "model.safetensors.index.json"]
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.exists(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
total = sum(os.path.getsize(os.path.join(d, s)) for s in shards if s not in bad) / 1e9
if missing or bad:
    sys.exit(f"[setup_model] VERIFY FAILED  missing={missing}  bad_shards={bad}")
cfg = json.load(open(os.path.join(d, "config.json")))
print(f"[setup_model] OK: {len(shards)} shards, {total:.1f} GB, arch={cfg['architectures'][0]}, "
      f"experts={cfg.get('num_experts')}, top_k={cfg.get('num_experts_per_tok')}")
PY

# Record provenance next to the weights (copy this into your run's env.json if useful)
python - <<PY
import json, time, os
d = "${MODEL_DIR}"
json.dump({"model_id": "${MODEL_ID}", "revision": open(os.path.join(d, ".revision")).read().strip(),
           "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "path": d},
          open(os.path.join(d, "model.lock.json"), "w"), indent=2)
PY

# ---- 4. Optional: back up to GCS ----------------------------------------------------------
if [[ -n "$GCS_URI" ]]; then
  log "backing up to $GCS_URI"
  gcloud storage rsync -r "$MODEL_DIR" "$GCS_URI" --exclude='^\.cache/.*$'
fi

# ---- 5. Next steps --------------------------------------------------------------------------
cat <<EOF

[setup_model] Done. Add to your shell / activate_env.sh so runs never hit the network:

  export HF_HOME=$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

And in configs/experiments.yaml:

  model:
    id: $MODEL_DIR
    served_name: qwen3-30b-a3b
EOF