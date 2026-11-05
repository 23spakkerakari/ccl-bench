# Usage guide

Run everything from the repo root. Settings are named `<gpu|tpu>-ep<4|8>-<allgather|alltoall>`
(e.g. `gpu-ep4-allgather`); see `configs/experiments.yaml`.

## Setup (once per machine)

```bash
bash setup_env.sh                 # auto-detects GPU/TPU; on Perlmutter use a LOGIN node
bash setup_env.sh --force         # rebuild the env
bash setup_env.sh --skip-download # env only, no model/dataset download
```

## Every shell / job

```bash
source activate_env.sh
EPB_ONLINE=1 source activate_env.sh       # allow Hugging Face downloads
EPB_NCCL_DEBUG=1 source activate_env.sh   # verbose NCCL (check 2-node transport)
```

## Download data only

```bash
python bench/prepare_data.py --data-dir $EPB_DATA_DIR   # --skip-model / --skip-dataset
```

## Run a benchmark

| Situation | Command |
|---|---|
| 1 node, script starts/stops the server | `python bench/run_benchmark.py --setting gpu-ep4-allgather --launch-server` |
| Quick smoke test | add `--concurrency 1 16` |
| Perlmutter batch, 4 GPUs | `sbatch -A <acct> -N 1 slurm/perlmutter_ep.sbatch gpu-ep4-allgather` |
| Perlmutter batch, 8 GPUs (2 nodes) | `sbatch -A <acct> -N 2 slurm/perlmutter_ep.sbatch gpu-ep8-allgather` |
| All 4 GPU settings | `bash scripts/submit_all_gpu.sh <acct>` |
| All 4 TPU settings (on the TPU VM) | `bash scripts/run_all_tpu.sh` |
| Server already running elsewhere | `python bench/run_benchmark.py --setting <s> --run-id <id> --host <h> --port <p>` |

Add `-C "gpu&hbm80g"` to `sbatch` for 80 GB A100s. Extra args after the setting name in `sbatch`
go to `run_benchmark.py`.

## Start the server by hand

```bash
python bench/launch_server.py --setting tpu-ep4-alltoall --run-id test         # start
python bench/launch_server.py --setting tpu-ep4-alltoall --run-id test --dry-run  # print command only
```

Use the same `--run-id` for the server and `run_benchmark.py` so traces land in one folder.

## Analyze

```bash
python bench/analyze.py                   # latest run per setting
python bench/analyze.py --all-runs        # every run
python bench/analyze.py --trace-metrics   # also run CCL-Bench metric tools (slow)
```

Writes to `$EPB_RESULTS_DIR/analysis/`: `summary.csv`, `pareto_total.png`, `pareto_per_chip.png`,
`pareto_points.csv`, `latency_vs_concurrency.png`, `coarse_<setting>.png`, `trace_metrics.csv`.

## Rebuild one CCL-Bench folder

```bash
python bench/collect_traces.py --run-dir <run> --setting gpu-ep4-allgather --concurrency 64
```

## Where results go

`$EPB_RESULTS_DIR/<setting>/<run_id>/`:
`bench/` latency results · `coarse/` per-second server metrics · `logs/` server and client logs ·
`profiler_raw/` raw traces · `ccl/<workload_name>/` CCL-Bench row (card + traces).

## Common config changes (`configs/experiments.yaml`)

| Want to… | Edit |
|---|---|
| Change load points | `sweep.max_concurrency` |
| Change prompts per point | `sweep.prompts_per_concurrency`, `min_prompts`, `max_prompts` |
| Trace more/fewer steps | `profiling.fine_grained_steps` |
| Trace only some points | `profiling.profile_points: first` or `[16, 64]` (or `none`) |
| Scrape metrics more often | `profiling.coarse_interval_s` |
| Pick the GPU all-to-all backend | `ep_impl.gpu.alltoall.all2all_backend` (unset by default — see README §5) |
| DP attention on TPU | `tpu.attn_dp: true` |
| Disable CUDA graphs (cleaner traces) | `serving.enforce_eager: true` |
| Pass extra `vllm serve` flags | `serving.extra_args` |

## Troubleshooting

- `/start_profile is 404` → server wasn't launched via `launch_server.py`.
- `all2all_backend is not set` → expected for `gpu-*-alltoall`; choose a backend first.
- TPU trace not converted → open `ccl/<name>/xplane/` in TensorBoard/XProf, export as `<host>.trace.json`.
- Server never gets healthy → check `logs/server_node0.log`.
