# Team 15 — Expert-parallel Qwen3-30B-A3B inference on A100 vs TPU v6e

Benchmark harness for the 8-setting study: **{GPU, TPU} × {4, 8 chips} × {all-gather, all-to-all EP}**,
Qwen3-30B-A3B served with vLLM on ShareGPT V3. Produces CCL-Bench trace directories with workload
cards, per-request latency results, and user-TPS vs system-TPS Pareto plots.

## 1. Environment

```bash
git clone <this repo> && cd team15-ep-bench
bash setup_env.sh            # once per machine; auto-detects GPU or TPU
source activate_env.sh       # every new shell / batch job
```

| | Perlmutter (A100) | Cloud TPU VM (v6e) |
|---|---|---|
| Where to run setup | **login node** (compute nodes can't download the 61 GB of weights) | the TPU VM |
| conda | `module load conda` | Miniforge installed to `~/miniforge3` if missing |
| env location | `$PSCRATCH/envs/ep-bench` | `~/envs/ep-bench` |
| vLLM | `vllm==0.32.0` (pip; `+cu129` wheel if the driver predates CUDA 13) | `vllm-tpu==0.32.0` (tpu-inference) |
| extras | FlashInfer kernels, nvtx | `xprof` (xplane → Chrome JSON) |

Both platforms use the **same vLLM release (0.32.0)** so scheduler behaviour is comparable. Override with
`EPB_VLLM_VERSION` / `EPB_VLLM_TPU_VERSION`; `EPB_CUDA_VARIANT=cu129|cu130` forces a CUDA build. Setup also
clones CCL-Bench into `third_party/ccl-bench`, downloads the model + ShareGPT, and writes `.ep_bench_env`
(read by `activate_env.sh`). Torch comes from pip, not conda — conda's torch statically links NCCL and
breaks vLLM.

`activate_env.sh` sets `HF_HOME` to scratch, `HF_HUB_OFFLINE=1` (set `EPB_ONLINE=1` to allow downloads),
`PYTHONNOUSERSITE=1`, compile caches on scratch, and `EPB_NCCL_DEBUG=1` → `NCCL_DEBUG=INFO`.

## 2. Running

**Perlmutter** (submit from the repo root):
```bash
sbatch -A <account> -N 1 slurm/perlmutter_ep.sbatch gpu-ep4-allgather
sbatch -A <account> -N 2 slurm/perlmutter_ep.sbatch gpu-ep8-allgather     # 8 GPUs = 2 nodes
bash scripts/submit_all_gpu.sh <account>                                  # all four GPU settings
# interactive, 1 node:
salloc -N 1 -C gpu -q interactive -t 01:00:00 --gpus-per-node 4 -A <account>
source activate_env.sh && python bench/run_benchmark.py --setting gpu-ep4-allgather --launch-server
```

**TPU VM:**
```bash
source activate_env.sh
python bench/run_benchmark.py --setting tpu-ep4-allgather --launch-server
bash scripts/run_all_tpu.sh                   # all four (the 8-chip ones need a v6e-8)
```

First time on real hardware, do a short smoke run: `--concurrency 1 16`.

**Analysis** (any machine with the env):
```bash
python bench/analyze.py                    # tables + Pareto + latency + coarse plots
python bench/analyze.py --trace-metrics    # also runs CCL-Bench tools on every trace dir
```

## 3. What a run does

For each `max_concurrency` in `configs/experiments.yaml` (default 1 … 256, closed loop, `ignore_eos` so both
platforms generate identical token counts):

1. **Warmup** at that concurrency (TPU compiles its batch bucket here). Never profiled.
2. **Fine-grained log:** `POST /start_profile`, then the **first 5 engine steps** of the measured run are
   traced. GPU: torch profiler, bounded by vLLM's `--profiler-config max_iterations=5` (exact). TPU: JAX
   profiler; tpu-inference ignores iteration bounds, so the client watches the engine step counter on
   `/metrics` and stops the profiler after 5 steps (≈5, ±1 — trim in post if needed).
3. **Measure:** `vllm bench serve` on ShareGPT with `--save-detailed` (per-request TTFT/ITL/lengths).
4. **Coarse-grained log, whole run:** `/metrics` scraped every 1 s (running/waiting requests, KV-cache
   usage, token counters, preemptions, engine steps) → `coarse/metrics.jsonl`.
5. **Collect:** raw traces → a CCL-Bench row directory with a workload card.

```
$EPB_RESULTS_DIR/<setting>/<run_id>/
  setting.json, env.json, summary.json
  bench/c<C>.json                  vllm bench serve results per point
  coarse/metrics.jsonl             /metrics scrapes (tagged by point)
  logs/                            server, warmup and bench logs; exact server command
  profiler_raw/                    untouched profiler output
  ccl/<workload_name>/             one CCL-Bench row per point:
      <workload_name>.yaml           workload card (template fields + EP details)
      kineto_trace_<rank>.json       GPU  (trace type "json")
      <host>.trace.json, xplane/     TPU  (trace type "json_tpu")
      serving/bench.json, coarse.jsonl
```

Workload names follow CCL-Bench conventions, e.g.
`qwen3-30b-a3b-vllm-dp_4_tp_1_ep_4-allgather_reducescatter-c64-perlmutter-group_15`.

`analyze.py` defines **user TPS = 1000 / median TPOT** and **system TPS = output tokens/s**; it plots both
total (equal chip budget) and per-chip frontiers, with a Pareto frontier per platform.

## 4. Files

| File | Purpose |
|---|---|
| `setup_env.sh` / `activate_env.sh` | create / activate the conda env |
| `configs/experiments.yaml` | model, dataset, serving knobs, sweep, profiling, the 8 settings |
| `bench/common.py` | config loading, `vllm serve` command builder, workload naming |
| `bench/prepare_data.py` | downloads weights + ShareGPT, normalizes to `sharegpt_v3.json` |
| `bench/launch_server.py` | starts `vllm serve` for a setting (`--dry-run` prints the command) |
| `bench/run_benchmark.py` | client driver: warmup, 5-step trace, coarse scraping, sweep |
| `bench/collect_traces.py` | raw traces → CCL-Bench dirs + workload cards (re-runnable by hand) |
| `bench/analyze.py` | summary CSV, Pareto/latency/coarse plots, CCL-Bench metrics |
| `slurm/perlmutter_ep.sbatch` | 1- or 2-node Perlmutter job (server per node + client) |

## 5. Open issues — please read before collecting data

1. **There is no A100-compatible *all-to-all* backend in current vLLM.** `--all2all-backend` options in
   0.32: `allgather_reducescatter` (default; our "all-gather"), DeepEP (Hopper+ only), FlashInfer NVLink
   (MNNVL systems), MoRI (AMD), and a few others targeting NVLink/RDMA on newer GPUs. `pplx` — the one that
   ran on A100 (CCL-Bench group 9 traces) — was removed in vLLM 0.17 and `naive` by 0.20, and vLLM now
   **silently rewrites them to `allgather_reducescatter`**. So `gpu-*-alltoall` is left unset in the config
   and refuses to start until you choose. Options: (a) pin vLLM 0.16.0 and build the deprecated
   NVSHMEM pplx-kernels; (b) keep the A2A-vs-AG axis TPU-only; (c) use Hopper GPUs for the A2A runs.
   Worth asking Eric. The TPU side maps cleanly: `USE_MOE_EP_KERNEL=0` → GMM_EP (all-gather +
   reduce-scatter), `=1` → fused Pallas EP kernel (remote-copy dispatch/combine).
2. **8 A100s on Perlmutter means 2 nodes**, so EP8 crosses Slingshot (~200 Gb/s per NIC) while TPU v6e-8 is
   one host on ICI. That's a real topology confound — fine if it's part of the story, but call it out. For
   2-node runs, check once with `EPB_NCCL_DEBUG=1` that NCCL uses the OFI/Slingshot transport, not
   sockets.
3. **Attention layout differs by default.** On GPU the all2all backend only matters with DP > 1, so GPU runs
   are DP-attention + EP (TP=1). tpu-inference defaults to TP-attention + EP. `tpu.attn_dp: true` in the
   config switches TPU to DP attention for parity; I haven't been able to verify it works for this model.
4. **"The metrics stated in the timeline"** weren't included in what you sent. The harness currently
   records TTFT/TPOT/ITL/E2EL (p50/p90/p99), request/output/total throughput, user and system TPS, coarse
   server metrics, and (via `--trace-metrics`) CCL-Bench's communication fraction, MoE fraction, comm
   time, compute/comm overlap, load imbalance, all-gather/all-to-all bandwidth utilization, collective
   volume, and step time. Send the list and I'll fill any gaps.
5. Per-engine limits: on GPU `max_num_seqs` / `max_num_batched_tokens` apply **per DP engine** (4–8 engines),
   on TPU to the single engine. At concurrency ≤ 256 `max_num_seqs` doesn't bind, but batched-token
   budgets per step differ.
6. The harness was tested end-to-end against a mock vLLM server (traces fed through the real CCL-Bench
   tools), not on Perlmutter or a TPU. Expect small fixes on the first real run, especially TPU trace
   conversion (`xprof`) and the 2-node launch.
