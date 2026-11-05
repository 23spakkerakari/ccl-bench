#!/usr/bin/env python3
"""Client-side driver: sweep load, log coarse metrics throughout, trace the first N steps.

For every max-concurrency value in configs/experiments.yaml:
  1. warmup     — a few requests at that concurrency (TPU compiles its batch bucket here)
  2. fine trace — POST /start_profile; the first `fine_grained_steps` engine steps of the
                  measured run are captured (GPU: engine-side max_iterations; TPU: this
                  script watches the engine step counter and POSTs /stop_profile)
  3. measure    — `vllm bench serve` on ShareGPT, per-request TTFT/TPOT/ITL/E2EL saved
  4. collect    — raw traces -> CCL-Bench directory + workload card (bench/collect_traces.py)
Throughout, /metrics is scraped every `coarse_interval_s` into coarse/metrics.jsonl.

Typical use (single node; starts and stops its own server like HW2's sweep.sh):
    python bench/run_benchmark.py --setting gpu-ep4-allgather --launch-server

Against a server that is already up (e.g. the 2-node sbatch):
    python bench/run_benchmark.py --setting gpu-ep8-allgather --run-id <same id as server>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import collect_traces  # noqa: E402
from bench.common import (RunPaths, Setting, base_url, get_setting, hostname, load_config,  # noqa: E402
                          new_run_id, num_prompts_for, run_dir_for, should_profile_point,
                          wait_for_health)

COARSE_METRICS = (
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:num_preemptions_total",
    "vllm:request_success_total", "vllm:iteration_tokens_total_count", "vllm:iteration_tokens_total_sum",
)
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


# ----------------------------------------------------------------------------
# /metrics scraping
# ----------------------------------------------------------------------------
def scrape(url: str, names=COARSE_METRICS) -> dict:
    """Aggregate Prometheus samples across engines: {name: {"sum", "max", "n"}}."""
    text = requests.get(f"{url}/metrics", timeout=5).text
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m or m.group(1) not in names:
            continue
        v = float(m.group(3))
        agg = out.setdefault(m.group(1), {"sum": 0.0, "max": float("-inf"), "n": 0})
        agg["sum"] += v
        agg["max"] = max(agg["max"], v)
        agg["n"] += 1
    return out


def engine_steps(url: str) -> float:
    return scrape(url, ("vllm:iteration_tokens_total_count",)).get(
        "vllm:iteration_tokens_total_count", {}).get("sum", 0.0)


class CoarsePoller(threading.Thread):
    """Coarse-grained logging for the whole run."""

    def __init__(self, url: str, out: Path, interval_s: float):
        super().__init__(daemon=True)
        self.url, self.out, self.interval = url, out, interval_s
        self.point = "setup"
        self._halt = threading.Event()

    def run(self) -> None:
        with open(self.out, "a") as f:
            while not self._halt.is_set():
                t0 = time.time()
                try:
                    row = {"t": t0, "point": self.point, "metrics": scrape(self.url)}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                except requests.RequestException:
                    pass
                self._halt.wait(max(0.0, self.interval - (time.time() - t0)))

    def stop(self) -> None:
        self._halt.set()


class StepWatcher(threading.Thread):
    """TPU only: stop the JAX profiler after N engine steps (worker ignores max_iterations)."""

    def __init__(self, url: str, n_steps: int, timeout_s: float = 900):
        super().__init__(daemon=True)
        self.url, self.n, self.timeout = url, n_steps, timeout_s
        self.baseline = engine_steps(url)
        self.steps_seen = 0.0
        self.stopped = threading.Event()

    def run(self) -> None:
        deadline = time.time() + self.timeout
        while time.time() < deadline and not self.stopped.is_set():
            try:
                self.steps_seen = engine_steps(self.url) - self.baseline
                if self.steps_seen >= self.n:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.02)
        stop_profile(self.url)
        self.stopped.set()


def start_profile(url: str, prefix: str | None) -> None:
    body = {"profile_prefix": prefix} if prefix else None
    r = requests.post(f"{url}/start_profile", json=body, timeout=120)
    if r.status_code == 404:
        raise SystemExit("/start_profile is 404: the server was started without --profiler-config. "
                         "Launch it with bench/launch_server.py (profiling enabled in the config).")
    r.raise_for_status()


def stop_profile(url: str) -> None:
    try:
        requests.post(f"{url}/stop_profile", timeout=600)
    except requests.RequestException as e:
        print(f"[bench] stop_profile failed: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# vllm bench serve
# ----------------------------------------------------------------------------
def bench_cmd(setting: Setting, url: str, concurrency: int, num_prompts: int, seed: int,
              result: Path | None) -> list[str]:
    cfg = setting.cfg
    cmd = [
        "vllm", "bench", "serve", "--backend", "vllm", "--base-url", url,
        "--model", cfg["model"]["id"], "--served-model-name", cfg["model"]["served_name"],
        "--dataset-name", "sharegpt", "--dataset-path", cfg["dataset"]["path"],
        "--num-prompts", str(num_prompts), "--max-concurrency", str(concurrency),
        "--request-rate", str(cfg["sweep"].get("request_rate", "inf")), "--seed", str(seed),
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
        "--ready-check-timeout-sec", "0", "--disable-tqdm",
    ]
    if cfg["dataset"].get("ignore_eos", False):
        cmd.append("--ignore-eos")
    if result is not None:
        cmd += ["--save-result", "--save-detailed", "--result-dir", str(result.parent),
                "--result-filename", result.name,
                "--metadata", f"setting={setting.name}", f"platform={setting.platform}",
                f"chips={setting.chips}", f"nodes={setting.nodes}", f"ep_impl={setting.ep_impl}",
                f"comm_impl={setting.comm_impl_label}", f"concurrency={concurrency}"]
    return cmd


def run_logged(cmd: list[str], log_path: Path) -> int:
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in ("throughput", "TTFT", "TPOT", "Error", "error")):
                sys.stdout.write("    " + line)
        return proc.wait()


# ----------------------------------------------------------------------------
# One sweep point
# ----------------------------------------------------------------------------
def run_point(setting: Setting, paths: RunPaths, url: str, concurrency: int, index: int,
              poller: CoarsePoller) -> dict:
    cfg = setting.cfg
    seed = int(cfg["dataset"].get("seed", 0))
    n = num_prompts_for(cfg, concurrency)
    label = f"c{concurrency}"
    print(f"\n=== {setting.name} | max_concurrency={concurrency} | {n} prompts ===", flush=True)

    poller.point = f"{label}-warmup"
    warm = int(cfg["sweep"].get("warmup_prompts", 0))
    if warm:
        rc = run_logged(bench_cmd(setting, url, concurrency, max(warm, concurrency), seed + 1000 + index, None),
                        paths.logs / f"warmup_{label}.log")
        if rc:
            print(f"[bench] warmup for {label} exited {rc}; continuing", file=sys.stderr)

    profiling = should_profile_point(cfg, concurrency, index)
    n_steps = int(cfg["profiling"]["fine_grained_steps"])
    watcher = None
    tpu_before = collect_traces.tpu_sessions(paths) if setting.platform == "tpu" else set()
    if profiling:
        if setting.platform == "gpu":
            start_profile(url, prefix=label)      # worker auto-stops after max_iterations
        else:
            start_profile(url, prefix=None)       # TPU parses the prefix as profiler options
            watcher = StepWatcher(url, n_steps)
            watcher.start()

    poller.point = label
    result = paths.bench / f"{label}.json"
    t0 = time.time()
    rc = run_logged(bench_cmd(setting, url, concurrency, n, seed, result), paths.logs / f"bench_{label}.log")
    wall = time.time() - t0
    poller.point = f"{label}-done"

    trace_files: list[Path] = []
    if profiling:
        if watcher is not None:
            watcher.stopped.wait(timeout=60)
            if not watcher.stopped.is_set():
                stop_profile(url)
            print(f"[bench] TPU profile stopped after ~{watcher.steps_seen:.0f} engine steps")
            trace_files = collect_traces.wait_for_files(
                lambda: [f for s in collect_traces.tpu_sessions(paths) - tpu_before
                         for f in collect_traces.tpu_session_files(s)], expected=1)
        else:
            stop_profile(url)                     # no-op if the worker already stopped itself
            trace_files = collect_traces.wait_for_files(
                lambda: collect_traces.gpu_trace_files(paths, label), expected=setting.chips)

    ccl_dir = collect_traces.assemble_point(setting, paths, concurrency, trace_files)
    summary = {"concurrency": concurrency, "num_prompts": n, "returncode": rc, "wall_s": round(wall, 1),
               "profiled": profiling, "trace_files": len(trace_files), "ccl_dir": str(ccl_dir)}
    if result.exists():
        r = json.loads(result.read_text())
        for k in ("completed", "failed", "output_throughput", "total_token_throughput",
                  "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"):
            summary[k] = r.get(k)
        print(f"[bench] {label}: out_tput={r.get('output_throughput', 0):.1f} tok/s  "
              f"TTFT p50={r.get('median_ttft_ms', 0):.1f} ms  TPOT p50={r.get('median_tpot_ms', 0):.2f} ms")
    return summary


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def write_env_info(paths: RunPaths, setting: Setting, url: str) -> None:
    info = {"host": hostname(), "setting": setting.name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "versions": collect_traces.platform_versions(setting)}
    try:
        info["server_version"] = requests.get(f"{url}/version", timeout=5).json()
    except Exception:  # noqa: BLE001
        pass
    (paths.root / "env.json").write_text(json.dumps(info, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--launch-server", action="store_true", help="start/stop bench/launch_server.py locally")
    ap.add_argument("--concurrency", type=int, nargs="+", default=None, help="override sweep.max_concurrency")
    ap.add_argument("--server-timeout", type=float, default=3600)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setting = get_setting(cfg, args.setting)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_for(setting, args.run_id or new_run_id())
    paths = RunPaths(run_dir).make()
    url = base_url(args.host, args.port or int(setting.serving["port"]))
    if not Path(cfg["dataset"]["path"]).exists():
        raise SystemExit(f"Dataset {cfg['dataset']['path']} missing — run bench/prepare_data.py on a login node.")
    print(f"[bench] run dir: {run_dir}")

    server = None
    if args.launch_server:
        if setting.nodes > 1:
            raise SystemExit(f"{setting.name} spans {setting.nodes} nodes; use slurm/perlmutter_ep.sbatch")
        server = subprocess.Popen([sys.executable, str(Path(__file__).with_name("launch_server.py")),
                                   "--setting", setting.name, "--run-dir", str(run_dir),
                                   *(["--config", args.config] if args.config else [])],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
    try:
        print(f"[bench] waiting for {url}/health ...", flush=True)
        wait_for_health(url, timeout_s=args.server_timeout, proc=server)
        write_env_info(paths, setting, url)

        poller = CoarsePoller(url, paths.coarse / "metrics.jsonl", float(cfg["profiling"]["coarse_interval_s"]))
        poller.start()
        summaries = []
        for i, c in enumerate(args.concurrency or cfg["sweep"]["max_concurrency"]):
            summaries.append(run_point(setting, paths, url, int(c), i, poller))
            (paths.root / "summary.json").write_text(json.dumps(summaries, indent=2))
        poller.stop()
        poller.join(timeout=10)
        print(f"\n[bench] done -> {paths.root}\n        analyze with: python bench/analyze.py {paths.root.parent.parent}")
        return 0
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)


if __name__ == "__main__":
    sys.exit(main())
