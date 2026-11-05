#!/usr/bin/env python3
"""Turn raw profiler output into CCL-Bench trace directories + workload cards.

One CCL-Bench row per (setting, concurrency) point:

    <run>/ccl/<workload_name>/
        <workload_name>.yaml          workload card (CCL-Bench template fields)
        kineto_trace_<rank>.json      GPU: torch-profiler trace per EP rank  (trace type "json")
        <host>.trace.json             TPU: JAX trace as Chrome JSON          (trace type "json_tpu")
        xplane/<host>.xplane.pb       TPU: raw JAX profile (TensorBoard/XProf)
        serving/bench.json            `vllm bench serve` result for this point
        serving/coarse.jsonl          /metrics scrapes for this point

Serving files live in a subfolder because CCL-Bench tools read every top-level *.json
as a trace. Usually called by run_benchmark.py; can be re-run by hand:

    python bench/collect_traces.py --run-dir <run> --setting gpu-ep4-allgather --concurrency 64
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import RunPaths, Setting, get_setting, load_config  # noqa: E402

_RANK_RE = [re.compile(r"(?:^|_)ep(\d+)(?:_|\.)"), re.compile(r"(?:^|_)dp(\d+)(?:_|\.)"),
            re.compile(r"(?:^|_)rank(\d+)(?:_|\.)")]


# ----------------------------------------------------------------------------
# Waiting for profiler output to land on disk
# ----------------------------------------------------------------------------
def _stable(files: list[Path], settle_s: float) -> bool:
    sizes = {f: f.stat().st_size for f in files if f.exists()}
    time.sleep(settle_s)
    return all(f.exists() and f.stat().st_size == s and s > 0 for f, s in sizes.items())


def wait_for_files(find, expected: int, timeout_s: float = 900, settle_s: float = 5.0) -> list[Path]:
    """Poll `find()` until >= expected files exist and their sizes stop changing."""
    deadline = time.time() + timeout_s
    files: list[Path] = []
    while time.time() < deadline:
        files = find()
        if len(files) >= expected and _stable(files, settle_s):
            return files
        time.sleep(2)
    print(f"[collect] WARNING: timed out waiting for {expected} trace files; have {len(files)}", file=sys.stderr)
    return files


def gpu_trace_files(paths: RunPaths, prefix: str) -> list[Path]:
    raw = paths.profiler_raw
    return sorted(p for p in raw.rglob(f"{prefix}_*") if p.name.endswith((".pt.trace.json.gz", ".pt.trace.json")))


def tpu_sessions(paths: RunPaths) -> set[Path]:
    return {p.parent for p in paths.profiler_raw.rglob("*.xplane.pb")}


def tpu_session_files(session: Path) -> list[Path]:
    return sorted(session.glob("*.xplane.pb"))


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------
def _rank_of(name: str, fallback: int) -> int:
    for rx in _RANK_RE:
        m = rx.search(name)
        if m:
            return int(m.group(1))
    return fallback


def collect_gpu(files: list[Path], out_dir: Path) -> list[Path]:
    written = []
    for i, f in enumerate(sorted(files)):
        rank = _rank_of(f.name, i)
        dst = out_dir / f"kineto_trace_{rank}.json"
        opener = gzip.open if f.name.endswith(".gz") else open
        with opener(f, "rb") as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        written.append(dst)
    return written


def collect_tpu(files: list[Path], out_dir: Path) -> list[Path]:
    (out_dir / "xplane").mkdir(exist_ok=True)
    written = []
    for f in files:
        host = f.name[: -len(".xplane.pb")]
        shutil.copy2(f, out_dir / "xplane" / f.name)
        try:
            from xprof.convert import raw_to_tool_data
            data, _ = raw_to_tool_data.xspace_to_tool_data([str(f)], "trace_viewer", {})
            if not data:
                raise RuntimeError("empty trace_viewer output")
            text = data.decode() if isinstance(data, (bytes, bytearray)) else data
            dst = out_dir / f"{host}.trace.json"
            dst.write_text(text)
            with gzip.open(out_dir / "xplane" / f"{host}.trace.json.gz", "wt") as gz:
                gz.write(text)    # the comm_kernel_breakdown_tpu tool globs *.json.gz
            written.append(dst)
        except Exception as e:  # noqa: BLE001
            print(f"[collect] could not convert {f.name} to Chrome JSON ({e}).\n"
                  f"          Open {out_dir/'xplane'} in TensorBoard/XProf and export the trace as "
                  f"{host}.trace.json into {out_dir}.", file=sys.stderr)
    return written


# ----------------------------------------------------------------------------
# Workload card
# ----------------------------------------------------------------------------
def model_arch(cfg: dict) -> dict:
    arch = dict(cfg["model"]["arch_fallback"])
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(cfg["model"]["id"], "config.json")
        hf = json.loads(Path(p).read_text())
        arch.update({
            "num_layers": hf.get("num_hidden_layers", arch["num_layers"]),
            "num_heads": hf.get("num_attention_heads", arch["num_heads"]),
            "num_kv_heads": hf.get("num_key_value_heads", arch["num_kv_heads"]),
            "head_dim": hf.get("head_dim", arch["head_dim"]),
            "hidden_size": hf.get("hidden_size", arch["hidden_size"]),
            "moe_intermediate_size": hf.get("moe_intermediate_size", arch["moe_intermediate_size"]),
            "num_experts": hf.get("num_experts", arch["num_experts"]),
            "num_experts_per_tok": hf.get("num_experts_per_tok", arch["num_experts_per_tok"]),
            "vocab_size": hf.get("vocab_size", arch["vocab_size"]),
        })
        arch["num_params_embedding"] = arch["vocab_size"] * arch["hidden_size"]
    except Exception:  # noqa: BLE001 — offline and not cached: keep fallbacks
        pass
    return arch


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def platform_versions(setting: Setting) -> dict:
    v: dict = {}
    if setting.platform == "gpu":
        drv = _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()
        v["driver"] = drv[0] if drv else ""
        try:
            import torch
            v["cuda"] = torch.version.cuda
            nccl = torch.cuda.nccl.version()
            v["nccl"] = ".".join(map(str, nccl)) if isinstance(nccl, tuple) else str(nccl)
        except Exception:  # noqa: BLE001
            pass
        v["gpu_name"] = (_sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
                         .splitlines() or [""])[0]
    else:
        for pkg in ("libtpu", "jax", "tpu_inference"):
            try:
                v[pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
    try:
        v["vllm"] = md.version("vllm") if setting.platform == "gpu" else md.version("vllm-tpu")
    except md.PackageNotFoundError:
        try:
            v["vllm"] = md.version("vllm")
        except md.PackageNotFoundError:
            pass
    return v


def write_workload_card(out_dir: Path, setting: Setting, concurrency: int, bench: dict | None,
                        trace_files: list[Path]) -> Path:
    cfg, srv = setting.cfg, setting.serving
    ver = platform_versions(setting)
    n_steps = int(cfg["profiling"]["fine_grained_steps"])
    gpu = setting.platform == "gpu"
    mean_in = mean_out = None
    if bench:
        if bench.get("input_lens"):
            mean_in = round(sum(bench["input_lens"]) / len(bench["input_lens"]), 1)
        if bench.get("output_lens"):
            mean_out = round(sum(bench["output_lens"]) / len(bench["output_lens"]), 1)

    card = {
        "version": 1,
        "description": (
            f"vLLM online serving of {cfg['model']['id']} on {setting.chips}x "
            f"{'NVIDIA A100 (Perlmutter)' if gpu else 'TPU v6e'}; EP={setting.chips} via "
            f"{setting.comm_impl_label} ({setting.ep_impl}); ShareGPT V3, closed loop at "
            f"max_concurrency={concurrency}. Trace = first {n_steps} engine steps of this point "
            f"(after warmup); serving/ holds the full-run latency results and /metrics scrapes."),
        "hf_url": f"https://huggingface.co/{cfg['model']['id']}",
        "trace_url": str(out_dir.resolve()),
        "contributor": cfg.get("contributor", ""),
        "contact": cfg.get("contact", ""),
        "workload": {
            "model": {
                "phase": "inference", "moe": True, "granularity": "model_fwd",
                "model_family": "qwen3-30b-a3b", "precision": cfg["model"]["precision"],
                "epochs": 1, "iteration": n_steps, "model_arch": model_arch(cfg),
            },
            "data": {
                "batch_size": concurrency, "seq_len": srv["max_model_len"],
                "input_len": mean_in, "output_len": mean_out,
                "dataset": cfg["dataset"]["name"],
                "num_prompts": (bench or {}).get("num_prompts"),
                "ignore_eos": bool(cfg["dataset"].get("ignore_eos", False)),
            },
            "hardware": {
                "network_topo": ({"topology": "slingshot" if setting.nodes > 1 else "nvlink",
                                  "bandwidth_gbps": [200, 2400]} if gpu else
                                 {"topology": "ici_2d", "bandwidth_gbps": [None, 3584]}),
                "xpu_spec": {"type": "GPU" if gpu else "TPU",
                             "model": "nvidia_a100" if gpu else "tpu_v6e",
                             "total_count": setting.chips,
                             "count_per_node": setting.chips_per_node},
                "driver_version": (f"cuda_{ver.get('cuda', '')} (driver {ver.get('driver', '')})" if gpu
                                   else f"libtpu_{ver.get('libtpu', '')}"),
            },
        },
        "Model-executor": {
            "framework": {
                "name": "vllm" if gpu else "vllm-tpu (tpu-inference)",
                "version": ver.get("vllm", ""),
                "compiler_tool_selection": ("eager" if srv.get("enforce_eager") else "cudagraph") if gpu else "xla",
            },
            "model_plan_parallelization": setting.parallel_layout,
            "communication_library": {
                "name": "NCCL" if gpu else "xla",
                "version": ver.get("nccl", "") if gpu else ver.get("jax", ""),
                "env": ({"all2all_backend": setting.comm_impl_label} if gpu
                        else dict(setting.impl_cfg.get("env", {}))),
            },
            "protocol_selection": (["rocev2" if setting.nodes > 1 else "none", "p2p"] if gpu
                                   else ["xla_collectives", "ici"]),
            "serving": {k: srv[k] for k in ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                                            "gpu_memory_utilization", "enable_prefix_caching",
                                            "enforce_eager")},
            "expert_parallel": {"mode": setting.ep_impl, "implementation": setting.comm_impl_label,
                                "eplb": False,
                                "top_k": cfg["model"]["arch_fallback"]["num_experts_per_tok"],
                                **({"tpu_attn_dp": setting.tpu_attn_dp} if not gpu else {})},
        },
        "metric_source": {
            "traces": ["json"] if gpu else ["json_tpu"],
            "metrics_specific_trace": ["serving/bench.json", "serving/coarse.jsonl"],
            "trace_files": [p.name for p in trace_files],
        },
    }
    name = out_dir.name
    path = out_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(card, f, sort_keys=False, default_flow_style=None, width=100)
    return path


# ----------------------------------------------------------------------------
# Entry point used by run_benchmark.py
# ----------------------------------------------------------------------------
def assemble_point(setting: Setting, paths: RunPaths, concurrency: int,
                   trace_files: list[Path]) -> Path:
    out_dir = paths.ccl / setting.workload_name(concurrency)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "serving").mkdir(exist_ok=True)

    written: list[Path] = []
    if trace_files:
        written = (collect_gpu if setting.platform == "gpu" else collect_tpu)(trace_files, out_dir)

    bench_path = paths.bench / f"c{concurrency}.json"
    bench = None
    if bench_path.exists():
        shutil.copy2(bench_path, out_dir / "serving" / "bench.json")
        bench = json.loads(bench_path.read_text())

    coarse = paths.coarse / "metrics.jsonl"
    if coarse.exists():
        with open(coarse) as src, open(out_dir / "serving" / "coarse.jsonl", "w") as dst:
            for line in src:
                if f'"point": "c{concurrency}"' in line:
                    dst.write(line)

    card = write_workload_card(out_dir, setting, concurrency, bench, written)
    print(f"[collect] c{concurrency}: {len(written)} trace file(s) + card -> {card.parent}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tpu-session", default=None, help="TPU: profile session dir to use (default: newest)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setting = get_setting(cfg, args.setting)
    paths = RunPaths(Path(args.run_dir))
    if setting.platform == "gpu":
        files = gpu_trace_files(paths, f"c{args.concurrency}")
    else:
        sessions = sorted(tpu_sessions(paths), key=lambda p: p.stat().st_mtime)
        session = Path(args.tpu_session) if args.tpu_session else (sessions[-1] if sessions else None)
        files = tpu_session_files(session) if session else []
    assemble_point(setting, paths, args.concurrency, files)


if __name__ == "__main__":
    main()
