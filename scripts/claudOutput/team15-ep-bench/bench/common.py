"""Shared config handling and command construction for the Team 15 EP benchmark.

Everything that both the server launcher and the client driver need to agree on
(paths, the `vllm serve` command line, workload names) lives here so the two can
never drift apart.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"

# Backends vLLM accepts on the CLI but silently rewrites to allgather_reducescatter.
_SILENT_FALLBACK_BACKENDS = {"naive", "pplx"}


# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------
def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return _expand(yaml.safe_load(f))


@dataclass
class Setting:
    name: str
    platform: str          # "gpu" | "tpu"
    chips: int
    nodes: int
    ep_impl: str           # "allgather" | "alltoall"
    cfg: dict = field(repr=False)

    # --- derived -------------------------------------------------------------
    @property
    def serving(self) -> dict:
        merged = dict(self.cfg["serving"])
        merged.update(self.cfg["settings"][self.name].get("serving", {}))
        return merged

    @property
    def impl_cfg(self) -> dict:
        return self.cfg["ep_impl"][self.platform][self.ep_impl] or {}

    @property
    def chips_per_node(self) -> int:
        return self.chips // self.nodes

    @property
    def tpu_attn_dp(self) -> bool:
        return self.platform == "tpu" and bool(self.cfg.get("tpu", {}).get("attn_dp", False))

    @property
    def comm_impl_label(self) -> str:
        if self.platform == "gpu":
            return str(self.impl_cfg.get("all2all_backend"))
        return "fused_ep_kernel" if self.ep_impl == "alltoall" else "gmm_ep"

    @property
    def parallel_layout(self) -> dict:
        n = self.chips
        if self.platform == "gpu":
            return {"dp_replicate": n, "dp_shard": 1, "tp": 1, "pp": 1, "cp": 1, "ep": n}
        if self.tpu_attn_dp:
            return {"dp_replicate": n, "dp_shard": 1, "tp": 1, "pp": 1, "cp": 1, "ep": n}
        return {"dp_replicate": 1, "dp_shard": 1, "tp": n, "pp": 1, "cp": 1, "ep": n}

    def workload_name(self, concurrency: int) -> str:
        p = self.parallel_layout
        hw = "perlmutter" if self.platform == "gpu" else "tpu_v6e"
        tag = self.cfg.get("group_tag", "group_15")
        return (f"qwen3-30b-a3b-vllm-dp_{p['dp_replicate']}_tp_{p['tp']}_ep_{p['ep']}-"
                f"{self.comm_impl_label}-c{concurrency}-{hw}-{tag}")


def get_setting(cfg: dict, name: str) -> Setting:
    if name not in cfg["settings"]:
        raise SystemExit(f"Unknown setting '{name}'. Choose from: {', '.join(cfg['settings'])}")
    s = cfg["settings"][name]
    setting = Setting(name=name, platform=s["platform"], chips=int(s["chips"]),
                      nodes=int(s.get("nodes", 1)), ep_impl=s["ep_impl"], cfg=cfg)
    if setting.chips % setting.nodes:
        raise SystemExit(f"{name}: chips ({setting.chips}) must divide evenly across nodes ({setting.nodes})")
    return setting


# ----------------------------------------------------------------------------
# Run directory layout
# ----------------------------------------------------------------------------
def results_root() -> Path:
    root = os.environ.get("EPB_RESULTS_DIR")
    return Path(root) if root else REPO_ROOT / "results"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def run_dir_for(setting: Setting, run_id: str) -> Path:
    return results_root() / setting.name / run_id


@dataclass
class RunPaths:
    root: Path

    @property
    def profiler_raw(self) -> Path: return self.root / "profiler_raw"
    @property
    def bench(self) -> Path: return self.root / "bench"
    @property
    def coarse(self) -> Path: return self.root / "coarse"
    @property
    def ccl(self) -> Path: return self.root / "ccl"
    @property
    def logs(self) -> Path: return self.root / "logs"

    def make(self) -> "RunPaths":
        for p in (self.root, self.profiler_raw, self.bench, self.coarse, self.ccl, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


# ----------------------------------------------------------------------------
# Profiling selection
# ----------------------------------------------------------------------------
def profiling_enabled(cfg: dict) -> bool:
    return cfg["profiling"].get("profile_points", "all") not in ("none", None, False)


def should_profile_point(cfg: dict, concurrency: int, index: int) -> bool:
    pts = cfg["profiling"].get("profile_points", "all")
    if pts in ("none", None, False):
        return False
    if pts == "all":
        return True
    if pts == "first":
        return index == 0
    if isinstance(pts, list):
        return concurrency in {int(x) for x in pts}
    raise SystemExit(f"profiling.profile_points: unrecognised value {pts!r}")


def num_prompts_for(cfg: dict, concurrency: int) -> int:
    sw = cfg["sweep"]
    n = int(sw["prompts_per_concurrency"]) * concurrency
    return max(int(sw["min_prompts"]), min(int(sw["max_prompts"]), n))


# ----------------------------------------------------------------------------
# Server command
# ----------------------------------------------------------------------------
def build_server_command(setting: Setting, paths: RunPaths, node_rank: int = 0,
                         head_addr: str | None = None, dp_rpc_port: int = 13345
                         ) -> tuple[list[str], dict[str, str]]:
    """Return (argv, extra_env) for `vllm serve` on this node."""
    cfg, srv = setting.cfg, setting.serving
    argv = [
        "vllm", "serve", cfg["model"]["id"],
        "--served-model-name", cfg["model"]["served_name"],
        "--host", "0.0.0.0", "--port", str(srv["port"]),
        "--max-model-len", str(srv["max_model_len"]),
        "--max-num-seqs", str(srv["max_num_seqs"]),
        "--max-num-batched-tokens", str(srv["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(srv["gpu_memory_utilization"]),
        "--seed", str(cfg["dataset"].get("seed", 0)),
        "--enable-expert-parallel",
    ]
    if not srv.get("enable_prefix_caching", False):
        argv.append("--no-enable-prefix-caching")
    if srv.get("enforce_eager", False):
        argv.append("--enforce-eager")

    env: dict[str, str] = {}

    if setting.platform == "gpu":
        backend = setting.impl_cfg.get("all2all_backend")
        if not backend:
            raise SystemExit(
                f"{setting.name}: ep_impl.gpu.{setting.ep_impl}.all2all_backend is not set.\n"
                "vLLM 0.32 has no all-to-all backend that runs on A100 (pplx/naive were removed;\n"
                "DeepEP needs Hopper+). Pick one in configs/experiments.yaml — see README 'Open issues'.")
        if backend in _SILENT_FALLBACK_BACKENDS:
            raise SystemExit(
                f"{setting.name}: all2all_backend '{backend}' was removed from vLLM and is silently\n"
                "replaced by allgather_reducescatter, which would mislabel the run. Refusing to start.")
        argv += ["--tensor-parallel-size", "1",
                 "--data-parallel-size", str(setting.chips),
                 "--all2all-backend", backend]
        if setting.nodes > 1:
            if not head_addr:
                raise SystemExit(f"{setting.name} spans {setting.nodes} nodes: pass --head-addr")
            argv += ["--data-parallel-size-local", str(setting.chips_per_node),
                     "--data-parallel-address", head_addr,
                     "--data-parallel-rpc-port", str(dp_rpc_port)]
            if node_rank > 0:
                argv += ["--data-parallel-start-rank", str(node_rank * setting.chips_per_node),
                         "--headless"]
    else:
        argv += ["--tensor-parallel-size", str(setting.chips)]
        if setting.tpu_attn_dp:
            sharding = {"sharding": {"sharding_strategy": {
                "enable_dp_attention": True, "attn_dp_size": setting.chips}}}
            argv += ["--additional-config", json.dumps(sharding)]
        env.update({k: str(v) for k, v in setting.impl_cfg.get("env", {}).items()})

    if profiling_enabled(cfg):
        prof = {"profiler": "torch", "torch_profiler_dir": str(paths.profiler_raw.resolve())}
        if setting.platform == "gpu":
            # Engine-side iteration bound: the worker stops itself after N steps.
            prof.update({
                "max_iterations": int(cfg["profiling"]["fine_grained_steps"]),
                "ignore_frontend": True,          # frontend profiler ignores iteration bounds
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": bool(cfg["profiling"].get("record_shapes", True)),
                "torch_profiler_use_gzip": True,
                # The default per-worker kernel summary (profiler_out.txt) takes minutes to
                # compute while the engines sit stalled; CCL-Bench doesn't use it.
                "torch_profiler_dump_cuda_time_total": False,
            })
        # TPU worker uses jax.profiler and ignores iteration bounds; the client
        # stops it after N steps by watching the engine's step counter.
        argv += ["--profiler-config", json.dumps(prof)]

    argv += [str(x) for x in srv.get("extra_args", [])]
    return argv, env


def format_command(argv: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return (prefix + " " if prefix else "") + shlex.join(argv)


# ----------------------------------------------------------------------------
# Small HTTP helpers
# ----------------------------------------------------------------------------
def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def wait_for_health(url: str, timeout_s: float = 3600, proc=None, poll_s: float = 5.0) -> None:
    import requests
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            if requests.get(f"{url}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"server at {url} not healthy after {timeout_s:.0f}s")


def hostname() -> str:
    return socket.gethostname()


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)