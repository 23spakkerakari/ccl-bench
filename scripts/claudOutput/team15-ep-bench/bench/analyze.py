#!/usr/bin/env python3
"""Turn benchmark results into tables, Pareto frontiers and CCL-Bench trace metrics.

    python bench/analyze.py                       # uses $EPB_RESULTS_DIR, latest run per setting
    python bench/analyze.py <results_root> --all-runs
    python bench/analyze.py --trace-metrics       # also run CCL-Bench tools on every ccl/ dir

Outputs (in <results_root>/analysis/):
    summary.csv                       one row per (setting, concurrency) point
    pareto_points.csv                 points on each platform's frontier
    pareto_total.png                  user TPS vs system TPS (equal chip budget comparison)
    pareto_per_chip.png               user TPS vs system TPS / chip
    latency_vs_concurrency.png        TTFT p50/p99 and TPOT p50 per setting
    coarse_<setting>.png              KV-cache usage and running/waiting requests over time
    trace_metrics.csv                 (with --trace-metrics) CCL-Bench metrics per trace dir

Definitions
    user TPS   = 1000 / median TPOT (ms)      decode tokens/s seen by one user
    system TPS = output_throughput            generated tokens/s across the whole system
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode), fixed order.
SERIES_COLORS = {
    ("4", "allgather"): "#2a78d6",   # blue
    ("4", "alltoall"): "#eb6834",    # orange
    ("8", "allgather"): "#1baf7a",   # aqua
    ("8", "alltoall"): "#eda100",    # yellow
}
PLATFORM_STYLE = {"gpu": {"marker": "o", "linestyle": "-"}, "tpu": {"marker": "s", "linestyle": "--"}}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

GPU_TRACE_METRICS = [
    "avg_step_time", "communication_fraction", "moe_fraction", "total_communication_time",
    "total_kernel_time", "compute_comm_overlap", "load_imbalance_ratio", "dominant_kernel_concentration",
    "bandwidth_utilization_allgather_group_6", "bandwidth_utilization_alltoall_group_6",
    "collective_vol_total_gb", "ttft", "tpot",
]
TPU_TRACE_METRICS = [
    "avg_step_time", "communication_fraction", "total_communication_time", "compute_comm_overlap",
    "dominant_kernel_concentration", "comm_kernel_breakdown_tpu", "collective_vol_total_gb", "ttft", "tpot",
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_runs(root: Path, all_runs: bool) -> list[Path]:
    runs = []
    for setting_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "analysis"):
        candidates = sorted(p for p in setting_dir.iterdir() if (p / "bench").is_dir())
        runs += candidates if all_runs else candidates[-1:]
    return runs


def _per_request_user_tps(r: dict) -> list[float]:
    out = []
    for itl in r.get("itls") or []:
        if itl:
            m = statistics.mean(itl)
            if m > 0:
                out.append(1.0 / m)
    return out


def load_points(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        meta = json.loads((run / "setting.json").read_text()) if (run / "setting.json").exists() else {}
        for f in sorted((run / "bench").glob("c*.json")):
            r = json.loads(f.read_text())
            setting = meta.get("setting") or r.get("setting") or run.parent.name
            chips = int(meta.get("chips") or r.get("chips") or 0)
            utps = _per_request_user_tps(r)
            tpot = r.get("median_tpot_ms") or float("nan")
            rows.append({
                "setting": setting, "run": run.name,
                "platform": meta.get("platform") or r.get("platform") or setting.split("-")[0],
                "chips": chips, "ep_impl": meta.get("ep_impl") or r.get("ep_impl"),
                "comm_impl": meta.get("comm_impl") or r.get("comm_impl"),
                "concurrency": int(r.get("max_concurrency") or f.stem[1:]),
                "completed": r.get("completed"), "failed": r.get("failed"), "duration_s": r.get("duration"),
                "request_throughput": r.get("request_throughput"),
                "output_throughput": r.get("output_throughput"),
                "total_token_throughput": r.get("total_token_throughput"),
                "system_tps": r.get("output_throughput"),
                "system_tps_per_chip": (r.get("output_throughput") or 0) / chips if chips else float("nan"),
                "user_tps": 1000.0 / tpot if tpot and tpot > 0 else float("nan"),
                "user_tps_p10": (sorted(utps)[max(0, len(utps) // 10 - 1)] if utps else float("nan")),
                **{f"{p}_{m}_ms": r.get(f"{p}_{m}_ms")
                   for m in ("ttft", "tpot", "itl", "e2el") for p in ("median", "p90", "p99")},
                "bench_file": str(f),
            })
    return pd.DataFrame(rows).sort_values(["setting", "concurrency"]) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------
# Pareto
# ----------------------------------------------------------------------------
def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Points not dominated in (x up, y up), sorted by x."""
    d = df.dropna(subset=[x, y]).sort_values(x, ascending=False)
    best, keep = float("-inf"), []
    for idx, row in d.iterrows():
        if row[y] > best:
            keep.append(idx)
            best = row[y]
    return d.loc[keep].sort_values(x)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def _series_kwargs(row) -> dict:
    color = SERIES_COLORS.get((str(row.chips), str(row.ep_impl)), INK_MUTED)
    st = PLATFORM_STYLE.get(row.platform, PLATFORM_STYLE["gpu"])
    return dict(color=color, marker=st["marker"], linestyle=st["linestyle"], linewidth=2,
                markersize=6, markeredgecolor="white", markeredgewidth=1)


def plot_pareto(df: pd.DataFrame, y: str, ylabel: str, out: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for setting, g in df.groupby("setting"):
        g = g.sort_values("concurrency")
        ax.plot(g["user_tps"], g[y], label=setting, alpha=0.7, **_series_kwargs(g.iloc[0]))
        # selective direct labels: lowest and highest concurrency only
        for _, r in g.iloc[[0, -1]].iterrows():
            ax.annotate(f"c{r.concurrency}", (r.user_tps, r[y]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=INK_MUTED)
    fronts = []
    for plat, g in df.groupby("platform"):
        f = pareto_front(g, "user_tps", y)
        if len(f):
            ax.plot(f["user_tps"], f[y], color=INK, linewidth=1.5, alpha=0.9,
                    linestyle=PLATFORM_STYLE.get(plat, {}).get("linestyle", "-"),
                    label=f"{plat.upper()} Pareto frontier", zorder=5)
            fronts.append(f.assign(frontier=plat, y_metric=y))
    _style_axes(ax, f"Qwen3-30B-A3B on ShareGPT — {ylabel} vs per-user speed",
                "User TPS (tokens/s per user = 1000 / median TPOT)", ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return pd.concat(fronts) if fronts else pd.DataFrame()


def plot_latency(df: pd.DataFrame, out: Path) -> None:
    panels = [("median_ttft_ms", "TTFT p50 (ms)"), ("p99_ttft_ms", "TTFT p99 (ms)"),
              ("median_tpot_ms", "TPOT p50 (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, (col, label) in zip(axes, panels):
        for setting, g in df.groupby("setting"):
            g = g.sort_values("concurrency")
            ax.plot(g["concurrency"], g[col], label=setting, **_series_kwargs(g.iloc[0]))
        ax.set_xscale("log", base=2)
        _style_axes(ax, label, "max concurrency", label)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_coarse(run: Path, out: Path) -> None:
    path = run / "coarse" / "metrics.jsonl"
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        m = r["metrics"]
        kv = m.get("vllm:kv_cache_usage_perc")
        rows.append({"t": r["t"], "point": r["point"],
                     "kv": 100 * kv["sum"] / kv["n"] if kv and kv["n"] else float("nan"),
                     "running": (m.get("vllm:num_requests_running") or {}).get("sum", float("nan")),
                     "waiting": (m.get("vllm:num_requests_waiting") or {}).get("sum", float("nan"))})
    if not rows:
        return
    d = pd.DataFrame(rows)
    d["t"] -= d["t"].min()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=150, sharex=True)
    axes[0].plot(d["t"], d["kv"], color="#2a78d6", linewidth=2)
    _style_axes(axes[0], f"{run.parent.name} — KV-cache usage (mean over engines)", "", "KV cache used (%)")
    axes[1].plot(d["t"], d["running"], color="#2a78d6", linewidth=2, label="running")
    axes[1].plot(d["t"], d["waiting"], color="#eb6834", linewidth=2, label="waiting")
    _style_axes(axes[1], "Requests (summed over engines)", "seconds since first scrape", "requests")
    axes[1].legend(frameon=False, fontsize=8)
    # shade measured (non-warmup) windows and label them
    for ax in axes:
        for point, g in d[~d["point"].str.contains("warmup|done|setup")].groupby("point"):
            ax.axvspan(g["t"].min(), g["t"].max(), color=GRID, alpha=0.5, linewidth=0)
    for point, g in d[~d["point"].str.contains("warmup|done|setup")].groupby("point"):
        axes[0].annotate(point, (g["t"].min(), axes[0].get_ylim()[1]), fontsize=7, color=INK_MUTED,
                         va="top")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CCL-Bench trace metrics
# ----------------------------------------------------------------------------
def run_trace_metrics(runs: list[Path], ccl_bench: Path, timeout_s: int) -> pd.DataFrame:
    main_py = ccl_bench / "tools" / "main.py"
    if not main_py.exists():
        raise SystemExit(f"CCL-Bench tools not found at {main_py} (set EPB_CCL_BENCH_DIR)")
    rows = []
    for run in runs:
        for d in sorted(p for p in (run / "ccl").glob("*") if p.is_dir()):
            has_gpu = any(d.glob("kineto_trace_*.json"))
            has_tpu = any(d.glob("*.trace.json"))
            if not (has_gpu or has_tpu):
                continue
            row = {"setting": run.parent.name, "run": run.name, "trace_dir": d.name,
                   "concurrency": int(d.name.split("-c")[-1].split("-")[0]) if "-c" in d.name else None}
            for metric in (GPU_TRACE_METRICS if has_gpu else TPU_TRACE_METRICS):
                try:
                    p = subprocess.run([sys.executable, str(main_py), "--trace", str(d), "--metric", metric],
                                       capture_output=True, text=True, timeout=timeout_s)
                    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
                    row[metric] = lines[-1] if p.returncode == 0 and lines else f"ERR: {p.stderr.strip()[-200:]}"
                except subprocess.TimeoutExpired:
                    row[metric] = "ERR: timeout"
                print(f"  {d.name} {metric}: {row[metric]}")
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", nargs="?", default=os.environ.get("EPB_RESULTS_DIR", "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-runs", action="store_true", help="include every run, not just the latest per setting")
    ap.add_argument("--trace-metrics", action="store_true", help="run CCL-Bench metric tools on ccl/ dirs")
    ap.add_argument("--ccl-bench-dir", default=os.environ.get("EPB_CCL_BENCH_DIR", "third_party/ccl-bench"))
    ap.add_argument("--metric-timeout", type=int, default=900)
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    runs = find_runs(root, args.all_runs)
    df = load_points(runs)
    if df.empty:
        print(f"No bench/c*.json results under {root}")
        return 1

    df.to_csv(out / "summary.csv", index=False)
    fronts = [plot_pareto(df, "system_tps", "System TPS (output tokens/s, all chips)", out / "pareto_total.png"),
              plot_pareto(df, "system_tps_per_chip", "System TPS per chip (output tokens/s/chip)",
                          out / "pareto_per_chip.png")]
    pd.concat([f for f in fronts if len(f)]).to_csv(out / "pareto_points.csv", index=False)
    plot_latency(df, out / "latency_vs_concurrency.png")
    for run in runs:
        plot_coarse(run, out / f"coarse_{run.parent.name}.png")

    cols = ["setting", "concurrency", "user_tps", "system_tps", "system_tps_per_chip",
            "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "failed"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    if args.trace_metrics:
        tm = run_trace_metrics(runs, Path(args.ccl_bench_dir), args.metric_timeout)
        tm.to_csv(out / "trace_metrics.csv", index=False)
    print(f"\n[analyze] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
