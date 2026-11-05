#!/usr/bin/env python3
"""Run the repository's trace metrics against a MoE benchmark trace directory.

Example:
    python scripts/moe_group15/analyze_trace.py /path/to/trace_dir

The selected metrics cover inference timing, MoE/communication kernel time, and
common GPU utilization and memory summaries. Metrics that do not apply to the
provided artifacts are reported with their error output; the script continues
so one missing artifact does not prevent the other metrics from running.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_MAIN = REPO_ROOT / "tools" / "main.py"
DEFAULT_METRICS = [
    "ttft",
    "tpot",
    "avg_step_time",
    "moe_fraction",
    "communication_fraction",
    "total_kernel_time",
    "total_communication_time",
    "dominant_kernel_concentration",
    "mean_sm_coverage",
    "average_memory_bandwidth",
    "memory_transfer_overhead",
    "load_imbalance_ratio",
    "communication_overlap_ratio",
    "compute_comm_overlap",
]


def run_metric(trace_dir: Path, metric: str) -> dict:
    command = [
        sys.executable,
        str(TOOLS_MAIN),
        "--trace",
        str(trace_dir),
        "--metric",
        metric,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    value = None
    if result.returncode == 0:
        try:
            value = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            value = stdout
    return {
        "value": value,
        "returncode": result.returncode,
        "error": stderr or (None if result.returncode == 0 else stdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CCL-Bench tools metrics over a trace directory."
    )
    parser.add_argument("trace_dir", type=Path, help="Directory containing profiler traces and workload YAML")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric name to run (repeatable). Defaults to the MoE/inference metric set.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON results to this path instead of stdout")
    args = parser.parse_args()

    trace_dir = args.trace_dir.expanduser().resolve()
    if not trace_dir.exists():
        parser.error(f"trace directory does not exist: {trace_dir}")
    if not trace_dir.is_dir():
        parser.error(f"trace path is not a directory: {trace_dir}")

    metrics = args.metrics or DEFAULT_METRICS
    report = {
        "trace_dir": str(trace_dir),
        "metrics": {metric: run_metric(trace_dir, metric) for metric in metrics},
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
