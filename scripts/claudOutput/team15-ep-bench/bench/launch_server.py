#!/usr/bin/env python3
"""Launch `vllm serve` for one experiment setting.

Single node (4 GPUs, or any TPU setting):
    python bench/launch_server.py --setting gpu-ep4-allgather --run-id myrun

Two-node GPU EP8 (normally done for you by slurm/perlmutter_ep.sbatch):
    node 0: python bench/launch_server.py --setting gpu-ep8-allgather --run-id R --node-rank 0 --head-addr <node0-ip>
    node 1: python bench/launch_server.py --setting gpu-ep8-allgather --run-id R --node-rank 1 --head-addr <node0-ip>

Server stdout/stderr is mirrored to <run-dir>/logs/server_node<rank>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import (RunPaths, build_server_command, format_command, get_setting,  # noqa: E402
                          load_config, new_run_id, run_dir_for)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", default=None, help="shared with run_benchmark.py so traces land in one dir")
    ap.add_argument("--run-dir", default=None, help="explicit run directory (overrides --run-id)")
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--head-addr", default=None, help="IP of node 0 (multi-node GPU only)")
    ap.add_argument("--dp-rpc-port", type=int, default=13345)
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setting = get_setting(cfg, args.setting)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_for(setting, args.run_id or new_run_id())
    paths = RunPaths(run_dir)

    argv, extra_env = build_server_command(setting, paths, node_rank=args.node_rank,
                                           head_addr=args.head_addr, dp_rpc_port=args.dp_rpc_port)
    cmd_str = format_command(argv, extra_env)
    if args.dry_run:
        print(cmd_str)
        return 0

    paths.make()
    (paths.logs / f"server_cmd_node{args.node_rank}.txt").write_text(cmd_str + "\n")
    if args.node_rank == 0:
        (paths.root / "setting.json").write_text(json.dumps({
            "setting": setting.name, "platform": setting.platform, "chips": setting.chips,
            "nodes": setting.nodes, "ep_impl": setting.ep_impl,
            "comm_impl": setting.comm_impl_label, "layout": setting.parallel_layout,
            "server_env": extra_env, "server_argv": argv,
        }, indent=2))

    log_path = paths.logs / f"server_node{args.node_rank}.log"
    print(f"[launch_server] {cmd_str}\n[launch_server] log -> {log_path}", flush=True)
    env = {**os.environ, **extra_env}
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def _forward(signum, _frame):
            proc.send_signal(signum)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _forward)

        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            log.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
