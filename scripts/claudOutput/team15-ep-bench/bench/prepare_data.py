#!/usr/bin/env python3
"""Download Qwen3-30B-A3B weights and the ShareGPT V3 dataset (run on a login node).

Writes <data-dir>/sharegpt_v3.json in the layout `vllm bench serve --dataset-name
sharegpt` expects: a JSON list of {"id", "conversations": [{"from", "value"}, ...]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import load_config  # noqa: E402


def _load_records(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def _normalize(rec: dict) -> dict | None:
    conv = rec.get("conversations")
    if conv is None:
        return None
    conv = list(conv)  # parquet gives numpy arrays
    turns = []
    for t in conv:
        t = dict(t)
        value = t.get("value") or t.get("text") or t.get("markdown")
        if value:
            turns.append({"from": t.get("from", ""), "value": value})
    if len(turns) < 2:
        return None
    return {"id": rec.get("id", ""), "conversations": turns}


def prepare_sharegpt(repo: str, out: Path) -> None:
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id=repo, repo_type="dataset",
                                   allow_patterns=["*.json", "*.jsonl", "*.parquet"]))
    files = sorted(p for p in local.rglob("*") if p.suffix in (".json", ".jsonl", ".parquet")
                   and p.name not in ("dataset_infos.json", "dataset_info.json"))
    if not files:
        raise SystemExit(f"No data files found in {local}")
    records = []
    for p in files:
        before = len(records)
        records += [r for r in map(_normalize, _load_records(p)) if r]
        print(f"[prepare_data] {p.relative_to(local)}: +{len(records) - before} conversations")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[prepare_data] wrote {len(records)} conversations -> {out}")


def prepare_model(model_id: str) -> None:
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=model_id,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    print(f"[prepare_data] model {model_id} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not args.skip_dataset:
        prepare_sharegpt(cfg["dataset"]["hf_repo"], Path(args.data_dir) / "sharegpt_v3.json")
    if not args.skip_model:
        prepare_model(cfg["model"]["id"])


if __name__ == "__main__":
    main()
