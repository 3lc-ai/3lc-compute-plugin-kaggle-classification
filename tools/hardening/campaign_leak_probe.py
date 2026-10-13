# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The leak hunt's step 3: N cycles of one job kind on a tester env whose service runs with the worker's
memory probe enabled (KAGGLE_CLASSIFICATION_LEAK_PROBE=1), the probe read after every cycle, the whole
series written to docs/campaign/<out>. ``--cpu`` hides CUDA from the service and its workers.

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_leak_probe.py --root <root> --repo <repo> --kind train --cycles 8 [--cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402
from campaign_f5_soak import lsq, worker_memory_mb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--kind", choices=["train", "predict"], default="train")
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    out_path = repo / "docs" / "campaign" / (args.out or f"leak_probe_{args.kind}{'_cpu' if args.cpu else ''}.json")
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    extra = {"KAGGLE_CLASSIFICATION_LEAK_PROBE": "1"}
    if args.cpu:
        extra["CUDA_VISIBLE_DEVICES"] = "-1"
    h.apply_env(h.env_for(root, catalog=str(catalog), extra=extra))
    record: dict = {"kind": args.kind, "cpu": args.cpu, "cycles": []}
    with h.host_client() as client:
        h.wait_ready(client, 900)
        basis = None
        if args.kind == "predict":
            basis = h.run_and_wait(client, "train", {"epochs": "1", "run_name": "probe basis"}, timeout=3600, quiet=True)
        else:   # one warm-up job so the first diff is not the imports
            h.run_and_wait(client, "train", {"epochs": "1", "run_name": "probe warmup"}, timeout=3600, quiet=True)
        first = client.get(f"{h.API}/debug/leak_probe?reset=1")
        if first.status_code != 200:
            record["error"] = f"probe route answered {first.status_code}: {first.text[:200]}"
            out_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            h.log(record["error"])
            return 1
        record["baseline"] = first.json()
        for n in range(1, args.cycles + 1):
            if args.kind == "train":
                job = h.run_and_wait(client, "train", {"epochs": "1", "run_name": f"probe {n:02d}"}, timeout=3600, quiet=True)
            else:
                job = h.run_and_wait(client, "predict", {"train_job_id": basis.get("id"), "device": ""}, timeout=3600, quiet=True)
            p = client.get(f"{h.API}/debug/leak_probe").json()
            mem = worker_memory_mb()
            record["cycles"].append({"n": n, "status": job.get("status"), "workers": mem, "probe": p})
            out_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
            h.log(json.dumps({"n": n, "status": job.get("status"), "process": p.get("process"), "tracemalloc_mb": p.get("tracemalloc_mb"),
                              "gc_objects": p.get("gc_objects"), "logging": p.get("logging", {}).get("named_total"),
                              "torch_live_tensors": p.get("torch", {}).get("live_tensors"), "torch_live_modules": p.get("torch", {}).get("live_modules")}))
    cyc = record["cycles"]
    record["summary"] = {
        "cycles": len(cyc),
        "private_mb_per_cycle_probe": round((cyc[-1]["probe"]["process"]["private_mb"] - cyc[0]["probe"]["process"]["private_mb"]) / max(len(cyc) - 1, 1), 2) if len(cyc) > 1 else None,
        "tracemalloc_mb_per_cycle": round((cyc[-1]["probe"]["tracemalloc_mb"] - cyc[0]["probe"]["tracemalloc_mb"]) / max(len(cyc) - 1, 1), 2) if len(cyc) > 1 else None,
        "slope_ws_mb_per_cycle": lsq([{"n": c["n"], "workers": c["workers"]} for c in [{"n": 0, "workers": []}] + cyc], "working_set_mb"),
    }
    out_path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    h.log(f"PROBE SUMMARY {json.dumps(record['summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
