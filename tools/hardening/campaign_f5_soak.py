# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section F5 of docs/TEST_PLAN.md: a soak of train -> predict cycles through the in-process host with the
plugin worker's memory tracked. Each cycle: one 1-epoch training run, one predict on it, then the working
set of every plugin worker process (``Get-Process``), written to the record as it goes. Stops after
``--hours`` or ``--cycles``, whichever comes first.

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_f5_soak.py --root <root> --repo <repo> --hours 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402


def worker_memory_mb() -> list[dict]:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*tlc_plugin_sdk.worker*' } | "
          "ForEach-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if ($p) { "
          "'{0} {1} {2}' -f $_.ProcessId, [math]::Round($p.WorkingSet64 / 1MB), [math]::Round($p.PrivateMemorySize64 / 1MB) } }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            rows.append({"pid": int(parts[0]), "working_set_mb": int(parts[1]), "private_mb": int(parts[2])})
    return rows


def lsq(cycles: list[dict], key: str) -> float | None:
    """Least-squares slope of the largest worker's key over cycles 2..N (cycle 1 carries the warm-up)."""
    pts = [(c["n"], max((w[key] for w in c["workers"]), default=0)) for c in cycles[1:] if c["workers"]]
    if len(pts) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return round(sum((x - mx) * (y - my) for x, y in pts) / den, 2) if den else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--cycles", type=int, default=10_000)
    ap.add_argument("--epochs", default="1")
    ap.add_argument("--out", default="section_f5.json", help="the record under docs/campaign/")
    ap.add_argument("--mode", choices=["both", "train", "predict"], default="both",
                    help="both: train + predict per cycle; train: train-only; predict: one train, then predict-only cycles")
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    out_path = repo / "docs" / "campaign" / args.out
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    h.apply_env(h.env_for(root, catalog=str(catalog)))
    record: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "hours": args.hours, "cycles": []}
    deadline = time.time() + args.hours * 3600
    with h.host_client() as client:
        h.wait_ready(client, 900)
        record["memory_before"] = worker_memory_mb()
        record["mode"] = args.mode
        basis = None
        if args.mode == "predict":   # one training run, then predict-only cycles on its checkpoint
            basis = h.run_and_wait(client, "train", {"epochs": args.epochs, "run_name": "soak basis"}, timeout=3600, quiet=True)
            record["basis_train"] = {"id": basis.get("id"), "status": basis.get("status")}
            record["memory_after_basis"] = worker_memory_mb()
        n = 0
        while time.time() < deadline and n < args.cycles:
            n += 1
            t0 = time.time()
            if args.mode in ("both", "train"):
                tr = h.run_and_wait(client, "train", {"epochs": args.epochs, "run_name": f"soak {n:04d}"}, timeout=3600, quiet=True)
            else:
                tr = {"status": "completed", "id": (basis or {}).get("id")}
            t1 = time.time()
            if args.mode in ("both", "predict"):
                pr = h.run_and_wait(client, "predict", {"train_job_id": tr.get("id"), "device": ""}, timeout=3600, quiet=True) if tr.get("id") else {"status": "skipped"}
            else:
                pr = {"status": "completed"}
            t2 = time.time()
            mem = worker_memory_mb()
            cyc = {"n": n, "at": time.strftime("%H:%M:%S"), "train": tr.get("status"), "train_s": round(t1 - t0, 1), "predict": pr.get("status"),
                   "predict_s": round(t2 - t1, 1), "workers": mem, "max_working_set_mb": max([m["working_set_mb"] for m in mem], default=0),
                   "error": str(tr.get("error") or pr.get("error") or "")[:160]}
            record["cycles"].append(cyc)
            out_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            h.log(json.dumps({k: v for k, v in cyc.items() if k != "workers"}))
            if tr.get("status") != "completed" or pr.get("status") != "completed":
                record["stopped"] = f"cycle {n}: train {tr.get('status')}, predict {pr.get('status')}"
                break
    cycles = record["cycles"]
    ws = [c["max_working_set_mb"] for c in cycles if c["max_working_set_mb"]]
    record["summary"] = {"cycles": len(cycles), "elapsed_h": round((time.time() - (deadline - args.hours * 3600)) / 3600, 2),
                         "working_set_mb_first": ws[0] if ws else None, "working_set_mb_last": ws[-1] if ws else None, "working_set_mb_max": max(ws) if ws else None,
                         "train_s_median": sorted(c["train_s"] for c in cycles)[len(cycles) // 2] if cycles else None,
                         "predict_s_median": sorted(c["predict_s"] for c in cycles)[len(cycles) // 2] if cycles else None,
                         "failures": [c["n"] for c in cycles if c["train"] != "completed" or c["predict"] != "completed"],
                         "slope_ws_mb_per_cycle": lsq(cycles, "working_set_mb"), "slope_private_mb_per_cycle": lsq(cycles, "private_mb")}
    out_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    h.log(f"SOAK {json.dumps(record['summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
