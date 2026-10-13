# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section A's scratch environments (docs/TEST_PLAN.md A1 stand-in, A3, A4, A8): each is a fresh tester
layout under the tester folder with its own host venv (3lc-compute 1.1.0 and the host 3lc / Python under
test), its own redirected home and project root, the API key file byte-copied from the tester home (never
read into memory as text), the plugin installed from the campaign catalog, then a smoke pass: the Doctor,
a kit download, an import, a one-epoch train and a predict.

  <tester>\\.venv\\Scripts\\python.exe tools/hardening/campaign_a2.py --root <tester> --repo <repo> --version 1.0.0rc14 --item A4 --python 3.11
  ... --item A8 --tlc 3.3.0      ... --item A8 --tlc 3.4.1      ... --item A3      ... --item A1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd)[:200], flush=True)
    return subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the tester folder (holds the key and the campaign catalog)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--item", required=True, choices=["A1", "A3", "A4", "A8", "SOAK"])
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--tlc", default=None, help="the host 3lc pin (A8); default: what 3lc-compute resolves")
    ap.add_argument("--results", default=None)
    ap.add_argument("--catalog", default=None, help="a catalog URL (default: the tester folder's newest catalog-campaign-*.json)")
    args = ap.parse_args()
    root, repo = Path(args.root), Path(args.repo)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.path.insert(0, str(HERE))
    import harness as h

    results = h.Results(Path(args.results or (repo / "docs" / "campaign" / "section_a.json")))
    tag = {"A1": "tést år ünïcode", "A3": "l" * 120, "A4": f"py{args.python}", "A8": f"tlc-{args.tlc or 'default'}", "SOAK": f"{args.version}"}[args.item]
    base = root / "campaign-scratch" / "envs"
    if args.item == "A3":
        # a tester folder whose plugin state path lands near 260 characters
        env_root = base / ("a3-" + "x" * 90) / ("y" * 60)
    else:
        env_root = base / f"{args.item}-{tag}"
    item_key = args.item if args.item != "A8" else f"A8-{args.tlc}"
    if env_root.exists():
        shutil.rmtree(env_root, ignore_errors=True)
    (env_root / "home" / "AppData" / "Local" / "3LC" / "3LC").mkdir(parents=True)
    (env_root / "home" / "AppData" / "Roaming").mkdir(parents=True)
    h.projects_dir(env_root).mkdir(parents=True)
    key_src = root / "home" / "AppData" / "Local" / "3LC" / "3LC" / "3lc_api_key"
    shutil.copyfile(key_src, env_root / "home" / "AppData" / "Local" / "3LC" / "3LC" / "3lc_api_key")
    state_path_len = len(str(h.plugin_state_dir(env_root) / "data" / "intel-scene" / "v1" / "starter_kit" / "data" / "train" / "buildings"))
    print(f"env root {env_root} (plugin state path length {state_path_len})", flush=True)

    venv = env_root / ".venv"
    r = run(["uv", "venv", "--python", args.python, str(venv)])
    if r.returncode:
        results.put(item_key, "FAIL", step="uv venv", stderr=r.stderr[-400:])
        return 1
    spec = ["3lc-compute==1.1.0"] + ([f"3lc=={args.tlc}"] if args.tlc else [])
    r = run(["uv", "pip", "install", "--python", str(venv / "Scripts" / "python.exe"), "--index-url", "https://pypi.org/simple", *spec])
    if r.returncode:
        results.put(item_key, "FAIL", step="uv pip install", stderr=r.stderr[-600:])
        return 1
    py = venv / "Scripts" / "python.exe"
    versions = run([str(py), "-c", "import sys, importlib.metadata as m; print(sys.version.split()[0], m.version('3lc'), m.version('3lc-compute'))"]).stdout.strip()
    print("host:", versions, flush=True)

    # The smoke pass runs inside the scratch venv (it needs that venv's tlc_compute): a child script.
    catalog = args.catalog or str(sorted(root.glob("catalog-campaign-*.json"))[-1])
    child = env_root / "smoke.py"
    child.write_text(SMOKE, encoding="utf-8")
    env = {**os.environ, **h.env_for(env_root, catalog=str(catalog)), "UV_CACHE_DIR": str(root / "home" / "AppData" / "Local" / "uv" / "cache"),
           "PYTHONIOENCODING": "utf-8", "CAMPAIGN_VERSION": args.version, "CAMPAIGN_HARNESS": str(HERE)}
    t0 = time.time()
    r = subprocess.run([str(py), str(child)], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    out_lines = [line for line in r.stdout.splitlines() if line.startswith("RESULT ")]
    facts = json.loads(out_lines[-1][7:]) if out_lines else {"error": "no RESULT line", "stdout_tail": r.stdout[-800:], "stderr_tail": r.stderr[-800:]}
    facts.update({"host": versions, "python_requested": args.python, "tlc_requested": args.tlc, "env_root": str(env_root), "state_path_len": state_path_len, "elapsed_s": round(time.time() - t0, 1)})
    ok = facts.get("install") == "succeeded" and facts.get("train") == "completed" and facts.get("predict") == "completed"
    results.put(item_key, "PASS" if ok else "FAIL", **facts)
    return 0 if ok else 1


SMOKE = r'''
import json, os, sys, time
sys.path.insert(0, os.environ["CAMPAIGN_HARNESS"])
import harness as h
h.setup_stdio()
version = os.environ["CAMPAIGN_VERSION"]
out = {}
try:
    with h.host_client() as client:
        facts = h.install(client, version)
        out["install"] = facts.get("state"); out["install_s"] = facts.get("install_s"); out["ready_s"] = facts.get("ready_s"); out["install_error"] = (facts.get("error") or "")[:300]
        if facts.get("state") != "succeeded":
            print("RESULT " + json.dumps(out)); sys.exit(0)
        doc = client.get(f"{h.API}/status/doctor?kaggle=0").json()
        out["doctor"] = {"version": (doc.get("plugin") or {}).get("version"), "python": (doc.get("versions") or {}).get("python"), "tlc": (doc.get("versions") or {}).get("tlc"),
                         "torch": (doc.get("versions") or {}).get("torch"), "gpu": ((doc.get("device") or {}).get("gpu") or {}).get("kernels_ok"), "home": (doc.get("plugin_home") or {}).get("path")}
        t0 = time.time(); dl = h.run_and_wait(client, "download_kit", {}, timeout=1800, quiet=True); out["download"] = dl.get("status"); out["download_s"] = round(time.time() - t0, 1); out["download_error"] = str(dl.get("error") or "")[:200]
        t0 = time.time(); imp = h.run_and_wait(client, "import", {}, timeout=1800, quiet=True); out["import"] = imp.get("status"); out["import_s"] = round(time.time() - t0, 1); out["import_error"] = str(imp.get("error") or "")[:200]
        t0 = time.time(); tr = h.run_and_wait(client, "train", {"epochs": "1", "run_name": "campaign smoke"}, timeout=3600, quiet=True); out["train"] = tr.get("status"); out["train_s"] = round(time.time() - t0, 1); out["train_error"] = str(tr.get("error") or "")[:200]
        st = client.get(f"{h.API}/train/state").json(); cur = st.get("current") or {}; out["device_label"] = (cur.get("facts") or {}).get("device_label")
        t0 = time.time(); pr = h.run_and_wait(client, "predict", {"train_job_id": tr.get("id"), "device": ""}, timeout=3600, quiet=True); out["predict"] = pr.get("status"); out["predict_s"] = round(time.time() - t0, 1); out["predict_error"] = str(pr.get("error") or "")[:200]
except Exception as exc:
    out["exception"] = f"{type(exc).__name__}: {str(exc)[:300]}"
print("RESULT " + json.dumps(out, default=str))
'''


if __name__ == "__main__":
    sys.exit(main())
