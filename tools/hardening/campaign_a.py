# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section A of docs/TEST_PLAN.md on a tester environment: install the build under test from a local
catalog (A9: over the rc13 install, with state), uninstall + reinstall (A10), a host restart (A11), the
CPU-only train / predict cycle with the ETA read (A5), and the Doctor's GPU facts. Run from the env's
venv python with the services stopped:

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_a.py --root <root> --repo <repo> --ref <sha> [--skip-cpu]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402


def snapshot(root: Path) -> dict:
    state = h.plugin_state_dir(root)
    out = {"exists": state.is_dir()}
    if not state.is_dir():
        return out
    for name in ("ui_config.json", "ledger.jsonl"):
        p = state / name
        out[name] = p.stat().st_size if p.exists() else None
    out["predictions"] = sorted(p.name for p in (state / "predictions").glob("*")) if (state / "predictions").is_dir() else []
    out["kit_records"] = sorted(p.name for p in (state / "kit").glob("*")) if (state / "kit").is_dir() else []
    out["runs"] = sorted(p.name for p in (root / "home" / "projects" / "intel-scene" / "runs").glob("*")) if (root / "home" / "projects" / "intel-scene" / "runs").is_dir() else []
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", required=True, help="the git ref (sha or tag) of the build under test")
    ap.add_argument("--version", default=None, help="the manifest version at that ref (default: read from the repo's plugin.toml)")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    manifest = h.plugin_manifest(repo)
    version = args.version or manifest["version"]
    catalog = h.write_catalog(root, version, args.ref, manifest)
    results = h.Results(Path(args.results or (repo / "docs" / "campaign" / "section_a.json")))
    h.apply_env(h.env_for(root, catalog=str(catalog)))

    # ── A9: upgrade with state ────────────────────────────────────────────────
    before = snapshot(root)
    h.log(f"state before: {json.dumps(before)[:300]}")
    with h.host_client() as client:
        facts = h.install(client, version)
        meta = client.get(f"{h.API}/config").json().get("_meta") or {}
        after = snapshot(root)
        doc = client.get(f"{h.API}/status/doctor?kaggle=0").json()
        same = {k: before.get(k) for k in ("ui_config.json", "ledger.jsonl", "predictions", "kit_records", "runs")} == \
               {k: after.get(k) for k in ("ui_config.json", "ledger.jsonl", "predictions", "kit_records", "runs")}
        results.put("A9", "PASS" if facts.get("state") == "succeeded" and same and meta.get("version") == version else "FAIL",
                    install=facts, version=meta.get("version"), state_same=same, before=before, after=after,
                    import_state=(meta.get("import_state") or {}).get("state"), kit_state=(meta.get("kit_state") or {}).get("state"),
                    doctor_gpu=(doc.get("device") or {}).get("gpu"), doctor_versions=doc.get("versions"), doctor_uv=doc.get("uv"),
                    doctor_images=(doc.get("import") or {}).get("images"))
        gpu = (doc.get("device") or {}).get("gpu") or {}
        results.put("P3", "PASS" if gpu.get("driver") else "FAIL", gpu=gpu)

        # ── A10: uninstall + reinstall ────────────────────────────────────────
        r = client.delete(f"/api/admin/plugins/{h.PLUGIN_ID}")
        h.log(f"DELETE plugin -> {r.status_code} {r.text[:160]}")
        gone = h.card_of(client)
        mid = snapshot(root)
        facts2 = h.install(client, version)
        after2 = snapshot(root)
        meta2 = client.get(f"{h.API}/config").json().get("_meta") or {}
        results.put("A10", "PASS" if r.status_code in (200, 202, 204) and facts2.get("state") == "succeeded" and mid.get("kit_records") == before.get("kit_records") and after2.get("ledger.jsonl") == before.get("ledger.jsonl") else "FAIL",
                    delete_http=r.status_code, card_after_delete=bool(gone) and not gone.get("placeholder"), state_between=mid, reinstall=facts2,
                    import_state=(meta2.get("import_state") or {}).get("state"), kit_state=(meta2.get("kit_state") or {}).get("state"))

    # ── A11: host restart with the plugin installed ───────────────────────────
    t0 = time.time()
    with h.host_client() as client:
        ready_s = h.wait_ready(client, 600)
        card = h.card_of(client) or {}
        meta3 = client.get(f"{h.API}/config").json().get("_meta") or {}
        results.put("A11", "PASS" if meta3.get("version") == version and ready_s < 120 else "FAIL",
                    ready_s=ready_s, boot_s=round(time.time() - t0, 1), provisioning=card.get("provisioning"), version=meta3.get("version"))

    # ── A5: CPU-only cycle (CUDA hidden from the host and its workers) ───────
    if not args.skip_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        with h.host_client() as client:
            h.wait_ready(client, 600)
            doc = client.get(f"{h.API}/status/doctor?kaggle=0").json()
            dev = doc.get("device") or {}
            pre = client.get(f"{h.API}/train/preflight").json()
            hint = {k: pre.get(k) for k in ("estimate", "eta", "duration_hint", "stats", "device_class") if k in pre}
            t0 = time.time()
            job = h.run_and_wait(client, "train", {"epochs": "2", "run_name": "campaign a5 cpu", "device": ""}, timeout=3600)
            train_s = round(time.time() - t0, 1)
            st = client.get(f"{h.API}/train/state").json()
            cur = st.get("current") or {}
            facts = cur.get("facts") or {}
            prog = cur.get("progress") or {}
            t0 = time.time()
            pj = h.run_and_wait(client, "predict", {"train_job_id": job.get("id"), "device": ""}, timeout=3600)
            predict_s = round(time.time() - t0, 1)
            ps = client.get(f"{h.API}/submit/state").json()
            results.put("A5", "PASS" if job.get("status") == "completed" and pj.get("status") == "completed" and str(facts.get("device")) == "cpu" else "FAIL",
                        doctor_device=dev, train_status=job.get("status"), train_s=train_s, device_label=facts.get("device_label"),
                        avg_epoch_s=prog.get("avg_epoch_s"), eta_at_epoch1=prog.get("eta_s"), setup_s=(cur.get("result") or {}).get("setup_s"),
                        collect_s=(cur.get("result") or {}).get("collect_s"), preflight_hint=hint,
                        predict_status=pj.get("status"), predict_s=predict_s, predict_device=((ps.get("record") or {}).get("facts") or {}).get("device_label") or (ps.get("facts") or {}).get("device_label"),
                        predict_checks=len((ps.get("record") or {}).get("checks") or ps.get("checks") or []))
        del os.environ["CUDA_VISIBLE_DEVICES"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
