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
    runs_dir = h.projects_dir(root) / "intel-scene" / "runs"
    out["runs"] = sorted(p.name for p in runs_dir.glob("*")) if runs_dir.is_dir() else []
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
        doc = h.doctor(client)
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
        code = h.uninstall(client)
        h.log(f"DELETE plugin (install path) -> {code}")

        class _R:   # the facts the rest of this block reads
            status_code = code
            text = ""

        r = _R()
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
        # "-1", not "": on Windows an empty value deletes the variable instead of setting it.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with h.host_client() as client:
            h.wait_ready(client, 600)
            doc = h.doctor(client)
            dev = doc.get("device") or {}
            t0 = time.time()
            code, body = h.start(client, "train", {"epochs": "3", "run_name": "campaign a5 cpu", "device": ""})
            jid = body.get("job_id") or body.get("id")
            # The ETA read at the first epoch boundary, compared with the actual time to completion.
            at1 = h.wait_terminal(client, jid, timeout=3600, quiet=True,
                                  until=lambda j: "Epoch 2/" in str((j.get("progress") or {}).get("label") or "") or j.get("status") != "running")
            t_eta = time.time()
            prog1 = (client.get(f"{h.API}/train/state").json().get("current") or {}).get("progress") or {}
            job = h.wait_terminal(client, jid, timeout=3600, quiet=True)
            job["id"] = jid
            actual_remaining = round(time.time() - t_eta, 1)
            train_s = round(time.time() - t0, 1)
            st = client.get(f"{h.API}/train/state").json()
            cur = st.get("current") or {}
            facts = cur.get("facts") or {}
            res = cur.get("result") or {}
            t0 = time.time()
            pj = h.run_and_wait(client, "predict", {"train_job_id": jid, "device": ""}, timeout=3600, quiet=True)
            predict_s = round(time.time() - t0, 1)
            ss = client.get(f"{h.API}/submit/state").json()
            eta1 = prog1.get("eta_s")
            ratio = round(eta1 / actual_remaining, 2) if isinstance(eta1, (int, float)) and eta1 and actual_remaining else None
            results.put("A5", "PASS" if job.get("status") == "completed" and pj.get("status") == "completed" and str(facts.get("device")) == "cpu" else "FAIL",
                        doctor_device=dev, train_status=job.get("status"), train_s=train_s, device_label=facts.get("device_label"),
                        eta_s_at_epoch1=eta1, avg_epoch_s_at_epoch1=prog1.get("avg_epoch_s"), actual_remaining_s=actual_remaining, eta_over_actual=ratio,
                        setup_s=res.get("setup_s"), collect_s=res.get("collect_s"), epoch_s_per_row=res.get("epoch_s_per_row"),
                        predict_status=pj.get("status"), predict_s=predict_s, predict_device=ss.get("device_label"),
                        predict_checks=sum(1 for c in (ss.get("checks") or []) if c.get("ok")), predict_checks_total=len(ss.get("checks") or []))
        del os.environ["CUDA_VISIBLE_DEVICES"]
    return 0


if __name__ == "__main__":
    sys.exit(main())
