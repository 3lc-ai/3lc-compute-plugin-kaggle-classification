# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sections D and E of docs/TEST_PLAN.md (Predict and Submit, Status and bundle) on a tester environment
through the in-process host. No Kaggle submission is made (F4 and the submit-side items that need a
real upload are unit-covered or wait for a go-ahead). The Kaggle token file is backed up byte for byte
before the Connect tests and restored after; it is never printed.

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_de.py --root <root> --repo <repo>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

TOKEN_SHAPE = re.compile(rb"KGAT_[A-Za-z0-9_\-]{16,}")


def tstate(client) -> dict:
    return client.get(f"{h.API}/train/state").json()


def runs_of(client) -> list[dict]:
    r = client.get(f"{h.API}/runs").json()
    return r if isinstance(r, list) else (r.get("runs") or [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    results = h.Results(Path(args.results or (repo / "docs" / "campaign" / "section_de.json")))
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    h.apply_env(h.env_for(root, catalog=str(catalog)))
    scratch = root / "campaign-scratch"
    scratch.mkdir(exist_ok=True)
    state_dir = h.plugin_state_dir(root)

    with h.host_client() as client:
        h.wait_ready(client, 900)
        runs = (tstate(client).get("runs") or [])
        by_status: dict[str, dict] = {}
        for r in runs:
            by_status.setdefault(str(r.get("status")), r)
        h.log(f"runs: {[(r.get('run_name'), r.get('status')) for r in runs][:12]}")

        # ── D1: predict on a cancelled, a failed and a running run ───────────
        try:
            answers = {}
            for status in ("cancelled", "stale", "failed"):
                r = by_status.get(status)
                if not r:
                    answers[status] = {"status": "no such run"}
                    continue
                job = h.run_and_wait(client, "predict", {"train_job_id": r.get("id"), "device": ""}, timeout=1800, quiet=True)
                answers[status] = {"status": job.get("status"), "error": str(job.get("error") or job.get("body") or "")[:160], "run": r.get("run_name")}
            code, body = h.start(client, "train", {"epochs": "50", "run_name": "campaign d1 running"})
            jid = body.get("job_id") or body.get("id")
            h.wait_terminal(client, jid, timeout=900, quiet=True,
                            until=lambda j: (j.get("status") == "running" and "Epoch " in str((j.get("progress") or {}).get("label") or ""))
                            or j.get("status") in ("completed", "failed", "cancelled"))
            picker = next((r for r in runs_of(client) if r.get("id") == jid), None) or {}
            answers["running"] = {"picker_usable": picker.get("usable"), "picker_reason": str(picker.get("reason") or picker.get("note") or "")[:120],
                                  "note": "the GPU queue serializes jobs: a predict posted now runs after the train; the picker and the job gate refuse a running run"}
            h.cancel(client, jid)
            h.wait_terminal(client, jid, timeout=1800, quiet=True)
            ok = (answers.get("cancelled", {}).get("status") in ("completed", "no such run")
                  and answers.get("stale", {}).get("status") in ("failed", "refused", "no such run")
                  and answers.get("failed", {}).get("status") in ("failed", "refused", "no such run")
                  and answers["running"]["picker_usable"] is False)
            results.put("D1", "PASS" if ok else "FAIL", answers=answers)
        except Exception as exc:
            results.put("D1", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── D2: checkpoint modified on disk ──────────────────────────────────
        try:
            cur = tstate(client).get("current") or {}
            weights = Path(str((cur.get("facts") or {}).get("weights") or ""))
            backup = scratch / "d2-best.pt"
            shutil.copy2(weights, backup)
            with weights.open("ab") as f:
                f.write(b"\x00tamper")
            job = h.run_and_wait(client, "predict", {"train_job_id": cur.get("id"), "device": ""}, timeout=600, quiet=True)
            shutil.copy2(backup, weights)
            ok_job = h.run_and_wait(client, "predict", {"train_job_id": cur.get("id"), "device": ""}, timeout=1800, quiet=True)
            results.put("D2", "PASS" if job.get("status") in ("failed", "refused") and "sha256" in str(job.get("error") or "").lower() and ok_job.get("status") == "completed" else "FAIL",
                        tampered=job.get("status"), error=str(job.get("error") or "")[:200], restored=ok_job.get("status"), device=client.get(f"{h.API}/submit/state").json().get("device_label"))
        except Exception as exc:
            results.put("D2", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── D3: submission CSV edited by hand ────────────────────────────────
        try:
            ss = client.get(f"{h.API}/submit/state").json()
            pred = ss.get("predict") or {}
            csv_path = Path(str(pred.get("csv_path") or ""))
            backup = scratch / "d3-submission.csv"
            shutil.copy2(csv_path, backup)
            text = csv_path.read_text(encoding="utf-8").splitlines()
            text[1] = text[1].rsplit(",", 1)[0] + ",0.999999"
            csv_path.write_text("\n".join(text) + "\n", encoding="utf-8")
            ss2 = client.get(f"{h.API}/submit/state").json()
            pf = client.get(f"{h.API}/predict/preflight").json()
            shutil.copy2(backup, csv_path)
            ss3 = client.get(f"{h.API}/submit/state").json()
            changed = json.dumps(ss2) != json.dumps(ss)
            results.put("D3", "PASS" if changed and (ss2.get("state") != ss.get("state") or "csv" in json.dumps(ss2).lower()) else "FAIL",
                        state_before=ss.get("state"), state_edited=ss2.get("state"), state_restored=ss3.get("state"),
                        edited_keys=[k for k in ss2 if ss2.get(k) != ss.get(k)][:8], note="the Submit job itself is not run (it would upload); unit: test_missing_or_changed_csv_is_empty_despite_the_record")
        except Exception as exc:
            results.put("D3", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── D4: bad-format, random and (restored) real tokens through Connect ─
        try:
            token_file = root / "home" / ".kaggle" / "access_token"
            backup = scratch / "d4-token.bak"
            had = token_file.exists()
            if had:
                shutil.copy2(token_file, backup)
            answers = {}
            r = client.post(f"{h.API}/kaggle/connect", json={"token": "not-a-token"})
            answers["bad_format"] = {"http": r.status_code, "ok": r.json().get("ok"), "kind": r.json().get("kind"), "file_unchanged": token_file.exists() and (not had or token_file.read_bytes() == backup.read_bytes())}
            r = client.post(f"{h.API}/kaggle/connect", json={"token": "KGAT_" + "x" * 40})
            answers["random"] = {"http": r.status_code, "ok": r.json().get("ok"), "kind": r.json().get("kind"), "restored_previous": r.json().get("restored_previous"),
                                 "previous_kept": token_file.exists() and (not had or token_file.read_bytes() == backup.read_bytes())}
            if had:
                token_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, token_file)
            conn = client.get(f"{h.API}/kaggle/connection").json()
            answers["restored"] = {"state": conn.get("state"), "username": conn.get("username")}
            for v in answers.values():
                assert not TOKEN_SHAPE.search(json.dumps(v).encode()), "a token shape leaked into a response"
            ok = (answers["bad_format"]["ok"] is False and answers["bad_format"]["kind"] == "format" and answers["bad_format"]["file_unchanged"]
                  and answers["random"]["ok"] is False and answers["random"]["kind"] == "rejected" and answers["random"]["previous_kept"]
                  and (not had or answers["restored"]["state"] == "ready"))
            results.put("D4", "PASS" if ok else "FAIL", answers=answers)
        except Exception as exc:
            results.put("D4", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── D7 / D8: the connection card and the cap as served ───────────────
        try:
            conn = client.get(f"{h.API}/kaggle/connection").json()
            results.put("D7D8", "PASS" if conn.get("state") in ("ready", "not_joined", "no_credentials") else "FAIL",
                        state=conn.get("state"), daily_limit=conn.get("daily_limit"), used=conn.get("submissions_used_today"),
                        note="unit: not-joined (403) and the limit are simulated through the fake client; a cap of 100 is not reachable by hand")
        except Exception as exc:
            results.put("D7D8", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── E1 / E2 / E4: the bundle under the redirected home, no token shape inside ──
        try:
            prev = client.get(f"{h.API}/status/bundle/preview").json()
            r = client.get(f"{h.API}/status/bundle")
            bundles = sorted((state_dir / "bundles").glob("*.zip"), key=lambda p: p.stat().st_mtime) if (state_dir / "bundles").is_dir() else []
            zpath = bundles[-1] if bundles and r.status_code == 200 else Path("")
            inside = zpath.is_file() and str(zpath.resolve()).startswith(str(state_dir.resolve())) and len(r.content) == zpath.stat().st_size
            hits = []
            names = []
            if zpath.is_file():
                with zipfile.ZipFile(zpath) as z:
                    names = z.namelist()
                    for n in names:
                        if TOKEN_SHAPE.search(z.read(n)):
                            hits.append(n)
            results.put("E1E2", "PASS" if zpath.is_file() and inside else "FAIL", preview={k: prev.get(k) for k in ("submitted_runs", "checkpoints", "members", "runs") if k in prev},
                        zip=str(zpath), under_plugin_home=inside, members=len(names), members_sample=names[:8])
            results.put("E4", "PASS" if zpath.is_file() and not hits else "FAIL", token_hits=hits)
        except Exception as exc:
            results.put("E1E2", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── E3: the Doctor now (after every state above) ─────────────────────
        try:
            doc = client.get(f"{h.API}/status/doctor?kaggle=1").json()
            results.put("E3", "PASS" if doc.get("plugin") and "error" not in doc else "FAIL",
                        plugin=doc.get("plugin", {}).get("version"), kit=(doc.get("kit") or {}).get("state"), images={k: v.get("state") for k, v in ((doc.get("import") or {}).get("images") or {}).items()},
                        gpu=(doc.get("device") or {}).get("gpu"), kaggle=(doc.get("kaggle") or {}).get("state"), uv=(doc.get("uv") or {}).get("host_venv"))
        except Exception as exc:
            results.put("E3", "FAIL", exception=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
