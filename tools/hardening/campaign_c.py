# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section C of docs/TEST_PLAN.md (Train) on a tester environment through the in-process host. Each
item records PASS / FAIL with its facts. Leaves the environment with a few extra runs (campaign c*).

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_c.py --root <root> --repo <repo>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402


def meta(client) -> dict:
    return client.get(f"{h.API}/config").json().get("_meta") or {}


def tstate(client) -> dict:
    return client.get(f"{h.API}/train/state").json()


def epoch_of(job: dict) -> int:
    """The epoch number in the host's 'Epoch k/N' label, 0 before the first one."""
    import re

    m = re.search(r"Epoch (\d+)/", str((job.get("progress") or {}).get("label") or ""))
    return int(m.group(1)) if m else 0


def label_epoch(job: dict, n: int) -> bool:
    """True once the run is at epoch n or later (a 2 s poll can skip a label on a fast GPU), or terminal."""
    return (job.get("status") == "running" and epoch_of(job) >= n) or job.get("status") in ("completed", "failed", "cancelled")


def edited_revision(train_url: str, edits: dict[int, int], name: str) -> str:
    """An EditedTable revision beside the train table (the Dashboard's shape): label edits."""
    import tlc
    from tlc._core.objects.tables.from_table.edited_table import EditedTable

    parent = tlc.Table.from_url(tlc.Url(train_url))
    runs_and_values: list = []
    for idx, value in edits.items():
        runs_and_values += [[idx], value]
    edited = EditedTable(input_table_url=parent, edits={"label": {"runs_and_values": runs_and_values}}, url=parent.url.create_sibling(name))
    edited.ensure_fully_defined()
    edited.write_to_url()
    return str(edited.url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    results = h.Results(Path(args.results or (repo / "docs" / "campaign" / "section_c.json")))
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    h.apply_env(h.env_for(root, catalog=str(catalog)))

    with h.host_client() as client:
        h.wait_ready(client, 900)
        facts = client.get(f"{h.API}/train/preflight").json()
        bounds = (facts.get("training") or facts.get("bounds") or {})
        h.log(f"preflight keys {list(facts)[:12]}")

        # ── C7: every bound on every exposed field ───────────────────────────
        try:
            cfg = meta(client)
            served = (cfg.get("training") or {}).get("bounds") or facts.get("bounds") or {}
            exposed = {}
            for key, spec in served.items():
                if isinstance(spec, (list, tuple)) and len(spec) == 2:
                    exposed[key] = (spec[0], spec[1])
                elif isinstance(spec, dict) and "min" in spec and "max" in spec:
                    exposed[key] = (spec["min"], spec["max"])
            answers = {}
            for key, (lo, hi) in exposed.items():
                whole = float(lo).is_integer() and float(hi).is_integer() and key != "lr" and key != "weight_decay"
                for tag, val in (("below", int(lo) - 1 if whole else (lo / 10 if lo else -0.01)), ("above", int(hi) + 1 if whole else hi * 10)):
                    job = h.run_and_wait(client, "train", {"epochs": "1", key: str(val), "run_name": f"campaign c7 {key} {tag}"}, timeout=300, quiet=True)
                    answers[f"{key} {tag}"] = {"status": job.get("status"), "error": str(job.get("error") or job.get("body") or "")[:140]}
            ok = bool(answers) and all(a["status"] in ("failed", "refused") and "between" in a["error"] for a in answers.values())
            results.put("C7", "PASS" if ok else ("SKIP" if not answers else "FAIL"), exposed=exposed, answers=answers)
        except Exception as exc:
            results.put("C7", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── C1: the largest batch the bound allows (OOM reach on this GPU) ───
        try:
            hi = (exposed.get("batch_size") or (None, None))[1] if "exposed" in dir() else None
            job = h.run_and_wait(client, "train", {"epochs": "1", "batch_size": str(hi or 256), "run_name": "campaign c1 batch max"}, timeout=1800, quiet=True)
            cur = (tstate(client).get("current") or {})
            f = cur.get("facts") or {}
            results.put("C1", "PASS" if job.get("status") == "completed" else "FAIL", batch=hi or 256, job_status=job.get("status"),
                        device_label=f.get("device_label"), fallback=f.get("device_fallback_reason"), error=str(job.get("error") or "")[:200],
                        note="an OOM is not reachable inside the bound on a 16 GB card; the fallback path is unit-tested and seen on rc13 (cu126)")
        except Exception as exc:
            results.put("C1", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── C2: cancel at epoch 1 ────────────────────────────────────────────
        try:
            code, body = h.start(client, "train", {"epochs": "5", "run_name": "campaign c2 cancel"})
            jid = body.get("job_id") or body.get("id")
            h.wait_terminal(client, jid, timeout=900, quiet=True, until=lambda j: label_epoch(j, 2))
            h.cancel(client, jid)
            job = h.wait_terminal(client, jid, timeout=300, quiet=True)
            cur = tstate(client).get("current") or {}
            weights = Path(str((cur.get("facts") or {}).get("weights") or ""))
            results.put("C2", "PASS" if job.get("status") == "cancelled" and cur.get("status") == "cancelled" and weights.is_file() else "FAIL",
                        job_status=job.get("status"), record_status=cur.get("status"), best_on_disk=weights.is_file(), best_epoch=(cur.get("result") or {}).get("best_epoch"))
            c2_job = jid
        except Exception as exc:
            results.put("C2", "FAIL", exception=f"{type(exc).__name__}: {exc}")
            c2_job = None

        # ── C5: two trains queued ────────────────────────────────────────────
        try:
            c1, b1 = h.start(client, "train", {"epochs": "1", "run_name": "campaign c5 first", "client_token": "c5-first"})
            c2, b2 = h.start(client, "train", {"epochs": "1", "run_name": "campaign c5 second", "client_token": "c5-second"})
            j1 = h.wait_terminal(client, b1.get("job_id") or b1.get("id"), timeout=1800, quiet=True)
            j2 = h.wait_terminal(client, b2.get("job_id") or b2.get("id"), timeout=1800, quiet=True)
            runs = client.get(f"{h.API}/runs").json()
            names = [r.get("run_name") for r in (runs if isinstance(runs, list) else runs.get("runs") or [])]
            results.put("C5", "PASS" if j1.get("status") == "completed" and j2.get("status") in ("completed", "failed") else "FAIL",
                        first=j1.get("status"), second=j2.get("status"), second_error=str(j2.get("error") or "")[:160], runs_named=[n for n in names if n and "c5" in n])
        except Exception as exc:
            results.put("C5", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── C8 + C9: epochs 1 with a unicode run name; epochs 50 cancelled at 3 with the ETA read ──
        try:
            name = "campaign c9 ünïcode 名前 run"
            job = h.run_and_wait(client, "train", {"epochs": "1", "run_name": name}, timeout=1800, quiet=True)
            cur = tstate(client).get("current") or {}
            run_url = str((cur.get("facts") or {}).get("run_url") or "")
            runs = tstate(client).get("runs") or []
            listed = any(name in json.dumps(r, ensure_ascii=False) for r in runs)
            results.put("C9", "PASS" if job.get("status") == "completed" and Path(run_url).is_dir() and listed else "FAIL",
                        job_status=job.get("status"), run_url=run_url, listed=listed)
            code, body = h.start(client, "train", {"epochs": "50", "run_name": "campaign c8 fifty"})
            jid = body.get("job_id") or body.get("id")
            h.wait_terminal(client, jid, timeout=1800, quiet=True, until=lambda j: label_epoch(j, 3))
            prog = (tstate(client).get("current") or {}).get("progress") or {}
            h.cancel(client, jid)
            done = h.wait_terminal(client, jid, timeout=300, quiet=True)
            results.put("C8", "PASS" if job.get("status") == "completed" and done.get("status") == "cancelled" and prog.get("total_epochs") == 50 else "FAIL",
                        one_epoch=job.get("status"), fifty=done.get("status"), eta_s_at_epoch2=prog.get("eta_s"), avg_epoch_s=prog.get("avg_epoch_s"), total=prog.get("total_epochs"))
        except Exception as exc:
            results.put("C8", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── C6: label edits in the Dashboard, then Use latest revision ───────
        try:
            st = meta(client).get("import_state") or {}
            train_url = ((st.get("tables") or {}).get("train") or {}).get("url")
            rev = edited_revision(train_url, {0: 1, 1: 2, 2: 3}, "campaign-c6-edited")
            job = h.run_and_wait(client, "train", {"epochs": "1", "use_latest": True, "run_name": "campaign c6 latest"}, timeout=1800, quiet=True)
            cur = tstate(client).get("current") or {}
            used = str((cur.get("facts") or {}).get("train_table_url") or "")
            results.put("C6", "PASS" if job.get("status") == "completed" and used.replace("\\", "/").rstrip("/").endswith("campaign-c6-edited") else "FAIL",
                        revision=rev, trained_on=used, job_status=job.get("status"), checks=[c.get("label") for c in (cur.get("checks") or []) if not c.get("ok")])
        except Exception as exc:
            results.put("C6", "FAIL", exception=f"{type(exc).__name__}: {exc}")

        # ── C3: the compute service killed mid-train ─────────────────────────
        try:
            code, body = h.start(client, "train", {"epochs": "50", "run_name": "campaign c3 killed"})
            jid = body.get("job_id") or body.get("id")
            h.wait_terminal(client, jid, timeout=900, quiet=True, until=lambda j: label_epoch(j, 3))
            killed = h.kill_workers()
            time.sleep(3)
            still = h.job_of(client, jid) or {}
            c3 = {"job": jid, "workers_killed": killed, "status_before_kill": str((still.get("progress") or {}).get("label") or "")}
        except Exception as exc:
            results.put("C3", "FAIL", exception=f"{type(exc).__name__}: {exc}")
            c3 = None
    if c3:
        try:
            with h.host_client() as client:
                h.wait_ready(client, 900)
                cur = tstate(client).get("current") or {}
                running = [j for j in h.jobs(client) if j.get("status") in ("running", "queued")]
                weights = Path(str((cur.get("facts") or {}).get("weights") or ""))
                results.put("C3", "PASS" if cur.get("status") == "stale" and "Interrupted" in str(cur.get("error") or "") and not running else "FAIL",
                            **c3, record_status=cur.get("status"), error=str(cur.get("error") or "")[:160], host_running_jobs=len(running), best_on_disk=weights.is_file())
                # C4 (the reload path): what a reopened browser reads right now
                results.put("C4", "PASS" if cur.get("status") == "stale" else "FAIL", note="unit: test_ui_train_reconcile; the reload path renders the record's state", record_status=cur.get("status"))
        except Exception as exc:
            results.put("C3", "FAIL", exception=f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
