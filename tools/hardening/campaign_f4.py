# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section F4 of docs/TEST_PLAN.md: real Kaggle submissions back to back, through the plugin's own Submit job
(`kind: kaggle_submit`) on the tester environment's in-process host, on Rishikesh's go-ahead (2026-10-12: at
most 5, messages tagged "campaign F4"). Records per submission: the job's outcome, Kaggle's status, ref and
reason, the read-back, and the timing from POST to terminal; never a token.

  <root>\\.venv\\Scripts\\python.exe tools/hardening/campaign_f4.py --root <root> --repo <repo> --count 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

HARD_CAP = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--count", type=int, default=HARD_CAP)
    args = ap.parse_args()
    count = min(args.count, HARD_CAP)
    h.setup_stdio()
    root, repo = Path(args.root), Path(args.repo)
    results = h.Results(repo / "docs" / "campaign" / "section_f4.json")
    catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
    h.apply_env(h.env_for(root, catalog=str(catalog)))
    with h.host_client() as client:
        h.wait_ready(client, 900)
        ss = client.get(f"{h.API}/submit/state").json()
        pred = ss.get("predict") or {}
        if ss.get("state") not in ("predicted", "submitted") or not pred.get("job_id"):
            results.put("F4", "FAIL", reason=f"no validated prediction to submit: state {ss.get('state')}")
            return 1
        conn = client.get(f"{h.API}/kaggle/connection").json()
        before = {"state": conn.get("state"), "used": conn.get("submissions_used_today"), "limit": conn.get("daily_limit")}
        h.log(f"basis: prediction {pred.get('job_id')} of run {pred.get('run_name')}; connection {before}")
        # Back to back: every POST goes out without waiting for the previous job; the host serializes them.
        posts = []
        t_all = time.time()
        for n in range(1, count + 1):
            t0 = time.time()
            code, body = h.start(client, "kaggle_submit", {"predict_job_id": pred["job_id"], "message": f"campaign F4 {n}/{count} {time.strftime('%H:%M:%S')}"})
            posts.append({"n": n, "http": code, "job_id": body.get("job_id"), "posted_at": round(t0 - t_all, 2)})
        outcomes = []
        for p in posts:
            job = h.wait_terminal(client, p["job_id"], timeout=600, quiet=True) if p["job_id"] else {"status": "refused"}
            outcomes.append({**p, "status": job.get("status"), "elapsed_s": job.get("elapsed_s"), "error": str(job.get("error") or "")[:200],
                             "finished_at": round(time.time() - t_all, 1)})
        # The ledger's view of each submission (Kaggle's status, ref, reason, the read-back), by submit job id
        # (the history route keeps one submission per prediction; all five share the prediction here).
        ledger_path = h.plugin_state_dir(root) / "ledger.jsonl"
        entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_job = {str(e.get("job_id")): e for e in entries if e.get("kind") == "submit"}
        for o in outcomes:
            s = by_job.get(str(o["job_id"])) or {}
            kg = s.get("kaggle") or {}
            o["kaggle"] = {k: kg.get(k) for k in ("status", "ref", "reason", "read_back_status", "read_back_public_score", "read_back_error_description", "submitted_at") if k in kg}
            o["message"] = s.get("message")
        # Kaggle's own list of this account's submissions (the newest ten): refs, statuses, messages.
        live = client.get(f"{h.API}/status/kaggle").json()
        kaggle_side = [s for s in (live.get("submissions") or []) if "campaign F4" in str(s.get("message") or "")]
        conn2 = client.get(f"{h.API}/kaggle/connection").json()
        after = {"state": conn2.get("state"), "used": conn2.get("submissions_used_today"), "limit": conn2.get("daily_limit")}
        submitted = [o for o in outcomes if (o.get("kaggle") or {}).get("status") == "submitted" or o["status"] == "completed"]
        results.put("F4", "PASS" if len(submitted) == count else "PARTIAL", count=count, wall_s=round(time.time() - t_all, 1),
                    connection_before=before, connection_after=after, outcomes=outcomes, kaggle_side=kaggle_side)
    return 0


if __name__ == "__main__":
    sys.exit(main())
