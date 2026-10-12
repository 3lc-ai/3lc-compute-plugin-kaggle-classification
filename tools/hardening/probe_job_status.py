# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the host's job list says while a training runs (status and label values): a campaign diagnostic."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

root = Path(sys.argv[1])
h.setup_stdio()
catalog = sorted(root.glob("catalog-campaign-*.json"))[-1]
h.apply_env(h.env_for(root, catalog=str(catalog)))
with h.host_client() as client:
    h.wait_ready(client, 600)
    code, body = h.start(client, "train", {"epochs": "50", "run_name": "probe status"})
    jid = body.get("job_id") or body.get("id")
    seen = set()
    for _ in range(40):
        job = h.job_of(client, jid) or {}
        key = (job.get("status"), str((job.get("progress") or {}).get("label") or "")[:40], job.get("state"))
        if key not in seen:
            seen.add(key)
            print("JOB", json.dumps({k: job.get(k) for k in ("status", "state", "progress", "subtitle")}, default=str)[:300], flush=True)
        if job.get("status") in ("completed", "failed", "cancelled"):
            break
        time.sleep(1)
    h.cancel(client, jid)
    h.wait_terminal(client, jid, timeout=120, quiet=True)
