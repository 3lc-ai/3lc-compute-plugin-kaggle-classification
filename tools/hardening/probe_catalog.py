# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the host's catalog route lists for the catalogs it was given (a campaign diagnostic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

root = Path(sys.argv[1])
catalog_url = sys.argv[2]
h.setup_stdio()
h.apply_env(h.env_for(root, catalog=catalog_url))
import os

print("TLC_COMPUTE_PLUGIN_CATALOG_URLS =", os.environ["TLC_COMPUTE_PLUGIN_CATALOG_URLS"])
with h.host_client() as client:
    r = client.get("/api/admin/plugins/catalog")
    print("status", r.status_code)
    body = r.json()
    cards = body.get("plugins") if isinstance(body, dict) else body
    for c in cards or []:
        if c.get("id") == h.PLUGIN_ID:
            print(json.dumps({k: c.get(k) for k in ("id", "version", "latest_version", "source", "installable", "compatible")}, indent=1)[:600])
            print("versions:", [(v.get("version"), str(v.get("source"))[-50:]) for v in (c.get("versions") or [])])
    if isinstance(body, dict):
        print("top-level keys", list(body)[:10], "| catalogs:", json.dumps(body.get("catalogs") or body.get("sources") or body.get("errors"))[:600])
