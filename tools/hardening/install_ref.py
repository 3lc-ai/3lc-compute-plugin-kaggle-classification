# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Install a git ref of the plugin into a tester env from a local one-entry catalog (the campaign's
build-under-test install), then print what the host reports and Kaggle's own submission list."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

root, repo, ref = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
h.setup_stdio()
manifest = h.plugin_manifest(repo)
for old in root.glob("catalog-campaign-*.json"):
    old.unlink()
catalog = h.write_catalog(root, manifest["version"], ref, manifest)
h.apply_env(h.env_for(root, catalog=str(catalog)))
with h.host_client() as client:
    facts = h.install(client, manifest["version"])
    print("INSTALL", json.dumps(facts))
    live = client.get(f"{h.API}/status/kaggle").json()
    print("KAGGLE", json.dumps({"connected": live.get("connected"), "submissions": (live.get("submissions") or [])[:7], "error": live.get("submissions_error")})[:1500])
