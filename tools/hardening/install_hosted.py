# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Install a release from the HOSTED test catalog into a tester env, the way a participant does (the catalog
URL the start script carries; the Hub's Plugins > Update is the same install route), and print what the host
and the Doctor report afterwards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

root, version, catalog_url = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
h.setup_stdio()
h.apply_env(h.env_for(root, catalog=catalog_url))
with h.host_client() as client:
    facts = h.install(client, version)
    print("INSTALL", json.dumps(facts))
    if facts.get("state") == "succeeded":
        doc = h.doctor(client)
        meta = client.get(f"{h.API}/config").json().get("_meta") or {}
        print("DOCTOR", json.dumps({"plugin": doc.get("plugin"), "versions": doc.get("versions"), "gpu": (doc.get("device") or {}).get("gpu"),
                                    "kit": (doc.get("kit") or {}).get("state"), "images": {k: v.get("state") for k, v in ((doc.get("import") or {}).get("images") or {}).items()},
                                    "manifest": (doc.get("manifest") or {}).get("manifest_source"), "import": (meta.get("import_state") or {}).get("state")}))
