# SPDX-License-Identifier: Apache-2.0
"""Generate config.json for the fixture harness: GET /config exactly as the plugin serves it, against a scratch
plugin home (nothing in any Hub environment is read or written). The manifest is fetched from the tier
KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL names (the dev tier by default), so the fixture numbers match what the
live tab derives from the served manifest.

    uv run python tools/fixture_harness/make_config.py <scratch home> [<out dir>]

Runs in the repo's own venv (the plugin package must import). <out dir> defaults to the harness folder."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    home = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE
    os.environ["KAGGLE_CLASSIFICATION_HOME"] = str(home / "plugin-home")
    os.environ.setdefault("KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL", "https://competitions.dev.3lc.ai")
    for var in ("HOME", "USERPROFILE"):
        os.environ[var] = str(home / "home")
    (home / "home").mkdir(parents=True, exist_ok=True)
    from kaggle_classification import manifest
    from kaggle_classification.routes import config_payload

    r = manifest.resolve(network=True)
    print("resolve:", r.manifest.source, r.manifest.source_detail, "kit", r.manifest.kit.version, "warnings", list(r.warnings))
    cfg = config_payload()
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("wrote", out / "config.json", "manifest_source", cfg["_meta"]["manifest_source"], "version", cfg["_meta"]["version"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
