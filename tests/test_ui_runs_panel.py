# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The previous-runs dropdown labels (item 5 of the 2026-09-29 re-check): unique names from the Run
folder, "interrupted" for the record's internal "stale" — the fragment's functions run in node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"
RUNS = [
    {"id": "a", "run_name": "g1_default", "status": "completed",
     "run_url": "C:/Users/p/AppData/Local/3LC/3LC/projects/intel-scene/runs/g1_default_0000"},
    {"id": "b", "run_name": "g1_default", "status": "completed",
     "run_url": "C:/Users/p/AppData/Local/3LC/3LC/projects/intel-scene/runs/g1_default/"},
    {"id": "c", "run_name": "g5c_restart", "status": "stale",
     "run_url": "C:\\Users\\p\\AppData\\Local\\3LC\\3LC\\projects\\intel-scene\\runs\\g5c_restart"},
    {"id": "d", "run_name": "refused-before-a-run", "status": "failed"},
]


def _fn(html: str, name: str, arg: str) -> str:
    m = re.search(rf"function {name}\({arg}\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def test_unique_names_and_interrupted_wording():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    script = (
        "function kgFmtAgo() { return ''; }\n" + _fn(html, "trStatusLabel", "status") + _fn(html, "trRunFolder", "r")
        + _fn(html, "trRunLabel", "r")
        + "process.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(trRunLabel)));\n"
    )
    out = subprocess.run([node, "-e", script, json.dumps(RUNS)], capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(out.stdout) == [
        "g1_default_0000 · completed", "g1_default · completed", "g5c_restart · interrupted",
        "refused-before-a-run · failed",
    ]
