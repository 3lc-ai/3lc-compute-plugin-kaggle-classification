# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The config load's startup patience (item 3 of the 2026-09-29 re-check): which failures count as
"the worker is starting" (retry with backoff) — the fragment's ``kgStartupError`` run in node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"
CASES = [
    ({"name": "AbortError", "message": "signal is aborted without reason"}, True),   # the Hub's fetch timeout
    ({"name": "TimeoutError", "message": "The operation timed out"}, True),
    ({"name": "Error", "message": "HTTP 503", "status": 503}, True),               # the worker not up yet
    ({"name": "Error", "message": "HTTP 502", "status": 502}, True),
    ({"name": "Error", "message": "HTTP 504", "status": 504}, True),
    ({"name": "Error", "message": "HTTP 500", "status": 500}, False),              # a real failure
    ({"name": "SyntaxError", "message": "Unexpected token < in JSON"}, False),
    ({"name": "Error", "message": "config update contains retired keys"}, False),
]


def test_startup_errors_are_the_retryable_ones():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    m = re.search(r"function kgStartupError\(e\) \{.*?\n      \}\n", html, re.S)
    assert m
    script = m.group(0) + "\nprocess.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(kgStartupError)));\n"
    out = subprocess.run([node, "-e", script, json.dumps([c for c, _ in CASES])], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert json.loads(out.stdout) == [want for _, want in CASES]


def test_the_retry_budget_covers_a_worker_start():
    """Ten retries over about two minutes: longer than a cold worker's torch import on the laptop."""
    html = UI.read_text(encoding="utf-8")
    m = re.search(r"var KG_CFG_DELAYS = \[([\d, ]+)\];", html)
    assert m
    delays = [int(x) for x in m.group(1).split(",")]
    assert len(delays) == 10 and 100_000 <= sum(delays) <= 180_000 and delays == sorted(delays)
