# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The in-run header's device wording (item 1 of the 2026-09-29 re-check), executed: the fragment's
own ``trDeviceLabel`` is extracted from ui.html and run in node for each case. The three sentences
are the trainer's ``device_label`` — the header must say what the log says."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"
CASES = [
    ({"device": "cuda", "device_requested": ""}, "· cuda (auto)"),
    ({"device": "cuda", "device_requested": "", "device_label": "cuda (auto)"}, "· cuda (auto)"),
    ({"device": "cpu", "device_requested": "cpu"}, "· cpu (forced in Advanced)"),
    ({"device": "cpu", "device_requested": "cpu", "device_label": "cpu (forced in Advanced)"}, "· cpu (forced in Advanced)"),
    ({"device": "cpu", "device_requested": "", "device_fallback_reason": "OutOfMemoryError: CUDA out of memory"},
     "· cpu (fallback: OutOfMemoryError: CUDA out of memory)"),
    ({"device": "cpu", "device_requested": "", "device_fallback_reason": "E: x", "device_label": "cpu (fallback: E: x)"},
     "· cpu (fallback: E: x)"),
    ({}, ""),
]


def _function_source(name: str) -> str:
    html = UI.read_text(encoding="utf-8")
    m = re.search(rf"function {name}\(facts\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def test_the_header_wording_in_each_case():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    src = _function_source("trDeviceLabel")
    script = src + "\nconst cases = JSON.parse(process.argv[1]);\nprocess.stdout.write(JSON.stringify(cases.map(trDeviceLabel)));\n"
    out = subprocess.run([node, "-e", script, json.dumps([c for c, _ in CASES])], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert json.loads(out.stdout) == [want for _, want in CASES]


def test_the_wording_is_the_trainers():
    """The fragment derives, for old records, exactly what ``trainer.device_label`` serves."""
    from kaggle_classification import trainer

    for facts, want in CASES:
        if not facts.get("device") or facts.get("device_label"):
            continue
        assert "· " + trainer.device_label(facts["device"], facts.get("device_requested"),
                                           facts.get("device_fallback_reason", "")) == want
