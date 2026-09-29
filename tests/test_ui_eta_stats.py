# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The duration estimate's inputs are keyed by the resolved device class (item 2 of the 2026-09-29
re-check): the fragment's ``trStatsFor`` and ``kgMedian`` are extracted from ui.html and run in
node against a mixed CPU / GPU history."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"
RUNS = [   # newest first, as the worker serves them
    {"status": "completed", "device_class": "cpu", "epoch_s_per_row": 0.045, "collect_s_per_row": 0.0105, "setup_s": 0.7},
    {"status": "completed", "device_class": "cuda", "epoch_s_per_row": 0.0061, "collect_s_per_row": 0.0031, "setup_s": 6.7},
    {"status": "cancelled", "device_class": "cpu", "epoch_s_per_row": 0.043, "collect_s": 77.0, "collect_rows": 7800},
    {"status": "completed", "device_class": "cuda", "epoch_s_per_row": 0.0069, "setup_s": 13.3},
    {"status": "failed", "device_class": "cuda", "epoch_s_per_row": 0.9},
    {"status": "completed", "epoch_s_per_row": 0.5},   # an old record without a class never enters
]
BENCH = {"cuda": {"epoch_s_per_row": 0.00673, "collect_s_per_row": 0.00285, "setup_s": 12.0},
         "cpu": {"epoch_s_per_row": 0.04231, "collect_s_per_row": 0.01049, "setup_s": 12.0}}


def _fn(html: str, name: str, arg: str) -> str:
    m = re.search(rf"function {name}\({arg}\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def test_the_estimate_inputs_are_keyed_by_device_class():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    script = (
        "var kgTrainState = {runs: " + json.dumps(RUNS) + "};\n"
        "var kgTraining = {benchmark: {per_device: " + json.dumps(BENCH) + "}};\n"
        + _fn(html, "kgMedian", "values") + _fn(html, "trStatsFor", "cls")
        + "process.stdout.write(JSON.stringify({cuda: trStatsFor('cuda'), cpu: trStatsFor('cpu'), none: trStatsFor(null), mps: trStatsFor('mps')}));\n"
    )
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    cuda, cpu, none, mps = out["cuda"], out["cpu"], out["none"], out["mps"]
    # GPU: the median of the two completed cuda runs only (0.0061, 0.0069 -> upper median 0.0069);
    # the failed run and the class-less record never enter; a missing term falls to the benchmark.
    assert cuda["source"] == "history" and cuda["perRow"] == 0.0069
    assert cuda["collectPerRow"] == 0.0031 and cuda["setup"] == 13.3
    # CPU: its own two runs (0.043, 0.045 -> 0.045); the derived collect term from collect_s / collect_rows.
    assert cpu["source"] == "history" and cpu["perRow"] == 0.045
    assert abs(cpu["collectPerRow"] - 0.0105) < 1e-9 and cpu["setup"] == 0.7
    # No class (probe pending): no estimate rather than a mixed one.
    assert none["source"] is None and none["perRow"] is None
    # A class without history and without a benchmark entry: nothing.
    assert mps["source"] is None
