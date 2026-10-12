# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Session 14: the Train tab reconciles with the job record. A missed completion event must not leave
"Training…" on screen: the fragment's ``trReconcileFromState`` (run in node) picks the terminal record
for the shown job out of a ``/train/state`` answer, and nothing else; the tab wires it to a poll
(``TR_RECONCILE_S``), to tab entry and to the reload path; the ``train-state3-reconciled`` fixture
renders the outcome."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"

COMPLETED = {"id": "j1", "status": "completed", "facts": {"run_url": "C:/p/runs/a"}, "result": {"best_epoch": 9}}
RUNNING = {"id": "j1", "status": "running", "progress": {"epoch": 9, "total_epochs": 10}}
STATES = [
    ("current completed", {"current": COMPLETED, "runs": [COMPLETED]}, "j1"),
    ("current still running", {"current": RUNNING, "runs": []}, "j1"),
    ("current is another job, ours completed in runs", {"current": {"id": "j2", "status": "running"}, "runs": [COMPLETED]}, "j1"),
    ("current is another job, ours absent", {"current": {"id": "j2", "status": "completed"}, "runs": []}, "j1"),
    ("failed record", {"current": {"id": "j1", "status": "failed", "error": "boom"}, "runs": []}, "j1"),
    ("queued is not terminal", {"current": {"id": "j1", "status": "queued"}, "runs": []}, "j1"),
    ("no state", None, "j1"),
    ("no id", {"current": COMPLETED, "runs": []}, ""),
]


def _fn(html: str, name: str, arg: str) -> str:
    m = re.search(rf"function {name}\({arg}\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def test_the_terminal_record_for_the_shown_job_and_nothing_else():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    script = (
        _fn(html, "trReconcileFromState", "st, id")
        + "process.stdout.write(JSON.stringify(JSON.parse(process.argv[1]).map(function (c) { return trReconcileFromState(c[1], c[2]); })));\n"
    )
    out = subprocess.run([node, "-e", script, json.dumps(STATES)], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(out.stdout)
    assert got[0] == COMPLETED                      # the current record, completed
    assert got[1] is None                           # still running: nothing to reconcile
    assert got[2] == COMPLETED                      # found among the runs when current is another job
    assert got[3] is None                           # absent: never guessed from another job's record
    assert got[4]["status"] == "failed" and got[4]["error"] == "boom"
    assert got[5] is None and got[6] is None and got[7] is None


def test_the_tab_wires_the_reconcile_to_the_poll_the_entry_and_the_reload_path():
    html = UI.read_text(encoding="utf-8")
    assert "var TR_RECONCILE_S = 15;" in html
    assert "trReconcileTimer = setInterval(trReconcile, TR_RECONCILE_S * 1000);" in html
    # The start and the reload path arm the poll; the terminal path disarms it; tab entry checks once.
    assert html.count("trStartReconcile();") == 2
    assert "trStopReconcile();\n        trRunning = false;" in html
    assert "trReconcile();               // session 14" in html
    # The reload path renders a settled record instead of the host's stale running listing.
    assert "var settled = running ? trReconcileFromState(kgTrainState, running.id) : null;" in html
    assert "'train-state3-reconciled'" in html
