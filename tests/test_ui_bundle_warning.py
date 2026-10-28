# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The bundle checklist's red warning (the 2026-10-13 hand-test finding), executed: the fragment's own
``stRenderBundlePreview`` is extracted from ui.html and run in node against a served preview. A submitted
run without its checkpoint shows the served warning in a red callout before Export; a run with its
checkpoint shows none."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kaggle_classification import status

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"

STUBS = r"""
var box = { innerHTML: '', querySelectorAll: function () { return []; } };
function el() { return box; }
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
function kgIcon(name) { return '<i data-icon="' + name + '"></i>'; }
function kgWhenSpan() { return 'today'; }
function fmtCount(n) { return String(n); }
function stFmtMB(b) { return String(b) + ' B'; }
function stLoadBundlePreview() {}
var stBundleSel = null;
var stBundleError = '';
function stExportGate() {}
"""


def _render(preview: dict) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    m = re.search(r"function stRenderBundlePreview\(pv\) \{.*?\n      \}\n", html, re.S)
    assert m
    script = STUBS + m.group(0) + "\nstRenderBundlePreview(JSON.parse(process.argv[1]));\nprocess.stdout.write(box.innerHTML);\n"
    out = subprocess.run([node, "-e", script, json.dumps(preview)], capture_output=True, text=True, encoding="utf-8", check=True)
    return out.stdout


def _row(job: str, name: str, *, available: bool) -> dict:
    return {"train_job_id": job, "run_name": name, "run_folder": name, "public_score": 0.64444, "submitted_at": 1.0,
            "ref": "57116805", "submissions": 1, "checkpoint_bytes": 45 if available else None, "available": available,
            "default": True, "selected": available, "skipped_reason": "",
            "warning": "" if available else status.CHECKPOINT_MISSING_WARNING}


def test_a_missing_checkpoint_renders_the_red_warning_before_export():
    out = _render({"eligible_runs": [_row("t1", "run_a", available=False), _row("t2", "run_b", available=True)],
                   "checkpoints": {"max": 2}, "counts": {"members": 3, "checkpoints": 1}, "bytes": 100})
    callouts = re.findall(r'<div class="kg-callout err">.*?</div>', out)
    assert len(callouts) == 1
    assert "run_a (#57116805): checkpoint missing. Organizers can&#39;t verify this submission without its checkpoint." in callouts[0]
    assert 'data-icon="x-circle"' in callouts[0]
    assert out.index("kg-callout err") < out.index("kg-bundle-size")


def test_no_warning_when_every_submitted_run_has_its_checkpoint():
    out = _render({"eligible_runs": [_row("t2", "run_b", available=True)], "checkpoints": {"max": 2},
                   "counts": {"members": 3, "checkpoints": 1}, "bytes": 100})
    assert "kg-callout err" not in out
