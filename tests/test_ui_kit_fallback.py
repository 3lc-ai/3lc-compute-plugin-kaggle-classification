# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Import tab's manual route when the kit download cannot complete (1.0.0), executed: the fragment's own
``kgShowKaggleFallback`` and ``stDoctorRows``' Kit fallback row are extracted from ui.html and run in node
against the served manifest. With the manifest's kit_fallback: the folder link, the zip's name and (for a
Google Drive link only) the virus-scan note; without one: the competition Discussion. The failure reason
stays as a muted line."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"

STUBS = r"""
var box = { innerHTML: '' };
function el() { return box; }
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
function kgIcon(name) { return '<i data-icon="' + name + '"></i>'; }
"""


def _fn(html: str, name: str) -> str:
    m = re.search(rf"      function {name}\([^)]*\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def _render(manifest: dict, reason: str) -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    src = "".join(_fn(html, n) for n in ("kgKaggleDiscussionUrl", "kgIsDriveUrl", "kgShowKaggleFallback"))
    script = (STUBS + "var kgManifest = JSON.parse(process.argv[1]);\n" + src +
              "kgShowKaggleFallback(process.argv[2]);\nprocess.stdout.write(box.innerHTML);\n")
    return subprocess.run([node, "-e", script, json.dumps(manifest), reason], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _served(manifest, **fallback) -> dict:
    d = manifest.to_dict()
    if fallback:
        d["kit_fallback"] = {**d["kit_fallback"], **fallback}
    return d


def test_with_a_drive_fallback_the_message_names_the_zip_and_links_the_folder(manifest):
    fb = manifest.kit_fallback
    out = _render(_served(manifest), "The download failed again.")
    assert ("<strong>Couldn’t download the starter kit.</strong> Open this folder, download " + fb.file +
            ", unzip it, and paste the path of the starter_kit folder here. If Google Drive says it can’t scan "
            "the file for viruses, click Download anyway.") in out
    assert f'href="{fb.url}"' in out and "<span>Open the folder</span>" in out
    assert 'class="kg-fallback-reason"' in out and "The download failed again.</div>" in out
    assert "/discussion" not in out and "/data" not in out


def test_a_fallback_outside_google_drive_has_no_virus_scan_note(manifest):
    out = _render(_served(manifest, url="https://example.org/kit"), "")
    assert "download " + manifest.kit_fallback.file in out and 'href="https://example.org/kit"' in out
    assert "Google Drive" not in out and "kg-fallback-reason" not in out


def test_without_a_fallback_the_message_points_at_the_competition_discussion(manifest):
    d = manifest.to_dict()
    d["kit_fallback"] = None
    out = _render(d, "The downloaded files did not pass verification.")
    assert "Ask the organizers for the starter kit in the competition Discussion on Kaggle." in out
    assert f'href="https://www.kaggle.com/competitions/{manifest.competition.slug}/discussion"' in out
    assert "Open the Discussion on Kaggle" in out and "The downloaded files did not pass verification." in out
    assert "Open the folder" not in out and "Google Drive" not in out


def test_the_doctor_row_shows_the_fallback_link(manifest):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = UI.read_text(encoding="utf-8")
    body = _fn(html, "stDoctorRows")
    stubs = ("function fmtCount(n) { return String(n); } function midTrunc(s) { return s; } function kgCleanPath(s) { return s; }"
             "function stAgoIso(s) { return s; } function kgFmtDate(s) { return s; } function stUvRow() { return {text: '', detail: ''}; }"
             "function stImagesRow() { return {text: '', detail: ''}; } function stGpuText() { return ''; } function stGpuDetail() { return ''; }")
    script = (stubs + body + "var docs = JSON.parse(process.argv[1]);\n"
              "process.stdout.write(JSON.stringify(docs.map(function (d) { return stDoctorRows(d, {}).filter(function (r) { return r[0] === 'Kit fallback'; })[0]; })));\n")
    fb = dataclasses.asdict(manifest.kit_fallback)
    rows = json.loads(subprocess.run([node, "-e", script, json.dumps([{"kit_fallback": fb}, {"kit_fallback": None}])],
                                     capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    assert rows[0][1] == fb["url"]
    assert rows[0][2] == f"{fb['file']} · {fb['bytes']} bytes · sha256 {fb['sha256']} · {fb['url']}"
    assert rows[1][1].startswith("none in the manifest") and rows[1][2] == ""
