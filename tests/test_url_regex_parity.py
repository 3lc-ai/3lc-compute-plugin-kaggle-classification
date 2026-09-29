# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""The fragment's ``kgUrlSeg`` regexes are a JS mirror of ``session.URL_SEG_PATTERNS`` (one predicate
per runtime). A one-sided edit fails here, and both sides are exercised on the same URLs across the
project-root shapes so the verdicts agree, not just the text."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import ROOT_SHAPES, table_url

from kaggle_classification.session import URL_SEG_PATTERNS, url_dataset, url_project, url_table

UI = Path(__file__).resolve().parents[1] / "src" / "kaggle_classification" / "ui" / "ui.html"


def _js_patterns() -> dict[str, str]:
    html = UI.read_text(encoding="utf-8")
    start = html.index("function kgUrlSeg(url, kind)")
    body = html[start : html.index("return m ? m[1] : null;", start)]
    found = dict(re.findall(r"(project|dataset|table): /(.+?)/[,\s}]", body))
    assert set(found) == {"project", "dataset", "table"}, found
    return found


def _js_to_python(js: str) -> str:
    # In a JS regex literal "\/" is an escaped slash; Python's pattern spells the slash bare.
    return js.replace("\\/", "/")


def test_js_patterns_are_the_python_patterns():
    js = _js_patterns()
    for kind, py in URL_SEG_PATTERNS.items():
        assert _js_to_python(js[kind]) == py, kind


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run the fragment's kgUrlSeg")
@pytest.mark.parametrize("backslashes", [False, True])
def test_both_runtimes_agree_on_every_root_shape(tmp_path, backslashes):
    html = UI.read_text(encoding="utf-8")
    start = html.index("function kgUrlSeg(url, kind)")
    end = html.index("\n      }\n", start) + len("\n      }\n")
    fn = html[start:end]
    cases = [
        table_url(tmp_path, shape, "intel-scene", "intel-scene_train", "initial-2", backslashes=backslashes)
        for shape in sorted(ROOT_SHAPES)
    ] + ["not a table url", ""]
    script = (
        fn + "\nconst cases = " + __import__("json").dumps(cases) + ";\n"
        "process.stdout.write(JSON.stringify(cases.map(u => "
        "[kgUrlSeg(u, 'project'), kgUrlSeg(u, 'dataset'), kgUrlSeg(u, 'table')])));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    js = __import__("json").loads(out.stdout)
    py = [[url_project(u), url_dataset(u), url_table(u)] for u in cases]
    assert js == py
    assert py[0] == ["intel-scene", "intel-scene_train", "initial-2"]
    assert py[-1] == [None, None, None]
