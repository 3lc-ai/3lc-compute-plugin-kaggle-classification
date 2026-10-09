# SPDX-License-Identifier: Apache-2.0
"""Copy rules for everything a participant reads (rc12, item 4): no em dash anywhere in the plugin.

The fragment, every module's string literals (run descriptions, job subtitles, the Doctor's text, the
bundle README, every refusal and log line), the plugin manifest, the catalog descriptions, the tester
kit and its start scripts. The scan is over the source text, which is a superset of every server-generated
user-facing string: a literal that never reaches a participant still fails, so the rule has no exceptions
to argue about.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import kaggle_classification

PKG = Path(kaggle_classification.__file__).parent
REPO = PKG.parent.parent
EM_DASH = "—"

SURFACES = sorted(PKG.glob("*.py")) + [
    PKG / "plugin.toml",
    PKG / "ui" / "ui.html",
    REPO / "catalog.json",
    REPO / "TESTING.md",
    REPO / "tester" / "start_tester.ps1",
    REPO / "tester" / "start_tester.sh",
]


def _offenders(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [f"{path.name}:{i}: {line.strip()[:120]}" for i, line in enumerate(text.splitlines(), 1) if EM_DASH in line]


def test_the_fragment_has_no_em_dash():
    assert _offenders(PKG / "ui" / "ui.html") == []


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: p.name)
def test_no_em_dash_on_any_participant_surface(path: Path):
    assert path.exists(), path
    assert _offenders(path) == []


def test_the_served_strings_have_no_em_dash(tmp_path, monkeypatch):
    """The strings the server composes at runtime, not only their literals: the config payload the
    fragment renders from (every copy string the manifest and the plugin contribute), the Doctor and
    the bundle README."""
    monkeypatch.setenv("KAGGLE_CLASSIFICATION_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL", "http://127.0.0.1:9")
    import json

    from kaggle_classification import routes, status

    payload = json.dumps(routes.config_payload(), default=str)
    assert EM_DASH not in payload
    assert EM_DASH not in status.BUNDLE_README
