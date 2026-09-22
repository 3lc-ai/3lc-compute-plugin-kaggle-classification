# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Shared plumbing for the pytest layer.

Every module under test is stdlib-light at import time; the package ``__init__`` imports the
SDK only to define the plugin class. When the SDK isn't installed (a plain pytest venv), a
two-name stub lets the package import — a real installed SDK wins.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

try:
    import tlc_plugin_sdk  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("tlc_plugin_sdk")
    _stub.ComputePlugin = type("ComputePlugin", (), {})
    _stub.JobContext = type("JobContext", (), {})
    sys.modules["tlc_plugin_sdk"] = _stub

from kaggle_classification import manifest as manifest_mod
from kaggle_classification import session

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

# tools/ is dev-only and not a package; tests that exercise the kit format import it by path.
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def manifest():
    """The bundled Intel manifest, as shipped."""
    return manifest_mod.load_bundled()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated plugin home: session store, kit record and kit data all under tmp_path."""
    root = tmp_path / "home" / ".3lc-kaggle-classification"
    monkeypatch.setenv("KAGGLE_CLASSIFICATION_HOME", str(root))
    monkeypatch.setattr(manifest_mod, "_refresh", dict(manifest_mod._refresh))
    manifest_mod.reset_refresh_state()
    return root


@pytest.fixture
def store(home):
    """The session module, pointed at the isolated home."""
    return session


# ── Project-root shapes ──────────────────────────────────────────────────
# The project root is configurable, so URL parsing must work when the root does not end in
# "projects" at all. Tests that parse table URLs parametrize over these three shapes.
ROOT_SHAPES = {
    "default": "AppData/Local/3LC/3LC/projects",
    "relocated": "relocated/projects",
    "bare": "3lc-data",
}


@pytest.fixture(params=sorted(ROOT_SHAPES))
def root_shape(request):
    return request.param


def table_url(root: Path, shape: str, project: str, dataset: str, table: str, *, backslashes: bool = False) -> str:
    url = f"{(root / ROOT_SHAPES[shape]).as_posix()}/{project}/datasets/{dataset}/tables/{table}"
    return url.replace("/", "\\") if backslashes else url
