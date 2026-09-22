# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-timm (tests/test_release_version.py, Apache-2.0, Copyright 2026 3LC Inc.).
"""A staged plugin must advertise the same version as its installable wheel."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_version.py"
STAGED = "0.1.0.20260922090000.42.1"
DIST = "3lc-compute-plugin-kaggle-classification"


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    for source in (ROOT / "src").glob("*/plugin.toml"):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
    return tmp_path


def run_version(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args], capture_output=True, text=True)


def test_sources_agree_as_committed() -> None:
    result = run_version(ROOT)
    assert result.returncode == 0, result.stderr
    assert "agree" in result.stdout


def test_stamp_advances_every_plugin(release_tree: Path) -> None:
    result = run_version(release_tree, "--stamp", STAGED)
    assert result.returncode == 0, result.stderr
    manifests = list((release_tree / "src").glob("*/plugin.toml"))
    assert len(manifests) == 1
    for path in [release_tree / "pyproject.toml", *manifests]:
        assert f'version = "{STAGED}"' in path.read_text()


def test_drift_refuses_to_stamp_any_file(release_tree: Path) -> None:
    manifest = next((release_tree / "src").glob("*/plugin.toml"))
    manifest.write_text(
        re.sub(r'^version = "[^"]+"', 'version = "9.0.0"', manifest.read_text(), count=1, flags=re.MULTILINE)
    )
    before = {path: path.read_bytes() for path in release_tree.rglob("*.toml")}
    result = run_version(release_tree, "--stamp", STAGED)
    assert result.returncode != 0
    assert "expected version" in result.stderr
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("broken_manifest", ["", "wrong-version", "missing"])
def test_validate_packaged_manifests(release_tree: Path, broken_manifest: str) -> None:
    assert run_version(release_tree, "--stamp", STAGED).returncode == 0
    wheel = release_tree / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(f"{DIST.replace('-', '_')}.dist-info/METADATA", f"Name: {DIST}\nVersion: {STAGED}\n")
        for i, path in enumerate((release_tree / "src").glob("*/plugin.toml")):
            if i == 0 and broken_manifest == "missing":
                continue
            text = path.read_text()
            if i == 0 and broken_manifest == "wrong-version":
                text = text.replace(STAGED, "0.1.0", 1)
            archive.writestr(str(path.relative_to(release_tree / "src")).replace("\\", "/"), text)
    result = run_version(release_tree, "--wheel", str(wheel))
    assert (result.returncode == 0) == (not broken_manifest), result.stderr
