# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""storage.plugin_home(): the resolution order, and that HOME is the last rule, not the first."""

from __future__ import annotations

from pathlib import Path

from kaggle_classification import storage


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(storage.HOME_ENV, str(tmp_path / "x"))
    monkeypatch.setattr(storage, "_state_root", tmp_path / "worker")
    assert storage.resolve() == (tmp_path / "x", "env")


def test_worker_state_root_beats_cwd_and_home(monkeypatch, tmp_path):
    monkeypatch.delenv(storage.HOME_ENV, raising=False)
    monkeypatch.setattr(storage, "_state_root", None)
    monkeypatch.chdir(tmp_path)
    assert storage.resolve() == (Path.home() / ".3lc-kaggle-classification", "home")

    # The SDK worker creates <cwd>/.plugin-state/<id> before it serves: seen -> used.
    (tmp_path / ".plugin-state" / storage.PLUGIN_ID).mkdir(parents=True)
    assert storage.resolve() == (tmp_path / ".plugin-state" / storage.PLUGIN_ID, "cwd")

    # A job revealing ctx.state_dir.parent pins it for the process.
    storage.remember_state_root(tmp_path / "root")
    assert storage.resolve() == (tmp_path / "root", "worker")
    monkeypatch.setattr(storage, "_state_root", None)


def test_describe_names_the_rule(home):
    assert storage.describe() == {"path": str(home), "resolved_by": "env"}
