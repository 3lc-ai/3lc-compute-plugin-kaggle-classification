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


# ── Session 5 (1.0.0rc3): the compute-home rule and the carry-forward on a plugin update ──────────


def _old_state(version_dir: Path, *, kit_name: str = "starter_kit") -> Path:
    """What a catalog install's worker leaves behind: <version>/.plugin-state/<id> with a session store
    whose paths point into itself, a kit record, a ledger line, the kit data and a prediction CSV."""
    import json

    state = version_dir / storage.WORKER_STATE_DIR_NAME / storage.PLUGIN_ID
    kit_dir = state / "data" / "comp" / "v1" / kit_name
    (kit_dir / "data" / "train" / "a").mkdir(parents=True)
    (kit_dir / "data" / "train" / "a" / "x.jpg").write_bytes(b"\xff\xd8")
    csv = state / "predictions" / "run1" / "submission_1.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("image_id,prediction,confidence\n", encoding="utf-8")
    (state / "kit").mkdir()
    (state / "kit" / "comp.json").write_text(json.dumps({"kit_dir": str(kit_dir), "dest_dir": str(kit_dir.parent)}),
                                             encoding="utf-8")
    (state / "ui_config.json").write_text(json.dumps({
        "session": {"kit_dir": str(kit_dir)},
        "import_state": {"kit_dir": str(kit_dir), "tables": {"train": {"url": "C:/root/p/datasets/d/tables/initial"}}},
        "predict_state": {"id": "p1", "facts": {"csv_path": str(csv)}},
    }, indent=1), encoding="utf-8")
    line = json.dumps({"kind": "predict", "job_id": "p1", "csv": {"path": str(csv)}}) + "\n"
    (state / "ledger.jsonl").write_text(line, encoding="utf-8")
    (state / "numba-cache").mkdir()
    (state / "numba-cache" / "blob").write_bytes(b"x")
    return state


def test_update_keeps_the_state_the_compute_home_rule_and_the_carry_forward(monkeypatch, tmp_path):
    import json

    monkeypatch.delenv(storage.HOME_ENV, raising=False)
    storage._reset_for_tests()
    compute_home = tmp_path / "home" / ".3lc-compute"
    managed = compute_home / storage.MANAGED_DIR_NAME / storage.PLUGIN_ID
    older = _old_state(managed / "0.1.0")
    newer = _old_state(managed / "1.0.0rc2")
    # The newest session store is the one carried (mtime order, not version order).
    import os
    import time

    os.utime(older / "ui_config.json", (time.time() - 1000, time.time() - 1000))
    new_version = managed / "1.0.0rc3"
    (new_version / ".venv").mkdir(parents=True)
    monkeypatch.chdir(new_version)

    home, rule = storage.resolve()
    assert rule == "compute-home"
    assert home == compute_home / storage.SHARED_STATE_DIR_NAME / storage.PLUGIN_ID
    # Carried: the session store, the kit record, the ledger, the kit data, the CSV; not the numba cache.
    assert (home / "ui_config.json").is_file() and (home / "ledger.jsonl").is_file()
    assert (home / "data" / "comp" / "v1" / "starter_kit" / "data" / "train" / "a" / "x.jpg").is_file()
    assert (home / "predictions" / "run1" / "submission_1.csv").is_file()
    assert not (home / "numba-cache").exists()
    # The paths inside the records now point at the new home, and nothing outside it was rewritten.
    cfg = json.loads((home / "ui_config.json").read_text(encoding="utf-8"))
    new_kit = str(home / "data" / "comp" / "v1" / "starter_kit")
    assert cfg["session"]["kit_dir"] == new_kit and cfg["import_state"]["kit_dir"] == new_kit
    assert cfg["import_state"]["tables"]["train"]["url"] == "C:/root/p/datasets/d/tables/initial"
    assert cfg["predict_state"]["facts"]["csv_path"] == str(home / "predictions" / "run1" / "submission_1.csv")
    assert json.loads((home / "kit" / "comp.json").read_text(encoding="utf-8"))["kit_dir"] == new_kit
    assert str(home) in (home / "ledger.jsonl").read_text(encoding="utf-8").replace("\\\\", "\\")
    # The source copy is untouched (never deleted, never rewritten).
    old_kit = str(newer / "data" / "comp" / "v1" / "starter_kit")
    assert json.loads((newer / "ui_config.json").read_text(encoding="utf-8"))["session"]["kit_dir"] == old_kit
    assert (newer / "numba-cache" / "blob").is_file()
    marker = json.loads((home / storage.MIGRATION_MARKER).read_text(encoding="utf-8"))
    assert marker["source"] == str(newer) and "ui_config.json" in marker["rewritten"]
    assert storage.migration_record()["source"] == str(newer)
    # Idempotent: a second resolution (another process) finds the store and carries nothing again.
    storage._reset_for_tests()
    (home / "ui_config.json").write_text("{}", encoding="utf-8")
    assert storage.resolve() == (home, "compute-home")
    assert (home / "ui_config.json").read_text(encoding="utf-8") == "{}"
    # The worker's own state root (rule 4) and the cwd default no longer win inside the managed layout.
    storage.remember_state_root(new_version / ".plugin-state" / storage.PLUGIN_ID)
    assert storage.resolve()[1] == "compute-home"
    storage._reset_for_tests()


def test_fresh_install_and_folder_source_use_the_compute_home_without_a_carry_forward(monkeypatch, tmp_path):
    monkeypatch.delenv(storage.HOME_ENV, raising=False)
    storage._reset_for_tests()
    compute_home = tmp_path / ".3lc-compute"
    fresh = compute_home / storage.MANAGED_DIR_NAME / storage.PLUGIN_ID / "1.0.0rc3"
    fresh.mkdir(parents=True)
    monkeypatch.chdir(fresh)
    home, rule = storage.resolve()
    assert (home, rule) == (compute_home / storage.SHARED_STATE_DIR_NAME / storage.PLUGIN_ID, "compute-home")
    assert not (home / storage.MIGRATION_MARKER).exists()
    # A folder source runs from <compute home>/managed-plugins/<id> itself: the same shared home.
    storage._reset_for_tests()
    monkeypatch.chdir(fresh.parent)
    assert storage.resolve() == (home, "compute-home")
    # Outside the managed layout the old rules stand.
    storage._reset_for_tests()
    monkeypatch.chdir(tmp_path)
    assert storage.resolve()[1] == "home"
    storage._reset_for_tests()


def test_a_garbage_collected_old_kit_tree_is_recreated_from_the_shared_copy(monkeypatch, tmp_path):
    """rc5: the host keeps three version dirs; tables imported before rc3 point into an older one's kit.
    The marker remembers that data dir and the plugin re-creates it from its own copy when it is gone."""
    import json
    import shutil

    monkeypatch.delenv(storage.HOME_ENV, raising=False)
    storage._reset_for_tests()
    compute_home = tmp_path / "home" / ".3lc-compute"
    managed = compute_home / storage.MANAGED_DIR_NAME / storage.PLUGIN_ID
    old = _old_state(managed / "1.0.0rc2")
    new_version = managed / "1.0.0rc3"
    new_version.mkdir(parents=True)
    monkeypatch.chdir(new_version)
    home, _ = storage.resolve()
    marker = json.loads((home / storage.MIGRATION_MARKER).read_text(encoding="utf-8"))
    assert marker["legacy_data_dirs"] == [str(old / "data")]
    # The host removes the old version dir (gc_old_versions); the next process start brings the kit back.
    shutil.rmtree(managed / "1.0.0rc2")
    assert not (old / "data").exists()
    storage._reset_for_tests()
    assert storage.resolve() == (home, "compute-home")
    assert (old / "data" / "comp" / "v1" / "starter_kit" / "data" / "train" / "a" / "x.jpg").is_file()
    assert not (old / "ui_config.json").exists()   # only the data tree, nothing else
    marker = json.loads((home / storage.MIGRATION_MARKER).read_text(encoding="utf-8"))
    assert marker["restored"][0]["dirs"] == [str(old / "data")]
    storage._reset_for_tests()
