# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The import stage against a synthetic kit and the REAL tlc (3.3.x) in an isolated project root.

Needs the heavy extra (tlc + PIL); skips without it, so a green run in a light venv is not a
green run for this module (CLAUDE.md §B, the ``test_kit_model.py`` rule).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from helpers import FakeCtx, make_kit_tree, small_manifest_data

import build_kit  # tools/ is on sys.path (conftest)
from kaggle_classification import importer, session
from kaggle_classification import manifest as manifest_mod
from kaggle_classification.kit import FILES_INDEX_NAME, KIT_DIR_NAME

tlc = pytest.importorskip("tlc")
pytest.importorskip("PIL")


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """The project root the importer writes under, redirected for the test: nothing lands in
    the real one. tlc resolves its own configuration once per process, so the seam is the
    importer's accessor, and every table URL the importer builds goes through it."""
    root = tmp_path / "3lc-root"
    root.mkdir()
    monkeypatch.setattr(importer, "project_root_url", lambda: root.as_posix())
    return root


@pytest.fixture
def kit_and_manifest(tmp_path, home):
    """A tiny REAL-image kit on disk with files.json, and the matching small manifest."""
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(tmp_path / "tree", data, real_images=True)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    manifest = manifest_mod.parse_manifest(data, source="test", source_detail="synthetic")
    session.publish_kit_dir(manifest, kit_root)
    return kit_root, manifest


def _run(manifest, **params):
    ctx = FakeCtx()
    result = importer.run_import(params, ctx, manifest)
    return result, ctx


def test_import_registers_train_and_val_with_undefined_and_weights(project_root, kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    result, ctx = _run(manifest)
    assert result["cancelled"] is False
    assert result["table_name"] == "initial"
    train = tlc.Table.from_url(tlc.Url(result["tables"]["train"]["url"]))
    val = tlc.Table.from_url(tlc.Url(result["tables"]["val"]["url"]))
    assert train.row_count == manifest.expected_rows("train")
    assert val.row_count == manifest.expected_rows("val")
    # Label map: manifest classes in id order, then undefined as the LAST entry.
    vm = train.get_value_map("label")
    names = [vm[k]["internal_name"] for k in sorted(vm)]
    assert names == [*manifest.class_names, importer.UNDEFINED_LABEL_NAME]
    assert int(max(vm)) == manifest.undefined_label_id
    rows = list(train.table_rows)
    labeled = [r for r in rows if r["label"] < manifest.num_classes]
    pool = [r for r in rows if r["label"] == manifest.undefined_label_id]
    assert len(pool) == manifest.splits.train.undefined and all(r["weight"] == 0.0 for r in pool)
    assert len(labeled) == manifest.splits.train.labeled_per_class * manifest.num_classes
    assert all(r["weight"] == 1.0 for r in labeled)
    assert all(r["weight"] == 1.0 for r in val.table_rows)
    assert all(Path(r["image"]).is_file() for r in rows)
    # test is never registered
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("test"), "initial")).exists()
    # every check passed, the record is in the session store with provenance and lineage
    assert ctx.checks and all(c["ok"] for c in ctx.checks)
    record = session.load()["import_state"]
    assert record["lineage_root"] == {"train_url": result["tables"]["train"]["url"], "val_url": result["tables"]["val"]["url"]}
    assert record["val_locked"]["url"] == result["tables"]["val"]["url"]
    assert record["manifest_provenance"]["competition_id"] == "intel-scene"
    assert record["timings"]["total_s"] >= 0
    assert ctx.facts["run_url"] == result["tables"]["train"]["url"]
    state = importer.import_state()
    assert state["state"] == "success"
    # No revision yet: the latest train/val revision IS the seed (the Loop's Dashboard target).
    assert state["latest"] == {"train": result["tables"]["train"]["url"], "val": result["tables"]["val"]["url"]}
    assert "3LC Scene Classification Challenge" in tlc.Table.from_url(tlc.Url(result["tables"]["train"]["url"])).description


def test_existing_tables_are_reused_and_revalidated_like_exdark(project_root, kit_and_manifest):
    """The ExDark mirror (docs/EXDARK_MIRROR.md #25): a table already at the target URL is REUSED,
    never rewritten, and the post-write checks run on it. ``mode=reimport`` (fresh ``-2`` names)
    stays available for the pending re-import decision (#26)."""
    kit_root, manifest = kit_and_manifest
    first, _ = _run(manifest)
    assert all(not t["reused"] for t in first["tables"].values())
    mtime = Path(first["tables"]["train"]["url"]).stat().st_mtime_ns
    second, ctx = _run(manifest)
    assert second["table_name"] == "initial"
    assert all(t["reused"] for t in second["tables"].values())
    assert second["tables"]["train"]["url"] == first["tables"]["train"]["url"]
    assert second["tables"]["train"]["rows"] == manifest.expected_rows("train")
    assert Path(first["tables"]["train"]["url"]).stat().st_mtime_ns == mtime  # untouched on disk
    assert any(c["label"] == "existing tables reused" and c["ok"] for c in ctx.checks)
    assert any(c["label"].startswith("train row count") and c["ok"] for c in ctx.checks)  # re-validated
    pre = importer.preflight({}, manifest)
    assert pre["all_ok"] is True and pre["existing"]["train"]["exists"] is True
    third, _ = _run(manifest, mode="reimport")
    assert third["table_name"] == "initial-2"
    assert tlc.Url(first["tables"]["train"]["url"]).exists() and tlc.Url(third["tables"]["val"]["url"]).exists()
    assert session.populated_session(manifest)["table_name"] == "initial-2"


def test_kit_defect_fails_before_any_table_and_names_every_problem(project_root, kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    # two defects at once: a missing val image and a test image without a submission row
    next(iter((kit_root / "data" / "val" / manifest.class_names[0]).iterdir())).unlink()
    (kit_root / "data" / "test" / "zzzz.jpg").write_bytes((kit_root / "data" / "test" / "t0000.jpg").read_bytes())
    with pytest.raises(importer.ImportRefused) as exc:
        _run(manifest)
    msg = str(exc.value)
    assert "val images per class" in msg and "ids are the test image ids" in msg and "kit defect" in msg
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial")).exists()
    assert importer.import_state()["state"] == "empty"


def test_undecodable_image_is_a_kit_defect(project_root, kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    victim = next(iter((kit_root / "data" / "train" / "undefined").iterdir()))
    victim.write_bytes(b"\xff\xd8not an image\xff\xd9")
    with pytest.raises(importer.ImportRefused, match="do not decode"):
        _run(manifest)
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial")).exists()


def test_failure_after_the_first_table_leaves_no_partial_tables(project_root, kit_and_manifest, monkeypatch):
    kit_root, manifest = kit_and_manifest
    real = importer.register_split
    calls = []

    def boom(split, *a, **k):
        calls.append(split)
        if split == "val":
            msg = "disk full"
            raise OSError(msg)
        return real(split, *a, **k)

    monkeypatch.setattr(importer, "register_split", boom)
    with pytest.raises(RuntimeError, match="disk full") as exc:
        _run(manifest)
    assert "Removed 1 table" in str(exc.value)
    assert calls == ["train", "val"]
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial")).exists()
    assert importer.import_state()["state"] == "empty"


def test_cancel_after_the_first_table_removes_it(project_root, kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    # is_cancelled is polled: after validate, after decode, then once per split -> cancel on the 4th
    ctx = FakeCtx(cancel_on_call=4)
    result = importer.run_import({}, ctx, manifest)
    assert result["cancelled"] is True
    assert len(result["removed"]) == 1
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial")).exists()


def test_preflight_without_a_kit_and_with_bad_params(home, manifest):
    pre = importer.preflight({}, manifest)
    assert pre["all_ok"] is False and "Download the starter kit" in pre["error"]
    assert pre["kit"] == {"state": "empty"} and pre["plugin_version"]
    with pytest.raises(importer.ImportRefused, match="plain name"):
        importer.resolve_params({"project_name": "a/b", "kit_dir": "x"}, manifest)
    with pytest.raises(importer.ImportRefused, match="mode"):
        importer.resolve_params({"mode": "overwrite"}, manifest)


def test_import_state_goes_stale_when_a_table_disappears(project_root, kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    result, _ = _run(manifest)
    tlc.Url(result["tables"]["val"]["url"]).delete()
    state = importer.import_state()
    assert state["state"] == "stale" and state["verified"] == {"train": True, "val": False}


def test_validate_kit_reports_wrong_class_dirs_and_version(kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    extra = kit_root / "data" / "train" / "zebra"
    extra.mkdir()
    index = json.loads((kit_root / FILES_INDEX_NAME).read_text(encoding="utf-8"))
    index["kit_version"] = "v0"
    (kit_root / FILES_INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
    checks = importer.validate_kit(importer.scan_kit(kit_root, manifest), manifest)
    failed = {c["label"] for c in checks if not c["ok"]}
    assert any("train class directories" in f for f in failed)
    assert any("kit version" in f for f in failed)
    assert kit_root.name == KIT_DIR_NAME


def test_build_rows_is_manifest_ordered_with_the_pool_last(kit_and_manifest):
    kit_root, manifest = kit_and_manifest
    rows = importer.build_rows(importer.scan_kit(kit_root, manifest), manifest)
    labels = rows["train"]["label"]
    n = manifest.splits.train.labeled_per_class
    assert labels[: n * manifest.num_classes] == [i for i in range(manifest.num_classes) for _ in range(n)]
    assert set(labels[n * manifest.num_classes :]) == {manifest.undefined_label_id}
    assert rows["train"]["weight"].count(0.0) == manifest.splits.train.undefined
    assert set(rows["val"]["weight"]) == {1.0}
    # a manifest whose class ids are not 0..N-1 in file order still maps by id
    shuffled = dataclasses.replace(manifest, classes=tuple(reversed(manifest.classes)))
    assert importer.label_map(shuffled)[shuffled.undefined_label_id] == importer.UNDEFINED_LABEL_NAME
