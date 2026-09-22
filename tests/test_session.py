# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""session.py: merge semantics, allowlist, atomicity, markers, the retired-key rejection (the
stale-fragment guard), manifest-derived defaults, and the URL parse across project-root shapes."""

from __future__ import annotations

import json

import pytest
from conftest import ROOT_SHAPES, table_url

from kaggle_classification.session import classify_override, url_dataset, url_project, url_table


def _write(store, data: dict) -> None:
    store.config_path().parent.mkdir(parents=True, exist_ok=True)
    store.config_path().write_text(json.dumps(data, indent=1), encoding="utf-8")


def test_load_missing_file_is_empty(store):
    assert store.load() == {}


def test_save_merges_per_tab(store):
    store.save({"train": {"epochs": "5"}})
    store.save({"predict": {"device": "cpu"}})
    data = store.load()
    assert data["train"] == {"epochs": "5"}
    assert data["predict"] == {"device": "cpu"}


def test_save_replaces_whole_tab(store):
    store.save({"train": {"epochs": "5", "batch": "16"}})
    store.save({"train": {"epochs": "7"}})
    assert store.load()["train"] == {"epochs": "7"}


def test_save_drops_unknown_tabs(store):
    store.save({"bogus": {"x": 1}, "train": {"epochs": "5"}})
    data = store.load()
    assert "bogus" not in data
    assert data["train"] == {"epochs": "5"}


def test_fresh_save_stamps_markers(store):
    store.save({"train": {"epochs": "5"}})
    assert store.load()["_migrations"]["session_v1"] == "fresh"


def test_load_stamps_markers_once_and_is_idempotent(store):
    _write(store, {"train": {"epochs": "3"}})
    store.load()
    first = store.config_path().read_text(encoding="utf-8")
    assert '"session_v1": "fresh"' in first
    store.load()
    assert store.config_path().read_text(encoding="utf-8") == first


def test_no_tmp_file_left_behind(store):
    store.save({"train": {"epochs": "5"}})
    assert not store.config_path().with_suffix(".json.tmp").exists()


def test_corrupt_file_reads_as_empty(store):
    _write(store, {"x": 1})
    store.config_path().write_text("{not json", encoding="utf-8")
    assert store.load() == {}


def test_retired_keys_are_rejected_and_write_nothing(store, monkeypatch):
    monkeypatch.setattr(store, "_RETIRED_TABS", ("import",))
    monkeypatch.setattr(store, "_RETIRED_TAB_KEYS", {"train": ("project_name",)})
    _write(store, {"train": {"epochs": "3"}, "_migrations": {"session_v1": "fresh"}})
    before = store.config_path().read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"train\.project_name"):
        store.save({"train": {"epochs": "9", "project_name": "x"}})
    with pytest.raises(ValueError, match="import"):
        store.save({"import": {"project_name": "x"}})
    assert store.config_path().read_text(encoding="utf-8") == before


@pytest.mark.parametrize("bad", ["../../etc", "Intel Scene", "", None, 7, "a/b", "..", r"x\y"])
def test_competition_id_is_validated_at_the_save_boundary(store, bad):
    with pytest.raises(ValueError, match="competition.id"):
        store.save({"competition": {"id": bad}})
    with pytest.raises(ValueError, match="competition.id"):
        store.save({"competition": "not-a-mapping"})
    assert not store.config_path().exists()
    store.save({"competition": {"id": "intel-scene"}})
    assert store.load()["competition"] == {"id": "intel-scene"}


def test_default_session_derives_from_the_manifest(store, manifest):
    sess = store.default_session(manifest)
    assert sess == {
        "project_name": manifest.competition.id,
        "table_name": "initial",
        "kit_dir": "",
        "device": "",
        "overrides": {},
    }


def test_populated_session_fills_missing_fields(store, manifest):
    store.save({"session": {"project_name": "probe-x"}})
    sess = store.populated_session(manifest)
    assert sess["project_name"] == "probe-x"
    assert sess["table_name"] == "initial" and sess["overrides"] == {}


def test_publish_kit_dir_merges_onto_the_freshest_session(store, manifest, tmp_path):
    store.save({"session": {**store.default_session(manifest), "device": "mps"}})
    store.publish_kit_dir(manifest, tmp_path / "kit")
    sess = store.load()["session"]
    assert sess["kit_dir"] == str(tmp_path / "kit")
    assert sess["device"] == "mps"


# ── URL parsing by position in the layout tail ───────────────────────────


@pytest.mark.parametrize("backslashes", [False, True])
def test_url_segments_parse_under_every_root_shape(tmp_path, root_shape, backslashes):
    url = table_url(tmp_path, root_shape, "intel-scene", "intel-scene_train", "round2", backslashes=backslashes)
    assert url_project(url) == "intel-scene"
    assert url_dataset(url) == "intel-scene_train"
    assert url_table(url) == "round2"
    assert url_table(url + " ") == "round2"  # trimmed like the fragment


def test_unparseable_urls_return_none():
    assert url_project("not-a-table-url") is None
    assert url_dataset("C:/x/tables/only") is None


def test_classify_override(tmp_path, root_shape):
    project, table = "intel-scene", "initial"
    same_split = table_url(tmp_path, root_shape, project, "intel-scene_train", "round2")
    derived = table_url(tmp_path, root_shape, project, "intel-scene_train", table)
    cross_split = table_url(tmp_path, root_shape, project, "intel-scene_val", "round2")
    cross_project = table_url(tmp_path, root_shape, "other", "intel-scene_train", "round2")
    assert classify_override(same_split, project, table, "intel-scene_train") == "keep"
    assert classify_override(derived, project, table, "intel-scene_train") == "suppress"
    assert classify_override(cross_split, project, table, "intel-scene_train") == "drop"
    assert classify_override(cross_project, project, table, "intel-scene_train") == "drop"
    assert classify_override("garbage", project, table, "intel-scene_train") == "drop"
    assert set(ROOT_SHAPES) == {"default", "relocated", "bare"}
