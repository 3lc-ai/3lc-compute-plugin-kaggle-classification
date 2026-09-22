# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Competition manifest schema v1: the bundled document loads, the derived facts are right,
and every rejection names its field. The resolution order (remote -> cache -> bundled) is
tested in test_manifest_resolution.py (Phase 2)."""

from __future__ import annotations

import copy
import json
import logging

import pytest

from kaggle_classification import manifest as m


def _data():
    return m.load_yaml_text(m.bundled_path().read_text(encoding="utf-8"))


def test_load_manifest_default_is_the_full_resolution(home, monkeypatch):
    calls = []
    monkeypatch.setattr(m, "resolve", lambda **kw: calls.append(kw) or m._local("intel-scene", [], (), allowed=None))
    m.load_manifest()
    assert calls == [{"network": True}]


def test_bundled_manifest_loads_with_the_locked_facts(manifest):
    assert manifest.schema_version == 1
    assert manifest.source == "bundled"
    assert manifest.competition.id == "intel-scene"
    assert manifest.num_classes == 6
    assert manifest.class_names == ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    assert manifest.undefined_label_id == 6
    assert manifest.model.arch == "resnet18" and manifest.model.pretrained is False and manifest.model.image_size == 150
    assert manifest.expected_rows("train") == 600 + 6000
    assert manifest.expected_rows("val") == 1200
    assert manifest.expected_rows("test") == 1800
    assert manifest.dataset_name("val") == "intel-scene_val"
    assert manifest.default_project == "intel-scene"
    assert manifest.kit.published is True and len(manifest.kit.shards) == 5  # the v1 build
    assert manifest.training.embeddings.n_components == 3
    assert manifest.submission.columns == ("image_id", "prediction", "confidence")
    assert manifest.warnings == ()


def test_to_dict_is_json_serializable(manifest):
    text = json.dumps(manifest.to_dict())
    assert '"arch": "resnet18"' in text


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["model"].__setitem__("pretrained", True), "pretrained"),
        (lambda d: d["classes"].__setitem__(2, {"id": 7, "name": "glacier"}), "contiguous"),
        (lambda d: d["training"]["bounds"].__setitem__("epochs", [50, 1]), "min 50 > max 1"),
        (lambda d: d["model"].pop("arch"), "model.arch"),
        (lambda d: d.__setitem__("schema_version", 2), "schema_version"),
        (lambda d: d["classes"].__setitem__(1, {"id": 1, "name": "buildings"}), "unique"),
        (lambda d: d["training"]["defaults"].__setitem__("epochs", 999), "outside bounds"),
        (lambda d: d["training"]["presets"].__setitem__("huge", {"epochs": 500}), "outside bounds"),
        (lambda d: d["kit"].__setitem__("shards", [{"name": "a.zip", "sha256": "zz", "bytes": 1}]), "sha256"),
        (lambda d: d["training"]["embeddings"].__setitem__("n_components", 5), "n_components"),
        (lambda d: d["competition"].__setitem__("id", "Intel Scene"), "competition.id"),
        (lambda d: d.pop("submission"), "submission"),
        (lambda d: d["kit"].__setitem__("base_url", "https://evil.example.com/kit/v1"), "not an allowed kit host"),
        (lambda d: d["kit"].__setitem__("base_url", "http://competitions.3lc.ai/kit/v1"), "must use https"),
        (
            lambda d: d["kit"].__setitem__("shards", [{"name": "../x.zip", "sha256": "a" * 64, "bytes": 1}]),
            "plain file name",
        ),
        (lambda d: d["ui"]["help_links"].__setitem__(0, {"label": "x", "url": "http://insecure.example"}), "https"),
        (lambda d: d["competition"].__setitem__("display_name", "<b>Scene</b>"), "markup"),
        (lambda d: d["classes"].__setitem__(0, {"id": 0, "name": "build<img>ings"}), "markup"),
        (lambda d: d["ui"].__setitem__("loop_banner_text", "line\x00break"), "markup"),
    ],
)
def test_rejections_name_the_field(mutate, match):
    data = _data()
    mutate(data)
    with pytest.raises(m.ManifestError, match=match):
        m.parse_manifest(data)


def test_unknown_fields_warn_but_load(caplog):
    data = _data()
    data["future_section"] = {"x": 1}
    data["model"]["dropout"] = 0.1
    with caplog.at_level(logging.WARNING, logger="kaggle_classification.manifest"):
        manifest = m.parse_manifest(data)
    assert "manifest.future_section: unknown field ignored" in manifest.warnings
    assert "model.dropout: unknown field ignored" in manifest.warnings
    assert "future_section" in caplog.text


def test_loopback_kit_hosts_are_allowed_over_http_when_allowlisted():
    data = _data()
    data["kit"]["base_url"] = "http://127.0.0.1:8765/kit/v1/"
    parsed = m.parse_manifest(data, allowed_kit_hosts={"127.0.0.1"})
    assert parsed.kit.base_url == "http://127.0.0.1:8765/kit/v1"
    with pytest.raises(m.ManifestError, match="not an allowed kit host"):
        m.parse_manifest(data)


def test_shards_validate_and_total(manifest):
    data = _data()
    data["kit"]["shards"] = [
        {"name": "part-00-root.zip", "sha256": "a" * 64, "bytes": 10},
        {"name": "part-01-data-train.zip", "sha256": "B" * 64, "bytes": 20},
    ]
    parsed = m.parse_manifest(data)
    assert parsed.kit.published and parsed.kit.total_bytes == 30
    assert parsed.kit.shards[1].sha256 == "b" * 64  # normalized lowercase
    data["kit"]["shards"].append(copy.deepcopy(data["kit"]["shards"][0]))
    with pytest.raises(m.ManifestError, match="unique"):
        m.parse_manifest(data)


def test_base_url_env_override(monkeypatch):
    monkeypatch.delenv(m.MANIFEST_BASE_URL_ENV, raising=False)
    assert m.base_url() == m.MANIFEST_BASE_URL
    monkeypatch.setenv(m.MANIFEST_BASE_URL_ENV, "http://127.0.0.1:8765/hackathon/")
    assert m.base_url() == "http://127.0.0.1:8765/hackathon"


def test_no_competition_literal_outside_the_manifest():
    """The class names, split sizes and arch live in the manifest ONLY. A literal copy in a
    module would be the two-sources divergence the ExDark suite exists to catch."""
    import pathlib

    src = pathlib.Path(m.__file__).parent
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("buildings", "glacier", "6000", "1800", '"resnet18"'):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, offenders
