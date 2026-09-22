# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GET /config`` serves a populated session and a ``_meta`` block the fragment renders and never
defines: version, repository, the manifest with its provenance, the kit destination and state."""

from __future__ import annotations

import kaggle_classification
from kaggle_classification import routes, session


def test_config_payload_is_populated_and_served_not_restated(home, manifest):
    out = routes.config_payload()
    assert out["session"] == session.default_session(manifest)
    meta = out["_meta"]
    assert meta["version"] == kaggle_classification.__version__
    assert meta["repository_url"] == kaggle_classification.REPOSITORY_URL
    assert meta["manifest_source"] == "bundled"
    assert meta["manifest"]["model"] == {"arch": "resnet18", "pretrained": False, "image_size": 150}
    assert [c["name"] for c in meta["manifest"]["classes"]] == manifest.class_names
    assert meta["kit_dest"] == str(home / "data" / manifest.competition.id)
    assert meta["kit_state"] == {"state": "empty"}


def test_config_payload_reflects_a_saved_session(home, manifest):
    session.save({"session": {**session.default_session(manifest), "project_name": "probe-x"}})
    out = routes.config_payload()
    assert out["session"]["project_name"] == "probe-x"
    assert out["session"]["table_name"] == "initial"


def test_route_handlers_build_when_litestar_is_present():
    import pytest

    pytest.importorskip("litestar")
    handlers = routes.get_route_handlers()
    paths = {p for h in handlers for p in h.paths}
    assert paths == {"/config"}


def test_plugin_compute_and_fragment():
    plugin = kaggle_classification.KaggleClassificationPlugin()
    info = plugin.compute({})
    assert info["plugin"] == "kaggle-classification" and info["implemented"] == ["download_kit"]
    html = plugin.get_ui_fragment()
    assert 'class="kgc"' in html and "kaggle-classification" in html
    for needle in ("buildings", "resnet18", "6000"):
        assert needle not in html, f"competition literal {needle!r} in the fragment"
