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
    assert meta["plugin_home"] == {"path": str(home), "resolved_by": "env"}
    assert meta["manifest_provenance"]["manifest_source"] == "bundled"
    assert meta["manifest_candidates"] == []
    # The config path never blocks on the network: it kicks the refresh and reports its state.
    assert meta["manifest_refresh"]["state"] in ("running", "done", "failed")


def test_config_path_never_fetches(home, monkeypatch):
    from kaggle_classification import manifest as manifest_mod

    def boom(*a, **k):
        raise AssertionError("GET /config fetched the network synchronously")

    monkeypatch.setattr(manifest_mod, "_http_get", boom)
    monkeypatch.setattr(manifest_mod, "refresh_in_background", lambda **kw: {"state": "skipped"})
    out = routes.config_payload()
    assert out["_meta"]["manifest_refresh"] == {"state": "skipped"}


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
    assert paths == {"/config", "/manifest", "/manifest/select"}


def test_route_handler_annotations_resolve_like_litestar_does():
    """The compute 1.1.0 worker died at startup because a handler's string return annotation
    (``from __future__ import annotations``) named a ``Response`` that was not in the routes
    module's globals. Litestar resolves hints with ``typing.get_type_hints``; so do we."""
    import typing

    import pytest

    pytest.importorskip("litestar")
    for handler in routes.get_route_handlers():
        hints = typing.get_type_hints(handler.fn, globalns=vars(routes))
        assert "return" in hints, handler.fn.__name__


def test_plugin_compute_and_fragment():
    plugin = kaggle_classification.KaggleClassificationPlugin()
    info = plugin.compute({})
    assert info["plugin"] == "kaggle-classification" and info["implemented"] == ["download_kit"]
    html = plugin.get_ui_fragment()
    assert 'class="kgc"' in html and "kaggle-classification" in html
    for needle in ("buildings", "resnet18", "6000"):
        assert needle not in html, f"competition literal {needle!r} in the fragment"


def test_fragment_renders_manifest_strings_only_through_textcontent():
    """Escaped at both ends: the server refuses markup in display strings, and the fragment never
    interprets HTML — no innerHTML, insertAdjacentHTML, outerHTML or document.write anywhere."""
    from pathlib import Path

    html = (Path(kaggle_classification.__file__).parent / "ui" / "ui.html").read_text(encoding="utf-8")
    for forbidden in ("innerHTML", "insertAdjacentHTML", "outerHTML", "document.write", "eval("):
        assert forbidden not in html, forbidden
    assert "textContent" in html
    assert "/manifest/select" in html and "/manifest'" in html  # the picker and the poll
