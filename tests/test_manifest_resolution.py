# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Manifest resolution: remote (reachable AND valid) -> cache -> bundled, against a real local
HTTP server, so urllib, the timeout budget and the retry are the code paths that run in the
worker. The remote points its kit at the allowed CDN host; nothing here downloads a kit."""

from __future__ import annotations

import copy
import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from kaggle_classification import manifest as m
from kaggle_classification import session

# ── A local CDN ───────────────────────────────────────────────────────────


class _Handler(http.server.SimpleHTTPRequestHandler):
    slow: set[str] = set()
    delay_s = 3.0

    def log_message(self, *args) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        if any(self.path.endswith(s) for s in _Handler.slow):
            time.sleep(_Handler.delay_s)
        super().do_GET()


class LocalCDN:
    def __init__(self, root: Path):
        self.root = root
        (root / "hackathon").mkdir(parents=True, exist_ok=True)
        handler = lambda *a, **k: _Handler(*a, directory=str(root), **k)  # noqa: E731
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/hackathon"

    def write_index(self, competitions: list[dict]) -> None:
        (self.root / "hackathon" / "index.json").write_text(
            json.dumps({"schema_version": 1, "competitions": competitions}), encoding="utf-8"
        )

    def write_manifest(self, cid: str, data: dict) -> bytes:
        d = self.root / "hackathon" / cid
        d.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data).encode("utf-8")
        (d / "manifest.json").write_bytes(raw)
        return raw

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _entry(cid: str, active: bool = True, display: str | None = None) -> dict:
    return {"id": cid, "display_name": display or cid.title(), "manifest_url": f"{cid}/manifest.json", "active": active}


def bundled_data(**over) -> dict:
    data = m.load_yaml_text(m.bundled_path().read_text(encoding="utf-8"))
    for key, value in over.items():
        data[key] = value
    return data


@pytest.fixture
def cdn(tmp_path, monkeypatch, home):
    server = LocalCDN(tmp_path / "cdn")
    monkeypatch.setenv(m.MANIFEST_BASE_URL_ENV, server.base)
    _Handler.slow = set()
    yield server
    server.close()


def _remote(cdn: LocalCDN, display: str = "REMOTE", cid: str = "intel-scene", **over) -> bytes:
    data = bundled_data(**over)
    data["competition"]["display_name"] = display
    data["competition"]["id"] = cid
    cdn.write_index([_entry(cid)])
    return cdn.write_manifest(cid, data)


def _seed_cache(display: str = "CACHED", kit_version: str = "v9", cid: str = "intel-scene") -> None:
    data = bundled_data()
    data["competition"]["display_name"] = display
    data["kit"]["version"] = kit_version
    raw = json.dumps(data).encode("utf-8")
    m.write_cache(cid, data, raw=raw, source_url="https://old/manifest.json", fetched_at="2026-01-01T00:00:00Z")


# ── The policy ────────────────────────────────────────────────────────────


def test_remote_valid_wins_over_a_newer_looking_cache(cdn):
    _seed_cache(kit_version="v9")  # "newer" than the remote's v1 by any version ordering
    raw = _remote(cdn)
    res = m.resolve()
    man = res.manifest
    assert man.source == "remote" and man.competition.display_name == "REMOTE" and man.kit.version == "v1"
    assert man.sha256 == m._sha256_bytes(raw) and man.fetched_at is not None
    assert res.warnings == () and res.candidates == ()
    # The cache is now the remote document, sidecar included.
    cached = m.read_cache("intel-scene")
    assert cached is not None and cached.competition.display_name == "REMOTE"
    meta = m.cache_meta("intel-scene")
    assert meta["sha256"] == man.sha256 and meta["source_url"].endswith("/intel-scene/manifest.json")
    assert meta["fetched_at"] == man.fetched_at


def test_remote_invalid_falls_to_cache_with_a_visible_warning(cdn):
    _seed_cache()
    data = bundled_data()
    data["model"]["pretrained"] = True  # a bad hotfix
    cdn.write_index([_entry("intel-scene")])
    cdn.write_manifest("intel-scene", data)
    res = m.resolve()
    assert res.manifest.source == "cache" and res.manifest.competition.display_name == "CACHED"
    assert len(res.warnings) == 1
    assert "invalid" in res.warnings[0] and "pretrained" in res.warnings[0]
    assert "cached copy from 2026-01-01T00:00:00Z" in res.warnings[0]
    assert m.read_cache("intel-scene").competition.display_name == "CACHED"  # untouched


def test_remote_unreachable_falls_to_cache_then_bundled(cdn, monkeypatch):
    cdn.close()  # nothing listens any more
    _seed_cache()
    t0 = time.monotonic()
    res = m.resolve()
    assert time.monotonic() - t0 < m.FETCH_BUDGET_S + 1
    assert res.manifest.source == "cache"
    assert any("could not be reached" in w and "cached copy" in w for w in res.warnings)

    for p in m._cache_paths("intel-scene"):
        p.unlink()
    res2 = m.resolve()
    assert res2.manifest.source == "bundled" and res2.manifest.competition.id == "intel-scene"
    assert any("bundled copy" in w for w in res2.warnings)


def test_no_active_competition_uses_bundled_with_warning(cdn):
    _seed_cache()
    cdn.write_index([_entry("intel-scene", active=False), _entry("other", active=False)])
    res = m.resolve()
    assert res.manifest.source == "bundled"
    assert any("no active competition" in w for w in res.warnings)


def test_one_active_competition_is_used_even_if_not_the_default(cdn):
    data = bundled_data()
    data["competition"]["id"] = "other-comp"
    data["competition"]["display_name"] = "Other"
    cdn.write_index([_entry("intel-scene", active=False), _entry("other-comp")])
    cdn.write_manifest("other-comp", data)
    res = m.resolve()
    assert res.manifest.source == "remote" and res.manifest.competition.id == "other-comp"
    assert m.read_cache("other-comp") is not None


def test_two_active_competitions_surface_a_picker(cdn):
    for cid, name in (("intel-scene", "Intel"), ("aerial", "Aerial")):
        data = bundled_data()
        data["competition"]["id"] = cid
        data["competition"]["display_name"] = name
        cdn.write_manifest(cid, data)
    cdn.write_index([_entry("intel-scene", display="Intel"), _entry("aerial", display="Aerial")])

    # The default is among the active ones: it is used, and the picker is offered.
    res = m.resolve()
    assert res.manifest.source == "remote" and res.manifest.competition.id == "intel-scene"
    assert [c["id"] for c in res.candidates] == ["intel-scene", "aerial"]

    # A selection wins.
    m.select_competition("aerial")
    res2 = m.resolve()
    assert res2.manifest.competition.id == "aerial" and res2.manifest.source == "remote"
    assert m.selected_competition_id() == "aerial"

    # Neither active one is the default or selected: warn, fall back locally, keep the picker.
    m.select_competition("nothing-active")
    cdn.write_index([_entry("a-comp"), _entry("b-comp")])
    res3 = m.resolve()
    assert res3.manifest.source in ("bundled", "cache")
    assert any("More than one competition is active" in w for w in res3.warnings)
    assert [c["id"] for c in res3.candidates] == ["a-comp", "b-comp"]


def test_kit_on_a_non_allowlisted_host_is_rejected_remotely_and_locally(cdn):
    _seed_cache()
    data = bundled_data()
    data["kit"]["base_url"] = "https://evil.example.com/kit/v1"
    with pytest.raises(m.ManifestError, match="not an allowed kit host"):
        m.parse_manifest(data)
    cdn.write_index([_entry("intel-scene")])
    cdn.write_manifest("intel-scene", data)
    res = m.resolve()
    assert res.manifest.source == "cache"
    assert any("not an allowed kit host" in w for w in res.warnings)


def test_timeout_path_returns_within_budget(cdn, monkeypatch):
    _remote(cdn)
    _Handler.slow = {"index.json"}
    _Handler.delay_s = 3.0
    monkeypatch.setattr(m, "FETCH_BUDGET_S", 0.6)
    t0 = time.monotonic()
    res = m.resolve()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, elapsed  # 0.6 s budget, retry included, not 3 s × attempts
    assert res.manifest.source == "bundled"
    assert any("could not be reached" in w for w in res.warnings)


def test_fetch_bytes_retries_once_within_the_budget(monkeypatch):
    calls: list[float] = []

    def flaky(url, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise OSError("first attempt fails")
        return b"ok"

    monkeypatch.setattr(m, "_http_get", flaky)
    assert m.fetch_bytes("http://x/y", budget_s=2.0) == b"ok"
    assert len(calls) == 2 and calls[1] <= 2.0

    def always(url, timeout):
        raise OSError("down")

    monkeypatch.setattr(m, "_http_get", always)
    with pytest.raises(m.FetchError, match="down"):
        m.fetch_bytes("http://x/y", budget_s=1.0)


def test_index_validation_rejects_offsite_manifest_urls():
    base = "https://competitions.3lc.ai/hackathon"
    with pytest.raises(m.ManifestError, match="must stay on"):
        m.parse_index(
            {"schema_version": 1, "competitions": [{**_entry("x"), "manifest_url": "https://evil/m.json"}]}, base
        )
    entries = m.parse_index({"schema_version": 1, "competitions": [_entry("x")]}, base)
    assert entries[0]["manifest_url"] == "https://competitions.3lc.ai/hackathon/x/manifest.json"
    with pytest.raises(m.ManifestError, match="schema_version"):
        m.parse_index({"schema_version": 2, "competitions": []}, base)
    with pytest.raises(m.ManifestError, match="unique"):
        m.parse_index({"schema_version": 1, "competitions": [_entry("x"), _entry("x")]}, base)


def test_local_resolution_never_touches_the_network(cdn, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network on the local path")

    monkeypatch.setattr(m, "_http_get", boom)
    res = m.resolve(network=False)
    assert res.manifest.source == "bundled"
    _seed_cache()
    assert m.resolve(network=False).manifest.source == "cache"


def test_resolve_manifest_for_job_returns_provenance(cdn):
    raw = _remote(cdn)
    manifest, prov = m.resolve_manifest_for_job()
    assert manifest.source == "remote"
    assert prov == manifest.provenance
    assert prov["manifest_sha256"] == m._sha256_bytes(raw)
    assert prov["manifest_source"] == "remote" and prov["competition_id"] == "intel-scene"
    assert prov["kit_version"] == "v1" and prov["manifest_fetched_at"] == manifest.fetched_at
    assert set(prov) == {
        "manifest_sha256",
        "manifest_source",
        "manifest_source_detail",
        "manifest_fetched_at",
        "competition_id",
        "kit_version",
        "schema_version",
    }


def test_bundled_manifest_has_a_sha256(manifest):
    assert len(manifest.sha256) == 64 and manifest.fetched_at is None
    assert manifest.provenance["manifest_source"] == "bundled"


def test_background_refresh_settles_and_is_rate_limited(cdn):
    _remote(cdn)
    status = m.refresh_in_background()
    assert status["state"] == "running"
    for _ in range(100):
        if m.refresh_status()["state"] != "running":
            break
        time.sleep(0.05)
    done = m.refresh_status()
    assert done["state"] == "done" and done["source"] == "remote"
    assert m.resolve(network=False).manifest.source == "cache"  # the refresh filled the cache
    assert m.refresh_in_background()["state"] == "done"  # within the TTL: no new run
    assert m.refresh_in_background(force=True)["state"] == "running"


def test_poisoned_selection_in_the_store_never_reaches_a_path(home):
    """A hand-edited ui_config.json bypasses save(); the reader must still refuse it."""
    session.config_path().parent.mkdir(parents=True, exist_ok=True)
    session.config_path().write_text(json.dumps({"competition": {"id": "../../outside"}}), encoding="utf-8")
    assert m.selected_competition_id() is None
    res = m.resolve(network=False)
    assert res.manifest.competition.id == "intel-scene"
    with pytest.raises(ValueError):
        m.select_competition("../../outside")


def test_cache_lives_under_the_plugin_home(home):
    assert m.cache_dir() == home / "manifest-cache"
    assert session.plugin_home() == home
    # And the cache sidecar shape is the one the spec names.
    data = bundled_data()
    meta = m.write_cache(
        "intel-scene", data, raw=b"x", source_url="https://s/m.json", fetched_at="2026-09-22T00:00:00Z"
    )
    assert {"fetched_at", "source_url", "sha256"} <= set(meta)


def test_manifest_document_may_be_yaml_or_json():
    data = bundled_data()
    assert m._decode_document(json.dumps(data).encode()) == data
    assert m._decode_document(m.bundled_path().read_bytes())["competition"]["id"] == "intel-scene"
    assert copy.deepcopy(data) == data
