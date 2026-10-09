# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The guided Kaggle connect flow (rc10): the write path under a redirected home on Windows and POSIX,
the token's format validation, the verification outcomes, and the no-token-leak guarantee across every
surface the plugin serves (the connection card, the Doctor, the Status tab, the session store, the
verification bundle). Light: the kaggle client is never imported — ``_authenticate_raw`` is stood in for."""

from __future__ import annotations

import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

from kaggle_classification import kaggle_client, routes, session, status

TOKEN = "KGAT_test0123456789abcdefghijklmnop_-XYZ"   # matches status.SECRET_PATTERNS' token pattern


@pytest.fixture
def redirected_home(tmp_path, monkeypatch):
    """A compute service whose home is redirected (tester/start_tester.ps1 / .sh): USERPROFILE and HOME
    both point into the tester folder; no Kaggle source of any kind."""
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(k, raising=False)
    home_dir = tmp_path / "tester" / "home"
    home_dir.mkdir(parents=True)
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.setenv("HOME", str(home_dir))
    return home_dir


class _Api:
    config_values = {"username": "participant"}


def _auth_ok():
    return _Api()


def _auth_rejected():
    raise SystemExit(1)   # what kaggle 2.x does when no source yields a usable credential


def _auth_unreachable():
    class ConnectionError(Exception):   # noqa: A001 — requests' name, matched by name
        pass

    raise ConnectionError("HTTPSConnectionPool(host='www.kaggle.com'): Max retries exceeded")


# ── The path ──────────────────────────────────────────────────────────────────


def test_token_path_is_what_kagglesdk_reads_under_the_redirected_home(redirected_home):
    """kagglesdk reads ``os.path.expanduser('~/.kaggle/access_token')``; the plugin writes exactly that,
    which under a redirected USERPROFILE / HOME is inside the tester folder, never the real profile."""
    path = kaggle_client.token_path()
    assert path == redirected_home / ".kaggle" / "access_token"
    assert path == Path(os.path.expanduser("~/.kaggle/access_token"))   # the same file, whichever separator
    assert kaggle_client.credential_sources()["access_token"] == str(path)


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_other_ways_carry_the_resolved_path_one_line_each(redirected_home, platform):
    help_ = kaggle_client.connect_help(platform)
    path = str(kaggle_client.token_path())
    assert help_["path"] == path
    for key in ("command", "env_var", "legacy"):
        assert "\n" not in help_[key]
    assert path in help_["command"] and "KGAT_<your token>" in help_["command"]
    if platform == "win32":
        assert help_["shell"] == "PowerShell" and "-NoNewline -Encoding ascii" in help_["command"]
    else:
        assert help_["shell"] == "Terminal" and "printf '%s'" in help_["command"] and "chmod 600" in help_["command"]
    assert help_["legacy"].startswith(str(kaggle_client.config_dir() / "kaggle.json"))


# ── The write ─────────────────────────────────────────────────────────────────


def test_save_token_writes_plain_ascii_without_bom_or_newline(redirected_home):
    path = kaggle_client.save_token(TOKEN)
    assert path == redirected_home / ".kaggle" / "access_token"
    raw = path.read_bytes()
    assert raw == TOKEN.encode("ascii")
    assert not raw.startswith(b"\xef\xbb\xbf") and not raw.endswith((b"\n", b"\r"))
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    # A second connect replaces the file in place (no stray copies next to it).
    kaggle_client.save_token(TOKEN + "2")
    assert path.read_bytes() == (TOKEN + "2").encode("ascii")
    assert sorted(p.name for p in path.parent.iterdir()) == ["access_token"]


# ── The format ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [
    TOKEN, f"  {TOKEN}  ", f"{TOKEN}\n", f"{TOKEN}\r\n", f"﻿{TOKEN}", f'"{TOKEN}"', f"'{TOKEN}'\n",
])
def test_validate_token_trims_what_a_paste_carries(raw):
    assert kaggle_client.validate_token(raw) == (TOKEN, "")


@pytest.mark.parametrize("raw", [
    "", "   ", None, "abc", "KGAT_short", "0123456789abcdef0123456789abcdef",   # a legacy 32-hex API key
    "kgat_lowercase0123456789abcdef", f"{TOKEN} extra", f"KGAT_{'é' * 20}", '{"username": "x", "key": "y"}',
])
def test_validate_token_refuses_anything_but_an_access_token(raw):
    token, reason = kaggle_client.validate_token(raw)
    assert token == "" and reason
    assert reason in (kaggle_client.REASONS["empty"], kaggle_client.REASONS["format"])
    assert "KGAT_" not in reason.replace("start with KGAT_", "")   # the sentence names the shape, never a value


# ── The flow ──────────────────────────────────────────────────────────────────


def test_connect_writes_verifies_and_reports_the_username(redirected_home, monkeypatch):
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_ok)
    out = kaggle_client.connect(f"  {TOKEN}\n")
    assert out == {"ok": True, "username": "participant", "path": str(redirected_home / ".kaggle" / "access_token")}
    assert (redirected_home / ".kaggle" / "access_token").read_bytes() == TOKEN.encode("ascii")
    assert kaggle_client.credentials_present() is True


def test_connect_removes_a_token_kaggle_rejects(redirected_home, monkeypatch):
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_rejected)
    out = kaggle_client.connect(TOKEN)
    assert out["ok"] is False and out["kind"] == "rejected"
    assert out["reason"] == kaggle_client.REASONS["rejected"]
    assert not (redirected_home / ".kaggle" / "access_token").exists()
    assert kaggle_client.credentials_present() is False


def test_connect_keeps_the_token_when_kaggle_is_unreachable(redirected_home, monkeypatch):
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_unreachable)
    out = kaggle_client.connect(TOKEN)
    assert out["ok"] is False and out["kind"] == "unreachable" and "no network" in out["reason"]
    assert (redirected_home / ".kaggle" / "access_token").read_bytes() == TOKEN.encode("ascii")


def test_connect_refuses_a_bad_format_without_writing(redirected_home, monkeypatch):
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_ok)
    out = kaggle_client.connect("not-a-token")
    assert out["ok"] is False and out["kind"] == "format"
    assert not (redirected_home / ".kaggle").exists()


def test_connect_refuses_when_the_env_var_would_shadow_the_file(redirected_home, monkeypatch):
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_ok)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_other0123456789abcdefghij")
    out = kaggle_client.connect(TOKEN)
    assert out["ok"] is False and out["kind"] == "env_override"
    assert not (redirected_home / ".kaggle").exists()


def test_a_client_error_is_one_scrubbed_sentence(redirected_home, monkeypatch):
    def boom():
        raise RuntimeError(f"500 Server Error for token {TOKEN}")

    monkeypatch.setattr(kaggle_client, "_authenticate_raw", boom)
    out = kaggle_client.connect(TOKEN)
    assert out["ok"] is False and out["kind"] == "error"
    assert TOKEN not in json.dumps(out) and "KGAT_…" in out["reason"]


def test_the_route_answers_the_card_on_success_and_the_sentence_on_failure(redirected_home, home, manifest, monkeypatch):
    pytest.importorskip("litestar")
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_ok)
    monkeypatch.setattr(kaggle_client, "competition_info",
                        lambda api, slug: {"user_has_entered": True, "max_daily_submissions": 0, "title": "T"})
    monkeypatch.setattr(kaggle_client, "submissions_used_today", lambda api, slug: None)
    handler = next(h for h in routes.get_route_handlers() if "/kaggle/connect" in h.paths)
    out = handler.fn({"token": TOKEN})
    assert out["ok"] is True and out["username"] == "participant"
    assert out["connection"]["state"] == "ready" and out["connection"]["username"] == "participant"
    assert TOKEN not in json.dumps(out)
    assert handler.fn({"token": ""})["kind"] == "empty"
    assert handler.fn(None)["kind"] == "empty"
    assert handler.fn({"nope": 1})["kind"] == "empty"


# ── No token leak ─────────────────────────────────────────────────────────────


def test_no_surface_ever_carries_the_token(redirected_home, home, manifest, monkeypatch):
    """After a connect, the token is on disk at the one path and nowhere else: not in the connection card,
    not in the Doctor (with and without the Kaggle block), not in the Status tab's live block, not in the
    session store, not in the verification bundle — and the bundle's secret scan would refuse it anyway."""
    monkeypatch.setattr(kaggle_client, "_authenticate_raw", _auth_ok)
    monkeypatch.setattr(kaggle_client, "competition_info",
                        lambda api, slug: {"user_has_entered": True, "max_daily_submissions": 5, "title": "T"})
    monkeypatch.setattr(kaggle_client, "submissions_used_today", lambda api, slug: 1)
    monkeypatch.setattr(kaggle_client, "list_submissions", lambda api, slug, page_size=10: [])
    assert kaggle_client.connect(TOKEN)["ok"] is True

    surfaces = {
        "connection": kaggle_client.connection(manifest.competition.slug, 100),
        "doctor": status.doctor(manifest, kaggle=True),
        "doctor_no_kaggle": status.doctor(manifest, kaggle=False),
        "kaggle_live": status.kaggle_live(manifest),
        "session_store": session.load(),
        "config": routes.config_payload(),
    }
    for name, payload in surfaces.items():
        text = json.dumps(payload, default=str)
        assert TOKEN not in text, name
        for label, pattern in status.SECRET_PATTERNS:
            assert not pattern.search(text), f"{name}: {label}"
    assert surfaces["doctor"]["kaggle"]["state"] == "ready"
    assert surfaces["doctor"]["kaggle"]["token_path"] == str(kaggle_client.token_path())

    path, _name = status.verification_bundle_file(manifest, checkpoints="none")
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            assert TOKEN not in zf.read(member).decode("utf-8", "replace"), member
    with pytest.raises(status.BundleRefused):
        status.scan_for_secrets("x", TOKEN)
