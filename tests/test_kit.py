# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""The kit download / verify stage (job kind ``download_kit``), offline.

A synthetic kit is served through a Range-aware stub in place of ``kit._open``, so every
network behaviour the ExDark CDN showed at staging (206 continuations, range-ignoring 200s,
corrupt bodies) plus the per-file ``files.json`` verification and the manifest split-count
gate are exercised in milliseconds.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from helpers import FakeCDN, FakeCtx, build_synthetic_kit, bump_version, read_index, small_manifest_data

from kaggle_classification import kit, session
from kaggle_classification import manifest as manifest_mod
from kaggle_classification.kit import FILES_INDEX_NAME, KIT_DIR_NAME


@pytest.fixture
def served(tmp_path, monkeypatch, home):
    """A published synthetic kit: (manifest, fake cdn)."""
    manifest, cdn_dir = build_synthetic_kit(tmp_path / "srv")
    fake = FakeCDN(cdn_dir)
    monkeypatch.setattr(kit, "_open", fake.open)
    return manifest, fake


def _dest(tmp_path: Path) -> Path:
    return tmp_path / "participant" / "data"


def _version_dir(tmp_path: Path, manifest) -> Path:
    return _dest(tmp_path) / manifest.kit.version


def _run(tmp_path, manifest, ctx=None, **extra):
    return kit.run_download({"dest_dir": str(_dest(tmp_path)), **extra}, ctx or FakeCtx(), manifest)


def test_fresh_download_end_to_end(served, tmp_path):
    manifest, _ = served
    ctx = FakeCtx()
    result = _run(tmp_path, manifest, ctx)
    kit_root = _version_dir(tmp_path, manifest) / KIT_DIR_NAME

    assert result["cancelled"] is False
    assert all(c["ok"] for c in ctx.checks), ctx.checks
    index = read_index(kit_root)
    assert result["file_count"] == index["file_count"] == result["verified_files"]
    assert result["kit_dir"] == str(kit_root)
    assert result["split_counts"] == kit.expected_split_counts(manifest)
    # The one server-side session write: Import now points at the kit.
    assert session.load()["session"]["kit_dir"] == str(kit_root)
    # The record backs the revisit state.
    assert kit.download_state(manifest)["state"] == "success"
    # Shards removed after verification; the tree and files.json stay.
    assert not list(_version_dir(tmp_path, manifest).glob("part-*.zip"))
    assert (kit_root / FILES_INDEX_NAME).is_file()
    assert ctx.progress[-1]["percent"] == 100.0
    assert any(p.get("phase") == "download" for p in ctx.progress)
    assert ctx.facts["kit_dir"] == str(kit_root)


def test_unpublished_kit_refuses_with_a_clear_message(home, tmp_path):
    manifest = manifest_mod.parse_manifest(small_manifest_data())
    assert manifest.kit.published is False
    with pytest.raises(RuntimeError, match="not been published"):
        _run(tmp_path, manifest)


def test_session_write_merges_not_replaces(served, tmp_path):
    manifest, _ = served
    session.save({"session": {**session.default_session(manifest), "project_name": "my-project"}})
    _run(tmp_path, manifest)
    after = session.load()["session"]
    assert after["project_name"] == "my-project"
    assert after["kit_dir"].endswith(KIT_DIR_NAME)


def test_completed_shard_skipped_and_partial_resumed(served, tmp_path):
    manifest, cdn = served
    names = [s.name for s in manifest.kit.shards]
    assert len(names) >= 2
    version_dir = _version_dir(tmp_path, manifest)
    version_dir.mkdir(parents=True)
    (version_dir / names[0]).write_bytes((cdn.dir / names[0]).read_bytes())
    (version_dir / (names[1] + ".part")).write_bytes((cdn.dir / names[1]).read_bytes()[:100])

    result = _run(tmp_path, manifest)
    assert result["cancelled"] is False
    assert names[0] not in [n for n, _ in cdn.requests]
    assert (names[1], 100) in cdn.requests


def test_range_ignored_falls_back_to_full_shard(served, tmp_path):
    manifest, cdn = served
    cdn.ignore_ranges = True
    name = manifest.kit.shards[0].name
    version_dir = _version_dir(tmp_path, manifest)
    version_dir.mkdir(parents=True)
    (version_dir / (name + ".part")).write_bytes((cdn.dir / name).read_bytes()[:100])

    ctx = FakeCtx()
    assert _run(tmp_path, manifest, ctx)["cancelled"] is False
    assert (name, 100) in cdn.requests
    assert any("range request not honored" in m for m in ctx.logs)


def test_corrupt_shard_on_disk_is_redownloaded(served, tmp_path):
    manifest, cdn = served
    entry = manifest.kit.shards[0]
    version_dir = _version_dir(tmp_path, manifest)
    version_dir.mkdir(parents=True)
    (version_dir / entry.name).write_bytes(b"\x00" * entry.bytes)
    assert _run(tmp_path, manifest)["cancelled"] is False
    assert (entry.name, None) in cdn.requests


def test_persistent_sha_mismatch_fails_with_participant_message(served, tmp_path):
    manifest, cdn = served
    name = manifest.kit.shards[0].name
    good = (cdn.dir / name).read_bytes()
    cdn.tamper[name] = good[:-1] + bytes([good[-1] ^ 0xFF])
    with pytest.raises(RuntimeError, match=name):
        _run(tmp_path, manifest)
    assert [n for n, _ in cdn.requests].count(name) == 2  # one retry


def test_files_index_disagreement_is_caught_per_file(tmp_path, monkeypatch, home):
    # A staging error: files.json promises a hash the (individually valid) shard does not carry.
    def corrupt_one(kit_root: Path) -> None:
        next((kit_root / "data" / "test").glob("*.jpg")).write_bytes(b"changed after indexing")

    manifest, cdn_dir = build_synthetic_kit(tmp_path / "srv", mutate_after_index=corrupt_one)
    monkeypatch.setattr(kit, "_open", FakeCDN(cdn_dir).open)
    ctx = FakeCtx()
    with pytest.raises(RuntimeError, match=f"do not match {FILES_INDEX_NAME}"):
        _run(tmp_path, manifest, ctx)
    failed = [c for c in ctx.checks if not c["ok"]]
    assert failed and failed[0]["label"] == f"extracted kit matches {FILES_INDEX_NAME}"
    assert kit.download_state(manifest)["state"] == "empty"  # nothing recorded


def test_split_counts_are_gated_against_the_manifest(served, tmp_path):
    manifest, _ = served
    import dataclasses

    wrong = dataclasses.replace(
        manifest,
        splits=dataclasses.replace(manifest.splits, train=dataclasses.replace(manifest.splits.train, undefined=99)),
    )
    ctx = FakeCtx()
    with pytest.raises(RuntimeError, match="image counts"):
        _run(tmp_path, wrong, ctx)
    assert any(c["label"] == "kit split counts match the manifest" and not c["ok"] for c in ctx.checks)


def test_kit_version_inside_the_kit_must_match_the_manifest(tmp_path, monkeypatch, home):
    manifest, cdn_dir = build_synthetic_kit(tmp_path / "srv", index_kit_version="v2")
    monkeypatch.setattr(kit, "_open", FakeCDN(cdn_dir).open)
    with pytest.raises(RuntimeError, match="asked for v1"):
        _run(tmp_path, manifest)


def test_zip_slip_entry_is_refused(served, tmp_path):
    manifest, cdn = served
    name = manifest.kit.shards[0].name
    buf = Path(tmp_path / "evil.zip")
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{KIT_DIR_NAME}/../evil.txt", b"x")
    data = buf.read_bytes()
    import dataclasses

    evil = dataclasses.replace(manifest.kit.shards[0], sha256=kit._sha256_file(buf), bytes=len(data))
    shards = (evil, *manifest.kit.shards[1:])
    m2 = dataclasses.replace(manifest, kit=dataclasses.replace(manifest.kit, shards=shards))
    cdn.tamper[name] = data
    with pytest.raises(RuntimeError, match="Unsafe"):
        _run(tmp_path, m2)


def test_cancel_between_shards_stays_resumable(served, tmp_path):
    manifest, _ = served
    names = [s.name for s in manifest.kit.shards]
    ctx = FakeCtx(cancel_on_call=2)  # first shard completes, then cancel
    result = _run(tmp_path, manifest, ctx)
    version_dir = _version_dir(tmp_path, manifest)
    assert result["cancelled"] is True and result["resumable"] is True
    assert (version_dir / names[0]).is_file()  # kept for resume
    assert not (version_dir / KIT_DIR_NAME).exists()  # never extracted
    assert not session.CONFIG_PATH.exists()  # session untouched
    assert kit.download_state(manifest)["state"] == "empty"


def test_extra_files_are_reported_not_fatal(served, tmp_path):
    manifest, _ = served
    _run(tmp_path, manifest)
    extra = _version_dir(tmp_path, manifest) / KIT_DIR_NAME / "my-notes.txt"
    extra.write_text("mine", encoding="utf-8")
    ctx = FakeCtx()
    result = _run(tmp_path, manifest, ctx)
    assert result["cancelled"] is False
    assert result["extra_files"] == 1
    assert all(c["ok"] for c in ctx.checks)
    assert extra.is_file()


def test_default_dest_lives_under_the_plugin_home(manifest, home):
    assert kit.default_dest(manifest) == home / "data" / manifest.competition.id
    assert kit.record_path(manifest) == home / "kit" / f"{manifest.competition.id}.json"


def test_resolve_params_defaults_and_rejections(manifest, home, tmp_path):
    params = kit.resolve_params({}, manifest)
    assert params == {"dest_dir": str(kit.default_dest(manifest)), "keep_archives": False}
    assert (kit.default_dest(manifest) / manifest.kit.version).is_dir()  # probed into existence
    with pytest.raises(ValueError, match="absolute"):
        kit.resolve_params({"dest_dir": "relative/path"}, manifest)
    quoted = f'"{_dest(tmp_path)}"'
    assert kit.resolve_params({"dest_dir": quoted, "keep_archives": 1}, manifest) == {
        "dest_dir": str(_dest(tmp_path)),
        "keep_archives": True,
    }


def test_keep_archives_keeps_shards(served, tmp_path):
    manifest, _ = served
    _run(tmp_path, manifest, keep_archives=True)
    assert len(list(_version_dir(tmp_path, manifest).glob("part-*.zip"))) == len(manifest.kit.shards)


def test_download_state_reverifies_disk_and_version(served, tmp_path):
    manifest, _ = served
    assert kit.download_state(manifest) == {"state": "empty"}
    result = _run(tmp_path, manifest)
    state = kit.download_state(manifest)
    assert state["state"] == "success"
    assert state["kit_dir"] == result["kit_dir"]
    assert state["file_count"] == result["file_count"]

    # The manifest moves to v2: same kit on disk is now superseded, not stale, not success.
    newer = bump_version(manifest, "v2")
    state2 = kit.download_state(newer)
    assert state2["state"] == "superseded"
    assert state2["kit_version"] == "v1" and state2["current_version"] == "v2"

    (Path(result["kit_dir"]) / FILES_INDEX_NAME).unlink()
    assert kit.download_state(manifest)["state"] == "stale"


def test_verify_now_passes_then_names_the_tamper(served, tmp_path):
    manifest, _ = served
    result = _run(tmp_path, manifest)
    v = kit.verify_now(manifest)
    assert v["ok"] is True and v["matched"] == result["file_count"]
    assert v["missing_count"] == 0 and v["mismatch_count"] == 0

    tampered = next((Path(result["kit_dir"]) / "data" / "val").rglob("*.jpg"))
    tampered.write_bytes(b"xx")
    v2 = kit.verify_now(manifest)
    assert v2["ok"] is False and v2["mismatch_count"] == 1
    assert tampered.name in v2["mismatch"][0]

    # A superseded kit still verifies against its OWN files.json.
    tampered.write_bytes(b"xx")
    assert kit.verify_now(bump_version(manifest, "v2"))["ok"] is False


def test_verify_now_requires_a_record(manifest, home):
    v = kit.verify_now(manifest)
    assert v["ok"] is False and "No completed download" in v["error"]
