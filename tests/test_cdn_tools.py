# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""tools/make_cdn_tree.py and tools/verify_cdn.py against a synthetic kit served by a local
HTTP server: the bucket layout, the upload plan, and the participant-side verification."""

from __future__ import annotations

import json

import make_cdn_tree
import pytest
import verify_cdn
import yaml
from helpers import build_synthetic_kit
from test_manifest_resolution import LocalCDN

from kaggle_classification import manifest as m


@pytest.fixture
def built(tmp_path):
    manifest, shards_dir = build_synthetic_kit(tmp_path / "srv")
    data = m.load_yaml_text(m.bundled_path().read_text(encoding="utf-8"))
    data["splits"] = {
        "train": {
            "labeled_per_class": manifest.splits.train.labeled_per_class,
            "undefined": manifest.splits.train.undefined,
        },
        "val": {"per_class": manifest.splits.val.per_class, "editable": True},
        "test": {"count": manifest.splits.test.count, "ids_from": "sample_submission.csv"},
    }
    data["kit"] = {
        "path": manifest.kit.path,
        "version": manifest.kit.version,
        "shards": [{"name": s.name, "sha256": s.sha256, "bytes": s.bytes} for s in manifest.kit.shards],
    }
    manifest_yaml = tmp_path / "manifest.yaml"
    manifest_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    return {"shards_dir": shards_dir, "manifest_yaml": manifest_yaml, "manifest": manifest, "out": tmp_path / "cdn"}


def test_make_cdn_tree_mirrors_the_bucket_layout(built):
    plan = make_cdn_tree.build_tree(
        shards_dir=built["shards_dir"], out=built["out"], manifest_yaml=built["manifest_yaml"]
    )
    out = built["out"]
    assert (out / "kaggle" / "classification-index.json").is_file()
    assert (out / "kaggle" / "intel-scene" / "manifest.json").is_file()
    shard_dir = out / "kaggle" / "intel-scene" / "starter-kit" / "v1"
    assert sorted(p.name for p in shard_dir.iterdir()) == sorted(s.name for s in built["manifest"].kit.shards)
    keys = [o["key"] for o in plan["objects"]]
    assert keys[:2] == ["kaggle/classification-index.json", "kaggle/intel-scene/manifest.json"]
    assert all(k.startswith("kaggle/intel-scene/starter-kit/v1/") for k in keys[2:])
    for o in plan["objects"]:
        if o["mutable"]:
            assert o["content_type"] == "application/json" and o["cache_control"] == "max-age=60"
        else:
            assert (
                o["content_type"] == "application/zip" and o["cache_control"] == "public, max-age=31536000, immutable"
            )
    assert (out / "upload-plan.json").is_file()
    # The served manifest is JSON of the same document and carries no absolute kit URL.
    served = json.loads((out / "kaggle" / "intel-scene" / "manifest.json").read_text(encoding="utf-8"))
    assert served["kit"]["path"] == "starter-kit/v1/" and "base_url" not in served["kit"]
    index = json.loads((out / "kaggle" / "classification-index.json").read_text(encoding="utf-8"))
    assert index["competitions"] == [
        {
            "id": "intel-scene",
            "display_name": built["manifest"].competition.display_name,
            "manifest_url": "intel-scene/manifest.json",
            "active": True,
        }
    ]
    with pytest.raises(SystemExit, match="--force"):
        make_cdn_tree.build_tree(shards_dir=built["shards_dir"], out=out, manifest_yaml=built["manifest_yaml"])


def test_make_cdn_tree_refuses_a_shard_that_disagrees_with_the_manifest(built):
    victim = next(built["shards_dir"].iterdir())
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="does not match the manifest"):
        make_cdn_tree.build_tree(shards_dir=built["shards_dir"], out=built["out"], manifest_yaml=built["manifest_yaml"])


def test_verify_cdn_against_a_local_server(built, monkeypatch):
    make_cdn_tree.build_tree(shards_dir=built["shards_dir"], out=built["out"], manifest_yaml=built["manifest_yaml"])
    server = LocalCDN(built["out"])
    try:
        # Serve the cdn/ tree as-is (LocalCDN makes kaggle/ if missing; ours exists).
        monkeypatch.setenv(m.MANIFEST_BASE_URL_ENV, server.base)
        notes: list[str] = []
        report = verify_cdn.verify(server.base, competition_id=None, skip_header_checks=False, log=notes.append)
        assert report["ok"], report["problems"]
        assert report["shards"] == len(built["manifest"].kit.shards)
        assert any("loopback" in n and "skipped" in n for n in notes)  # says so explicitly
        assert all(o["cache_control"] is None for o in report["objects"])  # http.server sends none

        # Tamper with a served shard: sha256 and bytes are caught, ETag never consulted.
        shard = next((built["out"] / "kaggle" / "intel-scene" / "starter-kit" / "v1").iterdir())
        shard.write_bytes(b"\x00" * 10)
        report2 = verify_cdn.verify(server.base, competition_id=None, skip_header_checks=True, log=lambda *a: None)
        assert not report2["ok"] and any("sha256" in p for p in report2["problems"])
    finally:
        server.close()


def test_verify_cdn_header_policy(monkeypatch):
    assert verify_cdn._max_age("public, max-age=31536000, immutable") == 31536000
    assert verify_cdn._max_age("max-age=60") == 60
    assert verify_cdn._max_age(None) is None and verify_cdn._max_age("no-store") is None
