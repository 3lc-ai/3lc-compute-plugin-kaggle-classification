# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic kits for the kit-stage tests.

A tiny kit tree with the real layout (``starter_kit/data/{train,val,test}``,
``sample_submission.csv``, ``files.json``) is built with the SAME format functions the
real builder uses (``tools/build_kit.py``), sharded with a tiny budget so several shards
exist, and served through a Range-aware stub in place of ``kit._open``. The manifest the
tests hand the downloader is the bundled one with small split counts and the synthetic
``kit{}`` block spliced in, so every check the real download runs is exercised offline.
"""

from __future__ import annotations

import dataclasses
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import build_kit  # tools/ is on sys.path (conftest)

from kaggle_classification import manifest as manifest_mod
from kaggle_classification.kit import FILES_INDEX_NAME, KIT_DIR_NAME

SMALL_SPLITS = {"labeled_per_class": 2, "undefined": 3, "val_per_class": 1, "test": 4}
DOC_URL = "https://cdn.test/kaggle/intel-scene/manifest.json"
ALLOWED = frozenset({"cdn.test"})


def small_manifest_data(kit_version: str = "v1") -> dict[str, Any]:
    """The bundled document with tiny split counts and an unpublished kit at ``kit_version``."""
    data = manifest_mod.load_yaml_text(manifest_mod.bundled_path().read_text(encoding="utf-8"))
    data["splits"] = {
        "train": {"labeled_per_class": SMALL_SPLITS["labeled_per_class"], "undefined": SMALL_SPLITS["undefined"]},
        "val": {"per_class": SMALL_SPLITS["val_per_class"], "editable": True},
        "test": {"count": SMALL_SPLITS["test"], "ids_from": "sample_submission.csv"},
    }
    data["kit"] = {"path": f"starter-kit/{kit_version}/", "version": kit_version, "shards": []}
    return data


def _fake_jpeg(seed: int) -> bytes:
    # Not a decodable image (session 1 never decodes); distinct, deterministic bytes, .jpg
    # suffix so the stored-not-deflated branch is exercised.
    return b"\xff\xd8" + bytes([seed % 251]) * 600 + b"\xff\xd9"


def make_kit_tree(root: Path, data: dict[str, Any]) -> Path:
    """``root/starter_kit/...`` matching ``data["classes"]`` and ``data["splits"]``."""
    kit = root / KIT_DIR_NAME
    classes = [c["name"] for c in data["classes"]]
    seed = 0
    for name in classes:
        d = kit / "data" / "train" / name
        d.mkdir(parents=True)
        for i in range(data["splits"]["train"]["labeled_per_class"]):
            (d / f"{name[:2]}{i:04x}.jpg").write_bytes(_fake_jpeg(seed))
            seed += 1
    und = kit / "data" / "train" / "undefined"
    und.mkdir(parents=True)
    for i in range(data["splits"]["train"]["undefined"]):
        (und / f"u{i:04x}.jpg").write_bytes(_fake_jpeg(seed))
        seed += 1
    for name in classes:
        d = kit / "data" / "val" / name
        d.mkdir(parents=True)
        for i in range(data["splits"]["val"]["per_class"]):
            (d / f"v{name[:2]}{i:04x}.jpg").write_bytes(_fake_jpeg(seed))
            seed += 1
    test = kit / "data" / "test"
    test.mkdir(parents=True)
    ids = []
    for i in range(data["splits"]["test"]["count"]):
        stem = f"t{i:04x}"
        (test / f"{stem}.jpg").write_bytes(_fake_jpeg(seed))
        ids.append(stem)
        seed += 1
    cols = data["submission"]["columns"]
    lines = [",".join(cols)] + [",".join([i] + ["0"] * (len(cols) - 1)) for i in sorted(ids)]
    (kit / data["splits"]["test"]["ids_from"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return kit


def build_synthetic_kit(
    srv: Path,
    data: dict[str, Any] | None = None,
    *,
    kit_version: str = "v1",
    shard_bytes: int = 2500,
    index_kit_version: str | None = None,
    mutate_after_index: Callable[[Path], None] | None = None,
) -> tuple[manifest_mod.Manifest, Path]:
    """Build a kit under ``srv`` and return (manifest with the kit block filled, cdn dir).

    ``index_kit_version`` lets a test write a files.json that claims another version;
    ``mutate_after_index`` edits the tree AFTER files.json is written (a staging error).
    """
    data = data or small_manifest_data(kit_version)
    kit = make_kit_tree(srv / "tree", data)
    build_kit.write_files_index(
        kit, competition_id=data["competition"]["id"], kit_version=index_kit_version or kit_version
    )
    if mutate_after_index is not None:
        mutate_after_index(kit)
    cdn = srv / "cdn" / kit_version
    shards = build_kit.shard_kit_tree(kit, cdn, shard_bytes=shard_bytes)
    data["kit"] = build_kit.kit_block(f"starter-kit/{kit_version}/", kit_version, shards)["kit"]
    return manifest_mod.parse_manifest(
        data, source="test", source_detail=str(cdn), document_url=DOC_URL, hosts=ALLOWED
    ), cdn


class _Resp(io.BytesIO):
    status = 200


class FakeCDN:
    """Serves a shard dir like the CDN and records every request."""

    def __init__(self, cdn_dir: Path):
        self.dir = cdn_dir
        self.requests: list[tuple[str, int | None]] = []
        self.tamper: dict[str, bytes] = {}
        self.ignore_ranges = False

    def open(self, url: str, start: int | None = None):
        name = url.rsplit("/", 1)[1]
        self.requests.append((name, start))
        data = self.tamper.get(name, (self.dir / name).read_bytes())
        if start and not self.ignore_ranges:
            resp = _Resp(data[start:])
            resp.status = 206
        else:
            resp = _Resp(data)
            resp.status = 200
        return resp


class FakeCtx:
    def __init__(self, cancel_on_call: int | None = None):
        self.logs: list[str] = []
        self.checks: list[dict] = []
        self.progress: list[dict] = []
        self.facts: dict = {}
        self._cancel_on = cancel_on_call
        self._calls = 0

    def log(self, m):
        self.logs.append(m)

    def set_checks(self, c):
        self.checks = [dict(x) for x in c]

    def set_progress(self, p):
        self.progress.append(dict(p))

    def set_field(self, k, v):
        self.facts[k] = v

    def is_cancelled(self):
        self._calls += 1
        return self._cancel_on is not None and self._calls >= self._cancel_on


def bump_version(manifest: manifest_mod.Manifest, version: str) -> manifest_mod.Manifest:
    """The same manifest shipping a newer kit version (the constant moved)."""
    kit = dataclasses.replace(manifest.kit, version=version, path=f"starter-kit/{version}/")
    return dataclasses.replace(manifest, kit=kit)


def read_index(kit_root: Path) -> dict[str, Any]:
    return json.loads((kit_root / FILES_INDEX_NAME).read_text(encoding="utf-8"))
