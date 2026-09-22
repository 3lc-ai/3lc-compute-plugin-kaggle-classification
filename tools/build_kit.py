# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project. The deterministic sharding is make_kit_manifest.py's; the
# salted rename, re-encode, files.json and mapping.csv are new.
"""Build the starter kit the plugin downloads — dev-only, never shipped in the wheel.

Phase 1 of session 1 ships the two FORMAT definitions the plugin's download stage
verifies against, so tests and the real build share one implementation:

* ``write_files_index(kit_root, ...)`` — ``files.json`` inside the kit: every file's
  relpath, sha256 and bytes (schema 1). ``kaggle_classification.kit`` verifies per file
  against it, not shard-only.
* ``shard_kit_tree(kit_root, out_dir, ...)`` — deterministic shard zips (fixed zip
  timestamps and attributes, sorted paths, size-cut per split group, JPEGs stored not
  deflated), each entry under ``starter_kit/``; returns the ``kit.shards[]`` entries.

Phase 3 adds the pipeline: walk the raw data dir, derive each image's new stem as
``sha256(salt + original_relpath)[:16]``, re-encode as RGB JPEG quality 92 with EXIF
stripped, keep the split/class folder structure, regenerate ``sample_submission.csv``,
write ``mapping.csv`` to the PRIVATE output dir only, gate on the manifest's split counts
and on stem collisions, shard, verify by unzipping, and emit the ``kit{}`` YAML block.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle_classification.kit import DATA_DIR_NAME, FILES_INDEX_NAME, FILES_INDEX_SCHEMA_VERSION, KIT_DIR_NAME

DEFAULT_SHARD_MB = 50
# Fixed zip metadata so regeneration is byte-identical (zip epoch, rw-r--r--).
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o644 << 16
# Already-compressed formats: store, don't deflate.
_STORED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SPLIT_GROUPS = ("train", "val", "test")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_files(kit_root: Path) -> list[tuple[str, Path]]:
    """Every file under the kit root as (posix relpath, path), sorted by relpath (bytewise —
    ``PurePath`` ordering case-folds on Windows, which would make shard hashes OS-specific)."""
    out = [
        (PurePosixPath(p.relative_to(kit_root).as_posix()).as_posix(), p) for p in kit_root.rglob("*") if p.is_file()
    ]
    out.sort(key=lambda t: t[0])
    return out


# ── files.json ─────────────────────────────────────────────────────────────


def write_files_index(kit_root: Path, *, competition_id: str, kit_version: str) -> dict[str, Any]:
    """Write ``<kit_root>/files.json`` covering every file in the tree except itself."""
    files = [
        {"path": rel, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for rel, path in _rel_files(kit_root)
        if rel != FILES_INDEX_NAME
    ]
    index = {
        "schema_version": FILES_INDEX_SCHEMA_VERSION,
        "competition_id": competition_id,
        "kit_version": kit_version,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    (kit_root / FILES_INDEX_NAME).write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")
    return index


# ── Shards ─────────────────────────────────────────────────────────────────


def _group_for(rel: str) -> str:
    parts = PurePosixPath(rel).parts
    if len(parts) > 2 and parts[0] == DATA_DIR_NAME and parts[1] in _SPLIT_GROUPS:
        return f"{DATA_DIR_NAME}-{parts[1]}"
    return "root"


def plan_shards(kit_root: Path, shard_bytes: int) -> list[dict[str, Any]]:
    """Deterministic shard plan: sorted groups, sorted paths, size-cut. A group only splits
    when it exceeds ``shard_bytes``. The root group (files.json, sample_submission.csv, ...)
    sorts first so a single-file top-up touches the smallest shard."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for rel, path in _rel_files(kit_root):
        groups.setdefault(_group_for(rel), []).append((rel, path.stat().st_size))
    shards: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda g: (g != "root", g)):
        chunks: list[list[str]] = [[]]
        size = 0
        for rel, nbytes in groups[group]:
            if chunks[-1] and size + nbytes > shard_bytes:
                chunks.append([])
                size = 0
            chunks[-1].append(rel)
            size += nbytes
        for chunk in chunks:
            if chunk:
                shards.append({"group": group, "files": chunk})
    return shards


def _shard_name(index: int, group: str, seq_in_group: int, group_shards: int) -> str:
    suffix = f"-{seq_in_group:02d}" if group_shards > 1 else ""
    return f"part-{index:02d}-{group}{suffix}.zip"


def shard_kit_tree(
    kit_root: Path, out_dir: Path, *, shard_bytes: int = DEFAULT_SHARD_MB * 1024 * 1024
) -> list[dict[str, Any]]:
    """Write the shard zips into ``out_dir`` and return the ``kit.shards[]`` entries
    (``name``, ``sha256``, ``bytes``, plus ``file_count`` for the build report)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_shards(kit_root, shard_bytes)
    group_totals: dict[str, int] = {}
    for shard in plan:
        group_totals[shard["group"]] = group_totals.get(shard["group"], 0) + 1
    entries: list[dict[str, Any]] = []
    group_seq: dict[str, int] = {}
    for index, shard in enumerate(plan):
        group = shard["group"]
        seq = group_seq.get(group, 0)
        group_seq[group] = seq + 1
        name = _shard_name(index, group, seq, group_totals[group])
        zip_path = out_dir / name
        with zipfile.ZipFile(zip_path, "w") as zf:
            for rel in shard["files"]:
                data = (kit_root / Path(rel)).read_bytes()
                entry = f"{KIT_DIR_NAME}/{rel}"
                info = zipfile.ZipInfo(entry, date_time=_ZIP_DATE_TIME)
                info.external_attr = _ZIP_EXTERNAL_ATTR
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if PurePosixPath(entry).suffix.lower() in _STORED_SUFFIXES
                    else zipfile.ZIP_DEFLATED
                )
                zf.writestr(info, data)
        entries.append({
            "name": name,
            "sha256": _sha256_file(zip_path),
            "bytes": zip_path.stat().st_size,
            "file_count": len(shard["files"]),
        })
    return entries


def kit_block(base_url: str, version: str, shards: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``kit{}`` mapping to paste into the competition manifest."""
    return {
        "kit": {
            "base_url": base_url,
            "version": version,
            "shards": [{"name": s["name"], "sha256": s["sha256"], "bytes": s["bytes"]} for s in shards],
        }
    }


def main(argv: list[str] | None = None) -> int:
    print("tools/build_kit.py: the build pipeline lands in Phase 3 of session 1 (docs/PLAN.md).", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
