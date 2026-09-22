# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Produce the ``cdn/`` folder that mirrors the competition bucket layout — dev-only, gitignored.

    python tools/make_cdn_tree.py --shards-dir "<kit>\\shards" --out cdn [--manifest <yaml>]

Layout under ``--out`` (docs/PLAN.md §A3; identical on the dev and prod tiers):

    kaggle/classification-index.json            mutable   application/json  max-age=60
    kaggle/<id>/manifest.json                   mutable   application/json  max-age=60
    kaggle/<id>/starter-kit/<version>/<shard>    IMMUTABLE application/zip   public, max-age=31536000, immutable
    upload-plan.json                            (beside the tree, not uploaded)

The manifest is the bundled YAML converted to JSON, validated as it will be served (its
``kit.path`` is relative, so the same bytes work on every tier). Every shard is copied from the
built kit and checked against the manifest's ``kit.shards`` (sha256 + bytes) — a shard the
manifest does not list, or a listed shard that is missing or differs, fails the build.
``upload-plan.json`` names every object key with its mutability, Content-Type and
Cache-Control so the console upload (docs/PROMOTION.md) has no judgement calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle_classification import manifest as manifest_mod

MUTABLE_CACHE_CONTROL = "max-age=60"
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tree(
    *,
    shards_dir: Path,
    out: Path,
    manifest_yaml: Path,
    display_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    data = manifest_mod.load_yaml_text(manifest_yaml.read_text(encoding="utf-8"))
    # Validate as prod will serve it; the document URL only fixes the host for the check.
    canonical = f"{manifest_mod.MANIFEST_BASE_URL}/{manifest_mod.manifest_path(str(data['competition']['id']))}"
    man = manifest_mod.parse_manifest(data, source="file", source_detail=str(manifest_yaml), document_url=canonical)
    if not man.kit.published:
        msg = f"{manifest_yaml}: kit.shards is empty; paste the kit block from tools/build_kit.py first"
        raise SystemExit(msg)

    kaggle = out / "kaggle"
    if kaggle.exists():
        if not force:
            msg = f"{kaggle} exists; pass --force to rebuild it"
            raise SystemExit(msg)
        shutil.rmtree(kaggle)
    cid = man.competition.id
    comp_dir = kaggle / cid
    kit_dir = comp_dir / PurePosixPath(man.kit.path)
    kit_dir.mkdir(parents=True)

    plan: list[dict[str, Any]] = []

    def add(key: str, path: Path, *, mutable: bool, content_type: str) -> None:
        plan.append({
            "key": key,
            "local_path": path.relative_to(out).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "mutable": mutable,
            "content_type": content_type,
            "cache_control": MUTABLE_CACHE_CONTROL if mutable else IMMUTABLE_CACHE_CONTROL,
        })

    # Shards: copied and checked against the manifest.
    for shard in man.kit.shards:
        src = shards_dir / shard.name
        if not src.is_file():
            msg = f"shard listed by the manifest is missing from {shards_dir}: {shard.name}"
            raise SystemExit(msg)
        if src.stat().st_size != shard.bytes or _sha256_file(src) != shard.sha256:
            msg = f"shard {shard.name} on disk does not match the manifest (bytes/sha256)"
            raise SystemExit(msg)
        dst = kit_dir / shard.name
        shutil.copyfile(src, dst)
        add(f"kaggle/{cid}/{man.kit.path}{shard.name}", dst, mutable=False, content_type="application/zip")
    extras = sorted(
        p.name for p in shards_dir.iterdir() if p.is_file() and p.name not in {s.name for s in man.kit.shards}
    )
    if extras:
        msg = f"{shards_dir} holds files the manifest does not list: {extras}"
        raise SystemExit(msg)

    # Manifest: YAML -> JSON, byte-identical on every tier.
    manifest_json = comp_dir / manifest_mod.MANIFEST_NAME
    manifest_json.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    add(manifest_mod.manifest_path(cid), manifest_json, mutable=True, content_type="application/json")

    # Index: one active competition, manifest_url relative to the index.
    index = {
        "schema_version": manifest_mod.INDEX_SCHEMA_VERSION,
        "competitions": [
            {
                "id": cid,
                "display_name": display_name or man.competition.display_name,
                "manifest_url": f"{cid}/{manifest_mod.MANIFEST_NAME}",
                "active": True,
            }
        ],
    }
    index_json = out / manifest_mod.INDEX_PATH
    index_json.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8", newline="\n")
    add(manifest_mod.INDEX_PATH, index_json, mutable=True, content_type="application/json")
    # Re-validate the index exactly as the plugin will read it.
    manifest_mod.parse_index(index, f"{manifest_mod.MANIFEST_BASE_URL}/{manifest_mod.INDEX_PATH}")

    plan.sort(key=lambda e: (not e["mutable"], e["key"]))
    upload_plan = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition_id": cid,
        "kit_version": man.kit.version,
        "manifest_sha256": _sha256_file(manifest_json),
        "rules": {
            "mutable": {"cache_control": MUTABLE_CACHE_CONTROL, "note": "invalidate after every change"},
            "immutable": {
                "cache_control": IMMUTABLE_CACHE_CONTROL,
                "note": "never overwrite; new content = new kit version",
            },
        },
        "objects": plan,
    }
    (out / "upload-plan.json").write_text(json.dumps(upload_plan, indent=1) + "\n", encoding="utf-8", newline="\n")
    return upload_plan


def print_tree(out: Path) -> None:
    for path in sorted(p for p in out.rglob("*")):
        rel = path.relative_to(out).as_posix()
        if path.is_file():
            print(f"  {rel}  ({path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("cdn"))
    parser.add_argument("--manifest", type=Path, default=manifest_mod.bundled_path())
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    plan = build_tree(
        shards_dir=args.shards_dir,
        out=args.out,
        manifest_yaml=args.manifest,
        display_name=args.display_name,
        force=args.force,
    )
    print(f"cdn tree at {args.out.resolve()}:")
    print_tree(args.out)
    print(f"\nupload-plan.json: {len(plan['objects'])} objects, {sum(o['bytes'] for o in plan['objects']):,} bytes")
    for o in plan["objects"]:
        print(
            f"  {'MUTABLE  ' if o['mutable'] else 'immutable'}  {o['key']}  {o['content_type']}  '{o['cache_control']}'"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
