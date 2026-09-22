# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project. The deterministic sharding is make_kit_manifest.py's; the
# salted rename, re-encode, files.json, mapping.csv and the verification pass are new.
"""Build the starter kit the plugin downloads — dev-only, never shipped in the wheel.

    python tools/build_kit.py --data-dir <raw data> --salt-file <file> \
        --kit-out <kit dir> --private-out <judge dir> [--shard-mb 50] [--kit-version v1]
        [--kit-path starter-kit/v1/] [--manifest <bundled yaml>] [--force]

Pipeline:

1. Read the salt from ``--salt-file`` (exact bytes; only a trailing newline is stripped;
   missing or shorter than 16 characters fails). The salt is never printed, logged or
   written anywhere; the verification pass scans every output for it.
2. Walk ``<data>/train/<class>/``, ``<data>/train/undefined/``, ``<data>/val/<class>/`` and
   the flat ``<data>/test/``; anything else in the tree is an error. Class folders must be
   the manifest's class names.
3. Every image's new stem is ``sha256(salt + original_relpath)[:16]``; a stem collision
   anywhere in the kit is a hard failure. Per-split counts must equal the manifest's
   ``splits{}`` before anything is written.
4. Re-encode every image as RGB JPEG quality 92 with EXIF and ICC stripped, into
   ``<kit-out>/tree/starter_kit/data/...`` with the split/class structure preserved
   (``undefined`` stays ``train/undefined/``, test stays flat).
5. Write ``sample_submission.csv`` (test ids sorted, ``prediction=0``, ``confidence=0.5``)
   and ``files.json`` inside the kit; write ``mapping.csv`` (original_relpath, new_relpath,
   split, class) to the PRIVATE output dir only — it is the judge's key. The class column is
   empty for test and ``undefined`` for pool rows: their true class is not known to us.
6. Shard the tree into ``<kit-out>/shards/<competition>-<version>-NN.zip`` (deterministic:
   fixed zip timestamps and attributes, bytewise-sorted paths, size-cut per split group,
   JPEGs stored not deflated), so the same salt and data rebuild byte-identical shards.
7. Write ``kit-manifest-block.yaml`` beside the kit output dir (its parent): the ``kit{}``
   block plus a ``splits{}`` block with the actual counts, ready to paste.
8. Verify by unzipping every shard to a temp dir: recount, no stem collisions, every image
   name an opaque 16-hex stem, no original filename anywhere, no class name in an
   ``undefined`` or test path, ``sample_submission.csv`` ids == test stems, every
   ``files.json`` entry present with matching sha256 and bytes and nothing outside
   ``data/train/``, ``data/val/``, ``data/test/`` and ``sample_submission.csv``, and the
   salt bytes absent from every output file.

The two FORMAT functions the plugin's download stage verifies against live here too, so
tests and the real build share one implementation: ``write_files_index`` and
``shard_kit_tree``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle_classification import manifest as manifest_mod
from kaggle_classification.kit import (
    DATA_DIR_NAME,
    FILES_INDEX_NAME,
    FILES_INDEX_SCHEMA_VERSION,
    KIT_DIR_NAME,
    UNDEFINED_DIR_NAME,
    expected_split_counts,
    split_counts,
    verify_tree,
)

DEFAULT_SHARD_MB = 50
JPEG_QUALITY = 92
STEM_LEN = 16
MIN_SALT_CHARS = 16
# Fixed zip metadata so regeneration is byte-identical (zip epoch, rw-r--r--).
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o644 << 16
# Already-compressed formats: store, don't deflate.
_STORED_SUFFIXES = {".jpg", ".jpeg", ".png"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SPLIT_GROUPS = ("train", "val", "test")
_STEM_RE = re.compile(r"^[0-9a-f]{16}$")


class BuildError(RuntimeError):
    """A hard failure of the build or its verification; the message says what and where."""


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


def _shard_name(index: int, group: str, seq_in_group: int, group_shards: int, name_prefix: str | None) -> str:
    if name_prefix:
        return f"{name_prefix}-{index:02d}.zip"
    suffix = f"-{seq_in_group:02d}" if group_shards > 1 else ""
    return f"part-{index:02d}-{group}{suffix}.zip"


def shard_kit_tree(
    kit_root: Path,
    out_dir: Path,
    *,
    shard_bytes: int = DEFAULT_SHARD_MB * 1024 * 1024,
    name_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Write the shard zips into ``out_dir`` and return the ``kit.shards[]`` entries
    (``name``, ``sha256``, ``bytes``, plus ``file_count`` for the build report).

    ``name_prefix`` gives ``<prefix>-NN.zip`` (the published naming); without it the shards
    are named after their split group (the synthetic-kit tests)."""
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
        name = _shard_name(index, group, seq, group_totals[group], name_prefix)
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


def kit_block(kit_path: str, version: str, shards: list[dict[str, Any]]) -> dict[str, Any]:
    """The ``kit{}`` mapping to paste into the competition manifest."""
    return {
        "kit": {
            "path": kit_path,
            "version": version,
            "shards": [{"name": s["name"], "sha256": s["sha256"], "bytes": s["bytes"]} for s in shards],
        }
    }


# ── The build ──────────────────────────────────────────────────────────────


def read_salt(path: Path) -> bytes:
    """The salt's exact bytes, minus one trailing newline. Fails loudly; never logs the value."""
    if not path.is_file():
        msg = f"salt file not found: {path}"
        raise BuildError(msg)
    raw = path.read_bytes()
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith(b"\n"):
        raw = raw[:-1]
    if len(raw) < MIN_SALT_CHARS:
        msg = f"salt file {path} holds fewer than {MIN_SALT_CHARS} characters after stripping a trailing newline"
        raise BuildError(msg)
    return raw


def derive_stem(salt: bytes, original_relpath: str) -> str:
    return hashlib.sha256(salt + original_relpath.encode("utf-8")).hexdigest()[:STEM_LEN]


@dataclass(frozen=True)
class Planned:
    original_relpath: str  # posix, relative to the data dir
    new_relpath: str  # posix, relative to the kit's data dir
    split: str  # train | val | test
    class_name: str  # "" for test, "undefined" for pool rows
    source: Path


def walk_source(data_dir: Path, class_names: list[str]) -> list[tuple[str, str, str, Path]]:
    """(original_relpath, split, class_name, path) for every image; anything unexpected is an error."""
    classes = set(class_names)
    found: list[tuple[str, str, str, Path]] = []
    expected_splits = set(_SPLIT_GROUPS)
    for split_dir in sorted(data_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name not in expected_splits:
            msg = f"unexpected entry in the data dir: {split_dir.name} (expected only {sorted(expected_splits)})"
            raise BuildError(msg)
    for split in _SPLIT_GROUPS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            msg = f"missing split directory: {split_dir}"
            raise BuildError(msg)
        if split == "test":
            for p in sorted(split_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in _IMAGE_SUFFIXES:
                    msg = f"test/ must be flat and hold only images: {p.name}"
                    raise BuildError(msg)
                found.append((f"test/{p.name}", "test", "", p))
            continue
        for class_dir in sorted(split_dir.iterdir()):
            allowed = classes | ({UNDEFINED_DIR_NAME} if split == "train" else set())
            if not class_dir.is_dir() or class_dir.name not in allowed:
                msg = f"unexpected entry {split}/{class_dir.name}: expected one of {sorted(allowed)}"
                raise BuildError(msg)
            for p in sorted(class_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in _IMAGE_SUFFIXES:
                    msg = f"non-image entry in {split}/{class_dir.name}: {p.name}"
                    raise BuildError(msg)
                found.append((f"{split}/{class_dir.name}/{p.name}", split, class_dir.name, p))
    return found


def plan_kit(data_dir: Path, salt: bytes, manifest: manifest_mod.Manifest) -> list[Planned]:
    planned: list[Planned] = []
    stems: dict[str, str] = {}
    for rel, split, class_name, path in walk_source(data_dir, manifest.class_names):
        stem = derive_stem(salt, rel)
        if stem in stems:
            msg = f"stem collision: {rel} and {stems[stem]} both derive {stem}; choose another salt"
            raise BuildError(msg)
        stems[stem] = rel
        folder = f"{split}/{class_name}" if class_name else split
        planned.append(Planned(rel, f"{folder}/{stem}.jpg", split, class_name, path))
    return planned


def planned_counts(planned: list[Planned]) -> dict[str, int]:
    counts = {"train_labeled": 0, "train_undefined": 0, "val": 0, "test": 0}
    for p in planned:
        if p.split == "train":
            counts["train_undefined" if p.class_name == UNDEFINED_DIR_NAME else "train_labeled"] += 1
        else:
            counts[p.split] += 1
    return counts


def gate_counts(planned: list[Planned], manifest: manifest_mod.Manifest) -> dict[str, int]:
    got, want = planned_counts(planned), expected_split_counts(manifest)
    if got != want:
        msg = f"per-split counts {got} do not match the manifest's splits {want}"
        raise BuildError(msg)
    per_class = dict.fromkeys(manifest.class_names, 0)
    for p in planned:
        if p.split == "train" and p.class_name in per_class:
            per_class[p.class_name] += 1
    if len(set(per_class.values())) != 1:
        msg = f"labeled train images are not balanced per class: {per_class}"
        raise BuildError(msg)
    return got


def reencode(src: Path, dst: Path) -> None:
    """RGB JPEG at quality 92; EXIF and ICC are not carried over."""
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, "JPEG", quality=JPEG_QUALITY, exif=b"", icc_profile=None)


def write_sample_submission(kit_root: Path, manifest: manifest_mod.Manifest, test_stems: list[str]) -> Path:
    cols = list(manifest.submission.columns)
    placeholders = {"prediction": "0", "confidence": "0.5"}
    lines = [",".join(cols)]
    for stem in sorted(test_stems):
        lines.append(",".join([stem] + [placeholders.get(c, "0") for c in cols[1:]]))
    path = kit_root / manifest.splits.test.ids_from
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_mapping(private_dir: Path, planned: list[Planned]) -> Path:
    private_dir.mkdir(parents=True, exist_ok=True)
    path = private_dir / "mapping.csv"
    lines = ["original_relpath,new_relpath,split,class"]
    for p in sorted(planned, key=lambda x: x.original_relpath):
        lines.append(f"{p.original_relpath},{DATA_DIR_NAME}/{p.new_relpath},{p.split},{p.class_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def write_manifest_block(
    path: Path, *, kit_path: str, version: str, shards: list[dict[str, Any]], counts: dict[str, int], manifest
) -> dict[str, Any]:
    import yaml

    n = manifest.num_classes
    block = {
        **kit_block(kit_path, version, shards),
        "splits": {
            "train": {"labeled_per_class": counts["train_labeled"] // n, "undefined": counts["train_undefined"]},
            "val": {"per_class": counts["val"] // n, "editable": manifest.splits.val.editable},
            "test": {"count": counts["test"], "ids_from": manifest.splits.test.ids_from},
        },
    }
    header = (
        f"# kit{{}} and splits{{}} blocks for {manifest.competition.id} kit {version}, built "
        f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} by tools/build_kit.py.\n"
        "# Paste both into the competition manifest (remote first, then the bundled copy).\n"
    )
    path.write_text(header + yaml.safe_dump(block, sort_keys=False), encoding="utf-8", newline="\n")
    return block


def build(
    *,
    data_dir: Path,
    salt_file: Path,
    kit_out: Path,
    private_out: Path,
    manifest: manifest_mod.Manifest,
    kit_version: str,
    kit_path: str,
    shard_bytes: int = DEFAULT_SHARD_MB * 1024 * 1024,
    force: bool = False,
    log=print,
) -> dict[str, Any]:
    """Run steps 1–7 and return the build report (counts, shards, paths, timings)."""
    t0 = time.monotonic()
    salt = read_salt(salt_file)
    tree_dir, shards_dir = kit_out / "tree", kit_out / "shards"
    for d in (tree_dir, shards_dir):
        if d.exists() and any(d.iterdir()):
            if not force:
                msg = f"{d} is not empty; pass --force to rebuild into it"
                raise BuildError(msg)
            import shutil

            shutil.rmtree(d)
    kit_root = tree_dir / KIT_DIR_NAME

    planned = plan_kit(data_dir, salt, manifest)
    counts = gate_counts(planned, manifest)
    log(f"planned {len(planned)} images: {counts}")

    for i, p in enumerate(planned, 1):
        reencode(p.source, kit_root / DATA_DIR_NAME / p.new_relpath)
        if i % 1000 == 0:
            log(f"  re-encoded {i}/{len(planned)}")
    test_stems = [PurePosixPath(p.new_relpath).stem for p in planned if p.split == "test"]
    write_sample_submission(kit_root, manifest, test_stems)
    mapping_path = write_mapping(private_out, planned)
    index = write_files_index(kit_root, competition_id=manifest.competition.id, kit_version=kit_version)
    prefix = f"{manifest.competition.id}-{kit_version}"
    shards = shard_kit_tree(kit_root, shards_dir, shard_bytes=shard_bytes, name_prefix=prefix)
    block_path = kit_out.parent / "kit-manifest-block.yaml"
    block = write_manifest_block(
        block_path, kit_path=kit_path, version=kit_version, shards=shards, counts=counts, manifest=manifest
    )
    elapsed = time.monotonic() - t0
    log(f"built {len(shards)} shards, {sum(s['bytes'] for s in shards):,} bytes, in {elapsed:.1f}s")
    return {
        "counts": counts,
        "images": len(planned),
        "shards": shards,
        "total_bytes": sum(s["bytes"] for s in shards),
        "file_count": index["file_count"],
        "kit_root": str(kit_root),
        "shards_dir": str(shards_dir),
        "mapping": str(mapping_path),
        "block": str(block_path),
        "manifest_block": block,
        "example": next((p for p in planned if p.split == "train" and p.class_name != UNDEFINED_DIR_NAME), None),
        "planned": planned,
        "build_elapsed_s": round(elapsed, 1),
    }


# ── Verification ───────────────────────────────────────────────────────────


def _text_tokens(text: str) -> set[str]:
    return set(re.split(r"[^A-Za-z0-9_]+", text))


def verify_build(
    report: dict[str, Any],
    manifest: manifest_mod.Manifest,
    *,
    salt_file: Path,
    private_out: Path,
    kit_out: Path,
    tmp_dir: Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Step 8: unzip every shard to a temp dir and check the kit from the participant's side."""
    t0 = time.monotonic()
    planned: list[Planned] = report["planned"]
    shards_dir = Path(report["shards_dir"])
    salt = read_salt(salt_file)
    problems: list[str] = []

    with tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None) as tmp:
        dest = Path(tmp)
        for s in report["shards"]:
            zp = shards_dir / s["name"]
            if _sha256_file(zp) != s["sha256"] or zp.stat().st_size != s["bytes"]:
                problems.append(f"shard on disk disagrees with the block: {s['name']}")
            with zipfile.ZipFile(zp) as zf:
                for info in zf.infolist():
                    n = info.filename.replace("\\", "/")
                    if n.startswith("/") or ".." in n.split("/") or not n.startswith(KIT_DIR_NAME + "/"):
                        problems.append(f"unsafe or misplaced entry {n} in {s['name']}")
                zf.extractall(dest)
        kit_root = dest / KIT_DIR_NAME

        # Recount from the unzipped tree.
        got, want = split_counts(kit_root), expected_split_counts(manifest)
        if got != want:
            problems.append(f"recount from shards {got} != manifest {want}")

        # Every image name is an opaque stem, unique, and no original name survives anywhere.
        original_stems = {PurePosixPath(p.original_relpath).stem for p in planned}
        original_stems |= {PurePosixPath(p.original_relpath).name for p in planned}
        class_names = set(manifest.class_names)
        seen: set[str] = set()
        for path in kit_root.rglob("*"):
            if not path.is_file():
                continue
            rel = PurePosixPath(path.relative_to(kit_root).as_posix())
            if rel.parts[0] == DATA_DIR_NAME:
                if not _STEM_RE.match(rel.stem) or rel.suffix != ".jpg":
                    problems.append(f"image name is not an opaque stem: {rel}")
                if rel.stem in seen:
                    problems.append(f"stem collision in the kit: {rel.stem}")
                seen.add(rel.stem)
                hidden = (len(rel.parts) > 2 and rel.parts[1] == "test") or (
                    len(rel.parts) > 3 and rel.parts[2] == UNDEFINED_DIR_NAME
                )
                if hidden and any(c in rel.as_posix().lower() for c in class_names):
                    problems.append(f"class name in a pool/test path: {rel}")
            for tok in _text_tokens(rel.as_posix()):
                if tok in original_stems:
                    problems.append(f"original file name survives in a kit path: {rel}")
        for text_name in (FILES_INDEX_NAME, manifest.splits.test.ids_from):
            tokens = _text_tokens((kit_root / text_name).read_text(encoding="utf-8"))
            leaked = tokens & original_stems
            if leaked:
                problems.append(f"original file names inside {text_name}: {sorted(leaked)[:5]}")

        # sample_submission ids == test stems, placeholders as specified.
        sub_lines = (kit_root / manifest.splits.test.ids_from).read_text(encoding="utf-8").splitlines()
        header, rows = sub_lines[0].split(","), [ln.split(",") for ln in sub_lines[1:]]
        if header != list(manifest.submission.columns):
            problems.append(f"sample_submission header {header} != {list(manifest.submission.columns)}")
        test_stems = sorted(p.stem for p in (kit_root / DATA_DIR_NAME / "test").iterdir())
        if [r[0] for r in rows] != test_stems:
            problems.append("sample_submission ids differ from the test stems")
        if any(r[1:] != ["0", "0.5"] for r in rows):
            problems.append("sample_submission placeholders are not prediction=0, confidence=0.5")

        # files.json: complete, correct, and confined to the kit's own layout.
        index = json.loads((kit_root / FILES_INDEX_NAME).read_text(encoding="utf-8"))
        delta = verify_tree(index, kit_root)
        if delta["missing"] or delta["mismatch"] or delta["extra"]:
            problems.append(f"files.json vs tree: {delta}")
        allowed_prefixes = (f"{DATA_DIR_NAME}/train/", f"{DATA_DIR_NAME}/val/", f"{DATA_DIR_NAME}/test/")
        for entry in index["files"]:
            path_ = str(entry["path"])
            if not (path_.startswith(allowed_prefixes) or path_ == manifest.splits.test.ids_from):
                problems.append(f"files.json path outside the kit layout: {path_}")
        if index.get("kit_version") != report["manifest_block"]["kit"]["version"]:
            problems.append("files.json kit_version disagrees with the block")

        # mapping.csv never ships; the private copy is complete.
        if (kit_root / "mapping.csv").exists() or any(p.name == "mapping.csv" for p in kit_root.rglob("*")):
            problems.append("mapping.csv found inside the kit")
        mapping_lines = (private_out / "mapping.csv").read_text(encoding="utf-8").splitlines()
        if len(mapping_lines) - 1 != len(planned):
            problems.append("mapping.csv row count differs from the plan")

    # The salt appears in no output file (kit, private dir minus the salt file itself, the block).
    scanned = 0
    for root in (kit_out, private_out, Path(report["block"]).parent):
        for path in root.rglob("*") if root.is_dir() else []:
            if not path.is_file() or path.resolve() == salt_file.resolve():
                continue
            if root == Path(report["block"]).parent and path.parent != root:
                continue  # the parent scan is for the block file only; the kit is scanned via kit_out
            if salt in path.read_bytes():
                problems.append(f"SALT BYTES FOUND IN {path}")
            scanned += 1

    elapsed = time.monotonic() - t0
    ok = not problems
    for p in problems:
        log("FAIL " + p)
    log(f"verify: {'PASS' if ok else 'FAIL'} in {elapsed:.1f}s ({scanned} output files scanned for the salt)")
    return {"ok": ok, "problems": problems, "verify_elapsed_s": round(elapsed, 1), "salt_scanned_files": scanned}


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--salt-file", required=True, type=Path, help="file holding the salt; the value is never printed"
    )
    parser.add_argument("--kit-out", required=True, type=Path)
    parser.add_argument("--private-out", required=True, type=Path, help="mapping.csv goes here and nowhere else")
    parser.add_argument("--shard-mb", type=int, default=DEFAULT_SHARD_MB)
    parser.add_argument("--kit-version", default=None, help="default: the bundled manifest's kit.version")
    parser.add_argument(
        "--kit-path", default=None, help="kit.path relative to the manifest (default: starter-kit/<version>/)"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="manifest YAML for the splits gate (default: bundled)"
    )
    parser.add_argument("--tmp-dir", type=Path, default=None, help="where the verification unzip happens")
    parser.add_argument("--force", action="store_true", help="rebuild into a non-empty kit-out")
    args = parser.parse_args(argv)

    if args.manifest:
        manifest = manifest_mod.parse_manifest(
            manifest_mod.load_yaml_text(args.manifest.read_text(encoding="utf-8")),
            source="file",
            source_detail=str(args.manifest),
        )
    else:
        manifest = manifest_mod.load_bundled()
    version = args.kit_version or manifest.kit.version
    kit_path = (args.kit_path or f"starter-kit/{version}/").rstrip("/") + "/"

    try:
        report = build(
            data_dir=args.data_dir,
            salt_file=args.salt_file,
            kit_out=args.kit_out,
            private_out=args.private_out,
            manifest=manifest,
            kit_version=version,
            kit_path=kit_path,
            shard_bytes=args.shard_mb * 1024 * 1024,
            force=args.force,
        )
        result = verify_build(
            report,
            manifest,
            salt_file=args.salt_file,
            private_out=args.private_out,
            kit_out=args.kit_out,
            tmp_dir=args.tmp_dir,
        )
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"counts: {report['counts']}")
    for s in report["shards"]:
        print(f"  {s['name']}  {s['bytes']:>12,} bytes  {s['file_count']:>5} files  sha256 {s['sha256']}")
    print(
        f"total: {len(report['shards'])} shards, {report['total_bytes']:,} bytes, {report['file_count']} files in files.json"
    )
    print(f"build {report['build_elapsed_s']}s, verify {result['verify_elapsed_s']}s")
    ex = report["example"]
    if ex:
        print(f"example: {ex.original_relpath} -> {DATA_DIR_NAME}/{ex.new_relpath}")
    print(f"kit block: {report['block']}")
    print(f"mapping (private): {report['mapping']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
