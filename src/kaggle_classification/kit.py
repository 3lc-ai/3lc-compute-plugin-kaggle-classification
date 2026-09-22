# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Starter-kit download and verification (job kind ``download_kit``).

Downloads every shard the competition manifest's ``kit{}`` block lists with
sha256 verification and Range-based resume, extracts the kit tree into
``<dest>/<kit.version>/``, verifies EVERY file against the ``files.json`` the
kit builder placed inside the kit, checks the per-split image counts against
the manifest's ``splits{}``, and finally publishes the kit directory into the
shared session so the Import form starts populated.

Integrity is sha256 from the manifest (shards) and from ``files.json``
(files), never the HTTP ETag: multipart-upload ETags are not content hashes.

Resume: a shard downloads to ``<name>.part``; an interrupted or cancelled job
leaves .part files behind, and the next job continues them with an HTTP Range
request (206 -> append; a 200 answer restarts that shard). A shard already
complete on disk with a matching sha256 is skipped without a request. Shard
archives are deleted only after the whole tree verifies, so a failed run
always resumes.

The ``ctx`` a job target receives is duck-typed (``log``, ``set_checks``,
``set_progress``, ``set_field``, ``is_cancelled``); ``__init__.run_job`` adapts
the SDK ``JobContext`` onto it, and tests pass a plain fake.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from kaggle_classification import session

if TYPE_CHECKING:
    from kaggle_classification.manifest import Kit, Manifest

_CHUNK = 1 << 20  # 1 MiB reads
# Cancellation/progress cadence in chunks: once per _CHUNK would be a record write per MB.
_POLL_EVERY = 4
_TIMEOUT = 30  # seconds, per request
_RETRIES = 2  # attempts per shard before the job fails
FILES_INDEX_SCHEMA_VERSION = 1

# Every shard zip carries its entries under this directory, so the extracted tree is
# <version_dir>/<KIT_DIR_NAME>/... and the shards beside it never count as extras.
KIT_DIR_NAME = "starter_kit"
FILES_INDEX_NAME = "files.json"
# The kit tree as tools/build_kit.py lays it out (and as the importer expects it).
DATA_DIR_NAME = "data"
UNDEFINED_DIR_NAME = "undefined"

# Participant-facing (renders in a UI callout).
_RERUN_RESUMES = "Run the download again. Completed shards are kept, and the job resumes where it stopped."


class _Cancelled(Exception):
    """Internal: unwinds the download loop on a cooperative cancel."""


# ── Locations ──────────────────────────────────────────────────────────────


def default_dest(manifest: Manifest) -> Path:
    """``<plugin home>/data/<competition id>`` — the kit lives with everything else the plugin owns."""
    return session.PLUGIN_HOME / "data" / manifest.competition.id


def record_path(manifest: Manifest) -> Path:
    """Where a completed download records its facts (dest, kit dir, version, file count)."""
    return session.PLUGIN_HOME / "kit" / f"{manifest.competition.id}.json"


def kit_root_of(version_dir: Path) -> Path:
    return version_dir / KIT_DIR_NAME


def shard_url(kit: Kit, name: str) -> str:
    return f"{kit.base_url}/{name}"


# ── Primitives ─────────────────────────────────────────────────────────────


def _open(url: str, start: int | None = None):
    """One thin seam over urllib (the tests' stub point). ``start`` adds an HTTP Range header
    for resume; callers must handle a 200 (range ignored)."""
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=_TIMEOUT)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_params(data: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """Resolve and probe the download params. Shared by the validation route and
    ``run_download`` (defense in depth: the host ``/run`` path skips validation).
    Raises ``ValueError`` with a participant-facing message.

    ``keep_archives`` defaults to FALSE: shards are only deleted AFTER the tree
    verifies, so every failure path still finds them on disk.
    """
    raw = str(data.get("dest_dir") or "").strip().strip('"')
    dest = Path(raw).expanduser() if raw else default_dest(manifest)
    if not dest.is_absolute():
        msg = f"Destination must be an absolute path, got: {dest}"
        raise ValueError(msg)
    version_dir = dest / manifest.kit.version
    try:
        version_dir.mkdir(parents=True, exist_ok=True)
        probe = version_dir / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        msg = f"Destination is not writable: {dest} ({exc})"
        raise ValueError(msg) from exc
    return {"dest_dir": str(dest), "keep_archives": bool(data.get("keep_archives"))}


def _free_space(version_dir: Path, kit: Kit) -> tuple[bool, str]:
    """Peak usage is shards + extracted tree (archives are deleted only after verification),
    minus whatever a previous attempt already left on disk."""
    have = sum(p.stat().st_size for p in version_dir.rglob("*") if p.is_file())
    needed = max(0, 2 * kit.total_bytes + (100 << 20) - have)
    free = shutil.disk_usage(version_dir).free
    return free >= needed, f"{free / 1e9:.1f} GB free, ~{needed / 1e9:.1f} GB needed"


def _download_shard(
    kit: Kit,
    entry: Any,
    version_dir: Path,
    log: Callable[[str], None],
    report: Callable[[int], None],
    is_cancelled: Callable[[], bool],
) -> None:
    """One shard: skip if already verified on disk, else resume/download, then sha256-verify
    and finalize (.part -> final rename)."""
    name, size, sha = entry.name, int(entry.bytes), str(entry.sha256)
    final = version_dir / name
    part = version_dir / (name + ".part")

    if final.is_file():
        if final.stat().st_size == size and _sha256_file(final) == sha:
            log(f"{name}: already on disk, sha256 verified, skipped")
            report(size)
            return
        log(f"{name}: on disk but does not match the manifest, re-downloading")
        final.unlink()

    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            start = part.stat().st_size if part.is_file() else 0
            if start >= size:  # over-long partial can only be corrupt
                part.unlink()
                start = 0
            if start:
                log(f"{name}: resuming at byte {start:,} of {size:,}")
            received = start
            with _open(shard_url(kit, name), start or None) as resp, part.open("r+b" if start else "wb") as f:
                if start:
                    if getattr(resp, "status", 200) == 206:
                        f.seek(0, 2)
                    else:
                        log(f"{name}: range request not honored, restarting the shard")
                        f.seek(0)
                        f.truncate()
                        received = 0
                chunks = 0
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)
                    chunks += 1
                    if chunks % _POLL_EVERY == 0:
                        report(min(received, size))
                        if is_cancelled():
                            raise _Cancelled()
            if received != size:
                msg = f"connection ended at {received:,} of {size:,} bytes"
                raise OSError(msg)
            if _sha256_file(part) != sha:
                part.unlink()  # nothing in it is trustworthy: restart clean
                msg = "sha256 mismatch after download"
                raise OSError(msg)
            part.replace(final)
            log(f"{name}: {size:,} bytes, sha256 verified")
            report(size)
            return
        except _Cancelled:
            raise
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt < _RETRIES:
                log(f"{name}: {exc}, retrying")
    msg = f"Could not download {name}: {last_error}. {_RERUN_RESUMES}"
    raise RuntimeError(msg)


# ── Verification ───────────────────────────────────────────────────────────


def load_files_index(kit_root: Path) -> dict[str, Any]:
    """The per-file index the kit builder wrote inside the kit."""
    path = kit_root / FILES_INDEX_NAME
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema_version") != FILES_INDEX_SCHEMA_VERSION:
        msg = (
            f"{FILES_INDEX_NAME} uses schema {index.get('schema_version')}, this plugin understands "
            f"schema {FILES_INDEX_SCHEMA_VERSION}. Update the plugin, then run the download again."
        )
        raise RuntimeError(msg)
    if not isinstance(index.get("files"), list):
        msg = f"{FILES_INDEX_NAME}: 'files' must be a list"
        raise RuntimeError(msg)
    return index


def verify_tree(files_index: dict[str, Any], kit_root: Path) -> dict[str, Any]:
    """Compare the extracted kit tree against ``files.json`` (path, bytes, sha256). Walks only
    the kit dir, so shards beside it never count as extras; the index itself is exempt."""
    expected = {str(f["path"]): f for f in files_index["files"]}
    actual: dict[str, Path] = {}
    if kit_root.is_dir():
        for p in kit_root.rglob("*"):
            if p.is_file():
                rel = PurePosixPath(p.relative_to(kit_root).as_posix()).as_posix()
                if rel != FILES_INDEX_NAME:
                    actual[rel] = p
    out: dict[str, Any] = {"matched": 0, "mismatch": [], "missing": [], "extra": []}
    for path, entry in expected.items():
        p = actual.get(path)
        if p is None:
            out["missing"].append(path)
        elif p.stat().st_size != int(entry["bytes"]) or _sha256_file(p) != str(entry["sha256"]):
            out["mismatch"].append(path)
        else:
            out["matched"] += 1
    out["extra"] = sorted(set(actual) - set(expected))
    out["mismatch"].sort()
    out["missing"].sort()
    return out


def split_counts(kit_root: Path) -> dict[str, int]:
    """Image counts by split as the kit lays them out: ``data/train/<class>/`` (labeled),
    ``data/train/undefined/``, ``data/val/<class>/``, ``data/test/`` (flat)."""
    data = kit_root / DATA_DIR_NAME
    counts = {"train_labeled": 0, "train_undefined": 0, "val": 0, "test": 0}
    train = data / "train"
    if train.is_dir():
        for class_dir in train.iterdir():
            if not class_dir.is_dir():
                continue
            n = sum(1 for p in class_dir.iterdir() if p.is_file())
            if class_dir.name == UNDEFINED_DIR_NAME:
                counts["train_undefined"] += n
            else:
                counts["train_labeled"] += n
    val = data / "val"
    if val.is_dir():
        counts["val"] = sum(1 for p in val.rglob("*") if p.is_file())
    test = data / "test"
    if test.is_dir():
        counts["test"] = sum(1 for p in test.iterdir() if p.is_file())
    return counts


def expected_split_counts(manifest: Manifest) -> dict[str, int]:
    return {
        "train_labeled": manifest.splits.train.labeled_per_class * manifest.num_classes,
        "train_undefined": manifest.splits.train.undefined,
        "val": manifest.splits.val.per_class * manifest.num_classes,
        "test": manifest.splits.test.count,
    }


# ── Record and revisit state ───────────────────────────────────────────────


def _write_record(manifest: Manifest, facts: dict[str, Any]) -> None:
    path = record_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**facts, "completed_at": time.time()}, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_record(manifest: Manifest) -> dict[str, Any] | None:
    path = record_path(manifest)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def download_state(manifest: Manifest) -> dict[str, Any]:
    """Revisit state for the Download section, re-verified against disk AND the shipped version.

      "empty"      — no completed download on record.
      "success"    — the kit on disk is the version the manifest ships.
      "superseded" — a complete kit of an OLDER version. Not a fault: the participant keeps
                     training; the copy names the newer kit.
      "stale"      — the recorded kit is gone from disk.

    "superseded" is not folded into "stale": one state string over two conditions is the
    divergence class the ExDark v1.2.12 release closed, and the two need opposite copy.
    """
    record = read_record(manifest)
    if not record:
        return {"state": "empty"}
    kit_dir = str(record.get("kit_dir") or "")
    kit_root = Path(kit_dir) if kit_dir else None
    if kit_root is None or not (kit_root / FILES_INDEX_NAME).is_file():
        return {"state": "stale", "reason": f"kit no longer on disk at {kit_dir or record.get('dest_dir')}"}
    recorded = str(record.get("kit_version") or "")
    return {
        "state": "success" if recorded == manifest.kit.version else "superseded",
        "dest_dir": record.get("dest_dir"),
        "kit_dir": kit_dir,
        "kit_version": recorded,
        "current_version": manifest.kit.version,
        "file_count": record.get("file_count"),
        "completed_at": record.get("completed_at"),
    }


def verify_now(manifest: Manifest) -> dict[str, Any]:
    """On-demand full re-verification for the revisit Verify action, against the ``files.json``
    kept inside the kit (its OWN version's, so a superseded kit verifies honestly)."""
    state = download_state(manifest)
    if state.get("state") not in ("success", "superseded"):
        return {"ok": False, "error": "No completed download on record. Download the starter kit first."}
    kit_root = Path(str(state["kit_dir"]))
    try:
        index = load_files_index(kit_root)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"ok": False, "error": f"{FILES_INDEX_NAME} could not be read at {kit_root}: {exc}"}
    delta = verify_tree(index, kit_root)
    ok = not delta["mismatch"] and not delta["missing"]
    return {
        "ok": ok,
        "file_count": len(index["files"]),
        "matched": delta["matched"],
        "missing_count": len(delta["missing"]),
        "mismatch_count": len(delta["mismatch"]),
        "missing": delta["missing"][:20],
        "mismatch": delta["mismatch"][:20],
        "extra_count": len(delta["extra"]),
    }


# ── The job ────────────────────────────────────────────────────────────────


def run_download(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """The download job. Raises with a participant-facing message on failure; returns
    ``{"cancelled": True, ...}`` when stopped (state stays resumable)."""
    log = ctx.log
    set_checks = ctx.set_checks
    set_progress = getattr(ctx, "set_progress", lambda p: None)
    set_field = getattr(ctx, "set_field", lambda k, v: None)
    is_cancelled = getattr(ctx, "is_cancelled", lambda: False)
    kit = manifest.kit

    checks: list[dict[str, Any]] = []

    def check(label: str, ok: bool, detail: str = "") -> bool:
        checks.append({"label": label, "ok": bool(ok), "detail": detail})
        set_checks(checks)
        log(("PASS " if ok else "FAIL ") + label + (f": {detail}" if detail else ""))
        return ok

    if not kit.published:
        msg = (
            f"The competition manifest ({manifest.source}) lists no kit shards for kit {kit.version}. "
            "The starter kit has not been published yet; there is nothing to fix on this machine."
        )
        raise RuntimeError(msg)

    resolved = resolve_params(params, manifest)  # re-validates: /run skips /validate
    dest_dir = Path(resolved["dest_dir"])
    keep_archives = resolved["keep_archives"]
    version_dir = dest_dir / kit.version
    kit_root = kit_root_of(version_dir)
    set_field("dest_dir", str(dest_dir))
    set_field("kit_version", kit.version)
    set_field("kit_dir", str(kit_root))

    check(
        "kit listed by the competition manifest",
        True,
        f"{kit.version} from {manifest.source}: {len(kit.shards)} shards, {kit.total_bytes:,} bytes",
    )
    space_ok, space_detail = _free_space(version_dir, kit)
    if not check("enough disk space at the destination", space_ok, space_detail):
        msg = (
            f"Not enough disk space at {dest_dir} ({space_detail}). Free up space or choose another "
            "destination, then run the download again."
        )
        raise RuntimeError(msg)

    # ── Download ────────────────────────────────────────────────────────
    total = kit.total_bytes or 1
    done = 0

    def cancelled_result() -> dict[str, Any]:
        log("Cancelled. Completed shards are kept; running the download again resumes.")
        return {"cancelled": True, "dest_dir": str(dest_dir), "resumable": True}

    try:
        for i, entry in enumerate(kit.shards):
            if is_cancelled():
                raise _Cancelled()

            def report(shard_done: int, _i: int = i, _name: str = entry.name) -> None:
                set_progress({
                    "percent": round(85.0 * (done + shard_done) / total, 1),
                    "label": f"Downloading shard {_i + 1}/{len(kit.shards)}",
                    "phase": "download",
                    "archive": _name,
                    "bytes_done": done + shard_done,
                    "bytes_total": total,
                })

            _download_shard(kit, entry, version_dir, log, report, is_cancelled)
            done += int(entry.bytes)
    except _Cancelled:
        return cancelled_result()
    check(f"all {len(kit.shards)} shards downloaded and sha256-verified", True, f"{total:,} bytes")

    # ── Extract ─────────────────────────────────────────────────────────
    for i, entry in enumerate(kit.shards):
        if is_cancelled():
            return cancelled_result()
        set_progress({
            "percent": round(85.0 + 8.0 * i / len(kit.shards), 1),
            "label": f"Extracting shard {i + 1}/{len(kit.shards)}",
            "phase": "extract",
        })
        with zipfile.ZipFile(version_dir / entry.name) as zf:
            for info in zf.infolist():
                n = info.filename.replace("\\", "/")
                if n.startswith("/") or ".." in n.split("/") or not n.startswith(KIT_DIR_NAME + "/"):
                    msg = f"Unsafe or misplaced path in {entry.name}: {info.filename}"
                    raise RuntimeError(msg)
            zf.extractall(version_dir)
    check(f"all {len(kit.shards)} shards extracted", True, str(kit_root))

    # ── Verify every file against files.json ────────────────────────────
    set_progress({"percent": 94.0, "label": "Verifying files", "phase": "verify"})
    if not check(f"{FILES_INDEX_NAME} present in the kit", (kit_root / FILES_INDEX_NAME).is_file(), str(kit_root)):
        msg = f"{FILES_INDEX_NAME} missing from the kit at {kit_root}. {_RERUN_RESUMES}"
        raise RuntimeError(msg)
    index = load_files_index(kit_root)
    served = str(index.get("kit_version") or "")
    if not check(
        "kit version inside the kit matches the manifest",
        served == kit.version,
        f"{served or '(unnamed)'} vs {kit.version}",
    ):
        msg = (
            f"The downloaded kit says it is {served or '(unnamed)'}, but the manifest asked for {kit.version}. "
            "The starter kit is misconfigured on the server. Report this to the organizers."
        )
        raise RuntimeError(msg)
    delta = verify_tree(index, kit_root)
    tree_ok = not delta["mismatch"] and not delta["missing"]
    detail = f"{delta['matched']}/{len(index['files'])} files verified"
    if not tree_ok:
        broken = delta["mismatch"] + delta["missing"]
        detail += "; first problems: " + ", ".join(broken[:5])
    if not check(f"extracted kit matches {FILES_INDEX_NAME}", tree_ok, detail):
        for path in (delta["mismatch"] + delta["missing"])[:50]:
            log(f"  DELTA {path}")
        broken_count = len(delta["mismatch"]) + len(delta["missing"])
        msg = f"{broken_count} files do not match {FILES_INDEX_NAME} after extraction. {_RERUN_RESUMES}"
        raise RuntimeError(msg)
    if delta["extra"]:
        # Not a failure: the kit is the participant's working copy — extras are theirs.
        check("no unexpected files in the kit tree", True, f"{len(delta['extra'])} extra files present, left in place")
        for path in delta["extra"][:20]:
            log(f"  EXTRA {path}")

    # ── Split counts against the manifest ───────────────────────────────
    got, want = split_counts(kit_root), expected_split_counts(manifest)
    counts_ok = got == want
    if not check("kit split counts match the manifest", counts_ok, f"found {got}, manifest {want}"):
        msg = (
            f"The kit's image counts {got} do not match the competition manifest {want}. The starter kit "
            "is misconfigured on the server. Report this to the organizers."
        )
        raise RuntimeError(msg)
    ids_from = kit_root / manifest.splits.test.ids_from
    if not check(f"{manifest.splits.test.ids_from} present at the kit root", ids_from.is_file(), str(ids_from)):
        msg = f"{manifest.splits.test.ids_from} missing from the kit at {kit_root}. {_RERUN_RESUMES}"
        raise RuntimeError(msg)

    # ── Publish + cleanup ───────────────────────────────────────────────
    facts = {
        "dest_dir": str(dest_dir),
        "kit_version": kit.version,
        "kit_dir": str(kit_root),
        "file_count": len(index["files"]),
    }
    _write_record(manifest, facts)
    session.publish_kit_dir(manifest, kit_root)
    log("Session updated: Import now points at the downloaded kit")

    if not keep_archives:
        for entry in kit.shards:
            (version_dir / entry.name).unlink(missing_ok=True)
        log("Shard archives removed after verification")

    set_progress({"percent": 100.0, "label": "Complete", "phase": "done"})
    return {
        "cancelled": False,
        **facts,
        "total_bytes": kit.total_bytes,
        "verified_files": int(delta["matched"]),
        "extra_files": len(delta["extra"]),
        "split_counts": got,
    }
