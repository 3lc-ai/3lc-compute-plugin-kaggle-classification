# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify a served CDN tier from the participant's side — dev-only.

    python tools/verify_cdn.py <base_url> [--competition-id intel-scene] [--skip-header-checks]

Fetches the index, the active competition's manifest and every shard exactly as the plugin
would (relative URLs resolved against the fetched documents), checks every shard's sha256 and
bytes against the manifest, and reports the Content-Type, Cache-Control and X-Cache headers
actually served. Header policy (docs/PLAN.md §A3): a MUTABLE object (index, manifest) must be
served with ``Cache-Control`` carrying ``max-age <= 300``; shards should carry ``immutable``.
Integrity is sha256 only — never ETag (multipart uploads). Header checks are skipped
automatically for a loopback base (``python -m http.server`` sends none), and the report says so.
Exit 0 on success, 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

from kaggle_classification import manifest as manifest_mod

TIMEOUT_S = 60
MAX_MUTABLE_AGE = 300
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _get(url: str) -> tuple[bytes, dict[str, str], str]:
    """(body, headers, sha256) with the body hashed as it streams."""
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-classification verify_cdn"})
    h = hashlib.sha256()
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            chunks.append(chunk)
    return b"".join(chunks), headers, h.hexdigest()


def _max_age(cache_control: str | None) -> int | None:
    if not cache_control:
        return None
    m = re.search(r"max-age=(\d+)", cache_control)
    return int(m.group(1)) if m else None


def verify(base_url: str, *, competition_id: str | None, skip_header_checks: bool, log=print) -> dict[str, Any]:
    base = base_url.rstrip("/")
    host = (urllib.parse.urlsplit(base).hostname or "").lower()
    loopback = host in _LOOPBACK
    if loopback and not skip_header_checks:
        skip_header_checks = True
        log(f"note: {host} is loopback; header policy checks are skipped (a local http.server sends no Cache-Control)")
    hosts = {host}
    problems: list[str] = []
    report: dict[str, Any] = {"base_url": base, "objects": []}

    def record(kind: str, url: str, headers: dict[str, str], nbytes: int, sha: str, *, mutable: bool) -> None:
        cc = headers.get("cache-control")
        entry = {
            "kind": kind,
            "url": url,
            "bytes": nbytes,
            "sha256": sha,
            "content_type": headers.get("content-type"),
            "cache_control": cc,
            "x_cache": headers.get("x-cache"),
            "mutable": mutable,
        }
        report["objects"].append(entry)
        log(
            f"  {kind:9s} {url}\n"
            f"            {nbytes:>12,} bytes  sha256 {sha[:16]}…  Content-Type={entry['content_type']!r}  "
            f"Cache-Control={cc!r}  X-Cache={entry['x_cache']!r}"
        )
        if skip_header_checks:
            return
        age = _max_age(cc)
        if mutable and (age is None or age > MAX_MUTABLE_AGE):
            problems.append(
                f"{kind} {url}: mutable object served with Cache-Control={cc!r}; need max-age<={MAX_MUTABLE_AGE}"
            )
        if not mutable and (cc is None or "immutable" not in cc):
            log(f"  warning: shard {url} served without 'immutable' in Cache-Control ({cc!r})")

    t0 = time.monotonic()
    index_url = f"{base}/{manifest_mod.INDEX_PATH}"
    log(f"index    {index_url}")
    raw, headers, sha = _get(index_url)
    record("index", index_url, headers, len(raw), sha, mutable=True)
    entries = manifest_mod.parse_index(json.loads(raw.decode("utf-8")), index_url)
    active = [e for e in entries if e["active"]]
    if competition_id:
        active = [e for e in active if e["id"] == competition_id]
    if len(active) != 1:
        problems.append(
            f"index lists {len(active)} matching active competitions ({[e['id'] for e in active]}); expected exactly 1"
        )
        return {**report, "ok": False, "problems": problems}
    entry = active[0]

    raw, headers, sha = _get(entry["manifest_url"])
    record("manifest", entry["manifest_url"], headers, len(raw), sha, mutable=True)
    man = manifest_mod.parse_manifest(
        manifest_mod._decode_document(raw),
        source="remote",
        source_detail=entry["manifest_url"],
        sha256=sha,
        document_url=entry["manifest_url"],
        hosts=hosts,
    )
    report["competition_id"] = man.competition.id
    report["kit_version"] = man.kit.version
    report["manifest_sha256"] = sha
    if not man.kit.published:
        problems.append("manifest lists no shards")

    total = 0
    for shard in man.kit.shards:
        url = man.shard_url(shard.name)
        raw, headers, sha = _get(url)
        record("shard", url, headers, len(raw), sha, mutable=False)
        total += len(raw)
        if len(raw) != shard.bytes:
            problems.append(f"{shard.name}: {len(raw)} bytes served, manifest says {shard.bytes}")
        if sha != shard.sha256:
            problems.append(f"{shard.name}: sha256 {sha} served, manifest says {shard.sha256}")
    elapsed = time.monotonic() - t0
    report.update({
        "ok": not problems,
        "problems": problems,
        "shards": len(man.kit.shards),
        "shard_bytes": total,
        "elapsed_s": round(elapsed, 1),
    })
    for p in problems:
        log("FAIL " + p)
    log(
        f"{'PASS' if not problems else 'FAIL'}: {len(man.kit.shards)} shards, {total:,} bytes, {elapsed:.1f}s"
        + (" (header checks skipped)" if skip_header_checks else "")
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--competition-id", default=None)
    parser.add_argument("--skip-header-checks", action="store_true")
    parser.add_argument(
        "--json", type=argparse.FileType("w", encoding="utf-8"), default=None, help="also write the report here"
    )
    args = parser.parse_args(argv)
    report = verify(args.base_url, competition_id=args.competition_id, skip_header_checks=args.skip_header_checks)
    if args.json:
        json.dump(report, args.json, indent=1)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
