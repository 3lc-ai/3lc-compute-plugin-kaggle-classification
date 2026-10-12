# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section F of docs/TEST_PLAN.md, the shared-resource load (no plugin involved): F1 N concurrent full
kit downloads from the competition CDN (every shard, Range requests like the plugin's resume path),
F2 N concurrent manifest + index fetches, F3 N concurrent catalog fetches. Throughput, latency
percentiles, status codes and CloudFront headers are recorded; nothing is written to disk but the record.

  python tools/hardening/campaign_f.py --downloads 50 --fetches 20 [--base https://competitions.dev.3lc.ai]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INDEX_PATH = "kaggle/classification-index.json"
CATALOG = "https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc13/catalog-test.json"


def fetch(url: str, *, headers: dict | None = None, timeout: float = 120) -> dict:
    t0 = time.time()
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            h = hashlib.sha256()
            n = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                h.update(chunk)
            return {"status": resp.status, "bytes": n, "sha256": h.hexdigest(), "s": round(time.time() - t0, 3),
                    "via": resp.headers.get("Via", ""), "x_cache": resp.headers.get("X-Cache", ""),
                    "cf_pop": resp.headers.get("X-Amz-Cf-Pop", ""), "accept_ranges": resp.headers.get("Accept-Ranges", "")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "bytes": 0, "s": round(time.time() - t0, 3), "error": str(exc)[:120]}
    except Exception as exc:
        return {"status": None, "bytes": 0, "s": round(time.time() - t0, 3), "error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))], 3)


def summarize(name: str, rows: list[dict], wall_s: float) -> dict:
    ok = [r for r in rows if r.get("status") in (200, 206)]
    bad = [r for r in rows if r not in ok]
    total = sum(r["bytes"] for r in ok)
    return {
        "name": name, "requests": len(rows), "ok": len(ok), "errors": len(bad), "status_codes": sorted({str(r.get("status")) for r in rows}),
        "bytes": total, "wall_s": round(wall_s, 1), "throughput_MBps": round(total / 1e6 / wall_s, 1) if wall_s else None,
        "latency_s": {"p50": pct([r["s"] for r in ok], 0.5), "p90": pct([r["s"] for r in ok], 0.9), "max": pct([r["s"] for r in ok], 1.0)},
        "x_cache": sorted({r.get("x_cache", "") for r in ok}), "cf_pops": sorted({r.get("cf_pop", "") for r in ok}),
        "errors_sample": [r.get("error") or r.get("status") for r in bad][:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://competitions.dev.3lc.ai")
    ap.add_argument("--competition", default="intel-scene")
    ap.add_argument("--downloads", type=int, default=50)
    ap.add_argument("--fetches", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "docs" / "campaign" / "section_f.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    index_url = f"{args.base}/{INDEX_PATH}"
    index = json.loads(urllib.request.urlopen(index_url, timeout=60).read())
    entry = next(c for c in index["competitions"] if c["id"] == args.competition)
    manifest_url = urllib.parse.urljoin(index_url, entry["manifest_url"])
    manifest = json.loads(urllib.request.urlopen(manifest_url, timeout=60).read())
    kit_base = urllib.parse.urljoin(manifest_url, manifest["kit"]["path"])
    shards = manifest["kit"]["shards"]
    print(f"index {index_url}\nmanifest {manifest_url}\nkit base {kit_base}: {len(shards)} shards, {sum(int(s['bytes']) for s in shards) / 1e6:.1f} MB", flush=True)
    out: dict = {"base": args.base, "manifest_url": manifest_url, "kit_base": kit_base, "shards": len(shards)}

    # F2: manifest + index fetches
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.fetches * 2) as ex:
        rows = list(ex.map(fetch, [index_url, manifest_url] * args.fetches))
    out["F2"] = summarize("manifest+index", rows, time.time() - t0)
    print(json.dumps(out["F2"]), flush=True)

    # F3: catalog fetches
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.fetches) as ex:
        rows = list(ex.map(fetch, [CATALOG] * args.fetches))
    out["F3"] = summarize("catalog", rows, time.time() - t0)
    print(json.dumps(out["F3"]), flush=True)

    # F1: concurrent full kit downloads, each worker fetches every shard in order (the plugin's order),
    # verifying the sha256 the manifest names; one worker in ten uses a Range request from byte 1024
    # (the resume path) on its first shard.
    lock = threading.Lock()
    mism: list[str] = []
    all_rows: list[dict] = []

    def one_download(i: int) -> list[dict]:
        rows = []
        for j, s in enumerate(shards):
            url = urllib.parse.urljoin(kit_base, s["name"])
            headers = {"Range": "bytes=1024-"} if (i % 10 == 0 and j == 0) else None
            r = fetch(url, headers=headers, timeout=600)
            r["worker"], r["shard"] = i, s["name"]
            if r.get("status") == 200 and r.get("sha256") != s["sha256"]:
                with lock:
                    mism.append(f"{i}:{s['name']}")
            if r.get("status") == 206 and r.get("bytes") != int(s["bytes"]) - 1024:
                with lock:
                    mism.append(f"{i}:{s['name']} range bytes {r.get('bytes')}")
            rows.append(r)
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.downloads) as ex:
        for rows in ex.map(one_download, range(args.downloads)):
            all_rows.extend(rows)
    out["F1"] = summarize(f"{args.downloads} concurrent kit downloads", all_rows, time.time() - t0)
    out["F1"]["sha256_mismatches"] = mism
    out["F1"]["range_requests"] = sum(1 for r in all_rows if r.get("status") == 206)
    out["F1"]["per_download_s"] = {"p50": pct([sum(r["s"] for r in all_rows if r.get("worker") == i) for i in range(args.downloads)], 0.5),
                                   "max": pct([sum(r["s"] for r in all_rows if r.get("worker") == i) for i in range(args.downloads)], 1.0)}
    print(json.dumps(out["F1"]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
