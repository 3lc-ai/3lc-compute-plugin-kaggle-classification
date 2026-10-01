# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ledger: an append-only JSON-lines record of every prediction and submission.

Contract (docs/PLAN.md "Ledger", docs/PREDICT_MIRROR.md §6): one line per step under the plugin
home (``ledger.jsonl``), carrying the inputs an organizer verifies a leaderboard entry against —
the run URL, the checkpoint hash (recorded and on disk), the manifest provenance, the CSV hash,
the Kaggle submission ref and its read-back status. Session 4 writes predict and submit entries;
session 5 adds the per-run verification bundle on top of them (``build_verification_bundle``).

Append-only by construction: the file is opened for append and every entry carries its own
timestamp and kind; nothing here rewrites or prunes. Import-light (stdlib only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kaggle_classification import storage

LEDGER_NAME = "ledger.jsonl"


def ledger_path() -> Path:
    return storage.plugin_home() / LEDGER_NAME


def append(entry: dict[str, Any]) -> dict[str, Any]:
    """Append one entry (``kind`` required); ``ts`` is stamped when absent. Returns the entry as written."""
    kind = str(entry.get("kind") or "").strip()
    if not kind:
        msg = "a ledger entry needs a kind"
        raise ValueError(msg)
    out = {"ts": entry.get("ts") or time.time(), **entry, "kind": kind}
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")
    return out


def read(kind: str | None = None) -> list[dict[str, Any]]:
    """Every entry in file order (oldest first), optionally one kind. A corrupt line is skipped."""
    path = ledger_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if isinstance(entry, dict) and (kind is None or entry.get("kind") == kind):
            out.append(entry)
    return out


def find(kind: str, **match: Any) -> dict[str, Any] | None:
    """The NEWEST entry of ``kind`` whose top-level fields equal ``match``."""
    for entry in reversed(read(kind)):
        if all(entry.get(k) == v for k, v in match.items()):
            return entry
    return None
