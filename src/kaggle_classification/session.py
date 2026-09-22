# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Disk-backed store for last-used UI values: one JSON file, one canonical **session**.

The session object holds the facts every tab shares — project, table name,
the kit directory, the device, and explicit per-field table-URL overrides.
Tabs render projections of it and no tab owns a default: backend defaults
derive from the competition manifest, and the fragment gets them via
``GET /config``. Per-tab keys keep only tab-local fields; the ``*_state``
keys are backend-written snapshots that back each tab's revisit view.

Keys this line retires later are REJECTED on save (``ValueError`` -> the
``/config`` route answers 400): the only writer that would still send them is
a stale browser-cached fragment, and silently dropping its writes would be a
new silent divergence. The retired sets are data (``_RETIRED_*``), empty on a
fresh line, so the enforcement exists before it is needed.

Never store secrets here — Kaggle credentials stay in the kaggle client's own files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaggle_classification.manifest import Manifest

_log = logging.getLogger(__name__)

# Everything the plugin owns lives under one home (ui_config.json, the kit record, the kit data):
# one directory to document, one to delete, and the redirected-home caveat stays a single caveat.
PLUGIN_HOME = Path.home() / ".3lc-kaggle-classification"
CONFIG_PATH = PLUGIN_HOME / "ui_config.json"

# Reentrant: save() holds it while load() may persist a migration.
_lock = threading.RLock()

# Keys the UI/backend may persist; anything else is dropped.
_ALLOWED_TABS = ("session", "train", "predict", "import_state", "predict_state", "submit_state")

# Retired keys — one logical fact must not reappear under a second key. Kept as data so save()
# can enforce it; empty until a migration retires something.
_RETIRED_TABS: tuple[str, ...] = ()
_RETIRED_TAB_KEYS: dict[str, tuple[str, ...]] = {}

_MIGRATIONS_KEY = "_migrations"
SPLITS = ("train", "val", "test")


def default_session(manifest: Manifest) -> dict[str, Any]:
    """The session a fresh install starts from, derived from the manifest.

    ``kit_dir`` is empty until a verified kit download publishes it (the one
    server-side session write); ``overrides`` holds only explicit per-field
    table-URL choices (hand-paste / revision picker).
    """
    from kaggle_classification.manifest import DEFAULT_TABLE_NAME

    return {
        "project_name": manifest.default_project,
        "table_name": DEFAULT_TABLE_NAME,
        "kit_dir": "",
        "device": "",
        "overrides": {},
    }


def populated_session(manifest: Manifest) -> dict[str, Any]:
    """The stored session with missing fields filled from the defaults — what
    ``GET /config`` serves, so the fragment carries no default literals."""
    stored = load().get("session")
    return {**default_session(manifest), **(stored if isinstance(stored, dict) else {})}


# ── URL helpers (public: trainer/predictor asserts reuse them) ──────────
# Table URLs follow the deterministic layout
# <project root>/<project>/datasets/<dataset>/tables/<table>; both slash styles occur in real
# configs. The PROJECT ROOT IS CONFIGURABLE, so "projects" is NOT a literal path segment — the
# project is identified by POSITION in the layout tail, anchored at end-of-string.
#
# MIRRORED IN ui.html once the pickers exist (session 2): the fragment classifies overrides at
# write time with these same three patterns. Edit both sides together; the parity test that
# guards it arrives with the fragment.
URL_SEG_PATTERNS = {
    "project": r"([^\\/]+)[\\/]datasets[\\/][^\\/]+[\\/]tables[\\/][^\\/]+[\\/]?$",
    "dataset": r"[\\/]datasets[\\/]([^\\/]+)[\\/]tables[\\/]",
    "table": r"[\\/]tables[\\/]([^\\/]+)[\\/]?$",
}


def _url_seg(url: str, kind: str) -> str | None:
    m = re.search(URL_SEG_PATTERNS[kind], str(url).strip())
    return m.group(1) if m else None


def url_project(url: str) -> str | None:
    return _url_seg(url, "project")


def url_dataset(url: str) -> str | None:
    return _url_seg(url, "dataset")


def url_table(url: str) -> str | None:
    return _url_seg(url, "table")


def classify_override(url: str, project: str, table_name: str, expected_dataset: str) -> str:
    """One predicate for whether a table-URL value may live in ``session.overrides``:

      "drop"     — wrong project, wrong dataset for the slot's split (individually valid,
                   wrong in context), or unparseable. Never stored.
      "suppress" — byte-equivalent to what derivation yields (right project, right dataset,
                   table == session table). Stored as an override it would silently freeze
                   future table-name changes — so it is not stored either.
      "keep"     — a genuine same-project, same-split revision choice.

    ``expected_dataset`` is ``manifest.dataset_name(split)`` — the caller names the split's
    dataset so this module stays manifest-free.
    """
    if url_project(url) != project or url_dataset(url) != expected_dataset:
        return "drop"
    if url_table(url) == table_name:
        return "suppress"
    return "keep"


# ── Migrations ───────────────────────────────────────────────────────────
# One-time migrations over the persisted snapshot, keyed by markers in "_migrations" so each
# runs exactly once per machine. The first marker on this line is the schema stamp; later
# migrations append here IN ORDER and record the branch that decided.


def _migrate(data: dict[str, Any]) -> bool:
    done = data.get(_MIGRATIONS_KEY)
    done = dict(done) if isinstance(done, dict) else {}
    changed = False
    if not done.get("session_v1"):
        done["session_v1"] = "fresh"
        changed = True
    if changed:
        data[_MIGRATIONS_KEY] = done
    return changed


def load() -> dict[str, Any]:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if _migrate(data):
        # The one deliberate read-path write: a one-shot, marker-guarded migration flush.
        with _lock:
            _write(data)
    return data


def _write(data: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def _retired_keys_in(update: dict[str, Any]) -> list[str]:
    bad = [tab for tab in _RETIRED_TABS if tab in update]
    for tab, keys in _RETIRED_TAB_KEYS.items():
        values = update.get(tab)
        if isinstance(values, dict):
            bad.extend(f"{tab}.{k}" for k in keys if k in values)
    return bad


def save(update: dict[str, Any]) -> dict[str, Any]:
    """Merge per-tab snapshots into the stored config; return the result.

    Raises ``ValueError`` if the update carries retired keys — nothing current
    writes them, so their presence means a stale cached fragment; rejecting
    the whole POST keeps the store coherent and makes the skew visible.
    """
    bad = _retired_keys_in(update)
    if bad:
        msg = (
            "config update contains retired keys (" + ", ".join(sorted(bad)) + "). If this write "
            "came from the plugin page, the browser is holding a stale fragment — hard-refresh the page."
        )
        raise ValueError(msg)
    with _lock:
        data = load()
        _migrate(data)  # fresh store: stamp the markers before first write
        for tab, values in update.items():
            if tab in _ALLOWED_TABS and isinstance(values, dict):
                data[tab] = values
        _write(data)
        return data


def publish_kit_dir(manifest: Manifest, kit_dir: Path | str) -> None:
    """The one server-side session writer: a verified kit download sets ``session.kit_dir``
    so the Import form starts populated. Merges onto the freshest load; a browser save racing
    this write wins or loses whole — last writer wins, by design."""
    session = populated_session(manifest)
    session["kit_dir"] = str(kit_dir)
    save({"session": session})
