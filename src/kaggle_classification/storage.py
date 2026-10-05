# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the plugin keeps its own state — resolved once, never from HOME first, and never lost
on a plugin update.

The ExDark plugin derived its home from ``Path.home()`` and paid for it: the worker's
``USERPROFILE`` was not the organizer's, so files landed in a directory nobody was
looking at (the redirected-home bug). This module resolves the plugin's directory in
this order and records which rule won, so ``GET /config`` can say where state lives:

1. ``KAGGLE_CLASSIFICATION_HOME`` — an explicit operator override.
2. A storage helper on the installed SDK, if a future SDK grows one (probed by name;
   0.3.x has none).
3. **The compute home** (session 5, 1.0.0rc3): when the worker runs from the host's managed
   layout ``<compute home>/managed-plugins/<id>/<version>`` (a catalog install) or
   ``<compute home>/managed-plugins/<id>`` (a folder source), the state lives in
   ``<compute home>/plugin-state/<id>`` — OUTSIDE the version dir, so a plugin update keeps the
   kit, the import record, the train / predict records and the ledger. The first resolution
   after an update carries the newest previous version's ``.plugin-state`` forward (a copy, the
   old copy is never deleted; the paths inside the records are rewritten to the new home).
4. The worker's state root, once a job has revealed it (``ctx.state_dir.parent``).
5. The SDK worker's default state root, ``<cwd>/.plugin-state/<plugin id>`` — the
   worker creates it before it serves, so inside a worker this always exists and is
   independent of HOME (the 1.0.x host passes no ``--state-root``).
6. ``~/.3lc-kaggle-classification`` — dev runs and tests only.

Import-light: stdlib plus the SDK's cheap contract surface.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

PLUGIN_ID = "kaggle-classification"
HOME_ENV = "KAGGLE_CLASSIFICATION_HOME"
_FALLBACK_DIR_NAME = ".3lc-kaggle-classification"
_SDK_HELPER_NAMES = ("plugin_state_dir", "plugin_data_dir", "plugin_storage_dir")

MANAGED_DIR_NAME = "managed-plugins"      # the host's layout: <compute home>/managed-plugins/<id>/<version>
SHARED_STATE_DIR_NAME = "plugin-state"    # ours, beside it: <compute home>/plugin-state/<id>
WORKER_STATE_DIR_NAME = ".plugin-state"   # the SDK worker's default under its cwd
MIGRATION_MARKER = "migrated_from.json"
# The files whose stored paths point into the plugin home and are rewritten on a carry-forward.
_PATH_BEARING = ("ui_config.json", "ledger.jsonl")
_PATH_BEARING_DIRS = ("kit",)
# Caches, never carried: rebuilt on demand.
_SKIP_ON_COPY = ("numba-cache",)
DATA_DIR_NAME = "data"                    # the kit data under the plugin home (kit.default_dest)

_state_root: Path | None = None
_shared_lock = threading.Lock()
_shared_ready: set[str] = set()   # shared homes whose carry-forward check ran in this process


def remember_state_root(path: Path | str) -> None:
    """Record the worker's state root (``ctx.state_dir.parent``) for the life of the process."""
    global _state_root
    _state_root = Path(path)


def _sdk_helper_dir() -> Path | None:
    try:
        import tlc_plugin_sdk
    except Exception:
        return None
    for name in _SDK_HELPER_NAMES:
        helper = getattr(tlc_plugin_sdk, name, None)
        if callable(helper):
            try:
                result = helper(PLUGIN_ID)
            except TypeError:
                result = helper()
            if result:
                return Path(str(result))
    return None


def managed_layout(cwd: Path) -> tuple[Path, Path] | None:
    """``(shared home, the managed plugin dir)`` when ``cwd`` is the host's managed layout for this
    plugin — ``<compute home>/managed-plugins/<id>/<version>`` (catalog install) or
    ``<compute home>/managed-plugins/<id>`` (folder source) — else None."""
    try:
        if cwd.parent.name == PLUGIN_ID and cwd.parent.parent.name == MANAGED_DIR_NAME:
            return cwd.parents[2] / SHARED_STATE_DIR_NAME / PLUGIN_ID, cwd.parent
        if cwd.name == PLUGIN_ID and cwd.parent.name == MANAGED_DIR_NAME:
            return cwd.parents[1] / SHARED_STATE_DIR_NAME / PLUGIN_ID, cwd
    except IndexError:
        return None
    return None


def previous_state_dirs(managed_dir: Path) -> list[Path]:
    """Every ``.plugin-state/<id>`` under the managed plugin dir that holds a session store — the
    version dirs' (catalog installs) and the folder source's own — newest store first."""
    found: list[Path] = []
    candidates = [managed_dir]
    try:
        candidates += [p for p in managed_dir.iterdir() if p.is_dir() and p.name != ".venv"]
    except OSError:
        pass
    for base in candidates:
        state = base / WORKER_STATE_DIR_NAME / PLUGIN_ID
        if (state / "ui_config.json").is_file():
            found.append(state)
    return sorted(found, key=lambda p: (p / "ui_config.json").stat().st_mtime, reverse=True)


def _path_forms(path: Path) -> list[str]:
    """The spellings a path takes inside the plugin's JSON records: JSON-escaped backslashes (what
    ``json.dumps`` writes on Windows), forward slashes, and the plain form."""
    raw = str(path)
    forms = [json.dumps(raw)[1:-1], path.as_posix(), raw]
    out: list[str] = []
    for f in forms:
        if f and f not in out:
            out.append(f)
    return out


def _rewrite_paths(text: str, old: Path, new: Path) -> str:
    for old_form, new_form in zip(_path_forms(old), _path_forms(new), strict=True):
        text = text.replace(old_form, new_form)
    return text


def carry_forward(shared: Path, managed_dir: Path) -> dict | None:
    """Copy the newest previous version's state into ``shared`` when ``shared`` holds no session
    store yet. Never deletes the source. Returns the marker written, or None when nothing to carry."""
    if (shared / "ui_config.json").is_file() or (shared / MIGRATION_MARKER).is_file():
        return None
    sources = [s for s in previous_state_dirs(managed_dir) if s.resolve() != shared.resolve()]
    if not sources:
        return None
    src = sources[0]
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, shared, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*_SKIP_ON_COPY))
    rewritten: list[str] = []
    targets = [shared / name for name in _PATH_BEARING]
    for d in _PATH_BEARING_DIRS:
        if (shared / d).is_dir():
            targets += sorted((shared / d).glob("*.json"))
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new_text = _rewrite_paths(text, src, shared)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8", newline="\n")
            rewritten.append(path.relative_to(shared).as_posix())
    marker = {
        "source": str(src), "destination": str(shared), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rewritten": rewritten, "skipped": list(_SKIP_ON_COPY),
        # Tables imported before the move keep their image paths into the source's kit tree; the host
        # garbage-collects old version dirs (three are kept), so the plugin re-creates that tree from its
        # own copy whenever it is missing (``ensure_legacy_data_dirs``, rc5).
        "legacy_data_dirs": [str(src / DATA_DIR_NAME)] if (src / DATA_DIR_NAME).is_dir() else [],
        "note": "The plugin's state moved out of the version dir (1.0.0rc3). The source copy was left in place; "
                "tables imported before this move keep their image paths into the source's kit dir, which the "
                "plugin re-creates from its own copy if the host removes the old version dir.",
    }
    (shared / MIGRATION_MARKER).write_text(json.dumps(marker, indent=1), encoding="utf-8")
    return marker


def ensure_legacy_data_dirs(shared: Path) -> list[str]:
    """Re-create every ``legacy_data_dirs`` entry of the carry-forward marker that is missing on disk from
    the shared home's own ``data`` tree (byte-identical: the kit is sha256-verified per file). The host's
    ``gc_old_versions`` keeps three version dirs; the tables imported before rc3 point into an older one.
    Returns the dirs re-created."""
    marker_path = shared / MIGRATION_MARKER
    if not marker_path.is_file():
        return []
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    restored: list[str] = []
    source = shared / DATA_DIR_NAME
    for raw in marker.get("legacy_data_dirs") or []:
        legacy = Path(str(raw))
        if legacy.exists() or not source.is_dir():
            continue
        try:
            shutil.copytree(source, legacy)
            restored.append(str(legacy))
        except OSError:
            continue
    if restored:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        marker.setdefault("restored", []).append({"at": stamp, "dirs": restored})
        marker_path.write_text(json.dumps(marker, indent=1), encoding="utf-8")
    return restored


def _shared_home(cwd: Path) -> Path | None:
    layout = managed_layout(cwd)
    if layout is None:
        return None
    shared, managed_dir = layout
    key = str(shared)
    if key not in _shared_ready:
        with _shared_lock:
            if key not in _shared_ready:
                try:
                    carry_forward(shared, managed_dir)
                    ensure_legacy_data_dirs(shared)
                finally:
                    _shared_ready.add(key)
    return shared


def resolve() -> tuple[Path, str]:
    """The plugin home and the rule that chose it
    (``env`` / ``sdk`` / ``compute-home`` / ``worker`` / ``cwd`` / ``home``)."""
    env = os.environ.get(HOME_ENV, "").strip()
    if env:
        return Path(env).expanduser(), "env"
    sdk_dir = _sdk_helper_dir()
    if sdk_dir is not None:
        return sdk_dir, "sdk"
    shared = _shared_home(Path.cwd())
    if shared is not None:
        return shared, "compute-home"
    if _state_root is not None:
        return _state_root, "worker"
    cwd_default = Path.cwd() / WORKER_STATE_DIR_NAME / PLUGIN_ID
    if cwd_default.is_dir():
        return cwd_default, "cwd"
    return Path.home() / _FALLBACK_DIR_NAME, "home"


def plugin_home() -> Path:
    return resolve()[0]


def describe() -> dict[str, str]:
    path, rule = resolve()
    return {"path": str(path), "resolved_by": rule}


def migration_record() -> dict | None:
    """The carry-forward marker of the current home, if a carry-forward happened (the Doctor shows it)."""
    path = plugin_home() / MIGRATION_MARKER
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _reset_for_tests() -> None:
    global _state_root
    _state_root = None
    _shared_ready.clear()
