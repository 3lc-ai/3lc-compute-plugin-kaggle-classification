# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the plugin keeps its own state — resolved once, never from HOME first.

The ExDark plugin derived its home from ``Path.home()`` and paid for it: the worker's
``USERPROFILE`` was not the organizer's, so files landed in a directory nobody was
looking at (the redirected-home bug). This module resolves the plugin's directory in
this order and records which rule won, so ``GET /config`` can say where state lives:

1. ``KAGGLE_CLASSIFICATION_HOME`` — an explicit operator override.
2. A storage helper on the installed SDK, if a future SDK grows one (probed by name;
   0.3.x has none).
3. The worker's state root, once a job has revealed it (``ctx.state_dir.parent``).
4. The SDK worker's default state root, ``<cwd>/.plugin-state/<plugin id>`` — the
   worker creates it before it serves, so inside a worker this always exists and is
   independent of HOME (the 1.0.x host passes no ``--state-root``).
5. ``~/.3lc-kaggle-classification`` — dev runs and tests only.

Import-light: stdlib plus the SDK's cheap contract surface.
"""

from __future__ import annotations

import os
from pathlib import Path

PLUGIN_ID = "kaggle-classification"
HOME_ENV = "KAGGLE_CLASSIFICATION_HOME"
_FALLBACK_DIR_NAME = ".3lc-kaggle-classification"
_SDK_HELPER_NAMES = ("plugin_state_dir", "plugin_data_dir", "plugin_storage_dir")

_state_root: Path | None = None


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


def resolve() -> tuple[Path, str]:
    """The plugin home and the rule that chose it (``env`` / ``sdk`` / ``worker`` / ``cwd`` / ``home``)."""
    env = os.environ.get(HOME_ENV, "").strip()
    if env:
        return Path(env).expanduser(), "env"
    sdk_dir = _sdk_helper_dir()
    if sdk_dir is not None:
        return sdk_dir, "sdk"
    if _state_root is not None:
        return _state_root, "worker"
    cwd_default = Path.cwd() / ".plugin-state" / PLUGIN_ID
    if cwd_default.is_dir():
        return cwd_default, "cwd"
    return Path.home() / _FALLBACK_DIR_NAME, "home"


def plugin_home() -> Path:
    return resolve()[0]


def describe() -> dict[str, str]:
    path, rule = resolve()
    return {"path": str(path), "resolved_by": rule}
