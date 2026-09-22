# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The UI fragment: one self-contained HTML+CSS+JS document, served at ``GET /ui``.

Session 1 ships the four-tab shell (Import / Train / Predict + Submit / Status) with the
config load and footer version; the tabs fill in over sessions 2 to 5. The SDK's job
tracker is injected so ``window.PluginJobs`` is available to every tab.
"""

from __future__ import annotations

from pathlib import Path

_UI = Path(__file__).resolve().parent / "ui.html"


def fragment() -> str:
    raw = _UI.read_text(encoding="utf-8")
    try:
        from tlc_plugin_sdk.shared.job_tracker import job_tracker_script
        from tlc_plugin_sdk.shared.ui_inject import inject_scripts

        return inject_scripts(raw, job_tracker_script())
    except Exception:  # an SDK without the shared helpers still serves the raw fragment
        return raw
