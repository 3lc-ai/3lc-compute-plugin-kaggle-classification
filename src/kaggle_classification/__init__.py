# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kaggle image-classification hackathon plugin for the 3LC Hub.

Tabbed workflow: Import / Train / Predict + Submit / Status — the tab bar
doubles as the pipeline stepper. Session 1 ships the scaffold, the competition
manifest, the session store and the kit download stage; the tabs are stubs.

SDK (venv-isolated) plugin: behavior-only. The manifest lives in plugin.toml
(read import-free by the host); this class subclasses the SDK's
``ComputePlugin`` and runs out-of-process in the plugin's own provisioned venv.
Long jobs run through ``run_job`` (the host dispatch channel) so they appear
in the Hub's generic Queue panel and honour host-side cancel.

Import-light by design: this module and everything it imports at module
level are stdlib + the SDK's cheap contract surface. torch/timm/tlc/yaml/
litestar are imported inside functions.

The package name ``kaggle_classification`` must stay distinct from every
distribution the host imports (a plugin root is prepended to ``sys.path``;
a package named ``kaggle`` would shadow the kaggle client for the process).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tlc_plugin_sdk import ComputePlugin, JobContext

DIST_NAME = "3lc-compute-plugin-kaggle-classification"
REPOSITORY_URL = "https://github.com/3lc-ai/3lc-compute-plugin-kaggle-classification"


def _read_version() -> str:
    # Derived, never hand-synced: the installed dist's metadata first, plugin.toml for source runs.
    try:
        from importlib.metadata import version

        return version(DIST_NAME)
    except Exception:
        try:
            import tomllib

            with open(Path(__file__).resolve().parent / "plugin.toml", "rb") as f:
                return str(tomllib.load(f)["version"])
        except Exception:
            return "unknown"


__version__ = _read_version()

TABS = ("import", "train", "predict_submit", "status")


class _JobCtxAdapter:
    """Duck-typed job context the stage modules program against, over the SDK ``JobContext``.

    ``log`` / ``set_progress`` / ``set_field`` feed the generic Queue panel; ``set_checks``
    and every fact also go out as plugin-private events for the fragment. Cancellation is
    the host's ``ctx.cancelled``.
    """

    def __init__(self, sdk_ctx: JobContext) -> None:
        self._sdk = sdk_ctx

    @property
    def job_id(self) -> str:
        return str(self._sdk.job_id)

    def log(self, message: str) -> None:
        self._sdk.log(message)

    def set_checks(self, checks: list[dict[str, Any]]) -> None:
        self._sdk.emit("checks", {"job_id": self.job_id, "checks": [dict(c) for c in checks]})

    def set_progress(self, progress: dict[str, Any]) -> None:
        if progress.get("percent") is not None:
            self._sdk.progress(percent=float(progress["percent"]), label=str(progress.get("label") or ""))
        self._sdk.emit("stage_progress", {"job_id": self.job_id, **progress})

    def set_field(self, key: str, value: Any) -> None:
        if key == "run_url" and value:
            self._sdk.result(str(value))
        self._sdk.emit("fact", {"job_id": self.job_id, "key": key, "value": value})

    def is_cancelled(self) -> bool:
        return bool(self._sdk.cancelled)


class KaggleClassificationPlugin(ComputePlugin):
    """Behavior class named by plugin.toml's ``runtime.entrypoint``."""

    id: str
    _ui_cache: str | None = None

    def get_ui_fragment(self) -> str:
        if self._ui_cache is None:
            from kaggle_classification.ui import fragment

            self._ui_cache = fragment()
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Generic info endpoint; real work goes through the custom routes and ``run_job``."""
        return {
            "plugin": "kaggle-classification",
            "version": __version__,
            "tabs": list(TABS),
            "implemented": ["download_kit"],
        }

    def get_route_handlers(self) -> list[Any]:
        from kaggle_classification.routes import get_route_handlers

        return get_route_handlers()

    def run_job(self, ctx: JobContext) -> None:
        """Host-dispatched job entry (``POST /api/plugins/kaggle-classification/run``).

        ``ctx.params`` carries ``{"kind": "download_kit" | "import" | "train" | "predict" |
        "submit", ...job params}``. Session 1 implements ``download_kit``; the other kinds
        fail cleanly through ``ctx.fail`` until their session lands.
        """
        from kaggle_classification import kit, manifest

        kind = str(ctx.params.get("kind", "")).strip()
        params = {k: v for k, v in ctx.params.items() if k != "kind"}
        current = manifest.load_manifest()
        adapter = _JobCtxAdapter(ctx)

        if kind == "download_kit":
            result = kit.run_download(params, adapter, current)
            if not result.get("cancelled"):
                ctx.progress(percent=100.0, label="Done")
            return
        if kind in ("import", "train", "predict", "submit"):
            ctx.fail(f"The {kind} step is not implemented in this build ({__version__}).")
        ctx.fail(f"Unknown job kind: {kind!r}. Expected one of download_kit, import, train, predict, submit.")
