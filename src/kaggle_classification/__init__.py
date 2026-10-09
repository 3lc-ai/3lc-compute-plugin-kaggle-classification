# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kaggle image-classification hackathon plugin for the 3LC Hub.

Tabbed workflow: Import / Train / Predict + Submit / Status, the tab bar
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


def generic_timing(progress: dict[str, Any]) -> dict[str, Any] | None:
    """The timing line of the Hub's Queue & Progress card, in the SDK's generic shape, what the timm
    plugin sends through ``tlc_plugin_sdk.shared.generic_job.epoch_progress``: ``{elapsed_s, eta_s,
    avg_step_s, step_label}`` (docs/TRAIN_MIRROR.md §15). Only the keys the payload carries are sent
    (the card renders "Elapsed | ETA | Per epoch" from whatever is present); ``None`` when the payload
    has no timing at all (Import, Predict, the download), so the card shows the bar alone."""
    out: dict[str, Any] = {}
    if progress.get("elapsed_s") is not None:
        out["elapsed_s"] = round(float(progress["elapsed_s"]), 1)
    if progress.get("eta_s") is not None:
        out["eta_s"] = round(float(progress["eta_s"]), 1)
    if progress.get("avg_epoch_s") is not None:
        out["avg_step_s"] = round(float(progress["avg_epoch_s"]), 1)
    if not out:
        return None
    out["step_label"] = "epoch"
    return out


JOB_LABELS = {"download_kit": "Download starter kit", "import": "Import", "train": "Train", "predict": "Predict",
              "kaggle_submit": "Submit to Kaggle"}


def job_label(kind: str, params: dict[str, Any], manifest: Any) -> str:
    """The Queue card's opening subtitle: ``<job> · <project>`` (rc11, item 17). Never a rows summary:
    that stays in the log lines."""
    from kaggle_classification import session

    label = JOB_LABELS.get(kind, kind or "Job")
    project = str(params.get("project_name") or "").strip()
    if not project:
        try:
            project = str(session.populated_session(manifest).get("project_name") or "")
        except Exception:
            project = ""
    return f"{label} · {project}" if project else label


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
        # The host keeps no job log the fragment can read back (PLAN §A3), so the line also goes
        # out as a plugin event: the Import tab's "Show log" accordion fills from it live.
        self._sdk.emit("log_line", {"job_id": self.job_id, "line": message})

    def set_checks(self, checks: list[dict[str, Any]]) -> None:
        self._sdk.emit("checks", {"job_id": self.job_id, "checks": [dict(c) for c in checks]})

    def set_progress(self, progress: dict[str, Any]) -> None:
        if progress.get("percent") is not None:
            # percent + label + the timing line: the project page's Queue & Progress card (the Hub's
            # "live progress bar on the run", docs/TRAIN_MIRROR.md §15), as the timm plugin feeds it.
            self._sdk.progress(percent=float(progress["percent"]), label=str(progress.get("label") or ""),
                               timing=generic_timing(progress))
        self._sdk.emit("stage_progress", {"job_id": self.job_id, **progress})

    def set_field(self, key: str, value: Any) -> None:
        if key == "run_url" and value:
            self._sdk.result(str(value))
        self._sdk.emit("fact", {"job_id": self.job_id, "key": key, "value": value})

    def set_metric(self, label: str, value: Any) -> None:
        """A key/value card on the Hub's Queue & Progress card, end-of-job counts only, as the sam3,
        image-metrics and importer plugins use it (session 6, TRAIN_MIRROR §15 S6-6): never a training
        metric, which the SDK guide keeps off the generic panel."""
        self._sdk.metric(str(label), value)

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
            "implemented": ["download_kit", "import", "train", "predict", "kaggle_submit"],
        }

    def get_route_handlers(self) -> list[Any]:
        from kaggle_classification.routes import get_route_handlers

        return get_route_handlers()

    def run_job(self, ctx: JobContext) -> None:
        """Host-dispatched job entry (``POST /api/plugins/kaggle-classification/run``).

        ``ctx.params`` carries ``{"kind": "download_kit" | "import" | "train" | "predict" |
        "kaggle_submit", ...job params}``. Every kind's refusals go out verbatim through
        ``ctx.fail``; an unknown kind fails the same way.
        """
        from kaggle_classification import kit, manifest, storage

        # The worker's state root is the plugin's home from here on (storage.py rule 3).
        state_dir = getattr(ctx, "state_dir", None)
        if state_dir:
            storage.remember_state_root(Path(state_dir).parent)

        kind = str(ctx.params.get("kind", "")).strip()
        params = {k: v for k, v in ctx.params.items() if k != "kind"}
        adapter = _JobCtxAdapter(ctx)
        # Every job re-resolves the manifest at start and records what it ran under.
        current, provenance = manifest.resolve_manifest_for_job()
        adapter.set_field("manifest", provenance)
        ctx.log(
            f"Manifest: {provenance['manifest_source']} ({provenance['competition_id']}, kit "
            f"{provenance['kit_version']}, sha256 {str(provenance['manifest_sha256'])[:12]})"
        )
        # rc11 (item 17): the Queue card's subtitle opens as "<job> · <project>": the host owns the title
        # (always the plugin's name, tlc_compute job_manager `to_generic`), the label is ours.
        ctx.progress(percent=0.0, label=job_label(kind, params, current))

        if kind == "download_kit":
            try:
                result = kit.run_download(params, adapter, current)
            except (RuntimeError, ValueError) as exc:
                # kit.py raises participant-facing messages (sha mismatch, disk space, kit
                # defects, resume hints); ctx.fail reports them verbatim, without the type
                # prefix and worker traceback an uncaught exception carries (seen live).
                ctx.fail(str(exc))
            if not result.get("cancelled"):
                ctx.progress(percent=100.0, label="Done")
            return
        if kind == "import":
            from kaggle_classification import importer

            try:
                result = importer.run_import(params, adapter, current)
            except (importer.ImportRefused, RuntimeError) as exc:
                # Participant-facing by construction (kit defect, collision, verification): the
                # message is the whole story, so it goes out verbatim, without a type prefix.
                ctx.fail(str(exc))
            if not result.get("cancelled"):
                ctx.progress(percent=100.0, label="Done")
                # F5: compile UMAP in the background now, so the first Train run's collection pass is warm.
                from kaggle_classification import trainer

                trainer.prewarm_umap_async()
            return
        if kind == "train":
            from kaggle_classification import trainer

            try:
                result = trainer.run_training(params, adapter, current)
            except trainer.TrainRefused as exc:
                # A refusal (bad table, nothing to train on, a duplicate start) is a message for the
                # participant, not a fault: verbatim, no type prefix (the SDK's JobFailed contract).
                ctx.fail(str(exc))
            if not result.get("cancelled"):
                ctx.progress(percent=100.0, label="Done")
            return
        if kind in ("predict", "kaggle_submit"):
            from kaggle_classification import predictor

            runner = predictor.run_predict if kind == "predict" else predictor.run_kaggle_submit
            try:
                result = runner(params, adapter, current)
            except predictor.PredictRefused as exc:
                # A refusal (a run that fails the provenance gate, unverifiable test images, a format
                # check, Kaggle's rejection) is the participant's message, verbatim.
                ctx.fail(str(exc))
            if not result.get("cancelled"):
                ctx.progress(percent=100.0, label="Done")
            return
        ctx.fail(f"Unknown job kind: {kind!r}. Expected one of download_kit, import, train, predict, kaggle_submit.")
