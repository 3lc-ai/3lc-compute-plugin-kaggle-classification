# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""``_JobCtxAdapter``: the stage-facing duck type over the SDK ``JobContext``.

The fake ctx carries the REAL 0.3.x plugin-facing signatures, so a call shape the SDK would
reject raises here too (the ExDark v1.2.8 lesson: ``result()`` went from keyword-only
``run_url=`` to positional ``url``, and a wrong call failed silently inside a try/except).
"""

from __future__ import annotations

from kaggle_classification import _JobCtxAdapter


class _Sdk03Ctx:
    def __init__(self) -> None:
        self.received: list[tuple] = []
        self.job_id = "sdkjob01"
        self.cancelled = False

    def log(self, message: str) -> None:
        self.received.append(("log", message))

    def progress(self, *, percent: float, label: str = "", timing: dict | None = None) -> None:
        self.received.append(("progress", percent, label))
        self.received.append(("timing", timing))

    def metric(self, label: str, value) -> None:
        self.received.append(("metric", label, value))

    def result(self, url: str) -> None:
        self.received.append(("result", url))

    def emit(self, name: str, payload: dict | None = None) -> None:
        if name == "job_update":
            msg = "reserved"
            raise ValueError(msg)
        self.received.append(("emit", name, payload or {}))


def _pair():
    sdk = _Sdk03Ctx()
    return _JobCtxAdapter(sdk), sdk


def test_run_url_fact_reaches_result_with_the_0_3_signature():
    ctx, sdk = _pair()
    ctx.set_field("run_url", "http://localhost:5016/objects/runs/p/run1")
    assert ("result", "http://localhost:5016/objects/runs/p/run1") in sdk.received


def test_other_facts_do_not_emit_result_but_are_relayed():
    ctx, sdk = _pair()
    ctx.set_field("kit_dir", "C:/x/kit")
    assert not [e for e in sdk.received if e[0] == "result"]
    assert ("emit", "fact", {"job_id": "sdkjob01", "key": "kit_dir", "value": "C:/x/kit"}) in sdk.received


def test_percent_progress_reaches_the_generic_panel():
    ctx, sdk = _pair()
    ctx.set_progress({"percent": 42.5, "label": "Downloading shard 5/10", "phase": "download"})
    assert ("progress", 42.5, "Downloading shard 5/10") in sdk.received
    assert any(e[0] == "emit" and e[1] == "stage_progress" and e[2]["phase"] == "download" for e in sdk.received)


def test_train_progress_carries_the_generic_timing_line_like_timm():
    """Session 6: the Hub's Queue & Progress card renders Elapsed | ETA | Per epoch from
    ``progress.timing`` in the SDK's generic shape; the trainer's payload names the fields ExDark's way."""
    ctx, sdk = _pair()
    ctx.set_progress({"percent": 33.3, "label": "Epoch 2/3", "phase": "train", "epoch": 1, "total_epochs": 3,
                      "elapsed_s": 25.04, "avg_epoch_s": 9.2, "eta_s": 18.4})
    assert ("timing", {"elapsed_s": 25.0, "eta_s": 18.4, "avg_step_s": 9.2, "step_label": "epoch"}) in sdk.received


def test_progress_without_timing_sends_none_so_the_card_shows_the_bar_alone():
    ctx, sdk = _pair()
    ctx.set_progress({"percent": 10.0, "label": "Downloading shard 1/10", "phase": "download"})
    assert ("timing", None) in sdk.received
    # A payload with only the elapsed time (the first flush before epoch 1) still carries the step label.
    ctx.set_progress({"percent": 0.0, "label": "Training: starting", "phase": "train", "elapsed_s": 13.2})
    assert ("timing", {"elapsed_s": 13.2, "step_label": "epoch"}) in sdk.received


def test_checks_go_out_as_a_plugin_event_and_never_as_job_update():
    ctx, sdk = _pair()
    ctx.set_checks([{"label": "x", "ok": True, "detail": ""}])
    assert (
        "emit",
        "checks",
        {"job_id": "sdkjob01", "checks": [{"label": "x", "ok": True, "detail": ""}]},
    ) in sdk.received


def test_log_and_cancellation():
    ctx, sdk = _pair()
    ctx.log("hello")
    assert ("log", "hello") in sdk.received
    assert ctx.is_cancelled() is False
    sdk.cancelled = True
    assert ctx.is_cancelled() is True
