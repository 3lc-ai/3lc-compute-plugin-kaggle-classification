# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Session 16 (the error-visibility audit): no job ends silently. The job entry keeps one note per kind
(running / completed / cancelled / failed, the pid rule on read), a refusal before a stage's own record
exists still leaves a failed record, and the fragment's pure selectors turn both into a visible message."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from kaggle_classification import KaggleClassificationPlugin, session
from kaggle_classification import manifest as manifest_mod

UI = Path(__file__).parent.parent / "src" / "kaggle_classification" / "ui" / "ui.html"


class _Ctx:
    """The SDK JobContext surface ``_run_job`` touches."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.job_id = "job-" + str(params.get("kind"))
        self.cancelled = False
        self.state_dir = None
        self.lines: list[str] = []

    def log(self, m: str) -> None:
        self.lines.append(m)

    def progress(self, *, percent: float, label: str = "", timing=None) -> None:
        pass

    def metric(self, label: str, value) -> None:
        pass

    def result(self, url: str) -> None:
        pass

    def emit(self, name: str, payload=None) -> None:
        pass

    def fail(self, message: str):
        from tlc_plugin_sdk.job_context import JobFailed

        raise JobFailed(message)


@pytest.fixture
def plugin(home, monkeypatch, manifest):
    monkeypatch.setattr(manifest_mod, "resolve_manifest_for_job",
                        lambda: (manifest, {"manifest_source": "test", "competition_id": "intel-scene",
                                            "kit_version": "v1", "manifest_sha256": "0" * 64}))
    return KaggleClassificationPlugin()


def test_a_stage_exception_leaves_a_failed_note_with_the_reason(plugin, monkeypatch):
    from kaggle_classification import importer

    def boom(params, ctx, manifest):
        raise ValueError("the disk is full")

    monkeypatch.setattr(importer, "run_import", boom)
    with pytest.raises(ValueError):
        plugin._run_job(_Ctx({"kind": "import"}))
    note = session.last_jobs()["import"]
    assert note["status"] == "failed" and note["error"] == "ValueError: the disk is full"
    assert note["id"] == "job-import" and note["finished_at"] and note["pid"] == os.getpid()


def test_a_refusal_leaves_a_failed_note_verbatim(plugin, monkeypatch):
    from tlc_plugin_sdk.job_context import JobFailed

    from kaggle_classification import importer

    def refuse(params, ctx, manifest):
        raise importer.ImportRefused("No starter kit on this machine yet. Download the starter kit first.")

    monkeypatch.setattr(importer, "run_import", refuse)
    with pytest.raises(JobFailed):
        plugin._run_job(_Ctx({"kind": "import"}))
    note = session.last_jobs()["import"]
    assert note["status"] == "failed" and note["error"].startswith("No starter kit on this machine yet.")


def test_a_finished_job_notes_completed_or_cancelled(plugin, monkeypatch):
    from kaggle_classification import kit

    monkeypatch.setattr(kit, "run_download", lambda params, ctx, manifest: {"cancelled": False})
    plugin._run_job(_Ctx({"kind": "download_kit"}))
    assert session.last_jobs()["download_kit"]["status"] == "completed"
    monkeypatch.setattr(kit, "run_download", lambda params, ctx, manifest: {"cancelled": True})
    plugin._run_job(_Ctx({"kind": "download_kit"}))
    assert session.last_jobs()["download_kit"]["status"] == "cancelled"


def test_an_unknown_kind_is_a_failed_note_too(plugin):
    from tlc_plugin_sdk.job_context import JobFailed

    with pytest.raises(JobFailed):
        plugin._run_job(_Ctx({"kind": "bogus"}))
    assert session.last_jobs()["bogus"]["status"] == "failed"


def test_a_running_note_of_a_dead_worker_reads_back_as_interrupted(home):
    now = time.time()
    session.note_job("import", id="j1", kind="import", status="running", pid=os.getpid() + 7, started_at=now,
                     heartbeat=now, finished_at=None, error=None)
    session.note_job("train", id="j2", kind="train", status="running", pid=os.getpid(), started_at=now - 2000,
                     heartbeat=now - 1000, finished_at=None, error=None)
    session.note_job("predict", id="j3", kind="predict", status="running", pid=os.getpid(), started_at=now,
                     heartbeat=now, finished_at=None, error=None)
    notes = session.last_jobs()
    assert notes["import"]["status"] == "interrupted"
    assert "the import was running. Start it again." in notes["import"]["error"]
    assert notes["train"]["status"] == "interrupted" and "training was running" in notes["train"]["error"]
    assert notes["predict"]["status"] == "running"
    # The stored note is untouched: the rule is applied on read.
    stored = json.loads(session.config_path().read_text(encoding="utf-8"))
    assert stored["last_jobs"]["import"]["status"] == "running"


def test_the_heartbeat_touches_only_the_running_note_of_that_job(home):
    now = time.time()
    session.note_job("train", id="j2", kind="train", status="running", pid=os.getpid(), started_at=now - 5000,
                     heartbeat=now - 5000, finished_at=None, error=None)
    session._touched.clear()
    session.touch_job("train", "other-job")
    assert session.last_jobs()["train"]["status"] == "interrupted"
    session._touched.clear()
    session.touch_job("train", "j2")
    assert session.last_jobs()["train"]["status"] == "running"


def test_config_serves_the_notes_with_the_rule_applied_and_never_raw(home):
    from kaggle_classification import routes

    now = time.time()
    session.note_job("download_kit", id="d1", kind="download_kit", status="running", pid=os.getpid() + 1,
                     started_at=now, heartbeat=now, finished_at=None, error=None)
    cfg = routes.config_payload()
    assert "last_jobs" not in cfg
    assert cfg["_meta"]["last_jobs"]["download_kit"]["status"] == "interrupted"


# ── The stages' own records for a refusal before they existed ────────────────────────────────


def test_a_train_refusal_before_the_record_leaves_a_failed_run_record(home):
    from kaggle_classification import trainer

    reason = "Import the kit first: no import record exists on this machine."
    assert trainer.record_refusal("t1", {"epochs": "3", "run_name": "first try"}, reason)
    state = trainer.train_state()
    assert state["current"]["id"] == "t1" and state["current"]["status"] == "failed"
    assert state["current"]["error"] == reason
    assert [r["id"] for r in state["runs"]] == ["t1"] and state["runs"][0]["status"] == "failed"
    assert state["runs"][0]["run_name"] == "first try"
    # The job that recorded itself, and a live run, are never overwritten.
    assert not trainer.record_refusal("t1", {}, "again")
    trainer._write_state({"current": {"id": "live", "status": "running", "pid": os.getpid(), "heartbeat": time.time(),
                                      "facts": {}, "log": []}, "runs": []})
    assert not trainer.record_refusal("t2", {"epochs": "1"}, "A training run is already in progress.")
    assert trainer.read_state()["current"]["id"] == "live"


def test_a_predict_or_submit_refusal_before_the_record_leaves_a_failed_record(home):
    from kaggle_classification import predictor

    reason = "The kit's test images could not be verified: missing. Run Import (tab 1) to restore the kit."
    assert predictor.record_refusal("predict", "p1", {"train_job_id": "t9"}, reason)
    st = predictor.predict_submit_state()
    assert st["state"] == "failed" and st["predict"]["job_id"] == "p1" and st["predict"]["error"] == reason
    submit_reason = "No validated prediction CSV found for that job. Run inference first."
    assert predictor.record_refusal("kaggle_submit", "s1", {"predict_job_id": "p0", "message": "m"}, submit_reason)
    rec = predictor.read_submit_record()
    assert rec["id"] == "s1" and rec["status"] == "failed" and rec["predict_job_id"] == "p0"
    assert not predictor.record_refusal("predict", "p1", {}, "again")


# ── The fragment's pure selectors (run in node) ──────────────────────────────────────────────


def _fn(html: str, name: str) -> str:
    m = re.search(rf"function {name}\([^)]*\) \{{.*?\n      \}}\n", html, re.S)
    assert m, name
    return m.group(0)


def _node(script: str, arg) -> object:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    out = subprocess.run([node, "-e", script, json.dumps(arg)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


def test_the_last_job_failure_selector():
    html = UI.read_text(encoding="utf-8")
    script = _fn(html, "kgLastJobFailure") + (
        "\nvar c = JSON.parse(process.argv[1]);"
        "process.stdout.write(JSON.stringify(c.map(function (x) {"
        " var r = kgLastJobFailure(x[0], x[1], x[2], x[3]); return r ? r.id : null; })));\n")
    failed = {"import": {"id": "j2", "status": "failed", "error": "boom", "finished_at": 200}}
    cases = [
        [failed, "import", None, None],            # no success record: shown
        [failed, "import", "j1", 100],             # an older success: shown
        [failed, "import", "j2", 300],             # the success record IS this job: not shown
        [failed, "import", "j1", 300],             # a newer success: not shown
        [{"import": {"id": "j3", "status": "interrupted", "finished_at": 5}}, "import", None, None],
        [{"import": {"id": "j4", "status": "completed"}}, "import", None, None],
        [failed, "download_kit", None, None],
        [None, "import", None, None],
    ]
    assert _node(script, cases) == ["j2", "j2", None, None, "j3", None, None, None]


def test_every_reason_gets_a_next_step_unless_it_has_one():
    html = UI.read_text(encoding="utf-8")
    m = re.search(r"var KG_NEXT_STEPS = \[.*?\n      function kgWithNextStep\(text\) \{.*?\n      \}\n", html, re.S)
    assert m
    script = m.group(0) + (
        "\nvar c = JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(c.map(kgWithNextStep)));\n")
    cases = [
        "[Errno 28] No space left on device",
        "[WinError 5] Access is denied: 'C:/kit'",
        "[Errno 2] No such file or directory: 'model/best.pt'",
        "CUDA error: no kernel image is available for execution on the device",
        "CUDA out of memory. Tried to allocate 2 GiB",
        "CUDA error: device-side assert triggered",
        "HTTPSConnectionPool: Read timed out",
        "Interrupted: the compute service restarted while training was running.",
        "something nobody foresaw",
        "Download the starter kit again (Re-verify on the Import tab names the files), then import.",
        "Import the kit first: no import record exists on this machine.",
        "",
    ]
    got = _node(script, cases)
    assert got[0].endswith("Free disk space, then run it again.")
    assert "Close what holds the folder" in got[1]
    assert "Run Import (tab 1) to restore the kit" in got[2]
    assert "update your NVIDIA driver to R580 or newer" in got[3]
    assert got[4].endswith("Lower the batch size or choose CPU, then run it again.")
    assert got[5].startswith("CUDA error: device-side assert triggered. Choose CPU and run it again")
    assert got[6].endswith("Check your internet connection, then run it again.")
    assert got[7].endswith("was running. Start it again.")
    assert got[8] == ("something nobody foresaw. Run it again. If it fails the same way, copy the diagnostics "
                      "and post them in the competition Discussion on Kaggle.")
    assert got[9] == cases[9] and got[10] == cases[10] and got[11] == ""


def test_the_predict_reconcile_selector_picks_the_terminal_record_of_the_shown_job():
    html = UI.read_text(encoding="utf-8")
    script = _fn(html, "psReconcileFromState") + (
        "\nvar c = JSON.parse(process.argv[1]);"
        "process.stdout.write(JSON.stringify(c.map(function (x) { return psReconcileFromState(x[0], x[1], x[2]); })));\n")
    cases = [
        [{"state": "failed", "predict": {"job_id": "p1", "status": "failed", "error": "boom"}}, "predict", "p1"],
        [{"state": "running", "predict": {"job_id": "p1", "status": "running"}}, "predict", "p1"],
        [{"state": "predicted", "predict": {"job_id": "p2", "status": "completed"}}, "predict", "p1"],
        [{"state": "stale", "predict": {"job_id": "p1", "status": "stale", "error": "Interrupted: x"}}, "predict", "p1"],
        [{"state": "predicted", "predict": {"job_id": "p1", "status": "completed"},
          "submission": {"job_id": "s1", "job_status": "failed", "reason": "Kaggle rejected the submission."}},
         "submit", "s1"],
        [{"state": "predicted", "predict": {"job_id": "p1", "status": "completed"},
          "submission": {"job_id": "s1", "job_status": "running"}}, "submit", "s1"],
        [None, "predict", "p1"],
        [{"state": "failed", "predict": {"job_id": "p1", "status": "failed"}}, "predict", ""],
    ]
    assert _node(script, cases) == [
        {"status": "failed", "error": "boom"}, None, None, {"status": "stale", "error": "Interrupted: x"},
        {"status": "failed", "error": "Kaggle rejected the submission."}, None, None, None,
    ]


def test_the_doctor_names_its_red_rows_and_the_import_tab_asks_once():
    """Item 3 of the audit: the Doctor's red rows, by title, drive both the Status panel and the Import tab's
    first-open banner ("Something needs attention before you start: <rows>. Open Status.")."""
    html = UI.read_text(encoding="utf-8")
    deps = "".join(_fn(html, n) for n in ("stImagesMissing", "stGpuMissingKernels", "stDoctorAttention"))
    script = deps + (
        "\nvar c = JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(c.map(stDoctorAttention)));\n")
    cases = [
        {"kit": {"state": "success"}, "device": {"cuda_available": True}},
        {"kit": {"state": "stale"}, "import": {"images": {"train": {"state": "missing"}}}},
        {"device": {"gpu": {"kernels_ok": False}}, "project_root": {"warning": "two roots"}},
        {"manifest": {"error": "x"}, "kaggle": {"state": "error"}},
        {"error": "the Doctor could not be read"},
        None,
    ]
    assert _node(script, cases) == [
        [], ["Kit", "Table images"], ["CUDA in the worker", "Project root"], ["Manifest", "Kaggle"], [], [],
    ]
    body = _fn(html, "kgDoctorAttentionCheck")
    assert "'/status/doctor?kaggle=0'" in body
    assert "Something needs attention before you start:" in body and "Open Status" in body
    assert "kgDoctorAttentionCheck(false);" in _fn(html, "kgInitImportTab")
