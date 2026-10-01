# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Predict + Submit stage (session 4, docs/PREDICT_MIRROR.md): the plugin-run-only gate (the
three-way sha256 match + green provenance, D2), the test-images gate (D3), the seven format checks
and the six-decimal CSV (D9), the distribution card (D5), the revisit resolution, the Kaggle client
(credentials never raise SystemExit, error classification, the D12 read-back, the soft outcomes), the
submit job's records and ledger entries, and — with the heavy extra — an end-to-end predict on the
synthetic kit after a real one-epoch run (the val check reproduces the recorded accuracy, a tampered
checkpoint is refused, an unreadable test image fails the job).

The heavy tests need tlc + torch + torchvision; they skip without them (CLAUDE.md §B)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import build_kit  # tools/ is on sys.path (conftest)
import pytest
from helpers import FakeCtx, make_kit_tree, small_manifest_data

from kaggle_classification import importer, kaggle_client, ledger, predictor, session, trainer
from kaggle_classification import manifest as manifest_mod

# ── Light: the run list and the plugin-run-only gate ─────────────────────────

SHA_A = "a" * 64


def _entry(tmp_path: Path, **over):
    weights = tmp_path / "runs" / over.get("run_name", "r1") / "model" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(over.pop("payload", b"plugin-trained"))
    e = {
        "id": over.pop("id", "t1"), "run_name": "r1", "run_url": str(weights.parent.parent), "status": "completed",
        "weights": str(weights), "best_checkpoint_sha256": predictor.sha256_of(weights), "provenance_ok": True,
        "best_val_accuracy": 57.58, "best_epoch": 2, "epochs_completed": 2, "created_at": time.time(),
    }
    e.update(over)
    return e


@pytest.fixture
def run_params_match(monkeypatch):
    """The Run's recorded sha agrees with the train record (the normal case)."""
    monkeypatch.setattr(predictor, "run_parameters", lambda url: {"best_checkpoint_sha256": predictor.sha256_of(
        Path(url) / "model" / "best.pt") if (Path(url) / "model" / "best.pt").is_file() else ""})


def _store_runs(store, entries, current=None):
    store.save({"train_state": {"current": current, "runs": entries}})


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"status": "running"}, "still training"),
        ({"status": "failed"}, "failed"),
        ({"status": "stale"}, "interrupted"),
        ({"weights": ""}, "no best checkpoint saved"),
        ({"provenance_ok": False}, "provenance check failed"),
        ({"best_checkpoint_sha256": SHA_A}, "best checkpoint changed on disk (sha256 mismatch)"),
        ({}, ""),
        ({"status": "cancelled"}, ""),
    ],
)
def test_assess_run_names_each_unusable_reason(tmp_path, run_params_match, over, reason):
    e = _entry(tmp_path, **over)
    usable, got = predictor.assess_run(e)
    assert got == reason and usable == (reason == "")


def test_assess_run_sees_a_missing_file_and_a_disagreeing_run(tmp_path, monkeypatch):
    e = _entry(tmp_path)
    Path(e["weights"]).unlink()
    assert predictor.assess_run(e) == (False, "best checkpoint missing on disk")
    e = _entry(tmp_path, id="t2", run_name="r2")
    monkeypatch.setattr(predictor, "run_parameters", lambda url: {"best_checkpoint_sha256": SHA_A})
    assert predictor.assess_run(e) == (False, "the Run’s record disagrees with the train record (sha256 mismatch)")


def test_list_runs_includes_the_running_current_record(store, tmp_path, run_params_match):
    done = _entry(tmp_path)
    cur = {"id": "t9", "kind": "train", "status": "running", "pid": os.getpid(), "created_at": time.time(),
           "started_at": time.time(), "heartbeat": time.time(), "params": {"epochs": 2, "run_name": "live"},
           "facts": {"run_name": "live"}, "progress": {"epoch": 1, "total_epochs": 2, "history": []}, "checks": [],
           "log": []}
    _store_runs(store, [done], current=cur)
    rows = predictor.list_runs()
    assert [r["job_id"] for r in rows] == ["t9", "t1"]
    assert rows[0]["usable"] is False and rows[0]["reason"] == "still training"
    assert rows[1]["usable"] is True and rows[1]["run_folder"] == "r1" and rows[1]["best_val_accuracy"] == 57.58


def test_run_lists_are_scoped_to_the_import_records_project(store, tmp_path, run_params_match):
    """After Start over into a new project, Train's Previous runs and Predict's picker list only that
    project's runs; the ETA history stays global; the gate still resolves any plugin run by id."""
    old = _entry(tmp_path, id="o1", run_name="old", project_name="intel-scene")
    new = _entry(tmp_path, id="n1", run_name="new", project_name="intel-scene-demo")
    # A summary without project_name: the project comes from the run URL (<root>/<project>/runs/<name>).
    legacy = _entry(tmp_path, id="l1", run_name="legacy")
    legacy["run_url"] = "C:/Users/p/AppData/Local/3LC/3LC/projects/intel-scene/runs/legacy"
    _store_runs(store, [new, old, legacy])
    assert trainer.run_project(legacy) == "intel-scene" and trainer.run_project(new) == "intel-scene-demo"
    # No import record: no filter.
    assert [r["job_id"] for r in predictor.list_runs()] == ["n1", "o1", "l1"]
    store.save({"import_state": {"project_name": "intel-scene-demo", "tables": {"train": {"url": "x"}}}})
    assert [r["job_id"] for r in predictor.list_runs()] == ["n1"]
    st = trainer.train_state()
    assert st["project"] == "intel-scene-demo" and [r["id"] for r in st["project_runs"]] == ["n1"]
    assert [r["id"] for r in st["runs"]] == ["n1", "o1", "l1"]
    store.save({"import_state": {"project_name": "intel-scene", "tables": {"train": {"url": "x"}}}})
    assert [r["job_id"] for r in predictor.list_runs()] == ["o1", "l1"]
    assert predictor.resolve_checkpoint({"train_job_id": "n1"})["run_name"] == "new"


def test_train_job_id_wins_and_a_supplied_path_is_ignored(store, tmp_path, run_params_match):
    """THE bypass case (ExDark's test_host_weights_gate): a real id plus an arbitrary path must load
    the record's checkpoint, never the path."""
    e = _entry(tmp_path)
    _store_runs(store, [e])
    smuggled = tmp_path / "smuggled.pt"
    smuggled.write_bytes(b"trained elsewhere")
    ck = predictor.resolve_checkpoint({"train_job_id": "t1", "weights_path": str(smuggled)})
    assert ck["weights"] == e["weights"] and ck["sha256_on_disk"] == ck["sha256_recorded"] == ck["sha256_on_run"]
    assert ck["run_name"] == "r1" and ck["run_folder"] == "r1" and ck["train_job_id"] == "t1"


def test_a_bare_weights_path_is_refused_for_everyone(store, tmp_path, run_params_match):
    """D1: no host mode — participants and organizers run the same build."""
    mine = tmp_path / "mine.pt"
    mine.write_bytes(b"x")
    with pytest.raises(predictor.PredictRefused) as exc:
        predictor.resolve_checkpoint({"weights_path": str(mine)})
    assert exc.value.args[0] == predictor.NOT_A_RUN
    with pytest.raises(predictor.PredictRefused, match="Select a run trained in this plugin"):
        predictor.resolve_checkpoint({})


def test_the_three_way_sha_gate_refuses_a_tampered_checkpoint(store, tmp_path, run_params_match):
    e = _entry(tmp_path)
    _store_runs(store, [e])
    Path(e["weights"]).write_bytes(b"tampered after training")
    with pytest.raises(predictor.PredictRefused) as exc:
        predictor.resolve_checkpoint({"train_job_id": "t1"})
    assert exc.value.args[0] == predictor.SHA_MISMATCH


def test_the_gate_refuses_when_the_run_disagrees(store, tmp_path, monkeypatch):
    e = _entry(tmp_path)
    _store_runs(store, [e])
    monkeypatch.setattr(predictor, "run_parameters", lambda url: {"best_checkpoint_sha256": SHA_A})
    with pytest.raises(predictor.PredictRefused) as exc:
        predictor.resolve_checkpoint({"train_job_id": "t1"})
    assert exc.value.args[0] == predictor.SHA_MISMATCH


def test_the_gate_blocks_a_failed_provenance_check_and_allows_cancelled_with_best(store, tmp_path, run_params_match):
    bad = _entry(tmp_path, id="p0", run_name="p0", provenance_ok=False)
    cancelled = _entry(tmp_path, id="c0", run_name="c0", status="cancelled")
    missing = _entry(tmp_path, id="m0", run_name="m0")
    Path(missing["weights"]).unlink()
    _store_runs(store, [bad, cancelled, missing])
    with pytest.raises(predictor.PredictRefused, match="failed a provenance check"):
        predictor.resolve_checkpoint({"train_job_id": "p0"})
    assert predictor.resolve_checkpoint({"train_job_id": "c0"})["run_name"] == "c0"
    with pytest.raises(predictor.PredictRefused, match="missing on disk"):
        predictor.resolve_checkpoint({"train_job_id": "m0"})
    with pytest.raises(predictor.PredictRefused, match="No plugin run with id"):
        predictor.resolve_checkpoint({"train_job_id": "nope"})


# ── Light: the CSV, the seven checks, the distribution card ──────────────────


def _rows(ids, pred=0, conf=0.5):
    return [[i, str(pred), f"{conf:.6f}"] for i in ids]


def test_the_seven_format_checks_pass_on_a_correct_file(manifest):
    ids = [f"{i:016x}" for i in range(manifest.splits.test.count)]
    checks = predictor.validate_submission(list(manifest.submission.columns), _rows(ids), manifest, ids)
    assert len(checks) == 7 and all(c["ok"] for c in checks)
    assert all(c["group"] == "Submission format" for c in checks)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda h, r, ids: (["image_id", "Prediction", "confidence"], r), "exactly the columns"),
        (lambda h, r, ids: (h, r[:-1]), "Re-run Import"),
        (lambda h, r, ids: (h, r[:-1] + [list(r[0])]), "duplicated image_id"),
        (lambda h, r, ids: (h, list(reversed(r))), "in its order"),
        (lambda h, r, ids: (h, [[i, "7", "0.5"] for i in ids]), "Predictions must be integer class ids 0–5"),
        (lambda h, r, ids: (h, [[i, "0.6", "0.5"] for i in ids]), "Predictions must be integer class ids"),
        (lambda h, r, ids: (h, [[i, "1", "1.5"] for i in ids]), "Confidence must be finite"),
        (lambda h, r, ids: (h, [[i, "1", "nan"] for i in ids]), "Confidence must be finite"),
        (lambda h, r, ids: (h, [[i, "1", ""] for i in ids]), "Confidence must be finite"),
    ],
)
def test_each_format_check_fails_with_its_remedy(manifest, mutate, needle):
    ids = [f"{i:016x}" for i in range(manifest.splits.test.count)]
    header, rows = mutate(list(manifest.submission.columns), _rows(ids, pred=1), ids)
    with pytest.raises(predictor.PredictRefused, match=needle) as exc:
        predictor.validate_submission(header, rows, manifest, ids)
    assert exc.value.checks and exc.value.checks[-1]["ok"] is False


def test_write_csv_is_lf_utf8_six_decimals_in_sample_order(tmp_path, manifest):
    path = tmp_path / "submission.csv"
    predictor.write_csv(path, manifest.submission.columns, [("b1", 3, 0.123456789), ("a2", 0, 1.0)])
    raw = path.read_bytes()
    assert raw == b"image_id,prediction,confidence\nb1,3,0.123457\na2,0,1.000000\n"
    assert not raw.startswith(b"\xef\xbb\xbf")
    header, rows = predictor.read_csv_rows(path)
    assert header == list(manifest.submission.columns) and rows == [["b1", "3", "0.123457"], ["a2", "0", "1.000000"]]


def test_distribution_card_counts_and_warns_on_skew(manifest):
    n = manifest.num_classes
    healthy = predictor.distribution([i % n for i in range(600)], [0.9] * 600, manifest)
    assert healthy["images"] == 600 and healthy["low_confidence"] == 0 and "warning" not in healthy
    assert set(healthy["per_class"]) == set(manifest.class_names) and all(v == 100 for v in healthy["per_class"].values())
    skewed = predictor.distribution([0] * 400 + [1] * 200, [0.3] * 300 + [0.8] * 300, manifest)
    assert skewed["low_confidence"] == 300 and skewed["mean_confidence"] == 0.55
    assert "Are these fully-trained weights? (Submitting is still fine.)" in skewed["warning"]
    assert manifest.class_names[0] + " 67 %" in skewed["skewed"] and manifest.class_names[2] + " 0 %" in skewed["skewed"]


# ── Light: the test-images gate on a synthetic kit ───────────────────────────


@pytest.fixture
def kit_record(tmp_path, home):
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(tmp_path / "tree", data)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    manifest = manifest_mod.parse_manifest(data, source="test", source_detail="synthetic")
    session.save({"import_state": {"kit_dir": str(kit_root), "tables": {"train": {"url": "x"}, "val": {"url": "y"}},
                                   "val_locked": {"url": "y"}}})
    return manifest, kit_root


def test_test_inputs_gate_verifies_every_file_against_files_json(kit_record):
    manifest, kit_root = kit_record
    info = predictor.test_inputs(manifest)
    assert info["ok"] and info["state"] == "ok" and info["count"] == 4 == info["verified"]
    assert info["ids"] == sorted(info["ids"]) and all(Path(p).is_file() for p in info["paths"])
    assert info["files_json_sha256"] and info["sample_submission_sha256"]
    assert "ids" not in predictor.preflight(manifest)
    # A tampered image (same size) is a mismatch; a deleted one is missing; no record is idle.
    target = Path(info["paths"][1])
    target.write_bytes(bytes(reversed(target.read_bytes())))
    info = predictor.test_inputs(manifest)
    assert not info["ok"] and info["mismatch"] == [info["ids"][1]] and "differ from the kit" in info["error"]
    target.unlink()
    info = predictor.test_inputs(manifest)
    assert not info["ok"] and info["missing"] == [info["ids"][1]] and "missing" in info["error"]
    session.save({"import_state": {}})
    assert predictor.test_inputs(manifest)["state"] == "idle"


# ── Light: the revisit resolution, the download path, the ledger ─────────────


def _predict_record(tmp_path, status="completed", pid=None, with_csv=True, sha=True):
    csv_path = tmp_path / "submission.csv"
    if with_csv:
        csv_path.write_text("image_id,prediction,confidence\n", encoding="utf-8")
    rec = {
        "id": "p1", "kind": "predict", "status": status, "pid": pid or os.getpid(), "created_at": time.time() - 10,
        "started_at": time.time() - 10, "heartbeat": time.time() - 5, "finished_at": time.time(), "params": {},
        "facts": {"run_name": "r1", "csv_path": str(csv_path), "csv_sha256": predictor.sha256_of(csv_path) if (sha and with_csv) else "",
                  "sanity": {"images": 1}, "local_score": {"kind": "val", "value": 50.0}},
        "checks": [{"label": "x", "ok": True}], "log": ["a"],
    }
    session.save({"predict_state": rec})
    return csv_path


def test_record_and_csv_present_is_predicted(store, tmp_path):
    _predict_record(tmp_path)
    out = predictor.predict_submit_state()
    assert out["state"] == "predicted" and out["predict"]["job_id"] == "p1" and out["predict"]["csv_on_disk"]
    assert predictor.csv_path_for("p1") == str(tmp_path / "submission.csv")


def test_missing_or_changed_csv_is_empty_despite_the_record(store, tmp_path):
    _predict_record(tmp_path, with_csv=False)
    assert predictor.predict_submit_state()["state"] == "empty"
    csv_path = _predict_record(tmp_path)
    csv_path.write_text("image_id,prediction,confidence\nx,0,0.5\n", encoding="utf-8")
    out = predictor.predict_submit_state()
    assert out["state"] == "empty" and "changed on disk" in out["note"]


def test_a_running_record_from_another_worker_reads_back_as_stale(store, tmp_path):
    _predict_record(tmp_path, status="running", pid=os.getpid() + 7919)
    out = predictor.predict_submit_state()
    assert out["state"] == "stale" and "restarted" in out["predict"]["error"]


def test_submitted_pairing_resolves_and_a_foreign_submission_does_not(store, tmp_path):
    _predict_record(tmp_path)
    session.save({"submit_state": {"id": "s1", "kind": "kaggle_submit", "status": "completed", "pid": os.getpid(),
                                   "predict_job_id": "p1", "params": {"message": "m"}, "finished_at": time.time(),
                                   "facts": {"submission": {"status": "submitted", "ref": "123", "kaggle": {"status": "COMPLETE", "public_score": 0.5}}}}})
    out = predictor.predict_submit_state()
    assert out["state"] == "submitted" and out["submission"]["ref"] == "123" and out["submission"]["kaggle"]["public_score"] == 0.5
    session.save({"submit_state": {"id": "s2", "kind": "kaggle_submit", "status": "completed", "pid": os.getpid(),
                                   "predict_job_id": "other", "facts": {"submission": {"status": "submitted", "ref": "9"}}}})
    out = predictor.predict_submit_state()
    assert out["state"] == "predicted" and "submission" not in out


def test_ledger_is_append_only_and_findable(home):
    assert ledger.read() == []
    ledger.append({"kind": "predict", "job_id": "a", "csv": {"path": "/tmp/a.csv"}})
    ledger.append({"kind": "submit", "job_id": "b", "predict_job_id": "a"})
    ledger.append({"kind": "predict", "job_id": "c", "csv": {"path": "/tmp/c.csv"}})
    assert [e["kind"] for e in ledger.read()] == ["predict", "submit", "predict"]
    assert ledger.find("predict", job_id="a")["csv"]["path"] == "/tmp/a.csv"
    assert ledger.find("submit", predict_job_id="a")["job_id"] == "b" and ledger.find("submit", job_id="zz") is None
    assert all("ts" in e for e in ledger.read())
    assert predictor.csv_path_for("c") == "/tmp/c.csv" and predictor.csv_path_for("none") == ""
    with pytest.raises(ValueError):
        ledger.append({"job_id": "x"})


# ── Light: the Kaggle client ─────────────────────────────────────────────────


@pytest.fixture
def no_credentials(tmp_path, monkeypatch):
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_credentials_absent_never_raises_system_exit(no_credentials):
    assert kaggle_client.credentials_present() is False
    api, reason = kaggle_client.authenticated_api()
    assert api is None and "Kaggle credentials not found" in reason and "upload it manually" in reason
    srcs = kaggle_client.credential_sources()
    assert srcs["access_token"].startswith(str(no_credentials)) and srcs["kaggle_json"].endswith("kaggle.json")


def test_credential_presence_follows_the_client_sources(no_credentials, monkeypatch):
    (no_credentials / ".kaggle").mkdir()
    (no_credentials / ".kaggle" / "access_token").write_text("x", encoding="ascii")
    assert kaggle_client.credentials_present() is True
    (no_credentials / ".kaggle" / "access_token").unlink()
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(no_credentials / "elsewhere"))
    assert kaggle_client.credentials_present() is False
    (no_credentials / "elsewhere").mkdir()
    (no_credentials / "elsewhere" / "kaggle.json").write_text("{}", encoding="ascii")
    assert kaggle_client.credentials_present() is True


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("You have reached your daily submission limit", "daily_limit"),
        ("Maximum of 5 submissions per day", "daily_limit"),
        ("You must accept the competition rules before submitting", "not_joined"),
        ("403 Forbidden", "error"),
    ],
)
def test_classify_error(text, kind):
    assert kaggle_client.classify_error(RuntimeError(text)) == kind


class _Sub:
    def __init__(self, ref, status, score=None, err=""):
        import enum

        self.ref = ref
        self.status = enum.Enum("S", ["PENDING", "COMPLETE", "ERROR"])[status]
        self.public_score = score
        self.private_score = None
        self.error_description = err
        self.date = None
        self.description = "m"
        self.file_name = "submission.csv"


class _Api:
    """A fake client: ``answers`` are the successive submissions-list replies; ``by_ref`` the
    per-ref replies of ``get_submission`` (absent = the call fails, like a 403)."""

    def __init__(self, answers=None, raise_on_submit=None, submit_ref="777", by_ref=None):
        self.answers = list(answers or [])
        self.by_ref = list(by_ref or [])
        self.raise_on_submit = raise_on_submit
        self.submit_ref = submit_ref
        self.submitted = []
        self.config_values = {"username": "participant"}

    def competition_submissions(self, slug, page_size=20):
        return self.answers.pop(0) if self.answers else []

    def get_submission(self, ref):
        if not self.by_ref:
            msg = "403 Forbidden"
            raise RuntimeError(msg)
        return kaggle_client._submission_dict(self.by_ref.pop(0))

    def competition_submit(self, file_name, message, competition):
        if self.raise_on_submit:
            raise RuntimeError(self.raise_on_submit)
        self.submitted.append((file_name, message, competition))

        class R:
            ref = self.submit_ref
            message = "Submission accepted"

        return R()


def _limits_403(api, slug):
    msg = "403 Forbidden"
    raise RuntimeError(msg)


@pytest.fixture(autouse=True)
def _fake_client_calls(monkeypatch):
    """The two per-call SDK wrappers go through the fake's methods (no kagglesdk requests in tests)."""
    monkeypatch.setattr(kaggle_client, "get_submission", lambda api, ref: api.get_submission(ref))
    monkeypatch.setattr(kaggle_client, "submission_limits", _limits_403)


def test_read_back_prefers_get_submission_by_ref(monkeypatch):
    api = _Api(by_ref=[_Sub("777", "PENDING"), _Sub("777", "COMPLETE", "0.65666")])
    naps = []
    out = kaggle_client.read_back(api, "slug", "777", schedule=(1, 1, 1), sleep=naps.append)
    assert out["status"] == "COMPLETE" and out["public_score"] == 0.65666 and naps == [1, 1]


def test_read_back_waits_for_a_non_pending_status(monkeypatch):
    api = _Api(answers=[[_Sub("777", "PENDING")], [_Sub("1", "COMPLETE", "0.1"), _Sub("777", "COMPLETE", "0.5733")]])
    naps = []
    out = kaggle_client.read_back(api, "slug", "777", schedule=(1, 1, 1), sleep=naps.append)
    assert out["status"] == "COMPLETE" and out["public_score"] == 0.5733 and naps == [1, 1]
    err = kaggle_client.read_back(_Api(answers=[[_Sub("777", "ERROR", err="Predictions must be 0–5")]]), "slug", "777",
                                  schedule=(1,), sleep=lambda s: None)
    assert err["status"] == "ERROR" and err["error_description"] == "Predictions must be 0–5"
    gone = kaggle_client.read_back(_Api(answers=[[]]), "slug", "777", schedule=(1,), sleep=lambda s: None)
    assert gone["status"] == "unknown"


def test_submit_outcomes_through_a_fake_client(monkeypatch, tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text("image_id,prediction,confidence\n", encoding="utf-8")
    logs = []
    info = {"user_has_entered": True, "max_daily_submissions": 100, "title": "T"}
    monkeypatch.setattr(kaggle_client, "competition_info", lambda api, slug: info)
    api = _Api(answers=[[_Sub("777", "COMPLETE", "0.57")]])
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (api, ""))
    monkeypatch.setattr(kaggle_client, "READ_BACK_SCHEDULE_S", (0,))
    out = kaggle_client.submit(str(csv), "msg", "slug", 100, logs.append)
    assert out["status"] == "submitted" and out["ref"] == "777" and out["kaggle"]["public_score"] == 0.57
    assert api.submitted == [(str(csv), "msg", "slug")]
    # Daily limit (Kaggle's refusal is authoritative; the manifest number is quoted).
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (_Api(raise_on_submit="daily submission limit exceeded"), ""))
    out = kaggle_client.submit(str(csv), "msg", "slug", 100, logs.append)
    assert out["status"] == "limit_reached" and "(100/day)" in out["reason"]
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (_Api(raise_on_submit="500 Server Error"), ""))
    out = kaggle_client.submit(str(csv), "msg", "slug", 100, logs.append)
    assert out["status"] == "failed" and out["reason"].startswith("Kaggle rejected the submission: 500")
    # Not joined, from the pre-probe (no attempt spent), and no credentials.
    info["user_has_entered"] = False
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (_Api(), ""))
    assert kaggle_client.submit(str(csv), "msg", "slug", 100, logs.append)["status"] == "not_joined"
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (None, "no creds"))
    assert kaggle_client.submit(str(csv), "msg", "slug", 100, logs.append) == {"status": "skipped", "reason": "no creds"}


def test_connection_card_states(monkeypatch):
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (None, "help text"))
    assert kaggle_client.connection("slug", 100)["state"] == "no_credentials"
    api = _Api(answers=[[_Sub("1", "COMPLETE", "0.1")]])
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (api, ""))
    monkeypatch.setattr(kaggle_client, "competition_info", lambda a, s: {"user_has_entered": True, "max_daily_submissions": 5, "title": "T"})
    monkeypatch.setattr(kaggle_client, "submission_limits",
                        lambda a, s: {"num_today": 2, "num_allowed_now": 98, "num_total": 2, "limited_by_total": False})
    logs = []
    out = kaggle_client.connection("slug", 100, logs.append)
    assert out["state"] == "ready" and out["username"] == "participant" and out["daily_limit"] == 100
    assert out["kaggle_daily_limit"] == 5 and out["submissions_used_today"] == 2 and logs
    assert kaggle_client.submissions_used_today(api, "slug") == 2
    monkeypatch.setattr(kaggle_client, "competition_info", lambda a, s: {"user_has_entered": False, "max_daily_submissions": 0, "title": ""})
    assert kaggle_client.connection("slug", 100)["state"] == "not_joined"

    def boom(a, s):
        raise RuntimeError("403")

    monkeypatch.setattr(kaggle_client, "competition_info", boom)
    out = kaggle_client.connection("slug", 100)
    assert out["state"] == "ready" and out["probe_error"] == "403"


def test_run_kaggle_submit_records_the_outcome_and_the_ledger(store, tmp_path, manifest, monkeypatch):
    csv_path = _predict_record(tmp_path)
    sub = {"status": "submitted", "ref": "777", "response": "ok", "kaggle": {"status": "COMPLETE", "public_score": 0.57}}
    seen = {}
    monkeypatch.setattr(kaggle_client, "submit", lambda path, message, slug, limit, log: seen.update(
        path=path, message=message, slug=slug, limit=limit) or dict(sub))
    ctx = FakeCtx()
    ctx.job_id = "s1"
    result = predictor.run_kaggle_submit({"predict_job_id": "p1", "message": ""}, ctx, manifest)
    assert result["submission"]["ref"] == "777" and seen["slug"] == manifest.competition.slug
    assert seen["limit"] == manifest.submission.daily_limit and seen["message"] == "r1 via 3LC plugin"
    assert seen["path"] == str(csv_path)
    st = predictor.predict_submit_state()
    assert st["state"] == "submitted" and st["submission"]["kaggle"]["public_score"] == 0.57
    assert predictor.read_predict_record()["facts"]["submission"]["ref"] == "777"
    entry = ledger.find("submit", job_id="s1")
    assert entry["csv_sha256"] == predictor.sha256_of(csv_path) and entry["kaggle"]["ref"] == "777"
    assert entry["kaggle"]["read_back_public_score"] == 0.57 and entry["slug"] == manifest.competition.slug
    assert entry["manifest_sha256"] == manifest.sha256
    # A rejection fails the job and leaves a failed submit record; a changed CSV is refused first.
    monkeypatch.setattr(kaggle_client, "submit", lambda *a: {"status": "failed", "reason": "Kaggle rejected the submission: 400"})
    with pytest.raises(predictor.PredictRefused, match="Kaggle rejected"):
        predictor.run_kaggle_submit({"predict_job_id": "p1"}, FakeCtx(), manifest)
    assert predictor.read_submit_record()["status"] == "failed"
    csv_path.write_text("image_id,prediction,confidence\nq,1,0.5\n", encoding="utf-8")
    with pytest.raises(predictor.PredictRefused, match="changed on disk"):
        predictor.run_kaggle_submit({"predict_job_id": "p1"}, FakeCtx(), manifest)
    with pytest.raises(predictor.PredictRefused, match="Run inference first"):
        predictor.run_kaggle_submit({"predict_job_id": "nope"}, FakeCtx(), manifest)


# ── Heavy: end to end on the synthetic kit ───────────────────────────────────

tlc = pytest.importorskip("tlc")
pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    root = tmp_path / "3lc-root"
    root.mkdir()
    monkeypatch.setattr(importer, "project_root_url", lambda: root.as_posix())
    return root


@pytest.fixture
def trained(tmp_path, home, project_root):
    """A real one-epoch CPU run on the synthetic kit: the import record + the train record."""
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(tmp_path / "tree", data, real_images=True)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    manifest = manifest_mod.parse_manifest(data, source="test", source_detail="synthetic")
    session.publish_kit_dir(manifest, kit_root)
    importer.run_import({}, FakeCtx(), manifest)
    ctx = FakeCtx()
    ctx.job_id = "train-e2e"
    result = trainer.run_training({"epochs": "1", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": "p_e2e"},
                                  ctx, manifest)
    assert result["cancelled"] is False and result["best_checkpoint_sha256"]
    return manifest, kit_root, result


def test_run_predict_end_to_end_on_cpu(trained):
    manifest, kit_root, train_result = trained
    ctx = FakeCtx()
    ctx.job_id = "predict-e2e"
    out = predictor.run_predict({"train_job_id": "train-e2e", "device": "cpu"}, ctx, manifest)
    assert out["cancelled"] is False and out["rows"] == 4 and out["device"] == "cpu"
    # The CSV: the sample ids in order, six decimals, the manifest's columns.
    csv_path = Path(out["csv_path"])
    assert csv_path.is_file() and csv_path.parent.name == "p_e2e" and csv_path.parent.parent.name == predictor.PREDICTIONS_DIR
    header, rows = predictor.read_csv_rows(csv_path)
    sample_ids = predictor.read_sample_ids(kit_root / manifest.splits.test.ids_from, manifest.submission.columns)
    assert header == list(manifest.submission.columns) and [r[0] for r in rows] == sample_ids
    assert all(r[1].isdigit() and 0 <= int(r[1]) < manifest.num_classes for r in rows)
    assert all(len(r[2].split(".")[1]) == 6 and 0 <= float(r[2]) <= 1 for r in rows)
    assert out["csv_sha256"] == predictor.sha256_of(csv_path)
    # The checks: the three-way sha, the test inputs, the val reproduction, the seven format checks.
    assert len(out["checks"]) == 10 and all(c["ok"] for c in out["checks"])
    assert out["local_score"]["kind"] == "val" and out["local_score"]["ok"] is True
    assert out["local_score"]["value"] == train_result["best_val_accuracy"] and out["local_score"]["rows"] == 6
    assert out["checkpoint_sha256"] == train_result["best_checkpoint_sha256"]
    assert set(out["sanity"]["per_class"]) == set(manifest.class_names) and out["sanity"]["images"] == 4
    # The record, the facts and the progress channel, the ledger entry.
    st = predictor.predict_submit_state()
    assert st["state"] == "predicted" and st["predict"]["job_id"] == "predict-e2e" and st["predict"]["csv_on_disk"]
    assert ctx.facts["csv_sha256"] == out["csv_sha256"] and ctx.facts["checkpoint_sha256"] == out["checkpoint_sha256"]
    assert any(p.get("phase") == "val" for p in ctx.progress) and any(p.get("phase") == "test" for p in ctx.progress)
    assert ctx.progress[-1]["percent"] == 100.0
    entry = ledger.find("predict", job_id="predict-e2e")
    assert entry["csv"]["sha256"] == out["csv_sha256"] and entry["checkpoint"]["sha256_recorded"] == out["checkpoint_sha256"]
    assert entry["run_url"] == train_result["run_url"] and entry["test_inputs"]["count"] == 4
    assert entry["contract"]["val_table_url"] and entry["local_score"]["value"] == out["local_score"]["value"]
    assert predictor.list_runs()[0]["usable"] is True
    # G2: a tampered best.pt is refused by the gate and shown as unusable in the run list.
    weights = Path(train_result["weights"])
    original = weights.read_bytes()
    weights.write_bytes(original + b"\x00")
    assert predictor.list_runs()[0]["reason"] == "best checkpoint changed on disk (sha256 mismatch)"
    with pytest.raises(predictor.PredictRefused) as exc:
        predictor.run_predict({"train_job_id": "train-e2e", "device": "cpu"}, FakeCtx(), manifest)
    assert exc.value.args[0] == predictor.SHA_MISMATCH
    weights.write_bytes(original)
    # G2: a run with a failed provenance check is blocked (D2).
    state = session.load()["train_state"]
    state["runs"][0]["provenance_ok"] = False
    session.save({"train_state": state})
    with pytest.raises(predictor.PredictRefused, match="failed a provenance check"):
        predictor.run_predict({"train_job_id": "train-e2e", "device": "cpu"}, FakeCtx(), manifest)
    state["runs"][0]["provenance_ok"] = True
    session.save({"train_state": state})
    # G2: an unreadable test image fails the job (D11) — the gate sees it first as a mismatch.
    target = kit_root / "data" / "test" / f"{sample_ids[0]}.jpg"
    good = target.read_bytes()
    target.write_bytes(b"\xff\xd8not an image" + b"\x00" * (len(good) - 15) + b"\xff\xd9")
    with pytest.raises(predictor.PredictRefused, match="could not be verified"):
        predictor.run_predict({"train_job_id": "train-e2e", "device": "cpu"}, FakeCtx(), manifest)
    target.write_bytes(good)
    assert predictor.read_predict_record()["status"] == "completed"   # the refusals happened before a record


def test_an_unreadable_test_image_fails_the_job_after_the_gate(trained, monkeypatch):
    """D11 proper: when the bytes on disk pass the gate but the decoder cannot read them, the job
    fails loudly with the id instead of predicting a placeholder."""
    manifest, kit_root, _ = trained

    def unreadable(self, i):
        raise predictor.PredictRefused(f"Test image {self.ids[i]} could not be read (OSError: truncated).")

    monkeypatch.setattr(predictor._TestImages, "__getitem__", unreadable)
    ctx = FakeCtx()
    ctx.job_id = "predict-d11"
    with pytest.raises(predictor.PredictRefused, match="could not be read"):
        predictor.run_predict({"train_job_id": "train-e2e", "device": "cpu"}, ctx, manifest)
    rec = predictor.read_predict_record()
    assert rec["id"] == "predict-d11" and rec["status"] == "failed" and "could not be read" in rec["error"]
    assert predictor.predict_submit_state()["state"] == "failed"
