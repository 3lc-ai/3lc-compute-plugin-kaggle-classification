# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Status tab's backend (docs/STATUS_MIRROR.md §2): the run history, the prediction / submission
history joined from the ledger with Kaggle's verdict per ref (stored, or read live by ref once and
cached), the live Kaggle section under the unlaunched answers, the Doctor payload, and the
verification bundle — its member list, that it carries no data or secret, and that a planted secret
refuses the whole export.

Light: every test runs on the session store + the ledger under an isolated home, with the kaggle
client's calls replaced."""

from __future__ import annotations

import io
import json
import os
import time
import zipfile

import pytest

from kaggle_classification import kaggle_client, ledger, session, status, trainer

PROJECT = "intel-scene"
ROOT = "C:/Users/p/AppData/Local/3LC/3LC/projects"


def _run_url(project: str, name: str) -> str:
    return f"{ROOT}/{project}/runs/{name}"


def _import_record(project: str = PROJECT) -> dict:
    return {
        "project_name": project, "table_name": "initial", "kit_dir": "C:/kit/starter_kit", "kit_version": "v1",
        "tables": {"train": {"url": f"{ROOT}/{project}/datasets/ds_train/tables/initial", "rows": 15},
                   "val": {"url": f"{ROOT}/{project}/datasets/ds_val/tables/initial", "rows": 6}},
        "lineage_root": {"train_url": f"{ROOT}/{project}/datasets/ds_train/tables/initial",
                         "val_url": f"{ROOT}/{project}/datasets/ds_val/tables/initial"},
        "val_locked": {"url": f"{ROOT}/{project}/datasets/ds_val/tables/initial", "editable": True},
        "label_map": {"0": "a", "1": "b", "2": "undefined"}, "checks": [], "timings": {},
        "manifest_provenance": {"manifest_sha256": "abc", "manifest_source": "bundled"}, "job_id": "imp1",
        "completed_at": time.time() - 3600,
    }


def _run(job_id: str, name: str, project: str = PROJECT, status_: str = "completed", **over) -> dict:
    base = {
        "id": job_id, "run_name": name, "run_url": _run_url(project, name), "project_name": project, "status": status_,
        "created_at": time.time() - 2000, "finished_at": time.time() - 1900, "epochs_requested": 2,
        "epochs_completed": 2,
        "best_epoch": 2, "best_val_accuracy": 51.5, "weights": "", "best_checkpoint_sha256": "ff" * 32,
        "device_class": "cuda", "device": "cuda", "device_label": "cuda (auto)", "usable_rows": 12,
        "provenance_ok": True, "train_table_url": f"{ROOT}/{project}/datasets/ds_train/tables/initial",
        "val_table_url": f"{ROOT}/{project}/datasets/ds_val/tables/initial", "params": {"epochs": 2},
        "elapsed_s": 100.0,
    }
    base.update(over)
    return base


def _predict_entry(job_id: str, train_id: str, name: str, project: str = PROJECT, ts: float | None = None) -> dict:
    return {
        "ts": ts or time.time() - 1000, "kind": "predict", "job_id": job_id, "manifest_sha256": "abc",
        "manifest_source": "bundled", "competition_id": "intel-scene", "kit_version": "v1", "plugin_version": "t",
        "train_job_id": train_id, "run_url": _run_url(project, name), "run_name": name, "run_folder": name,
        "checkpoint": {"path": "C:/x/best.pt", "sha256_recorded": "ff" * 32, "sha256_on_disk": "ff" * 32,
                       "sha256_on_run": "ff" * 32},
        "contract": {"seed": 42}, "test_inputs": {"count": 4}, "device": "cuda",
        "csv": {"path": f"C:/home/predictions/{name}/submission_x.csv", "sha256": "aa" * 32, "rows": 4},
        # The ledger stores a prediction's checks as [label, ok] pairs (predictor.py), not dicts — the
        # shape the tester proof on 1.0.0rc1 caught (an empty History after a real prediction).
        "checks": [["columns are image_id, prediction, confidence", True], ["no missing values", True]], "sanity": {},
        "local_score": {"kind": "val", "value": 51.5, "recorded": 51.5},
    }


def _submit_entry(job_id: str, predict_id: str, name: str, ref: str, verdict: dict | None, ts: float | None = None,
                  project: str = PROJECT, status_: str = "submitted") -> dict:
    kaggle = {"status": status_, "ref": ref if status_ == "submitted" else None, "response": "ok", "reason": None,
              "detail": None, "submitted_at": ts or time.time() - 900}
    if verdict:
        kaggle.update({f"read_back_{k}": v for k, v in verdict.items()})
    return {
        "ts": ts or time.time() - 900, "kind": "submit", "job_id": job_id, "predict_job_id": predict_id,
        "csv_sha256": "aa" * 32, "csv_path": f"C:/home/predictions/{name}/submission_x.csv", "run_name": name,
        "run_url": _run_url(project, name), "checkpoint_sha256": "ff" * 32, "manifest_sha256": "abc",
        "manifest_source": "bundled", "competition_id": "intel-scene", "kit_version": "v1", "plugin_version": "t",
        "slug": "slug", "message": f"{name} via 3LC plugin", "kaggle": kaggle,
    }


@pytest.fixture
def seeded(store):
    """An import record, three runs (one in another project), two predictions with submissions."""
    store.save({"import_state": _import_record()})
    runs = [
        _run("t3", "run_c", status_="cancelled", best_val_accuracy=48.0),
        _run("t2", "run_b", best_val_accuracy=55.0),
        _run("t1", "run_a"),
        _run("tx", "run_x", project="other"),
    ]
    store.save({"train_state": {"current": None, "runs": runs}})
    t0 = time.time() - 5000
    ledger.append(_predict_entry("p1", "t1", "run_a", ts=t0))
    ledger.append(_submit_entry("s1", "p1", "run_a", "100",
                                {"status": "COMPLETE", "public_score": 0.5, "error_description": ""}, ts=t0 + 10))
    ledger.append(_predict_entry("p2", "t2", "run_b", ts=t0 + 100))
    ledger.append(_submit_entry("s2", "p2", "run_b", "200", {"status": "unknown", "error": "HTTPError: 403"},
                                ts=t0 + 110))
    ledger.append(_predict_entry("px", "tx", "run_x", project="other", ts=t0 + 200))
    ledger.append(_predict_entry("p3", "t3", "run_c", ts=t0 + 300))
    ledger.append(_submit_entry("s3", "p3", "run_c", "", None, ts=t0 + 310, status_="limit_reached"))
    return store


@pytest.fixture(autouse=True)
def _no_probe(monkeypatch):
    monkeypatch.setattr(trainer, "probe_device_async", lambda: {"state": "done", "device_class": "cpu"})
    monkeypatch.setattr(trainer, "_device_probe", {"state": "done", "device_class": "cpu"})


# ── Runs ──────────────────────────────────────────────────────────────────────


def test_run_history_is_project_scoped_newest_first_and_includes_a_running_current(seeded):
    rows = status.run_history()
    assert [r["run_name"] for r in rows] == ["run_c", "run_b", "run_a"]
    assert rows[0]["status"] == "cancelled" and rows[1]["best_val_accuracy"] == 55.0
    assert all(r["project_name"] == PROJECT for r in rows)
    assert rows[1]["train_table_url"].endswith("ds_train/tables/initial") and rows[1]["usable_rows"] == 12
    assert rows[1]["weights_on_disk"] is False and rows[1]["device_label"] == "cuda (auto)"
    # A running current record (this process, fresh heartbeat) leads the list.
    cur = {"id": "t9", "kind": "train", "status": "running", "pid": os.getpid(), "created_at": time.time(),
           "started_at": time.time(), "heartbeat": time.time(), "params": {"run_name": "run_live", "epochs": 3},
           "progress": {"epoch": 1, "total_epochs": 3}, "facts": {"run_name": "run_live", "project_name": PROJECT,
           "run_url": _run_url(PROJECT, "run_live")}, "checks": [], "result": None, "log": []}
    st = seeded.load()["train_state"]
    st["current"] = cur
    seeded.save({"train_state": st})
    rows = status.run_history()
    assert rows[0]["job_id"] == "t9" and rows[0]["status"] == "running"


# ── History ───────────────────────────────────────────────────────────────────


def test_prediction_history_joins_the_ledger_scopes_the_project_and_computes_the_delta(seeded, manifest):
    rows = status.prediction_history(manifest, live=False)
    assert [r["job_id"] for r in rows] == ["p3", "p2", "p1"]   # newest first; run_x (other project) excluded
    a, b, c = rows[2], rows[1], rows[0]
    assert a["submission"]["ref"] == "100" and a["public_score"] == 0.5 and a["delta"] is None
    assert a["submission"]["kaggle"]["source"] == "ledger" and a["val_accuracy"] == 51.5 and a["checks_ok"]
    assert b["submission"]["ref"] == "200" and b["public_score"] is None   # unresolved at submit time, no live read
    assert c["submission"]["status"] == "limit_reached" and c["public_score"] is None and c["csv_path"]
    assert status.best_public_score(rows) == {"score": 0.5, "run_name": "run_a", "ref": "100"}


def test_live_history_reads_unresolved_refs_by_ref_once_and_caches(seeded, manifest, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (object(), ""))

    def fake_get(api, ref):
        calls.append(str(ref))
        return {"ref": str(ref), "status": "COMPLETE", "public_score": 0.6, "error_description": "", "date": "d"}

    monkeypatch.setattr(kaggle_client, "get_submission", fake_get)
    status._live_cache.clear()
    rows = status.prediction_history(manifest, live=True)
    b = next(r for r in rows if r["job_id"] == "p2")
    assert calls == ["200"]   # only the unresolved ref; the COMPLETE one is never re-asked
    assert b["public_score"] == 0.6 and b["submission"]["kaggle"]["source"] == "live"
    assert b["delta"] == pytest.approx(0.1)   # vs the previous Kaggle-scored submission (0.5)
    rows = status.prediction_history(manifest, live=True)
    assert calls == ["200"]   # cached
    assert status.best_public_score(rows)["score"] == 0.6


def test_a_failed_latest_prediction_without_a_ledger_entry_is_listed(seeded, manifest):
    seeded.save({"predict_state": {
        "id": "p9", "kind": "predict", "status": "failed", "pid": 1, "created_at": time.time(),
        "started_at": time.time(),
        "heartbeat": time.time(), "finished_at": time.time(), "cancelled": False, "params": {}, "progress": {},
        "facts": {"run_name": "run_b", "project_name": PROJECT, "run_url": _run_url(PROJECT, "run_b")}, "checks": [],
        "result": None, "error": "Submission must contain exactly 4 rows; got 3.", "log": [],
    }})
    rows = status.prediction_history(manifest, live=False)
    assert rows[0]["job_id"] == "p9" and rows[0]["status"] == "failed" and "rows" in rows[0]["error"]


# ── Kaggle live ───────────────────────────────────────────────────────────────


class _Entry:
    def __init__(self, team, score):
        self.team_name = team
        self.score = score


class _Api:
    def __init__(self, board=None, list_error=None):
        self.config_values = {"username": "participant"}
        self._board = board or []
        self._list_error = list_error

    def competition_leaderboard_view(self, slug):
        return list(self._board)


def test_kaggle_live_reports_unlaunched_when_the_list_403s_and_the_board_is_empty(seeded, manifest, monkeypatch):
    monkeypatch.setattr(kaggle_client, "credentials_present", lambda: True)
    monkeypatch.setattr(kaggle_client, "authenticated_api", lambda: (_Api(), ""))

    def forbidden(api, slug, page_size=20):
        msg = "HTTPError: 403 Client Error: Forbidden for url: .../ListSubmissions"
        raise RuntimeError(msg)

    monkeypatch.setattr(kaggle_client, "list_submissions", forbidden)
    out = status.kaggle_live(manifest)
    assert out["connected"] and out["launched"] is False
    assert "403" in out["submissions_error"] and out["leaderboard_top"] == [] and out["my_rank"] is None
    assert out["best_public_score"] == 0.5   # from the ledger's read-backs


def test_kaggle_live_lists_the_board_and_the_rank_once_the_api_answers(seeded, manifest, monkeypatch):
    monkeypatch.setattr(kaggle_client, "credentials_present", lambda: True)
    monkeypatch.setattr(kaggle_client, "authenticated_api",
                        lambda: (_Api(board=[_Entry("someone", 0.9), _Entry("participant", 0.7)]), ""))
    monkeypatch.setattr(kaggle_client, "list_submissions", lambda api, slug, page_size=20: [
        {"ref": "1", "date": "d", "status": "COMPLETE", "public_score": 0.7, "private_score": None,
         "error_description": "", "description": "m", "file_name": "f"}])
    out = status.kaggle_live(manifest)
    assert out["launched"] is True and out["leaderboard_top"][0]["team"] == "someone"
    assert out["my_rank"] == {"rank": 2, "team": "participant", "score": "0.7"}
    assert out["submissions"][0]["public_score"] == 0.7 and out["best_public_score"] == 0.7


def test_kaggle_live_without_credentials_is_a_friendly_state(seeded, manifest, monkeypatch):
    monkeypatch.setattr(kaggle_client, "credentials_present", lambda: False)
    out = status.kaggle_live(manifest)
    assert out["connected"] is False and "Connect your Kaggle account" in out["reason"]


# ── Doctor ────────────────────────────────────────────────────────────────────


def test_doctor_reports_the_facts_without_the_network(seeded, manifest):
    doc = status.doctor(manifest, kaggle=False)
    assert set(doc) >= {"plugin", "versions", "device", "manifest", "kit", "import", "records", "plugin_home", "time"}
    assert doc["plugin"]["version"] and doc["plugin"]["commit"]
    assert set(doc["versions"]) >= {"sdk", "tlc", "torch", "torchvision", "kaggle", "python", "platform"}
    assert doc["device"]["device_class"] == "cpu" and doc["device"]["cuda_available"] is False
    assert doc["manifest"]["manifest_source"] in ("bundled", "cache") and doc["manifest"]["manifest_sha256"]
    assert doc["kit"]["state"] == "empty" and doc["import"]["project_name"] == PROJECT
    assert doc["records"] == {"runs": 4, "project_runs": 3, "ledger_predict": 4, "ledger_submit": 3,
                              "ledger_path": str(ledger.ledger_path())}
    assert doc["plugin_home"]["resolved_by"] == "env" and isinstance(doc["plugin_home"]["disk"]["free_bytes"], int)
    assert "kaggle" not in doc
    json.dumps(doc)   # serializable as served


# ── The verification bundle ───────────────────────────────────────────────────

EXPECTED_MEMBERS = {
    "README.txt", "plugin.json", "manifest_provenance.json", "import_record.json", "train_revisions.json",
    "runs/t1.json", "runs/t2.json", "runs/t3.json",
    "predictions/p1.json", "predictions/p2.json", "predictions/p3.json",
    "submissions/s1.json", "submissions/s2.json", "submissions/s3.json", "ledger.jsonl",
}


def test_verification_bundle_carries_the_records_and_nothing_else(seeded, manifest):
    data, name = status.verification_bundle(manifest)
    assert name.startswith("verification-bundle_intel-scene_") and name.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert names == EXPECTED_MEMBERS
        texts = {n: zf.read(n).decode("utf-8") for n in names}
    # No data: no image, checkpoint or CSV member; the other project's prediction is not there.
    assert not any(n.lower().endswith((".jpg", ".jpeg", ".png", ".pt", ".csv", ".parquet")) for n in names)
    assert "run_x" not in texts["ledger.jsonl"].split("px")[0] or "predictions/px.json" not in names
    # No secret pattern anywhere (the same scan the export runs, plus the token-shaped needles).
    for n, text in texts.items():
        status.scan_for_secrets(n, text)
        assert "KGAT_" not in text and "3lc_api_key" not in text and "solution" not in text.lower()
    run = json.loads(texts["runs/t2.json"])
    assert run["summary"]["best_val_accuracy"] == 55.0 and run["best_checkpoint_sha256_recorded"] == "ff" * 32
    sub = json.loads(texts["submissions/s1.json"])
    assert sub["kaggle"]["read_back_public_score"] == 0.5 and sub["csv_sha256"] == "aa" * 32
    assert json.loads(texts["import_record.json"])["lineage_root"]["train_url"].endswith("tables/initial")
    assert "revisions" in texts["train_revisions.json"] or "error" in texts["train_revisions.json"] \
        or "datasets" in texts["train_revisions.json"]


@pytest.mark.parametrize("needle", ["KGAT_abcdefghijklmnop0123", "mapping.csv", "solution_kit_v1.csv", "3lc_api_key"])
def test_a_planted_secret_refuses_the_whole_export(seeded, manifest, needle):
    ledger.append({"kind": "predict", "job_id": "p-evil", "run_url": _run_url(PROJECT, "run_a"), "note": needle})
    with pytest.raises(status.BundleRefused, match="refused"):
        status.verification_bundle(manifest)


def test_bundle_without_an_import_record_still_exports(store, manifest):
    data, _ = status.verification_bundle(manifest)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert {"README.txt", "plugin.json", "manifest_provenance.json", "import_record.json", "train_revisions.json",
                "ledger.jsonl"} <= set(zf.namelist())
        assert "no import record" in zf.read("train_revisions.json").decode()


def test_the_session_store_never_holds_a_secret_pattern(seeded):
    """The records the bundle is built from are themselves clean (a guard on what the plugin writes)."""
    text = json.dumps(session.load())
    for label, pattern in status.SECRET_PATTERNS:
        assert not pattern.search(text), label
