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
from pathlib import Path

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
    "project/intel-scene/files.json",
}


def _assert_data_rule(names):
    """Session 6: tables and runs ride along (3LC records: object.3lc.json + parquet under project/), the
    only checkpoint member is a run's best.pt; never an image, a CSV, last.pt or a file outside project/."""
    for n in names:
        low = n.lower()
        assert not low.endswith((".jpg", ".jpeg", ".png", ".csv")), n
        if low.endswith(".parquet"):
            assert n.startswith("project/") and ("/tables/" in n or "/metrics_" in n), n
        if low.endswith(".pt"):
            assert n.startswith("project/") and n.endswith("/model/best.pt"), n


def test_verification_bundle_carries_the_records_and_nothing_else(seeded, manifest):
    """The records on a project whose folders are not on this disk: the member set is the session-5 one
    plus the (empty) files index; no data member can exist."""
    data, name = status.verification_bundle(manifest)
    assert name.startswith("verification-bundle_intel-scene_") and name.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert names == EXPECTED_MEMBERS
        texts = {n: zf.read(n).decode("utf-8") for n in names}
    _assert_data_rule(names)
    index = json.loads(texts["project/intel-scene/files.json"])
    assert index["files"] == [] and index["checkpoints"]["mode"] == "default"
    # The rule on the seeded ledger: run_a (the only scored ref) + run_b (the most recent ref); neither best.pt is on disk.
    assert index["checkpoints"]["default_runs"] == ["t1", "t2"]
    assert [s["reason"] for s in index["checkpoints"]["skipped"]] == ["best.pt is not on disk"] * 2
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


# ── The bundle on a project that IS on disk (session 6): tables, runs, the checkpoint rule ────────

import hashlib


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_project(tmp_path, monkeypatch):
    """A project folder in the 3LC layout under tmp_path: the seed train table (object + row cache), a
    labeled revision (EditedTable pointing at ``../initial``), a stray root table outside the lineage, the
    locked val table, and four runs with metrics tables, best.pt / last.pt and a stray image."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kaggle_classification import importer

    root = tmp_path / "3lc" / "projects"
    proj = root / PROJECT
    monkeypatch.setattr(importer, "project_root_url", lambda: root.as_posix())
    monkeypatch.setattr(importer, "latest_url", lambda url, **kw: url)

    def table(ds, name, parent=None, rows=3, secret=None):
        d = proj / "datasets" / ds / "tables" / name
        d.mkdir(parents=True)
        obj = {"type": "EditedTable" if parent else "TableFromPydict", "row_count": rows, "created": "2026-10-06"}
        if parent:
            obj["input_table_url"] = f"../{parent}"
        else:
            obj["row_cache_url"] = "./row_cache.parquet"
            paths = [f"C:/kit/starter_kit/data/train/a/{i}.jpg" for i in range(rows)]
            if secret:
                paths[0] = secret
            pq.write_table(pa.table({"image": paths, "label": [0] * rows, "weight": [1.0] * rows}), d / "row_cache.parquet")
        (d / "object.3lc.json").write_text(json.dumps(obj), encoding="utf-8")
        return d.as_posix()

    def run(name, best: bytes):
        d = proj / "runs" / name
        (d / "model").mkdir(parents=True)
        (d / "object.3lc.json").write_text(json.dumps({
            "type": "Run", "status": 1.0,
            "constants": {"parameters": {"best_checkpoint_sha256": _sha(best), "best_checkpoint": "model/best.pt"},
                          "outputs": [{"epoch": 1, "val_accuracy": 40.0}]},
        }), encoding="utf-8")
        (d / "model" / "best.pt").write_bytes(best)
        (d / "model" / "last.pt").write_bytes(b"last-" + best)
        m = d / "metrics_0000"
        m.mkdir()
        (m / "object.3lc.json").write_text(json.dumps({"type": "TableFromParquet", "input_url": "./metrics_0000.parquet"}), encoding="utf-8")
        pq.write_table(pa.table({"predicted": [0, 1, 2], "loss": [0.1, 0.2, 0.3]}), m / "metrics_0000.parquet")
        (d / "thumb.jpg").write_bytes(b"\xff\xd8\xff")
        return d.as_posix(), (d / "model" / "best.pt").as_posix(), _sha(best)

    seed = table("ds_train", "initial")
    rev = table("ds_train", "labels-1", parent="initial")
    stray = table("ds_train", "my-own-root", rows=2)
    val = table("ds_val", "initial")
    runs = {n: run(n, f"weights-of-{n}".encode()) for n in ("run_a", "run_b", "run_c", "run_d")}
    return {"root": root, "proj": proj, "seed": seed, "rev": rev, "stray": stray, "val": val, "runs": runs}


@pytest.fixture
def on_disk(store, tmp_path, monkeypatch):
    """The seeded records re-pointed at a project that exists on disk. Submissions: run_a (oldest, public
    0.5), run_b (public 0.7 — the best), run_c (the most recent, unscored, its ledger sha256 differs from
    the file); run_d trained but never submitted."""
    fp = _fake_project(tmp_path, monkeypatch)
    rec = _import_record()
    rec["tables"]["train"]["url"] = fp["seed"]
    rec["tables"]["val"]["url"] = fp["val"]
    rec["lineage_root"] = {"train_url": fp["seed"], "val_url": fp["val"]}
    rec["val_locked"] = {"url": fp["val"], "editable": True}
    store.save({"import_state": rec})
    runs = []
    for job, name, train in (("t4", "run_d", fp["seed"]), ("t3", "run_c", fp["rev"]), ("t2", "run_b", fp["rev"]), ("t1", "run_a", fp["seed"])):
        url, weights, sha = fp["runs"][name]
        runs.append(_run(job, name, run_url=url, weights=weights, best_checkpoint_sha256=sha, train_table_url=train))
    store.save({"train_state": {"current": None, "runs": runs}})
    t0 = time.time() - 5000

    def predict(job, train, name, ts, sha):
        e = _predict_entry(job, train, name, ts=ts)
        e["run_url"] = fp["runs"][name][0]
        e["checkpoint"] = {"path": fp["runs"][name][1], "sha256_recorded": sha, "sha256_on_disk": sha, "sha256_on_run": sha}
        return e

    def submit(job, pid, name, ref, verdict, ts):
        e = _submit_entry(job, pid, name, ref, verdict, ts=ts)
        e["run_url"] = fp["runs"][name][0]
        return e

    ledger.append(predict("p1", "t1", "run_a", t0, fp["runs"]["run_a"][2]))
    ledger.append(submit("s1", "p1", "run_a", "100", {"status": "COMPLETE", "public_score": 0.5, "error_description": ""}, t0 + 10))
    ledger.append(predict("p2", "t2", "run_b", t0 + 100, fp["runs"]["run_b"][2]))
    ledger.append(submit("s2", "p2", "run_b", "200", {"status": "COMPLETE", "public_score": 0.7, "error_description": ""}, t0 + 110))
    ledger.append(predict("p3", "t3", "run_c", t0 + 300, "00" * 32))   # the ledger's sha256 does not match the file
    ledger.append(submit("s3", "p3", "run_c", "300", None, t0 + 310))
    ledger.append(predict("p4", "t4", "run_d", t0 + 400, fp["runs"]["run_d"][2]))   # predicted, never submitted
    return fp


def _members(data: bytes) -> tuple[set, dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        index = json.loads(zf.read(f"project/{PROJECT}/files.json").decode("utf-8"))
        return names, index


def test_bundle_copies_the_lineage_tables_the_runs_and_the_rule_s_checkpoints(on_disk, manifest):
    data, _ = status.verification_bundle(manifest)
    names, index = _members(data)
    _assert_data_rule(names)
    P = f"project/{PROJECT}/"
    # The seed (object + row cache), the labeled revision (object only: an EditedTable has no cache), the
    # locked val; the stray root table outside the lineage is NOT copied.
    assert {P + "datasets/ds_train/tables/initial/object.3lc.json", P + "datasets/ds_train/tables/initial/row_cache.parquet",
            P + "datasets/ds_train/tables/labels-1/object.3lc.json", P + "datasets/ds_val/tables/initial/object.3lc.json",
            P + "datasets/ds_val/tables/initial/row_cache.parquet"} <= names
    assert not any("my-own-root" in n for n in names)
    # Every run: the object and its metrics table; never last.pt or the stray image.
    for r in ("run_a", "run_b", "run_c", "run_d"):
        assert {P + f"runs/{r}/object.3lc.json", P + f"runs/{r}/metrics_0000/object.3lc.json",
                P + f"runs/{r}/metrics_0000/metrics_0000.parquet"} <= names
        assert P + f"runs/{r}/model/last.pt" not in names and P + f"runs/{r}/thumb.jpg" not in names
    # The rule: best public score (run_b) + most recent submission (run_c); run_c fails the ledger check.
    assert index["checkpoints"]["mode"] == "default" and index["checkpoints"]["default_runs"] == ["t2", "t3"]
    assert [i["train_job_id"] for i in index["checkpoints"]["included"]] == ["t2"]
    assert P + "runs/run_b/model/best.pt" in names and P + "runs/run_c/model/best.pt" not in names
    assert P + "runs/run_a/model/best.pt" not in names and P + "runs/run_d/model/best.pt" not in names
    skipped = index["checkpoints"]["skipped"]
    assert len(skipped) == 1 and skipped[0]["train_job_id"] == "t3" and "the ledger" in skipped[0]["reason"]
    assert index["checkpoints"]["included"][0]["verified_against"] == ["the ledger", "the run record", "the Run"]
    # files.json: every copied file with its real sha256 and size.
    by_path = {f["path"]: f for f in index["files"]}
    assert set(by_path) == {n for n in names if n.startswith(P) and not n.endswith("files.json")}
    ckpt = by_path[P + "runs/run_b/model/best.pt"]
    assert ckpt["sha256"] == on_disk["runs"]["run_b"][2] and ckpt["bytes"] == len(b"weights-of-run_b") and ckpt["kind"] == "checkpoint"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert _sha(zf.read(P + "datasets/ds_train/tables/initial/row_cache.parquet")) == by_path[P + "datasets/ds_train/tables/initial/row_cache.parquet"]["sha256"]
    assert index["project_dir"].replace("\\", "/").endswith(f"projects/{PROJECT}")
    assert {t["role"] for t in index["tables"]} == {"seed", "train revision", "locked val"}


def test_bundle_preview_is_the_plan_without_the_bytes(on_disk, manifest):
    pv = status.bundle_preview(manifest)
    assert pv["checkpoints"]["default_runs"] == ["t2", "t3"] and pv["counts"]["checkpoints"] == 1
    runs = {e["train_job_id"]: e for e in pv["eligible_runs"]}
    assert set(runs) == {"t1", "t2", "t3"}   # submitted runs only (run_d was never submitted)
    assert runs["t2"]["public_score"] == 0.7 and runs["t2"]["default"] and runs["t2"]["selected"]
    assert runs["t3"]["default"] and "the ledger" in runs["t3"]["skipped_reason"]
    assert runs["t1"]["public_score"] == 0.5 and not runs["t1"]["default"] and not runs["t1"]["selected"]
    assert runs["t1"]["available"] and runs["t1"]["checkpoint_bytes"] == len(b"weights-of-run_a")
    assert pv["bytes"] > pv["bytes_checkpoints"] == len(b"weights-of-run_b")
    assert pv["counts"] == {"members": len(pv["members"]), "tables": 3, "runs": 4, "metrics_tables": 4, "checkpoints": 1}
    assert f"project/{PROJECT}/files.json" in pv["members"] and "README.txt" in pv["members"]
    assert "text_members" not in pv and "files" not in pv


def test_bundle_checkpoint_modes_selected_none_and_all(on_disk, manifest):
    P = f"project/{PROJECT}/"
    # selected: the participant's checklist, at most two submitted runs.
    names, index = _members(status.verification_bundle(manifest, checkpoints="selected", runs="t1")[0])
    assert {n for n in names if n.endswith(".pt")} == {P + "runs/run_a/model/best.pt"}
    assert index["checkpoints"]["requested"] == ["t1"] and index["checkpoints"]["mode"] == "selected"
    names, _ = _members(status.verification_bundle(manifest, checkpoints="selected", runs=["t1", "t2"])[0])
    assert {n for n in names if n.endswith(".pt")} == {P + "runs/run_a/model/best.pt", P + "runs/run_b/model/best.pt"}
    with pytest.raises(ValueError, match="At most 2"):
        status.bundle_plan(manifest, checkpoints="selected", runs="t1,t2,t3")
    with pytest.raises(ValueError, match="Not a submitted run"):
        status.bundle_plan(manifest, checkpoints="selected", runs="t4")
    with pytest.raises(ValueError, match="checkpoints must be one of"):
        status.bundle_plan(manifest, checkpoints="some")
    # none: the records, tables and runs without any checkpoint.
    names, index = _members(status.verification_bundle(manifest, checkpoints="none")[0])
    assert not any(n.endswith(".pt") for n in names) and index["checkpoints"]["included"] == []
    # all (organizers): every run's best.pt that matches its record and its Run; run_c's ledger mismatch
    # still excludes it because the ledger is checked whenever the run was submitted.
    names, index = _members(status.verification_bundle(manifest, checkpoints="all")[0])
    assert {n for n in names if n.endswith(".pt")} == {P + f"runs/{r}/model/best.pt" for r in ("run_a", "run_b", "run_d")}
    assert [s["train_job_id"] for s in index["checkpoints"]["skipped"]] == ["t3"]


def test_default_checkpoint_rule_picks_the_best_and_the_most_recent_once(on_disk, manifest):
    eligible = status.eligible_checkpoint_runs(manifest)
    assert [e["train_job_id"] for e in eligible] == ["t3", "t2", "t1"]   # newest submission first
    assert status.default_checkpoint_runs(eligible) == ["t2", "t3"]
    # One id when the best-scored run is also the most recently submitted one.
    one = [e for e in eligible if e["train_job_id"] != "t3"]
    assert status.default_checkpoint_runs(one) == ["t2"]
    # No scored submission: the most recent one alone; nothing submitted: no checkpoint at all.
    assert status.default_checkpoint_runs([{**e, "public_score": None} for e in eligible]) == ["t3"]
    assert status.default_checkpoint_runs([]) == []


def test_a_secret_inside_a_table_s_row_cache_refuses_the_export(store, tmp_path, monkeypatch, manifest):
    """The parquet members are scanned through their string columns (session 6): a token-shaped value in
    the seed's image column aborts the whole export, as a text member would."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    fp = _fake_project(tmp_path, monkeypatch)
    cache = Path(fp["seed"]) / "row_cache.parquet"
    pq.write_table(pa.table({"image": ["KGAT_abcdefghijklmnop0123", "C:/kit/b.jpg"], "label": [0, 1], "weight": [1.0, 1.0]}), cache)
    rec = _import_record()
    rec["lineage_root"] = {"train_url": fp["seed"], "val_url": fp["val"]}
    rec["val_locked"] = {"url": fp["val"], "editable": True}
    store.save({"import_state": rec})
    assert "KGAT_" in status.parquet_text(cache)
    with pytest.raises(status.BundleRefused, match="row_cache.parquet"):
        status.verification_bundle(manifest)


def test_bundle_file_lands_under_the_plugin_home_and_old_ones_are_pruned(on_disk, manifest, monkeypatch):
    from kaggle_classification import storage

    bundles = storage.plugin_home() / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    stale = bundles / "verification-bundle_intel-scene_20200101_000000Z.zip"
    stale.write_bytes(b"old")
    os.utime(stale, (time.time() - 2 * status.BUNDLE_KEEP_S, time.time() - 2 * status.BUNDLE_KEEP_S))
    path, name = status.verification_bundle_file(manifest, checkpoints="none")
    assert path.parent == bundles and path.name == name and path.is_file()
    assert not stale.exists()


def test_the_session_store_never_holds_a_secret_pattern(seeded):
    """The records the bundle is built from are themselves clean (a guard on what the plugin writes)."""
    text = json.dumps(session.load())
    for label, pattern in status.SECRET_PATTERNS:
        assert not pattern.search(text), label


def test_doctor_reports_the_refresh_result_not_the_local_label(seeded, manifest, monkeypatch):
    """A fresh start resolves cache / bundled without the network; the Doctor's manifest row says what the
    background refresh found (rc3, item 3), kicks it, and names a failed fetch."""
    from kaggle_classification import manifest as manifest_mod

    calls = []

    def done(**kw):
        calls.append(1)
        return {"state": "done", "source": "remote", "error": None, "started_at": 1, "finished_at": 2}

    monkeypatch.setattr(manifest_mod, "refresh_in_background", done)
    doc = status.doctor(manifest, kaggle=False)
    assert calls == [1]
    assert doc["manifest"]["manifest_source"] == "remote"
    assert doc["manifest"]["manifest_source_local"] in ("bundled", "cache")
    assert doc["manifest"]["refresh_state"] == "done" and doc["manifest"]["refresh_error"] is None
    monkeypatch.setattr(manifest_mod, "refresh_in_background",
                        lambda **kw: {"state": "failed", "source": None, "error": "index unreachable: HTTP 403"})
    doc = status.doctor(manifest, kaggle=False)
    assert doc["manifest"]["manifest_source"] in ("bundled", "cache") and "403" in doc["manifest"]["refresh_error"]
    assert doc["plugin_home"]["migrated_from"] is None
