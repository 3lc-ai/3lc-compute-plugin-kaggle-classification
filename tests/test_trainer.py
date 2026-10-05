# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Train stage: params and bounds (light), the durable record and orphan rule (light), and —
with the heavy extra — the preflight's lineage and usable-row summary, the weight semantics the
brief pins (gate G4), an end-to-end CPU run on the synthetic kit (checkpoints, provenance, the
per-sample metrics contract, tables never modified), the duplicate-start guard and cancel.

The heavy tests need tlc + torch + torchvision; they skip without them, so a green run in a light venv
is not a green run for this module (CLAUDE.md §B)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from conftest import table_url
from helpers import FakeCtx, make_kit_tree, small_manifest_data

import build_kit  # tools/ is on sys.path (conftest)
from kaggle_classification import importer, session, trainer
from kaggle_classification import manifest as manifest_mod

# ── Light: facts, params, bounds ──────────────────────────────────────────────


def test_training_facts_serve_defaults_bounds_and_the_locked_facts(manifest):
    facts = trainer.training_facts(manifest)
    assert facts["defaults"]["epochs"] == 10 and facts["defaults"]["optimizer"] == "adam"
    assert facts["defaults"]["seed"] == 42 and facts["defaults"]["workers"] == trainer.default_workers()
    assert facts["bounds"]["epochs"] == [1, 50] and facts["bounds"]["seed"] == [0, 2147483647]
    assert facts["bounds"]["workers"] == [0, 16]
    assert facts["optimizer"] == "adam" and facts["schedule"]["kind"] == "step"
    assert facts["schedule"]["step_size"] == 5 and facts["schedule"]["gamma"] == 0.1
    # Part B: what the form opens and what a manifest may allow; this event locks the choice fields.
    assert facts["editable"] == ["epochs", "batch_size", "lr", "weight_decay", "seed"]
    assert facts["options"] == {"optimizer": ["adam"], "schedule": ["steplr"]}
    assert set(facts["option_labels"]["optimizer"]) == {"adam", "adamw", "sgd"}
    assert set(facts["option_labels"]["schedule"]) == {"steplr", "cosine", "none"}
    assert facts["max_rows"]["train"] == manifest.expected_rows("train")
    assert set(facts["benchmark"]["per_device"]) == {"cuda", "mps", "cpu"}


def test_seed_bound_falls_back_when_the_manifest_has_none(manifest):
    import dataclasses

    bounds = {k: v for k, v in manifest.training.bounds.items() if k != "seed"}
    m = dataclasses.replace(manifest, training=dataclasses.replace(manifest.training, bounds=bounds))
    assert trainer.effective_bounds(m)["seed"] == (0.0, 2147483647.0)


def test_build_train_kwargs_merges_defaults_and_locks_the_contract(home, manifest):
    kw = trainer.build_train_kwargs({"epochs": "3", "lr": "", "batch_size": "16.0"}, manifest)
    assert kw["epochs"] == 3 and kw["batch_size"] == 16 and kw["lr"] == 0.0001 and kw["weight_decay"] == 0.0
    assert kw["seed"] == 42 and kw["workers"] == trainer.default_workers()
    assert kw["optimizer"] == "adam" and kw["schedule"] == "steplr" and kw["schedule_params"]["step_size"] == 5
    assert kw["backbone"] == manifest.model.backbone and kw["head"] == manifest.model.head
    assert kw["arch"] == manifest.model.arch and kw["image_size"] == manifest.model.image_size
    assert kw["pretrained"] is False and kw["use_latest"] is True
    assert kw["run_name"].startswith(f"{manifest.competition.id}_run_")
    assert kw["project_name"] == manifest.default_project


@pytest.mark.parametrize(
    ("params", "needle"),
    [
        ({"epochs": "999"}, "epochs must be between 1 and 50 (got 999)."),
        ({"batch_size": "4"}, "batch_size must be between 8 and 128"),
        ({"lr": "1"}, "lr must be between"),
        ({"seed": "-1"}, "seed must be between 0 and 2147483647"),
        ({"workers": "99"}, "workers must be between 0 and 16"),
        ({"epochs": "2.5"}, "whole number"),
        ({"epochs": "abc"}, "Invalid value for epochs"),
        ({"optimizer": "sgd"}, "optimizer is locked for this competition"),
        ({"run_name": "a/b"}, "run name must be a plain name"),
        ({"use_latest": "false", "epochs": "1"}, None),
    ],
)
def test_build_train_kwargs_refuses_out_of_bounds_and_locked_overrides(home, manifest, params, needle):
    if needle is None:
        assert trainer.build_train_kwargs(params, manifest)["use_latest"] is False
        return
    with pytest.raises(trainer.TrainRefused, match=__import__("re").escape(needle)):
        trainer.build_train_kwargs(params, manifest)


def test_validate_train_url_enforces_split_identity(tmp_path, manifest, root_shape):
    ok = table_url(tmp_path, root_shape, "intel-scene", manifest.dataset_name("train"), "initial")
    trainer.validate_train_url(ok, manifest)
    val = table_url(tmp_path, root_shape, "intel-scene", manifest.dataset_name("val"), "initial")
    with pytest.raises(trainer.TrainRefused, match="points at intel-scene_val; expected intel-scene_train"):
        trainer.validate_train_url(val, manifest)
    with pytest.raises(trainer.TrainRefused, match="Missing the train table URL"):
        trainer.validate_train_url("", manifest)


def test_effective_weights_and_the_usable_row_summary(manifest):
    n = manifest.num_classes
    undefined = manifest.undefined_label_id
    labels = [0, 1, 2, 3, 4, 5, undefined, undefined, 0]
    weights = [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.5, 1.0]
    eff = trainer.effective_weights(labels, weights, manifest)
    assert eff == [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    s = trainer.summarize_rows(labels, weights, manifest)
    assert s["total"] == 9 and s["labeled_in_use"] == 6 and s["excluded_undefined"] == 2
    assert s["excluded_zero_weight"] == 1 and s["undefined_with_weight"] == 1
    assert s["per_class"][manifest.class_names[0]] == 2 and s["classes_without_rows"] == [manifest.class_names[2]]
    assert n == 6


# ── Light: the durable record and the orphan rule ──────────────────────────


def _record(**over):
    base = {
        "id": "job1", "kind": "train", "status": "running", "pid": os.getpid(), "created_at": time.time() - 30,
        "started_at": time.time() - 30, "heartbeat": time.time(), "finished_at": None, "cancelled": False,
        "params": {"epochs": 3, "backbone": "x", "head": "y", "arch": "x", "image_size": 1, "pretrained": False, "seed": 42},
        "client_token": "tok1", "progress": {"epoch": 1, "total_epochs": 3, "history": [], "batch_i": 0, "batch_n": 2},
        "facts": {"run_name": "r1", "project_name": "p", "usable": {"labeled_in_use": 12}}, "checks": [],
        "result": None, "error": None, "log": [], "gaps": [],
    }
    base.update(over)
    return base


def test_train_state_is_empty_without_a_record(store):
    st = trainer.train_state()
    assert st["state"] == "empty" and st["current"] is None and st["runs"] == []


def test_a_running_record_from_another_worker_process_reads_back_as_stale(store):
    store.save({"train_state": {"current": _record(pid=999_999_999)}})
    st = trainer.train_state()
    assert st["state"] == "stale" and st["current"]["status"] == "stale"
    assert "restarted" in st["current"]["error"]
    # Persisted, and moved into the run history so the ETA never counts it as running.
    assert store.load()["train_state"]["current"]["status"] == "stale"
    assert store.load()["train_state"]["runs"][0]["status"] == "stale"


def test_a_running_record_of_this_process_with_a_fresh_heartbeat_stays_running(store):
    store.save({"train_state": {"current": _record()}})
    assert trainer.train_state()["state"] == "running"


def test_a_running_record_of_this_process_without_a_heartbeat_reads_back_as_stale(store):
    store.save({"train_state": {"current": _record(heartbeat=time.time() - trainer.STALE_HEARTBEAT_S - 1)}})
    st = trainer.train_state()
    assert st["state"] == "stale" and "stopped reporting" in st["current"]["error"]


def test_run_summary_carries_what_predict_and_the_eta_need(store):
    rec = _record(status="completed", finished_at=time.time(), result={
        "best_epoch": 2, "best_val_accuracy": 50.0, "epoch_s_per_row": 0.01, "collect_s": 3.0, "collect_rows": 20,
    })
    rec["facts"].update({"weights": "C:/x/best.pt", "best_checkpoint_sha256": "ab", "device_class": "cpu"})
    rec["checks"] = [{"label": "a", "ok": True}]
    s = trainer.run_summary(rec)
    assert s["usable_rows"] == 12 and s["best_epoch"] == 2 and s["epoch_s_per_row"] == 0.01
    assert s["provenance_ok"] is True and s["weights"] == "C:/x/best.pt" and s["device_class"] == "cpu"


def test_older_run_summaries_are_backfilled_from_the_run_once(store, tmp_path):
    """Item 6 of the re-check: a summary from before part E (no params, no elapsed) gets its settings
    from the Run's recorded parameters and its elapsed time from the record's timestamps, written
    back once; a Run that cannot be read leaves params_missing (Use these settings disabled)."""
    import json

    run_dir = tmp_path / "runs" / "review-1"
    run_dir.mkdir(parents=True)
    (run_dir / "object.3lc.json").write_text(json.dumps({"constants": {"parameters": {
        "epochs": 10, "batch_size": 16, "lr": 0.0001, "weight_decay": 0.0, "seed": 42, "optimizer": "adam",
        "schedule": "steplr(step_size=5, gamma=0.1)", "device": "cuda", "device_requested": "", "device_fallback_reason": "",
    }}}), encoding="utf-8")
    old = {"id": "old1", "run_name": "review-1", "status": "completed", "created_at": 1000.0, "finished_at": 1070.4,
           "run_url": run_dir.as_posix(), "device": "cuda", "device_class": "cuda", "epoch_s_per_row": 0.006}
    gone = {"id": "old2", "run_name": "g1_cpu", "status": "completed", "created_at": 2000.0, "finished_at": 2100.0,
            "run_url": (tmp_path / "runs" / "missing").as_posix(), "device": "cpu"}
    store.save({"train_state": {"runs": [old, gone]}})
    st = trainer.train_state()
    a, b = st["runs"]
    assert a["params"] == {"epochs": 10, "batch_size": 16, "lr": 0.0001, "weight_decay": 0.0, "seed": 42,
                           "optimizer": "adam", "schedule": "steplr"}
    assert a["params_source"] == "run" and a["elapsed_s"] == 70.4 and a["device_label"] == "cuda (auto)"
    assert "params" not in b and "could not be read" in b["params_missing"] and b["elapsed_s"] == 100.0
    # Written back once: the store now carries the filled summaries.
    saved = store.load()["train_state"]["runs"]
    assert saved[0]["params"]["epochs"] == 10 and saved[1]["params_missing"]
    assert trainer._run_parameters_on_disk("s3://bucket/run") is None and trainer._run_parameters_on_disk("") is None


def test_device_label_names_the_reason_in_each_case():
    """Item 1 of the 2026-09-29 re-check: the log line and the in-run header share one sentence
    that says WHY the run is on its device."""
    assert trainer.device_label("cuda", "") == "cuda (auto)"
    assert trainer.device_label("cuda", None) == "cuda (auto)"
    assert trainer.device_label("cpu", "cpu") == "cpu (forced in Advanced)"
    assert trainer.device_label("cuda:1", "1") == "cuda:1 (forced in Advanced)"
    assert trainer.device_label("cpu", "", "OutOfMemoryError: CUDA out of memory") == "cpu (fallback: OutOfMemoryError: CUDA out of memory)"
    assert trainer.device_label("cpu", "cuda", "RuntimeError: x") == "cpu (fallback: RuntimeError: x)"
    # A record from before the field derives the same label in run_summary.
    rec = _record(status="completed", finished_at=time.time())
    rec["facts"].update({"device": "cpu", "device_requested": "cpu"})
    assert trainer.run_summary(rec)["device_label"] == "cpu (forced in Advanced)"
    rec["facts"].update({"device_requested": "", "device_fallback_reason": "E: boom"})
    assert trainer.run_summary(rec)["device_label"] == "cpu (fallback: E: boom)"


# ── Heavy: the real tlc, torch and torchvision on the synthetic kit ────────────

tlc = pytest.importorskip("tlc")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")


def test_auto_never_resolves_to_cpu_while_cuda_is_available(monkeypatch):
    """Item 1: with torch.cuda.is_available() True in this process, a blank Device field is cuda —
    no environment switch or earlier forced-CPU run can turn auto into CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer.resolve_device("") == "cuda"
    assert trainer.resolve_device(None) == "cuda"
    assert trainer.resolve_device("cpu") == "cpu"          # forced stays forced
    assert trainer.resolve_device("0") == "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert trainer.resolve_device("") == "cpu"


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    root = tmp_path / "3lc-root"
    root.mkdir()
    monkeypatch.setattr(importer, "project_root_url", lambda: root.as_posix())
    return root


@pytest.fixture
def imported(tmp_path, home, project_root):
    """A tiny REAL-image kit imported into the isolated project root: the seed tables + record."""
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(tmp_path / "tree", data, real_images=True)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    manifest = manifest_mod.parse_manifest(data, source="test", source_detail="synthetic")
    session.publish_kit_dir(manifest, kit_root)
    result = importer.run_import({}, FakeCtx(), manifest)
    return manifest, result["tables"]["train"]["url"], result["tables"]["val"]["url"]


def _edit(train_url: str, column: str, edits: dict[int, float], name: str) -> str:
    """An EditedTable revision beside the train table (the Dashboard's shape)."""
    from tlc._core.objects.tables.from_table.edited_table import EditedTable

    parent = tlc.Table.from_url(tlc.Url(train_url))
    runs_and_values: list = []
    for idx, value in edits.items():
        runs_and_values += [[idx], value]
    edited = EditedTable(
        input_table_url=parent, edits={column: {"runs_and_values": runs_and_values}},
        url=parent.url.create_sibling(name),
    )
    edited.ensure_fully_defined()
    edited.write_to_url()
    return str(edited.url)


def _snapshot(url: str) -> dict[str, str]:
    """Every file under a table folder by content hash. Content, not mtime: reading a table through
    a ``TableView`` re-stamps ``object.3lc.json`` with byte-identical bytes (tlc 3.3 behaviour)."""
    import hashlib

    root = Path(url)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }


def test_preflight_reports_existence_lineage_and_the_usable_row_summary(imported):
    manifest, train_url, val_url = imported
    pf = trainer.preflight({"train_url": train_url}, manifest)
    assert pf["exists"] and pf["split_ok"] and pf["import_state"] == "success"
    assert pf["seed_url"] == train_url and pf["val_locked_url"] == val_url
    assert pf["base"]["descends_from_seed"] is True and pf["base"]["steps_from_seed"] == 0
    s = pf["base"]["summary"]
    assert s["labeled_in_use"] == 12 and s["excluded_undefined"] == 3 and s["excluded_zero_weight"] == 0
    assert s["undefined_with_weight"] == 0 and s["classes_without_rows"] == []
    assert pf["label_map_ok"] is True and pf["has_revisions"] is False and pf["max_rows"] == 15
    # A revision that gives an undefined row weight 1 and zero-weights a labeled one: latest differs.
    undefined_row = 12  # rows are labeled first, the pool last
    rev = _edit(train_url, "weight", {undefined_row: 1.0, 0: 0.0}, "initial_w")
    pf2 = trainer.preflight({"train_url": train_url}, manifest)
    assert pf2["has_revisions"] and pf2["latest_url"] == rev and pf2["latest"]["descends_from_seed"] is True
    assert pf2["latest"]["steps_from_seed"] == 1
    s2 = pf2["latest"]["summary"]
    assert s2["labeled_in_use"] == 11 and s2["undefined_with_weight"] == 1 and s2["excluded_zero_weight"] == 1
    assert s2["classes_without_rows"] == []
    # The base revision's summary is unchanged (the picker can still choose it).
    assert pf2["base"]["summary"]["labeled_in_use"] == 12
    # A val URL in the train slot is refused (split identity), and a missing table says so.
    bad = trainer.preflight({"train_url": val_url}, manifest)
    assert bad["split_ok"] is False and "expected" in bad["error"]
    gone = trainer.preflight({"train_url": train_url + "-nope"}, manifest)
    assert gone["exists"] is False and gone["split_ok"] is True


def test_a_table_outside_the_seed_lineage_is_reported_and_refused(imported):
    manifest, _train_url, _ = imported
    # A fresh table in the same dataset that was NOT derived from the seed.
    stranger = tlc.Table.from_dict(
        {"image": [], "label": [], "weight": []}, schema=importer._schema(manifest),
        table_url=tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "stranger")),
        if_exists="raise", add_weight_column=False,
    )
    pf = trainer.preflight({"train_url": str(stranger.url)}, manifest)
    assert pf["base"]["descends_from_seed"] is False
    params = {"train_table_url": str(stranger.url), "use_latest": False, "device": "cpu"}
    with pytest.raises(trainer.TrainRefused, match="does not descend from the imported seed"):
        trainer.run_training(params, FakeCtx(), manifest)


def test_weight_semantics_gate_g4(imported):
    """weight 0 excluded; undefined at weight > 0 skipped (and warned); weight 2 drawn about twice as
    often; the epoch length is the usable-row count; the table's weights are never written."""
    manifest, train_url, _ = imported
    rev = _edit(train_url, "weight", {0: 2.0, 1: 0.0, 12: 1.0}, "initial_g4")
    table = tlc.Table.from_url(tlc.Url(rev))
    before = _snapshot(train_url)
    labels, weights = trainer.scan_rows(table, manifest)
    s = trainer.summarize_rows(labels, weights, manifest)
    assert s["labeled_in_use"] == 11 and s["excluded_zero_weight"] == 1 and s["undefined_with_weight"] == 1
    eff = trainer.effective_weights(labels, weights, manifest)
    assert eff[1] == 0.0 and eff[12] == 0.0 and eff[0] == 2.0
    trainer.set_seed(42)
    sampler = trainer.build_sampler(eff)
    assert len(sampler) == 11
    draws = 40_000
    counts = [0] * len(eff)
    torch.manual_seed(42)
    idx = torch.multinomial(torch.as_tensor(eff, dtype=torch.double), draws, replacement=True)
    for i in idx.tolist():
        counts[i] += 1
    assert counts[1] == 0 and counts[12] == 0, "weight 0 and undefined-at-weight-1 rows must never be drawn"
    others = [counts[i] for i in range(len(eff)) if eff[i] == 1.0]
    mean_one = sum(others) / len(others)
    ratio = counts[0] / mean_one
    assert 1.8 < ratio < 2.2, f"weight 2 should be drawn about twice as often ({ratio:.2f}x)"
    # The sampler the trainer uses draws the same way (same generator, same semantics).
    drawn = list(iter(sampler))
    assert len(drawn) == 11 and 1 not in drawn and 12 not in drawn
    # Reading weights, summarising and sampling never touched the table on disk.
    assert _snapshot(train_url) == before
    assert list(tlc.Table.from_url(tlc.Url(rev)).table_rows)[1]["weight"] == 0.0


def test_zero_usable_rows_is_a_refusal_before_any_run_exists(imported, project_root):
    manifest, train_url, _ = imported
    rev = _edit(train_url, "weight", dict.fromkeys(range(12), 0.0), "initial_zero")
    runs_dir = project_root / "intel-scene" / "runs"
    runs_before = sorted(runs_dir.glob("*")) if runs_dir.is_dir() else []
    with pytest.raises(trainer.TrainRefused, match="No usable rows"):
        trainer.run_training({"train_table_url": rev, "use_latest": False, "device": "cpu"}, FakeCtx(), manifest)
    runs_after = sorted(runs_dir.glob("*")) if runs_dir.is_dir() else []
    assert runs_after == runs_before
    assert trainer.train_state()["state"] == "empty"


def test_run_training_end_to_end_on_cpu(imported, project_root):
    """Two epochs on the synthetic kit: checkpoints under the run, eight provenance checks, the
    per-sample metrics contract on train (undefined rows: no loss) and val, the durable record."""
    import math

    manifest, train_url, val_url = imported
    before_train, before_val = _snapshot(train_url), _snapshot(val_url)
    ctx = FakeCtx()
    params = {"epochs": "2", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": "t_e2e",
              "use_latest": True, "client_token": "click-1"}
    result = trainer.run_training(params, ctx, manifest)
    assert result["cancelled"] is False and result["epochs_completed"] == 2 and result["best_epoch"] in (1, 2)
    assert result["device"] == "cpu" and result["usable_rows"] == 12 and result["reducer"] in ("umap", "pca")
    run_url = result["run_url"]
    run_dir = trainer.run_local_dir(run_url)
    assert run_dir is not None and (run_dir / "model" / "best.pt").is_file()
    assert (run_dir / "model" / "last.pt").is_file()
    assert result["weights"] == str(run_dir / "model" / "best.pt")
    assert trainer._sha256_file(run_dir / "model" / "best.pt") == result["best_checkpoint_sha256"]
    assert not list((run_dir / "model").glob("*.tmp"))
    # Provenance: eight checks, all green, read back from the Run's own record.
    checks = ctx.checks
    assert len(checks) == 9 and all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    run = tlc.Run.from_url(tlc.Url(run_url))
    p = trainer.get_run_parameters(run)
    assert p["backbone"] == manifest.model.backbone and p["head"] == manifest.model.head and p["arch"] == manifest.model.arch
    assert p["pretrained"] is False and p["seed"] == 42 and p["torchvision_version"]
    assert p["train_table_url"] == train_url and p["val_table_url"] == val_url
    assert p["best_checkpoint_sha256"] == result["best_checkpoint_sha256"] and p["optimizer"] == "adam"
    assert p["manifest_sha256"] == manifest.sha256 or p.get("manifest_source") == "test"
    # Per-sample metrics: one table per split with the contract's columns.
    infos = list(run.metrics_tables) if hasattr(run, "metrics_tables") else []
    assert len(infos) == 2
    tables = {}
    for info in infos:
        url = getattr(info, "url", None) or (info.get("url") if isinstance(info, dict) else None)
        t = tlc.Table.from_url(tlc.Url(str(url)).to_absolute(run.url))
        t.ensure_fully_defined()
        rows = list(t.table_rows)
        tables[len(rows)] = rows
    train_rows, val_rows = tables[15], tables[6]
    names = manifest.class_names
    for rows in (train_rows, val_rows):
        for r in rows:
            assert 0 <= int(r["predicted"]) < manifest.num_classes
            assert 0.0 <= float(r["confidence"]) <= 1.0
            assert len(r["embeddings"]) == 3
            assert not any(k.startswith("prob_") for k in r)
            assert "accuracy" not in r   # item 8 of the re-check: label, weight, predicted, confidence, loss, embeddings
            assert r["epoch"] == result["best_epoch"]
    # Loss: present for labeled rows, NaN (absent) for the three undefined rows — never fabricated.
    labeled = [r for r in train_rows if int(r["example_id"]) < 12]
    pool = [r for r in train_rows if int(r["example_id"]) >= 12]
    assert len(pool) == 3 and all(math.isnan(float(r["loss"])) for r in pool)
    assert all(not math.isnan(float(r["loss"])) and float(r["loss"]) >= 0 for r in labeled)
    assert all(not math.isnan(float(r["loss"])) for r in val_rows)
    assert names
    # The record: completed, in the history, with the ETA stats and the checkpoint facts.
    st = trainer.train_state()
    assert st["state"] == "completed" and st["current"]["weights_on_disk"] is True
    assert st["current"]["facts"]["train_table_url"] == train_url and st["current"]["client_token"] == "click-1"
    assert st["current"]["progress"]["epoch"] == 2 and len(st["current"]["progress"]["history"]) == 2
    assert {"e", "tl", "vl", "va"} <= set(st["current"]["progress"]["history"][0])
    assert st["runs"][0]["id"] == st["current"]["id"] and st["runs"][0]["epoch_s_per_row"] > 0
    assert st["runs"][0]["collect_s"] is not None and st["runs"][0]["provenance_ok"] is True
    # The train and val tables were never modified.
    assert _snapshot(train_url) == before_train and _snapshot(val_url) == before_val
    # The progress channel carried ExDark's shape: epochs, history, batch counters, the ETA fields.
    epoch_payloads = [p for p in ctx.progress if p.get("epoch") == 2 and p.get("phase") == "train"]
    assert epoch_payloads and "avg_epoch_s" in epoch_payloads[-1] and epoch_payloads[-1]["total_epochs"] == 2
    assert any(p.get("stage") == "collect" for p in ctx.progress)
    # Facts the fragment and session 4 read.
    assert ctx.facts["run_url"] == run_url and ctx.facts["weights"] == result["weights"]
    assert ctx.facts["device"] == "cpu"
    # Item 1: the header and the log say why (this run forced cpu); the requested device reaches the host.
    assert ctx.facts["device_label"] == "cpu (forced in Advanced)" and ctx.facts["device_requested"] == "cpu"
    assert "Device: cpu (forced in Advanced)" in ctx.logs
    assert st["runs"][0]["device_label"] == "cpu (forced in Advanced)"


def test_a_second_start_is_refused_while_a_run_is_in_progress_and_a_reused_click_token_is_refused(imported, store):
    manifest, _train_url, _ = imported
    store.save({"train_state": {"current": _record(client_token="click-9", facts={"run_name": "busy"})}})
    with pytest.raises(trainer.TrainRefused, match="already in progress \\(busy\\)"):
        trainer.run_training({"device": "cpu", "client_token": "click-10"}, FakeCtx(), manifest)
    rec = _record(status="completed", client_token="click-9", finished_at=time.time())
    store.save({"train_state": {"current": rec}})
    with pytest.raises(trainer.TrainRefused, match="already used"):
        trainer.run_training({"device": "cpu", "client_token": "click-9"}, FakeCtx(), manifest)


def test_cancel_mid_run_keeps_the_best_checkpoint_and_marks_the_run_cancelled(imported):
    manifest, _train_url, _ = imported
    # FakeCtx cancels from its Nth is_cancelled() call: after epoch 1 has completed (2 batches +
    # the epoch check + the first-batch probe), so epoch 2 stops at its first batch.
    ctx = FakeCtx(cancel_on_call=6)
    result = trainer.run_training(
        {"epochs": "3", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": "t_cancel"}, ctx, manifest
    )
    assert result["cancelled"] is True and 1 <= result["epochs_completed"] < 3
    assert Path(result["weights"]).is_file() and result["best_checkpoint_sha256"]
    run = tlc.Run.from_url(tlc.Url(result["run_url"]))
    assert 'status="cancelled"' in repr(run), repr(run)
    st = trainer.train_state()
    assert st["state"] == "cancelled" and st["current"]["weights_on_disk"] is True
    assert st["runs"][0]["status"] == "cancelled"
    # Provenance still recorded (Predict may use a cancelled run's best checkpoint, D10).
    assert ctx.checks and all(c["ok"] for c in ctx.checks)
    assert not any(p.get("stage") == "collect" for p in ctx.progress)


# ── Part B: manifest-driven editability ────────────────────────────────────────


def _opened(manifest, **over):
    """The bundled manifest with the choice fields opened (what another event's document would say)."""
    import dataclasses

    data = manifest_mod.load_yaml_text(manifest_mod.bundled_path().read_text(encoding="utf-8"))
    data["training"]["editable"] = ["epochs", "batch_size", "lr", "weight_decay", "seed", "optimizer", "schedule"]
    data["training"]["options"] = {"optimizer": ["adam", "adamw", "sgd"], "schedule": ["steplr", "cosine", "none"]}
    for k, v in over.items():
        data["training"][k] = v
    m = manifest_mod.parse_manifest(data)
    assert not any("editable" in w for w in m.warnings)
    return dataclasses.replace(m)


def test_the_bundled_manifest_locks_the_choice_fields_and_the_server_refuses_them(home, manifest):
    assert not manifest.training.is_editable("optimizer") and not manifest.training.is_editable("schedule")
    kw = trainer.build_train_kwargs({"epochs": "2"}, manifest)
    assert kw["optimizer"] == "adam" and kw["schedule"] == "steplr" and kw["schedule_params"]["step_size"] == 5
    for locked in ({"optimizer": "sgd"}, {"optimizer": "adam"}, {"schedule": "cosine"}, {"schedule": "steplr"}):
        with pytest.raises(trainer.TrainRefused, match="is locked for this competition"):
            trainer.build_train_kwargs(locked, manifest)


def test_a_manifest_that_opens_the_choice_fields_accepts_and_validates_them(home, manifest):
    m = _opened(manifest)
    assert trainer.training_facts(m)["editable"][-2:] == ["optimizer", "schedule"]
    kw = trainer.build_train_kwargs({"optimizer": "SGD", "schedule": "cosine", "lr": "0.01"}, m)
    assert kw["optimizer"] == "sgd" and kw["schedule"] == "cosine" and kw["lr"] == 0.01
    with pytest.raises(trainer.TrainRefused, match="optimizer must be one of adam, adamw, sgd"):
        trainer.build_train_kwargs({"optimizer": "lamb"}, m)
    with pytest.raises(trainer.TrainRefused, match="schedule must be one of steplr, cosine, none"):
        trainer.build_train_kwargs({"schedule": "plateau"}, m)
    # A manifest may lock a numeric field too: then the client must not send it.
    locked_epochs = _opened(manifest, editable=["batch_size", "lr"])
    with pytest.raises(trainer.TrainRefused, match="epochs is locked"):
        trainer.build_train_kwargs({"epochs": "3"}, locked_epochs)
    assert trainer.build_train_kwargs({}, locked_epochs)["epochs"] == 10


def test_the_manifest_validates_editable_and_options(manifest):
    data = manifest_mod.load_yaml_text(manifest_mod.bundled_path().read_text(encoding="utf-8"))
    data["training"]["editable"] = ["epochs", "momentum"]
    with pytest.raises(manifest_mod.ManifestError, match="training.editable: 'momentum'"):
        manifest_mod.parse_manifest(data)
    data["training"]["editable"] = ["epochs"]
    data["training"]["options"] = {"optimizer": ["lamb"]}
    with pytest.raises(manifest_mod.ManifestError, match="training.options.optimizer"):
        manifest_mod.parse_manifest(data)
    data["training"]["options"] = {"optimizer": ["sgd"]}   # the default adam is not in the allowed list
    with pytest.raises(manifest_mod.ManifestError, match="training.defaults.optimizer"):
        manifest_mod.parse_manifest(data)
    # Absent editable/options (a pre-part-B document): the numeric fields open, the choices lock.
    del data["training"]["editable"], data["training"]["options"]
    m = manifest_mod.parse_manifest(data)
    assert m.training.editable == manifest_mod.DEFAULT_EDITABLE
    assert m.training.options == {"optimizer": ("adam",), "schedule": ("steplr",)}


def test_an_opened_manifest_trains_with_the_chosen_optimizer_and_schedule(imported):
    manifest, train_url, _ = imported
    m = _opened(manifest)
    ctx = FakeCtx()
    result = trainer.run_training(
        {"epochs": "1", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": "t_sgd",
         "optimizer": "sgd", "schedule": "cosine", "lr": "0.01"}, ctx, m,
    )
    assert result["cancelled"] is False and result["contract"]["optimizer"] == "sgd"
    assert result["contract"]["schedule"] == "cosine"
    p = trainer.get_run_parameters(tlc.Run.from_url(tlc.Url(result["run_url"])))
    assert p["optimizer"] == "sgd" and p["schedule"] == "cosine" and p["lr"] == 0.01
    assert any("sgd, cosine" in line for line in ctx.logs)


# ── Part D: the revision tree the picker renders ─────────────────────────────


def test_tables_list_is_the_seed_lineage_tree_with_counts(imported, store):
    manifest, train_url, val_url = imported
    a = _edit(train_url, "weight", {0: 2.0}, "round-1")
    b = _edit(a, "weight", {1: 0.0}, "round-2")
    stranger = tlc.Table.from_dict(
        {"image": [], "label": [], "weight": []}, schema=importer._schema(manifest),
        table_url=tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial-2")),
        if_exists="raise", add_weight_column=False,
    )
    used = {a: 2, str(stranger.url): 1}
    listing = importer.list_project_tables(manifest, "intel-scene", seed_url=train_url, runs_used=used)
    ds = next(d for d in listing["datasets"] if d["name"] == manifest.dataset_name("train"))
    by_name = {t["name"]: t for t in ds["tables"]}
    assert [t["name"] for t in ds["tables"]][:3] == ["initial", "round-1", "round-2"]
    assert [t["depth"] for t in ds["tables"]][:3] == [0, 1, 2]
    assert by_name["round-1"]["parent"] == train_url and by_name["round-2"]["parent"] == a
    assert by_name["initial"]["in_lineage"] and by_name["round-2"]["in_lineage"]
    assert by_name["initial-2"]["in_lineage"] is False and by_name["initial-2"]["depth"] == 0
    assert ds["tables"][-1]["name"] == "initial-2"   # a different import sorts last
    assert by_name["initial"]["labeled_rows"] == 12 and by_name["round-2"]["labeled_rows"] == 11
    assert by_name["round-1"]["runs_used"] == 2 and by_name["initial"]["runs_used"] == 0
    assert by_name["round-2"]["latest"] is True and by_name["initial"]["latest"] is False
    assert ds["latest_url"] == b
    # The val dataset lists no labeled counts (val is locked) and is in lineage by default.
    val_ds = next(d for d in listing["datasets"] if d["name"] == manifest.dataset_name("val"))
    assert val_ds["tables"][0]["labeled_rows"] is None
    # Cached by URL: a second listing does not re-scan the rows.
    assert importer._LABELED_ROWS_CACHE[importer._norm(train_url)] == 12


# ── Part F: the Run's status on interruption; the numba cache; the pre-warm ──────


def test_an_interrupted_run_is_marked_on_the_run_itself(imported, store):
    """F4: a running record whose worker died reads back stale AND the Run it created is set to
    cancelled with interrupted = True, so the Hub's Runs list stops showing EMPTY."""
    manifest, train_url, _ = imported
    result = trainer.run_training(
        {"epochs": "1", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": "t_interrupted"}, FakeCtx(), manifest
    )
    state = trainer.read_state()
    state["current"].update({"status": "running", "pid": 999_999_999, "finished_at": None})
    store.save({"train_state": state})
    st = trainer.train_state()
    assert st["state"] == "stale"
    run = tlc.Run.from_url(tlc.Url(result["run_url"]))
    assert 'status="cancelled"' in repr(run)
    p = trainer.get_run_parameters(run)
    assert p["interrupted"] is True and "restarted" in p["interrupted_reason"]
    assert any("marked cancelled (interrupted)" in line for line in st["current"]["log"])


def test_numba_cache_lives_under_the_plugin_home_and_the_prewarm_is_idempotent(home, monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    path = trainer.numba_cache_env()
    assert Path(path) == home / trainer.NUMBA_CACHE_DIR_NAME and Path(path).is_dir()
    assert trainer.numba_cache_env() == path
    # The pre-warm runs once per process; a second call reports the same state.
    first = trainer.prewarm_umap_async()
    assert first["state"] in ("running", "done")
    second = trainer.prewarm_umap_async()
    assert second["state"] in ("running", "done")
    deadline = time.time() + 300
    while trainer.prewarm_status()["state"] == "running" and time.time() < deadline:
        time.sleep(1)
    status = trainer.prewarm_status()
    assert status["state"] == "done", status
    assert trainer.train_state()["umap_prewarm"]["state"] == "done"


# ── Session 5, part B3: Hub table operations cannot smuggle val or test rows into a train revision ──


def _join(manifest, urls: list[str], name: str, *, with_lineage: bool) -> str:
    """What a Hub Table Op (merge / concatenate) produces: a JoinedTable over several inputs, with or
    without the lineage pointer that makes it descend from the seed."""
    tables = [tlc.Table.from_url(tlc.Url(u)) for u in urls]
    joined = tlc.Table.join_tables(
        tables,
        table_url=tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), name)),
        input_tables=[urls[0]] if with_lineage else None,
    )
    return str(joined.url)


def test_a_merged_train_and_val_table_passes_the_lineage_walk_and_is_refused_by_the_row_gate(imported):
    manifest, train_url, val_url = imported
    rev = _edit(train_url, "weight", {0: 1.0}, "initial_b3")
    merged = _join(manifest, [rev, val_url], "merged-with-val", with_lineage=True)
    pf = trainer.preflight({"train_url": merged}, manifest)
    # The lineage walk alone would let this through (the merge records the train revision as its input)...
    assert pf["base"]["descends_from_seed"] is True
    # ...but every val row's image lives outside <kit>/data/train, so the gate reports and refuses it.
    val_rows = manifest.splits.val.per_class * manifest.num_classes
    assert pf["base"]["foreign_rows"] == val_rows and pf["base"]["rows"] == 15 + val_rows
    assert pf["base"]["foreign_examples"] and "/data/val/" in pf["base"]["foreign_examples"][0].replace("\\", "/")
    assert pf["base"]["kit_train_dir"].replace("\\", "/").endswith("/data/train")
    params = {"train_table_url": merged, "use_latest": False, "device": "cpu"}
    with pytest.raises(trainer.TrainRefused, match="not in the kit's train folder"):
        trainer.run_training(params, FakeCtx(), manifest)
    # A merge WITHOUT the lineage pointer fails the lineage walk first, and still counts the foreign rows.
    orphan = _join(manifest, [rev, val_url], "merged-no-lineage", with_lineage=False)
    pf2 = trainer.preflight({"train_url": orphan}, manifest)
    assert pf2["base"]["descends_from_seed"] is False and pf2["base"]["foreign_rows"] == val_rows
    with pytest.raises(trainer.TrainRefused, match="does not descend from the imported seed"):
        trainer.run_training({"train_table_url": orphan, "use_latest": False, "device": "cpu"}, FakeCtx(), manifest)
    # The honest revisions still read as clean.
    assert trainer.preflight({"train_url": rev}, manifest)["base"]["foreign_rows"] == 0


# ── Session 5, part B2: val edits newer than the locked revision are reported ──


def test_import_state_flags_val_revisions_newer_than_the_locked_one(imported):
    manifest, _train_url, val_url = imported
    state = importer.import_state()
    assert state["state"] == "success" and state["val_edited"] is False
    assert importer._norm(state["val_latest_url"]) == importer._norm(val_url)
    rev = _edit(val_url, "weight", {0: 0.0}, "initial_edited")
    state = importer.import_state()
    assert state["val_edited"] is True and importer._norm(state["val_latest_url"]) == importer._norm(rev)
    # The locked URL is untouched: the record still names the seed revision.
    assert importer._norm(state["record"]["val_locked"]["url"]) == importer._norm(val_url)
