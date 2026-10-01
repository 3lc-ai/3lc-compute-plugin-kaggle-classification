# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Predict + Submit stage (job kinds ``predict`` and ``kaggle_submit``), docs/PREDICT_MIRROR.md.

* **Plugin-run-only, provably** (§2, D1, D2): a predict request carries only ``train_job_id`` (+
  device). The checkpoint path comes from the train record entry with that id; a path in the
  request is ignored, a bare path refused. The file must exist and its sha256 must equal BOTH the
  train record's ``best_checkpoint_sha256`` and the Run's own ``best_checkpoint_sha256`` parameter
  (the three-way match), and every provenance check the run recorded must be green. Enforced in
  the job (``resolve_checkpoint``); ``GET /runs`` only displays the same verdicts.
* **Test inputs** (D3): the ids of the kit's ``sample_submission.csv``, each image under
  ``<kit_dir>/data/test`` verified by sha256 against the kit's ``files.json`` before inference.
  The test split is never a table.
* **Val check** (D6): the checkpoint is run over the LOCKED val revision first; the accuracy must
  reproduce the run's recorded ``best_val_accuracy`` (a provenance check in disguise) and is the
  hero stat ("Val accuracy · Your locked validation split, not the leaderboard.").
* **Inference parity** with ``intel-kit/predict.py`` (§4): the val transform, ``build_model``,
  ``weights_only`` strict load, eval + no_grad, batch 32, softmax argmax / max; an unreadable test
  image fails the job (D11).
* **The CSV** (§3, D9): the manifest's columns, one row per sample_submission id in that order,
  ``confidence`` at six decimals, LF, no BOM; seven format checks before anything else, the file
  kept as ``.INVALID.csv`` when one fails; the predicted-class distribution card (D5).
* **Submit** (§7, D7, D8, D12): the slug from the manifest, the message defaulting to
  "<run> via 3LC plugin", the kaggle client's soft outcomes, one status read-back after submit.
* **Records**: durable ``predict_state`` / ``submit_state`` in the session store (the revisit view,
  the pid orphan rule), a ledger entry per prediction and per submission (§6).

Import-light: torch / torchvision / tlc / PIL live inside functions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from kaggle_classification import importer, kaggle_client, kit, ledger, session, storage, trainer

if TYPE_CHECKING:
    from kaggle_classification.manifest import Manifest

PREDICTIONS_DIR = "predictions"
BATCH_SIZE = 32                 # the kit's predict.py batch (a plugin constant, not a competition fact)
PROGRESS_FLUSH_S = 1.0          # ExDark: per-image counts land at most ~1/s
CONFIDENCE_DECIMALS = 6         # D9
LOW_CONFIDENCE = 0.5            # D5: the "low confidence" stat threshold
SKEW_LOW, SKEW_HIGH = 0.05, 0.50   # D5: a class below 5 % or above 50 % of the predictions warns
VAL_TOLERANCE_PP = 0.25         # D6: recorded vs recomputed val accuracy, in percentage points
LOG_KEEP = 300
INVALID_SUFFIX = ".INVALID.csv"

NOT_A_RUN = (
    "Direct weights files are not accepted. Select a run trained in this plugin — predictions must "
    "carry verified provenance."
)
SHA_MISMATCH = (
    "The best checkpoint on disk no longer matches the run’s record (sha256 differs). Re-train, or pick "
    "another run."
)
TEST_REMEDY = "Re-run Import (tab 1) to rebuild the test images, then predict again."


class PredictRefused(RuntimeError):
    """A participant-facing refusal; the job fails with the message verbatim."""


# ── Checkpoint hashes (cached by path + size + mtime: GET /runs hashes every run's best.pt) ────
_sha_cache: dict[str, tuple[tuple[int, float], str]] = {}
_sha_lock = threading.Lock()


def sha256_of(path: Path | str) -> str:
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return ""
    key = (st.st_size, st.st_mtime)
    with _sha_lock:
        hit = _sha_cache.get(str(p))
        if hit and hit[0] == key:
            return hit[1]
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _sha_lock:
        _sha_cache[str(p)] = (key, value)
    return value


def run_parameters(run_url: str) -> dict[str, Any] | None:
    """The Run's recorded parameters: through tlc when it is importable, else the torch-free JSON read."""
    try:
        import tlc

        return dict(trainer.get_run_parameters(tlc.Run.from_url(tlc.Url(run_url))))
    except Exception:
        return trainer._run_parameters_on_disk(run_url)


def run_folder(run_url: str, run_name: str) -> str:
    """The Run folder's name (unique on disk, equals the run name unless tlc suffixed it)."""
    tail = str(run_url or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or str(run_name or "")


# ── The run list and the plugin-run-only gate ───────────────────────────────


def run_entries() -> list[dict[str, Any]]:
    """Every train record entry Predict may consider, newest first: the finished-run summaries plus
    the current record when it is still running (not yet in the history)."""
    state = trainer.train_state()
    runs = [dict(r) for r in state.get("runs") or [] if isinstance(r, dict) and r.get("id")]
    cur = state.get("current")
    if isinstance(cur, dict) and cur.get("id") and cur.get("status") == "running":
        if not any(r.get("id") == cur["id"] for r in runs):
            runs.insert(0, trainer.run_summary(cur))
    for r in runs:
        if r.get("status") == "failed" and isinstance(cur, dict) and cur.get("id") == r.get("id"):
            r["error"] = cur.get("error")
    return runs


def assess_run(r: dict[str, Any], *, check_run: bool = True) -> tuple[bool, str]:
    """``(usable, reason)`` — the display-side verdicts ``GET /runs`` shows (ExDark's four, plus ours:
    interrupted, provenance failed, the sha mismatches). ``resolve_checkpoint`` repeats the hard ones."""
    status = str(r.get("status") or "")
    weights = str(r.get("weights") or "")
    if status == "running":
        return False, "still training"
    if status == "failed":
        err = str(r.get("error") or "").strip()
        return False, f"failed: {err}" if err else "failed"
    if status == "stale":
        return False, "interrupted"
    if not weights:
        return False, "no best checkpoint saved"
    if not Path(weights).is_file():
        return False, "best checkpoint missing on disk"
    if not r.get("provenance_ok"):
        return False, "provenance check failed"
    recorded = str(r.get("best_checkpoint_sha256") or "")
    if not recorded or sha256_of(weights) != recorded:
        return False, "best checkpoint changed on disk (sha256 mismatch)"
    if check_run:
        p = run_parameters(str(r.get("run_url") or "")) or {}
        if str(p.get("best_checkpoint_sha256") or "") != recorded:
            return False, "the Run’s record disagrees with the train record (sha256 mismatch)"
    return True, ""


def list_runs() -> list[dict[str, Any]]:
    """``GET /runs``: the Run picker's rows, newest first, with ``usable`` + ``reason``."""
    out = []
    for r in run_entries():
        usable, reason = assess_run(r)
        out.append({
            "job_id": r.get("id"),
            "run_name": r.get("run_name") or "",
            "run_folder": run_folder(str(r.get("run_url") or ""), str(r.get("run_name") or "")),
            "run_url": r.get("run_url"),
            "status": r.get("status"),
            "created_at": r.get("created_at"),
            "finished_at": r.get("finished_at"),
            "epochs_completed": r.get("epochs_completed"),
            "best_epoch": r.get("best_epoch"),
            "best_val_accuracy": r.get("best_val_accuracy"),
            "provenance_ok": bool(r.get("provenance_ok")),
            "best_checkpoint_sha256": r.get("best_checkpoint_sha256") or "",
            "weights": r.get("weights") or "",
            "device": r.get("device"),
            "device_label": r.get("device_label"),
            "contract": r.get("contract"),
            "usable": usable,
            "reason": reason,
        })
    return out


def resolve_checkpoint(params: dict[str, Any]) -> dict[str, Any]:
    """THE plugin-run-only gate (§2): ``train_job_id`` wins and any path in the request is ignored;
    a bare path is refused; the file must exist; its sha256 must equal the train record's AND the
    Run's; the run's provenance checks must all be green (D2). Raises ``PredictRefused``."""
    job_id = str(params.get("train_job_id") or "").strip()
    if not job_id:
        if str(params.get("weights_path") or "").strip():
            raise PredictRefused(NOT_A_RUN)
        msg = "Select a run trained in this plugin."
        raise PredictRefused(msg)
    entry = next((r for r in run_entries() if str(r.get("id")) == job_id), None)
    if entry is None:
        msg = f"No plugin run with id {job_id}. Refresh the runs list and pick a run trained here."
        raise PredictRefused(msg)
    status = str(entry.get("status") or "")
    name = entry.get("run_name") or job_id
    if status == "running":
        msg = f"{name} is still training. Wait for it to finish, then predict."
        raise PredictRefused(msg)
    if status not in ("completed", "cancelled"):
        msg = f"{name} did not finish ({'interrupted' if status == 'stale' else status}). Pick a completed run."
        raise PredictRefused(msg)
    weights = str(entry.get("weights") or "")
    if not weights:
        msg = f"{name} saved no best checkpoint. Pick another run."
        raise PredictRefused(msg)
    if not Path(weights).is_file():
        msg = f"The best checkpoint of {name} is missing on disk ({weights}). Re-train, or pick another run."
        raise PredictRefused(msg)
    if not entry.get("provenance_ok"):
        msg = (
            f"{name} failed a provenance check at training time, so it cannot be used for a submission. "
            "Re-train, or pick another run."
        )
        raise PredictRefused(msg)
    recorded = str(entry.get("best_checkpoint_sha256") or "")
    on_disk = sha256_of(weights)
    run_url = str(entry.get("run_url") or "")
    p = run_parameters(run_url) or {}
    on_run = str(p.get("best_checkpoint_sha256") or "")
    if not recorded or on_disk != recorded or on_run != recorded:
        raise PredictRefused(SHA_MISMATCH)
    return {
        "train_job_id": job_id,
        "run_name": str(entry.get("run_name") or ""),
        "run_folder": run_folder(run_url, str(entry.get("run_name") or "")),
        "run_url": run_url,
        "project_name": entry.get("project_name"),
        "weights": weights,
        "sha256_recorded": recorded,
        "sha256_on_disk": on_disk,
        "sha256_on_run": on_run,
        "best_val_accuracy": entry.get("best_val_accuracy"),
        "best_epoch": entry.get("best_epoch"),
        "epochs_completed": entry.get("epochs_completed"),
        "run_parameters": p,
        "contract": {k: p.get(k) for k in (
            "backbone", "head", "arch", "image_size", "pretrained", "torch_version", "torchvision_version", "seed",
            "epochs", "batch_size", "lr", "weight_decay", "optimizer", "schedule", "train_table_url",
            "val_table_url", "usable_rows", "plugin", "job_id", "manifest_sha256", "manifest_source",
            "competition_id", "kit_version",
        )},
    }


# ── Test inputs (D3) ──────────────────────────────────────────────────────


def read_sample_ids(path: Path, columns: tuple[str, ...]) -> list[str]:
    """The ids of the kit's sample submission, in file order (the CSV's row order is theirs)."""
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != list(columns):
            msg = f"{path.name} has the columns {header}; expected {list(columns)}."
            raise PredictRefused(msg)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def test_inputs(manifest: Manifest, *, verify: bool = True) -> dict[str, Any]:
    """``GET /predict/preflight`` and the job's gate: every sample id's image under
    ``<kit_dir>/data/test`` present and matching the kit's ``files.json`` by sha256 and size."""
    record = importer.read_record()
    if not record:
        return {"state": "idle", "ok": False}
    kit_dir = Path(str(record.get("kit_dir") or ""))
    out: dict[str, Any] = {
        "state": "missing", "ok": False, "kit_dir": str(kit_dir), "test_dir": str(kit_dir / kit.DATA_DIR_NAME / "test"),
        "expected": int(manifest.splits.test.count), "count": 0,
    }
    if not kit_dir.is_dir():
        out["error"] = "The kit directory is gone."
        return out
    try:
        index = kit.load_files_index(kit_dir)
    except Exception as exc:
        out["error"] = f"{kit.FILES_INDEX_NAME} could not be read: {exc}"
        return out
    out["files_json_sha256"] = sha256_of(kit_dir / kit.FILES_INDEX_NAME)
    sample = kit_dir / manifest.splits.test.ids_from
    if not sample.is_file():
        out["error"] = f"{manifest.splits.test.ids_from} is missing from the kit."
        return out
    try:
        ids = read_sample_ids(sample, manifest.submission.columns)
    except PredictRefused as exc:
        out["error"] = str(exc)
        return out
    prefix = f"{kit.DATA_DIR_NAME}/test/"
    by_stem: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for entry in index.get("files") or []:
        rel = str(entry.get("path") or "")
        if rel.startswith(prefix) and "/" not in rel[len(prefix):]:
            stem = PurePosixPath(rel).stem
            if stem in by_stem:
                duplicates.append(stem)
            by_stem[stem] = entry
    missing: list[str] = []
    mismatch: list[str] = []
    paths: list[str] = []
    for i in ids:
        entry = by_stem.get(i)
        if entry is None:
            missing.append(i)
            paths.append("")
            continue
        p = kit_dir / PurePosixPath(str(entry["path"]))
        paths.append(str(p))
        if not p.is_file():
            missing.append(i)
        elif verify and (p.stat().st_size != int(entry["bytes"]) or sha256_of(p) != str(entry["sha256"])):
            mismatch.append(i)
    out.update({
        "count": len(ids), "verified": len(ids) - len(missing) - len(mismatch),
        "missing_count": len(missing), "mismatch_count": len(mismatch), "duplicate_count": len(duplicates),
        "missing": missing[:20], "mismatch": mismatch[:20], "ids": ids, "paths": paths,
        "sample_submission": str(sample), "sample_submission_sha256": sha256_of(sample),
    })
    ok = not missing and not mismatch and not duplicates and len(ids) == int(manifest.splits.test.count)
    if len(ids) != int(manifest.splits.test.count):
        out["error"] = (
            f"{manifest.splits.test.ids_from} lists {len(ids):,} ids; the competition test split has "
            f"{manifest.splits.test.count:,}."
        )
    elif missing:
        out["error"] = f"{len(missing):,} test image(s) are missing (first: {missing[0]})."
    elif mismatch:
        out["error"] = f"{len(mismatch):,} test image(s) differ from the kit’s files.json (first: {mismatch[0]})."
    elif duplicates:
        out["error"] = f"{len(duplicates):,} test id(s) appear twice in files.json."
    out["state"] = "ok" if ok else "missing"
    out["ok"] = ok
    return out


def preflight(manifest: Manifest) -> dict[str, Any]:
    """The route's view of the gate: everything but the per-id lists (ids and paths stay in the job)."""
    info = test_inputs(manifest)
    return {k: v for k, v in info.items() if k not in ("ids", "paths")}


# ── The CSV, its checks, the distribution card ────────────────────────────


def format_confidence(value: float) -> str:
    return f"{float(value):.{CONFIDENCE_DECIMALS}f}"


def write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(columns))
        for image_id, pred, conf in rows:
            w.writerow([image_id, int(pred), format_confidence(conf)])
    os.replace(tmp, path)


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, []) or []
        return list(header), list(reader)


def validate_submission(header: list[str], rows: list[list[str]], manifest: Manifest, sample_ids: list[str]
                        ) -> list[dict[str, Any]]:
    """The seven format checks (§1 #20, §3), stricter than the metric on purpose. Raises
    ``PredictRefused`` on the first failure with the remedy; the checks so far are on the exception."""
    checks: list[dict[str, Any]] = []
    columns = list(manifest.submission.columns)
    expected = int(manifest.splits.test.count)
    n_classes = manifest.num_classes

    def check(label: str, ok: bool, detail: str, message: str) -> None:
        checks.append({"label": label, "ok": bool(ok), "detail": detail, "group": "Submission format"})
        if not ok:
            exc = PredictRefused(message)
            exc.checks = checks  # type: ignore[attr-defined]
            raise exc

    check(f"columns are {', '.join(columns)}", header == columns, f"got {header}",
          f"Submission must have exactly the columns {columns}.")
    check(f"exactly {expected:,} rows (one per test image)", len(rows) == expected, f"got {len(rows):,}",
          f"Submission must contain exactly {expected:,} rows; got {len(rows):,}. {TEST_REMEDY}")
    ids = [r[0] if r else "" for r in rows]
    seen: set[str] = set()
    dup: list[str] = []
    for i in ids:
        if i in seen:
            dup.append(i)
        seen.add(i)
    check("no duplicated image_id rows", not dup, f"{len(dup)} duplicates",
          f"Submission contains {len(dup)} duplicated image_id row(s); submit exactly one row per test image.")
    same = ids == list(sample_ids)
    check(f"image_id set and order equal to {manifest.splits.test.ids_from}", same,
          "aligned" if same else ("same ids, different order" if sorted(ids) == sorted(sample_ids) else "ids differ"),
          f"Submission ids must be the test ids of {manifest.splits.test.ids_from}, in its order. {TEST_REMEDY}")
    bad_pred = []
    for r in rows:
        v = r[1] if len(r) > 1 else ""
        try:
            f = float(v)
            if not f.is_integer() or not 0 <= int(f) < n_classes:
                bad_pred.append(v)
        except ValueError:
            bad_pred.append(v)
    check(f"prediction is an integer class id 0–{n_classes - 1}", not bad_pred,
          f"{len(bad_pred)} invalid" if bad_pred else "all valid",
          f"Predictions must be integer class ids 0–{n_classes - 1}. Found invalid: {bad_pred[:5]}.")
    bad_conf = []
    for r in rows:
        v = r[2] if len(r) > 2 else ""
        try:
            f = float(v)
            if not math.isfinite(f) or not 0.0 <= f <= 1.0:
                bad_conf.append(v)
        except ValueError:
            bad_conf.append(v)
    check("confidence is finite and in [0, 1]", not bad_conf, f"{len(bad_conf)} invalid" if bad_conf else "all valid",
          f"Confidence must be finite and in [0, 1]. Found invalid: {bad_conf[:5]}.")
    blanks = sum(1 for r in rows if len(r) < len(columns) or any(not str(v).strip() for v in r[: len(columns)]))
    check("no missing values", blanks == 0, f"{blanks} blank cells" if blanks else "none",
          f"Submission has {blanks} row(s) with a missing value.")
    return checks


def distribution(preds: list[int], confs: list[float], manifest: Manifest) -> dict[str, Any]:
    """D5: the predicted-class distribution card. Informational, never blocks."""
    names = manifest.class_names
    per_class = dict.fromkeys(names, 0)
    for p in preds:
        if 0 <= int(p) < len(names):
            per_class[names[int(p)]] += 1
    n = len(preds)
    mean_conf = (sum(confs) / n) if n else 0.0
    low = sum(1 for c in confs if c < LOW_CONFIDENCE)
    out: dict[str, Any] = {
        "images": n, "mean_confidence": round(mean_conf, 3), "low_confidence": low,
        "low_confidence_threshold": LOW_CONFIDENCE, "per_class": per_class,
        "skew_low_pct": round(SKEW_LOW * 100), "skew_high_pct": round(SKEW_HIGH * 100),
    }
    skewed = []
    for name in names:
        share = per_class[name] / n if n else 0.0
        if share < SKEW_LOW or share > SKEW_HIGH:
            skewed.append(f"{name} {share * 100:.0f} %")
    if skewed and n:
        out["skewed"] = skewed
        out["warning"] = (
            f"The predicted-class distribution is skewed ({' · '.join(skewed)} of {n:,} predictions; "
            f"expected between {out['skew_low_pct']} % and {out['skew_high_pct']} % per class). "
            "Are these fully-trained weights? (Submitting is still fine.)"
        )
    return out


# ── The durable records ────────────────────────────────────────────────────


def read_predict_record() -> dict[str, Any] | None:
    data = session.load().get("predict_state")
    return data if isinstance(data, dict) and data.get("id") else None


def read_submit_record() -> dict[str, Any] | None:
    data = session.load().get("submit_state")
    return data if isinstance(data, dict) and data.get("id") else None


class _Record:
    """The durable predict / submit record: log tail, heartbeat, terminal state, one JSON key."""

    def __init__(self, key: str, record: dict[str, Any]) -> None:
        self.key = key
        self.record = record
        self._lock = threading.Lock()
        self._last = 0.0

    def log(self, line: str) -> None:
        with self._lock:
            self.record["log"].append(line)
            del self.record["log"][:-LOG_KEEP]

    def touch(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            self.record["heartbeat"] = now
            if not force and now - self._last < trainer.HEARTBEAT_S:
                return
            self._last = now
            session.save({self.key: self.record})

    def flush(self) -> None:
        self.touch(force=True)

    def finish(self, status: str, *, error: str | None = None, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.record["status"] = status
            self.record["finished_at"] = time.time()
            self.record["heartbeat"] = self.record["finished_at"]
            if error:
                self.record["error"] = error
            if result is not None:
                self.record["result"] = result
            self.record["cancelled"] = status == "cancelled"
            session.save({self.key: self.record})


def _new_record(kind: str, job_id: str, params: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "id": job_id, "kind": kind, "status": "running", "pid": os.getpid(), "created_at": now, "started_at": now,
        "heartbeat": now, "finished_at": None, "cancelled": False, "params": dict(params), "progress": {},
        "facts": {}, "checks": [], "result": None, "error": None, "log": [],
    }


def _orphaned(record: dict[str, Any], what: str) -> dict[str, Any]:
    """The Train pid rule: a running record owned by another worker process died with it."""
    if record.get("status") != "running":
        return record
    same_pid = record.get("pid") == os.getpid()
    hb = float(record.get("heartbeat") or record.get("started_at") or 0)
    if same_pid and time.time() - hb < trainer.STALE_HEARTBEAT_S:
        return record
    record["status"] = "stale"
    record["error"] = f"Interrupted: the compute service restarted while {what} was running."
    if record.get("finished_at") is None:
        record["finished_at"] = time.time()
    return record


def predict_submit_state() -> dict[str, Any]:
    """``GET /submit/state``: ``empty`` / ``running`` / ``predicted`` / ``submitted`` / ``stale`` / ``failed``,
    the CSV re-verified on disk (its sha256 too) and the submit record attached when it belongs to
    this prediction."""
    ps = read_predict_record()
    if not ps:
        return {"state": "empty"}
    ps = _orphaned(dict(ps), "inference")
    facts = ps.get("facts") or {}
    csv_path = str(facts.get("csv_path") or "")
    on_disk = bool(csv_path) and Path(csv_path).is_file()
    view = {
        "job_id": ps.get("id"), "status": ps.get("status"), "run_name": facts.get("run_name"),
        "run_folder": facts.get("run_folder"), "run_url": facts.get("run_url"), "train_job_id": facts.get("train_job_id"),
        "csv_path": csv_path, "csv_on_disk": on_disk, "csv_sha256": facts.get("csv_sha256"),
        "csv_sha256_now": sha256_of(csv_path) if on_disk else "",
        "checkpoint_sha256": facts.get("checkpoint_sha256"), "sanity": facts.get("sanity"),
        "local_score": facts.get("local_score"), "checks": ps.get("checks") or [], "error": ps.get("error"),
        "created_at": ps.get("created_at"), "finished_at": ps.get("finished_at"), "log": ps.get("log") or [],
    }
    status = str(ps.get("status") or "")
    if status == "running":
        return {"state": "running", "predict": view}
    if status != "completed":
        return {"state": status or "failed", "predict": view}
    if not on_disk:
        return {"state": "empty", "predict": view, "note": "The last prediction's CSV is gone from disk."}
    if view["csv_sha256"] and view["csv_sha256_now"] != view["csv_sha256"]:
        return {"state": "empty", "predict": view, "note": "The last prediction's CSV changed on disk since it was validated."}
    out: dict[str, Any] = {"state": "predicted", "predict": view}
    ss = read_submit_record()
    if ss and ss.get("predict_job_id") == ps.get("id"):
        ss = _orphaned(dict(ss), "the submission")
        sub = ((ss.get("facts") or {}).get("submission")) or {}
        out["submission"] = {
            "job_id": ss.get("id"), "predict_job_id": ss.get("predict_job_id"), "status": sub.get("status") or ss.get("status"),
            "job_status": ss.get("status"), "ref": sub.get("ref"), "reason": sub.get("reason") or ss.get("error"),
            "kaggle": sub.get("kaggle"), "message": (ss.get("params") or {}).get("message"),
            "finished_at": ss.get("finished_at"), "log": ss.get("log") or [],
        }
        if sub.get("status") == "submitted":
            out["state"] = "submitted"
    return out


def csv_path_for(job_id: str) -> str:
    """``GET /submissions/{job_id}/download``: the record's CSV, else the ledger's for an older job."""
    ps = read_predict_record()
    if ps and str(ps.get("id")) == str(job_id):
        return str((ps.get("facts") or {}).get("csv_path") or "")
    entry = ledger.find("predict", job_id=str(job_id))
    return str(((entry or {}).get("csv") or {}).get("path") or "")


# ── Inference ───────────────────────────────────────────────────────────────


def reclaim_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_model(weights: str, manifest: Manifest, device: str) -> Any:
    """The manifest's model with the checkpoint's raw state dict: ``weights_only``, strict, eval."""
    import torch

    model = trainer.build_model(manifest.model.backbone, manifest.model.head, manifest.num_classes)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        msg = "The checkpoint is not a model state dict. Re-train, or pick another run."
        raise PredictRefused(msg)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        msg = f"The checkpoint does not fit the competition model ({type(exc).__name__}: {str(exc)[:160]}). Re-train, or pick another run."
        raise PredictRefused(msg) from exc
    return model.to(device).eval()


class _TestImages:
    """``(tensor, index)`` per test id; an unreadable image raises (D11: the job fails, loudly)."""

    def __init__(self, ids: list[str], paths: list[str], transform: Any) -> None:
        self.ids, self.paths, self.transform = ids, paths, transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Any:
        from PIL import Image

        try:
            with Image.open(self.paths[i]) as im:
                image = im.convert("RGB")
        except Exception as exc:
            msg = (
                f"Test image {self.ids[i]} could not be read ({type(exc).__name__}: {str(exc)[:120]}). "
                "Re-run Import (tab 1) to restore the kit, then predict again."
            )
            raise PredictRefused(msg) from exc
        return self.transform(image), i


def _forward_batches(model: Any, loader: Any, device: str, on_batch: Any, is_cancelled: Any) -> bool:
    """Runs the loader through the model; ``on_batch(preds, confs, labels)`` per batch. False on cancel."""
    import torch

    with torch.no_grad():
        for images, extra in loader:
            if is_cancelled():
                return False
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            on_batch(pred.cpu().tolist(), conf.float().cpu().tolist(), extra)
    return True


def _is_device_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and bool(trainer._DEVICE_ERROR_RE.search(str(exc)))


# ── The predict job ────────────────────────────────────────────────────────


def run_predict(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """The Predict job: resolve + verify the checkpoint, verify the test inputs, the val check, the
    test pass, the CSV and its checks, the distribution card, the record and the ledger entry."""

    log = ctx.log
    set_progress = getattr(ctx, "set_progress", lambda p: None)
    set_field = getattr(ctx, "set_field", lambda k, v: None)
    set_checks = getattr(ctx, "set_checks", lambda c: None)
    is_cancelled = getattr(ctx, "is_cancelled", lambda: False)
    job_id = str(getattr(ctx, "job_id", "") or "") or f"local-{int(time.time())}"

    existing = read_predict_record()
    if existing and _orphaned(dict(existing), "inference").get("status") == "running":
        msg = "A prediction is already in progress. Wait for it to finish."
        raise PredictRefused(msg)

    # ── The gates, before any record exists ────────────────────────────────
    ck = resolve_checkpoint(params)
    inputs = test_inputs(manifest)
    if inputs.get("state") == "idle":
        msg = "Import the kit first: no import record exists on this machine."
        raise PredictRefused(msg)
    if not inputs.get("ok"):
        msg = f"The kit’s test images could not be verified: {inputs.get('error') or 'unknown'} {TEST_REMEDY}"
        raise PredictRefused(msg)
    record_imp = importer.read_record() or {}
    val_url = str((record_imp.get("val_locked") or {}).get("url") or "")
    if not val_url or not trainer._url_exists(val_url):
        msg = f"The locked val table is missing on disk ({val_url or 'no URL recorded'}). Run Import (tab 1) again."
        raise PredictRefused(msg)

    rec = _Record("predict_state", _new_record("predict", job_id, {
        "train_job_id": ck["train_job_id"], "device": str(params.get("device") or ""),
    }))
    sample_ids: list[str] = list(inputs["ids"])
    rec.record["facts"].update({
        "run_name": ck["run_name"], "run_folder": ck["run_folder"], "run_url": ck["run_url"],
        "train_job_id": ck["train_job_id"], "weights": ck["weights"], "checkpoint_sha256": ck["sha256_recorded"],
        "checkpoint_sha256_on_disk": ck["sha256_on_disk"], "checkpoint_sha256_on_run": ck["sha256_on_run"],
        "best_val_accuracy": ck["best_val_accuracy"], "contract": ck["contract"],
        "test_inputs": {k: v for k, v in inputs.items() if k not in ("ids", "paths", "missing", "mismatch")},
        "val_table_url": val_url, "manifest": dict(manifest.provenance),
    })
    rec.flush()
    for k in ("run_name", "run_folder", "run_url", "train_job_id", "weights", "checkpoint_sha256", "best_val_accuracy"):
        set_field(k, rec.record["facts"][k])

    def rlog(line: str) -> None:
        rec.log(line)
        log(line)

    checks: list[dict[str, Any]] = [{
        "label": "best checkpoint sha256 matches the train record and the Run", "ok": True, "group": "Checkpoint",
        "detail": f"{ck['sha256_recorded'][:12]}… recorded · on disk · on the Run",
    }, {
        "label": f"{inputs['count']:,} test images match the kit’s {kit.FILES_INDEX_NAME}", "ok": True,
        "group": "Test inputs", "detail": f"{inputs['verified']:,} verified by sha256",
    }]
    rec.record["checks"] = checks
    set_checks(checks)
    rlog(f"Run: {ck['run_name']} ({ck['run_url']})")
    rlog(f"Checkpoint: {ck['weights']} — sha256 {ck['sha256_recorded'][:12]}… matches the train record and the Run")
    rlog(f"Test images: {inputs['count']:,} verified against {kit.FILES_INDEX_NAME} under {inputs['test_dir']}")

    try:
        return _predict_core(ck, inputs, sample_ids, val_url, params, rec, rlog, set_progress, set_field,
                             set_checks, is_cancelled, manifest, checks)
    except PredictRefused as exc:
        extra = getattr(exc, "checks", None)
        if extra:
            rec.record["checks"] = checks + [c for c in extra if c not in checks]
            set_checks(rec.record["checks"])
        rec.finish("failed", error=str(exc))
        raise
    except Exception as exc:
        rec.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _predict_core(ck, inputs, sample_ids, val_url, params, rec, log, set_progress, set_field, set_checks,
                  is_cancelled, manifest, checks) -> dict[str, Any]:
    import tlc
    from torch.utils.data import DataLoader

    image_size = int(manifest.model.image_size)
    workers = trainer.default_workers()
    _, val_tf = trainer.build_transforms(image_size)
    device_requested = str(params.get("device") or "")
    device = trainer.resolve_device(device_requested)
    reclaim_gpu_memory()

    last_flush = [0.0]

    def progress(phase: str, done: int, total: int, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - last_flush[0] < PROGRESS_FLUSH_S:
            return
        last_flush[0] = now
        label = ("Val check: " if phase == "val" else "Inference: ") + f"{done:,} / {total:,} images"
        # One bar across both passes: the val check is the first part, the test pass the rest.
        n_val, n_test = int(inputs_val_total[0]), len(sample_ids)
        base = 0 if phase == "val" else n_val
        percent = 100.0 * (base + done) / max(n_val + n_test, 1)
        payload = {"percent": round(percent, 2), "label": label, "phase": phase, "images": done, "total_images": total}
        rec.record["progress"] = {k: v for k, v in payload.items() if k != "percent"}
        set_progress(payload)
        rec.touch()

    inputs_val_total = [0]

    # ── Model (with the Train stage's CPU retry at load time) ───────────────
    model: Any = None
    fallback_reason = ""
    while True:
        try:
            model = load_model(ck["weights"], manifest, device)
            break
        except Exception as exc:
            if device != "cpu" and _is_device_error(exc):
                fallback_reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                log(f"Accelerator failed ({fallback_reason}). Retrying on CPU.")
                device = "cpu"
                continue
            raise
    label = trainer.device_label(device, device_requested, fallback_reason)
    rec.record["facts"].update({"device": device, "device_label": label})
    set_field("device", device)
    set_field("device_label", label)
    log(f"Device: {label}")
    log(f"Model: {manifest.model.backbone} + {manifest.model.head}, {image_size}px (locked), batch {BATCH_SIZE}, "
        f"{trainer.INFERENCE}; weights loaded strict (weights_only)")

    # ── D6: the val check on the locked revision ────────────────────────────
    val_table = tlc.Table.from_url(tlc.Url(val_url))
    val_view = val_table.with_transform(trainer._SampleTransform(val_tf))
    n_val = int(val_table.row_count)
    inputs_val_total[0] = n_val
    log(f"Val check: {val_url} ({n_val:,} rows, the locked revision)")
    progress("val", 0, n_val, force=True)
    correct = [0]
    seen = [0]

    def on_val(preds: list[int], confs: list[float], labels: Any) -> None:
        lab = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        correct[0] += sum(1 for p, y in zip(preds, lab, strict=True) if int(p) == int(y))
        seen[0] += len(preds)
        progress("val", seen[0], n_val)

    val_loader = DataLoader(val_view, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=False)
    if not _forward_batches(model, val_loader, device, on_val, is_cancelled):
        return _cancelled(rec, log, model)
    progress("val", n_val, n_val, force=True)
    val_acc = round(100.0 * correct[0] / max(seen[0], 1), 2)
    recorded = ck.get("best_val_accuracy")
    delta = (abs(val_acc - float(recorded)) if isinstance(recorded, (int, float)) else None)
    val_ok = delta is not None and delta <= VAL_TOLERANCE_PP
    local_score = {
        "kind": "val", "value": val_acc, "recorded": recorded, "delta": round(delta, 2) if delta is not None else None,
        "rows": seen[0], "table_url": val_url, "ok": val_ok, "tolerance_pp": VAL_TOLERANCE_PP,
    }
    rec.record["facts"]["local_score"] = local_score
    set_field("local_score", local_score)
    checks.append({
        "label": "checkpoint reproduces the recorded val accuracy", "ok": val_ok, "group": "Checkpoint",
        "detail": (f"{recorded:.2f} % recorded · {val_acc:.2f} % now" if isinstance(recorded, (int, float))
                   else f"no recorded accuracy · {val_acc:.2f} % now"),
    })
    rec.record["checks"] = checks
    set_checks(checks)
    log(("PASS " if val_ok else "FAIL ") + checks[-1]["label"] + f" — {checks[-1]['detail']}")
    if not val_ok:
        msg = (
            f"The checkpoint does not reproduce the run’s recorded val accuracy (recorded "
            f"{recorded if recorded is not None else '—'} %, now {val_acc:.2f} % on the locked val split). The checkpoint "
            "or the val table changed since training. Re-train, or pick another run."
        )
        raise PredictRefused(msg)

    # ── The test pass ───────────────────────────────────────────────────────
    n_test = len(sample_ids)
    log(f"Running inference: {Path(ck['weights']).name}, {n_test:,} test images, device={device}")
    progress("test", 0, n_test, force=True)
    preds: list[int] = [-1] * n_test
    confs: list[float] = [0.0] * n_test
    done = [0]

    def on_test(p: list[int], c: list[float], idx: Any) -> None:
        ix = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        for i, pv, cv in zip(ix, p, c, strict=True):
            preds[int(i)] = int(pv)
            confs[int(i)] = float(cv)
        done[0] += len(p)
        progress("test", done[0], n_test)

    test_loader = DataLoader(_TestImages(sample_ids, list(inputs["paths"]), val_tf), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=workers, pin_memory=False)
    if not _forward_batches(model, test_loader, device, on_test, is_cancelled):
        return _cancelled(rec, log, model)
    progress("test", n_test, n_test, force=True)
    del model
    reclaim_gpu_memory()
    if any(p < 0 for p in preds):
        msg = "Inference skipped some test images. Predict again."
        raise PredictRefused(msg)

    # ── The CSV: write, check, hash (the file on disk is what is checked and hashed) ──────
    out_dir = storage.plugin_home() / PREDICTIONS_DIR / ck["run_folder"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows = list(zip(sample_ids, preds, confs, strict=True))
    csv_path = out_dir / f"submission_{stamp}.csv"
    write_csv(csv_path, manifest.submission.columns, rows)
    header, csv_rows = read_csv_rows(csv_path)
    try:
        fmt = validate_submission(header, csv_rows, manifest, sample_ids)
    except PredictRefused:
        bad = out_dir / f"submission_{stamp}{INVALID_SUFFIX}"
        os.replace(csv_path, bad)
        log(f"Pre-flight failed; unvalidated CSV kept for debugging: {bad}")
        raise
    checks.extend(fmt)
    rec.record["checks"] = checks
    set_checks(checks)
    for c in fmt:
        log(("PASS " if c["ok"] else "FAIL ") + c["label"] + f" — {c['detail']}")
    csv_sha = sha256_of(csv_path)
    rec.record["facts"].update({"csv_path": str(csv_path), "csv_sha256": csv_sha, "rows": len(rows)})
    set_field("csv_path", str(csv_path))
    set_field("csv_sha256", csv_sha)
    log(f"submission.csv written: {csv_path} (sha256 {csv_sha[:12]}…)")

    sanity = distribution(preds, confs, manifest)
    rec.record["facts"]["sanity"] = sanity
    set_field("sanity", sanity)
    log("Distribution: " + " · ".join(f"{k} {v:,}" for k, v in sanity["per_class"].items()) +
        f" — mean confidence {sanity['mean_confidence']}, {sanity['low_confidence']:,} below {LOW_CONFIDENCE}")
    if sanity.get("warning"):
        log(f"WARNING: {sanity['warning']}")

    result = {
        "run_name": ck["run_name"], "run_folder": ck["run_folder"], "run_url": ck["run_url"], "weights": ck["weights"],
        "checkpoint_sha256": ck["sha256_recorded"], "csv_path": str(csv_path), "csv_sha256": csv_sha, "rows": len(rows),
        "sanity": sanity, "local_score": local_score, "checks": checks, "device": device, "cancelled": False,
    }
    rec.finish("completed", result=result)
    ledger.append({
        "kind": "predict", "job_id": rec.record["id"], **dict(manifest.provenance),
        "plugin_version": _plugin_version(), "train_job_id": ck["train_job_id"], "run_url": ck["run_url"],
        "run_name": ck["run_name"], "run_folder": ck["run_folder"],
        "checkpoint": {"path": ck["weights"], "sha256_recorded": ck["sha256_recorded"], "sha256_on_disk": ck["sha256_on_disk"],
                       "sha256_on_run": ck["sha256_on_run"]},
        "contract": ck["contract"],
        "test_inputs": {"count": inputs["count"], "files_json_sha256": inputs.get("files_json_sha256"),
                        "sample_submission_sha256": inputs.get("sample_submission_sha256")},
        "device": device, "csv": {"path": str(csv_path), "sha256": csv_sha, "rows": len(rows)},
        "checks": [(c["label"], bool(c["ok"])) for c in checks], "sanity": sanity, "local_score": local_score,
    })
    set_progress({"percent": 100.0, "label": "Done", "phase": "done", "images": n_test, "total_images": n_test})
    return result


def _cancelled(rec: _Record, log: Any, model: Any) -> dict[str, Any]:
    log("Inference stopped by cancellation request.")
    rec.finish("cancelled")
    del model
    reclaim_gpu_memory()
    return {"cancelled": True}


def _plugin_version() -> str:
    try:
        import kaggle_classification

        return str(kaggle_classification.__version__)
    except Exception:
        return "unknown"


# ── The submit job ─────────────────────────────────────────────────────────


def run_kaggle_submit(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """The Submit job: the predict record's CSV (re-hashed), the manifest's slug, the kaggle client's
    outcome, the D12 read-back, the submit record, the ledger entry, the outcome written back onto
    the predict record. A Kaggle rejection fails the job; the soft states complete it."""
    log = ctx.log
    set_field = getattr(ctx, "set_field", lambda k, v: None)
    set_progress = getattr(ctx, "set_progress", lambda p: None)
    job_id = str(getattr(ctx, "job_id", "") or "") or f"local-{int(time.time())}"

    predict_job_id = str(params.get("predict_job_id") or "").strip()
    if not predict_job_id:
        msg = "Missing required field 'predict_job_id'"
        raise PredictRefused(msg)
    ps = read_predict_record()
    if not ps or str(ps.get("id")) != predict_job_id or ps.get("status") != "completed":
        msg = "No validated prediction CSV found for that job. Run inference first."
        raise PredictRefused(msg)
    facts = ps.get("facts") or {}
    csv_path = str(facts.get("csv_path") or "")
    if not csv_path or not Path(csv_path).is_file():
        msg = f"The prediction's CSV is missing on disk ({csv_path or 'no path recorded'}). Run inference again."
        raise PredictRefused(msg)
    csv_sha = sha256_of(csv_path)
    if facts.get("csv_sha256") and csv_sha != facts.get("csv_sha256"):
        msg = "The prediction's CSV changed on disk since it was validated. Run inference again."
        raise PredictRefused(msg)
    run_name = str(facts.get("run_name") or "run")
    message = str(params.get("message") or "").strip() or f"{run_name} via 3LC plugin"
    slug = manifest.competition.slug
    daily_limit = int(manifest.submission.daily_limit)

    rec = _Record("submit_state", _new_record("kaggle_submit", job_id, {"predict_job_id": predict_job_id, "message": message}))
    rec.record["predict_job_id"] = predict_job_id
    rec.record["facts"].update({"run_name": run_name, "csv_path": csv_path, "csv_sha256": csv_sha, "slug": slug,
                                "predict_job_id": predict_job_id})
    rec.flush()
    for k in ("run_name", "csv_path", "csv_sha256", "slug", "predict_job_id"):
        set_field(k, rec.record["facts"][k])

    def rlog(line: str) -> None:
        rec.log(line)
        log(line)

    set_progress({"percent": 10.0, "label": "Submitting to Kaggle…", "phase": "submit"})
    rlog(f"Submitting {Path(csv_path).name} for {run_name!r} to {slug}: {message!r}")
    try:
        submission = kaggle_client.submit(csv_path, message, slug, daily_limit, rlog)
    except Exception as exc:
        submission = {"status": "failed", "reason": f"Kaggle rejected the submission: {exc}", "detail": str(exc)}
    rec.record["facts"]["submission"] = submission
    set_field("submission", submission)
    if submission["status"] != "submitted":
        rlog(f"Submit step: {submission['status']} — {submission.get('reason', '')}")

    # The outcome also lands on the predict record (ExDark writes facts.submission back) and in the ledger.
    try:
        ps_now = read_predict_record()
        if ps_now and str(ps_now.get("id")) == predict_job_id:
            ps_now.setdefault("facts", {})["submission"] = {
                "job_id": job_id, "status": submission.get("status"), "ref": submission.get("ref"),
                "kaggle": submission.get("kaggle"), "finished_at": time.time(),
            }
            session.save({"predict_state": ps_now})
    except Exception as exc:
        rlog(f"WARNING: could not write the outcome onto the predict record ({exc}).")
    ledger.append({
        "kind": "submit", "job_id": job_id, "predict_job_id": predict_job_id, "csv_sha256": csv_sha, "csv_path": csv_path,
        "run_name": run_name, "run_url": facts.get("run_url"), "checkpoint_sha256": facts.get("checkpoint_sha256"),
        **dict(manifest.provenance), "plugin_version": _plugin_version(),
        "slug": slug, "message": message,
        "kaggle": {"status": submission.get("status"), "ref": submission.get("ref"), "response": submission.get("response"),
                   "reason": submission.get("reason"), "detail": submission.get("detail"),
                   "submitted_at": time.time() if submission.get("status") == "submitted" else None,
                   **{f"read_back_{k}": v for k, v in (submission.get("kaggle") or {}).items()}},
    })
    result = {"run_name": run_name, "csv_path": csv_path, "csv_sha256": csv_sha, "predict_job_id": predict_job_id,
              "slug": slug, "submission": submission}
    if submission["status"] == "failed":
        rec.finish("failed", error=submission.get("reason") or "Kaggle rejected the submission.", result=result)
        raise PredictRefused(submission.get("reason") or "Kaggle rejected the submission.")
    rec.finish("completed", result=result)
    set_progress({"percent": 100.0, "label": "Done", "phase": "done"})
    return result


def kaggle_connection(manifest: Manifest) -> dict[str, Any]:
    """``GET /kaggle/connection``: the card's payload, slug and daily limit from the manifest."""
    return kaggle_client.connection(manifest.competition.slug, int(manifest.submission.daily_limit))


def state_for_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


__all__ = [
    "NOT_A_RUN", "SHA_MISMATCH", "PredictRefused", "assess_run", "csv_path_for", "distribution", "kaggle_connection",
    "list_runs", "predict_submit_state", "preflight", "resolve_checkpoint", "run_kaggle_submit", "run_predict",
    "test_inputs", "validate_submission", "write_csv",
]
