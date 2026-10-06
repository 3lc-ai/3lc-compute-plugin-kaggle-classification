# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""The Status tab's backend (docs/STATUS_MIRROR.md §2): the run history, the prediction / submission
history joined from the ledger with Kaggle's verdict per ref, the live Kaggle section, the Doctor
panel and the verification bundle.

Everything here reads: the session store's records, the ledger, the Run folders on disk and — only
on the explicit live paths — the Kaggle API (``GetSubmission`` by ref, the submissions list, the
leaderboard view). Nothing is rewritten: the ledger stays append-only, the verdicts Kaggle gives
after the fact are cached in memory per worker. No credential value is ever read or reported; the
bundle refuses to include a member that matches a secret pattern. Import-light: tlc, torch and the
kaggle client are imported inside functions.
"""

from __future__ import annotations

import io
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaggle_classification import importer, kaggle_client, kit, ledger, storage, trainer

if TYPE_CHECKING:
    from kaggle_classification.manifest import Manifest

LIVE_CACHE_S = 300.0        # a Kaggle verdict read live is kept this long before it is asked again
LIVE_CALLS_MAX = 10         # GetSubmission calls per history request, at most
LEADERBOARD_TOP = 5
BUNDLE_README = (
    "3LC Kaggle Classification plugin — verification bundle\n"
    "\n"
    "What this is: the records an organizer verifies a leaderboard entry against, exported by the\n"
    "plugin's Status tab. Every link in the chain is a hash or a URL an earlier stage produced:\n"
    "a Kaggle submission ref -> the submit entry (CSV sha256) -> the predict entry (checkpoint sha256,\n"
    "run URL, test-inputs hashes) -> the run's provenance (contract, train revision, locked val,\n"
    "checkpoint sha256 on the Run and on disk) -> the train revision chain -> the import record.\n"
    "\n"
    "Members: README.txt, plugin.json, manifest_provenance.json, import_record.json,\n"
    "train_revisions.json, runs/<job id>.json, predictions/<job id>.json, submissions/<job id>.json,\n"
    "ledger.jsonl (verbatim), and project/<project>/ — the 3LC records copied from the project folder in\n"
    "its own layout: datasets/<dataset>/tables/<revision>/object.3lc.json (+ row_cache.parquet where one\n"
    "exists) for every train revision in the seed lineage and for the locked val table; runs/<run>/\n"
    "object.3lc.json with the per-sample metrics tables (metrics_*/object.3lc.json + .parquet) for every\n"
    "run of the project; runs/<run>/model/best.pt for the runs the checkpoint rule selects; and\n"
    "files.json listing every copied file with its sha256, size and source path.\n"
    "\n"
    "Checkpoint rule: by default best.pt of at most two runs — the run behind the best public score and\n"
    "the run behind the most recent submission (one file when they are the same run) — each verified\n"
    "against the ledger's checkpoint sha256 (and the run record's and the Run's) before it is included;\n"
    "the participant may pick other submitted runs (at most two) in the Status tab; organizers can export\n"
    "with ?checkpoints=none or ?checkpoints=all. A checkpoint that fails verification is skipped and\n"
    "named in files.json.\n"
    "\n"
    "Never included: images, last.pt, prediction CSVs, Kaggle or 3LC credentials, the competition's\n"
    "answer keys.\n"
    "\n"
    "Machine-specific paths. The table and run records under project/<project>/ are byte-for-byte copies\n"
    "from the exporting machine's 3LC project root and contain that machine's absolute paths: the image\n"
    "column of each row_cache.parquet points at the starter kit images (kit version and shard hashes in\n"
    "manifest_provenance.json; the images are not included), and each run's parameters train_table_url /\n"
    "val_table_url point at <project root>/datasets/…, which is project/<project>/datasets/… in this\n"
    "bundle. The verification chain does not depend on these paths: every link is a sha256 or a URL\n"
    "compared as a string. To browse in the Dashboard, copy project/<project>/ into your own projects\n"
    "root; thumbnails are missing unless the kit sits at the recorded path or an alias maps it.\n"
    "\n"
    "GitHub: every member stays under GitHub's 100 MB per-file limit (a resnet18 best.pt is about 45 MB);\n"
    "the zip itself passes 100 MB with two checkpoints, so unzip it before committing the tree.\n"
)

CHECKPOINT_MODES = ("default", "none", "all", "selected")
CHECKPOINT_MAX = 2            # best.pt files a participant's export carries (the default and the checklist)
BUNDLE_KEEP_S = 3600.0        # exported zips under <plugin home>/bundles/ older than this are removed

# A member of the bundle that matches one of these is refused (the export aborts, never a partial zip).
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("kaggle access token", re.compile(r"KGAT_[A-Za-z0-9_\-]{8,}")),
    ("kaggle.json credential", re.compile(r'"key"\s*:\s*"[A-Za-z0-9]{16,}"')),
    ("kaggle api key assignment", re.compile(r"(?i)\bkaggle_key\b\s*[:=]\s*['\"]?[A-Za-z0-9]{16,}")),
    ("3lc api key file", re.compile(r"3lc_api_key")),
    ("judge mapping", re.compile(r"(?i)\bmapping\.csv\b")),
    ("answer key", re.compile(r"(?i)\bsolution[A-Za-z0-9_\-]*\.csv\b")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}")),
)


class BundleRefused(RuntimeError):
    """A member matched a secret pattern: the export is refused as a whole."""


# ── The run history (#7) ───────────────────────────────────────────────────────


def _run_folder(run_url: str, run_name: str) -> str:
    tail = str(run_url or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or str(run_name or "")


def run_history() -> list[dict[str, Any]]:
    """The current project's runs, newest first, with what the Runs table shows."""
    state = trainer.train_state()
    runs = [dict(r) for r in state.get("project_runs") or [] if isinstance(r, dict) and r.get("id")]
    cur = state.get("current")
    if (
        isinstance(cur, dict) and cur.get("id") and cur.get("status") == "running"
        and state.get("current_in_project") and not any(r.get("id") == cur["id"] for r in runs)
    ):
        runs.insert(0, trainer.run_summary(cur))
    rows = []
    for r in runs:
        weights = str(r.get("weights") or "")
        error = str(cur.get("error") or "") if isinstance(cur, dict) and cur.get("id") == r.get("id") else ""
        rows.append({
            "job_id": r.get("id"),
            "run_name": r.get("run_name") or "",
            "run_folder": _run_folder(str(r.get("run_url") or ""), str(r.get("run_name") or "")),
            "run_url": r.get("run_url"),
            "project_name": trainer.run_project(r),
            "status": r.get("status"),
            "error": error,
            "created_at": r.get("created_at"),
            "finished_at": r.get("finished_at"),
            "elapsed_s": r.get("elapsed_s"),
            "train_table_url": r.get("train_table_url"),
            "usable_rows": r.get("usable_rows"),
            "best_val_accuracy": r.get("best_val_accuracy"),
            "best_epoch": r.get("best_epoch"),
            "epochs_completed": r.get("epochs_completed"),
            "epochs_requested": r.get("epochs_requested"),
            "device": r.get("device"),
            "device_class": r.get("device_class"),
            "device_label": r.get("device_label"),
            "provenance_ok": bool(r.get("provenance_ok")),
            "weights_on_disk": bool(weights) and Path(weights).is_file(),
            "best_checkpoint_sha256": r.get("best_checkpoint_sha256") or "",
        })
    return rows


# ── The prediction / submission history (#8) ───────────────────────────────────

_live_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_live_lock = threading.Lock()


def _in_project(run_url: Any, project_name: Any = None) -> bool:
    return bool(trainer.runs_in_project([{"run_url": run_url, "project_name": project_name}]))


def _verdict_from_submit_entry(s: dict[str, Any]) -> dict[str, Any] | None:
    """The read-back stored at submit time (PREDICT D12), as the history's ``kaggle`` block."""
    k = s.get("kaggle") or {}
    if not isinstance(k, dict):
        return None
    status = str(k.get("read_back_status") or "").upper()
    if not status:
        return None
    return {
        "status": status,
        "public_score": k.get("read_back_public_score"),
        "error_description": k.get("read_back_error_description") or "",
        "date": k.get("read_back_date") or "",
        "source": "ledger",
    }


def _live_verdict(ref: str, api_holder: dict[str, Any], budget: dict[str, int]) -> dict[str, Any] | None:
    """One GetSubmission by ref, cached per worker for ``LIVE_CACHE_S``; None when it cannot be read."""
    now = time.time()
    with _live_lock:
        hit = _live_cache.get(ref)
        if hit and now - hit[0] < LIVE_CACHE_S:
            return dict(hit[1])
    if budget["left"] <= 0:
        return None
    if "api" not in api_holder:
        api, reason = kaggle_client.authenticated_api()
        api_holder["api"] = api
        api_holder["reason"] = reason
    api = api_holder.get("api")
    if api is None:
        return None
    budget["left"] -= 1
    try:
        s = kaggle_client.get_submission(api, ref)
    except Exception as exc:
        return {"status": "unknown", "error": f"{type(exc).__name__}: {exc}", "source": "live"}
    verdict = {
        "status": str(s.get("status") or "").upper(), "public_score": s.get("public_score"),
        "error_description": s.get("error_description") or "", "date": s.get("date") or "", "source": "live",
    }
    with _live_lock:
        _live_cache[ref] = (now, dict(verdict))
    return verdict


def _check_ok(c: Any) -> bool:
    """A check as the ledger writes it (``[label, ok]`` pairs, PREDICT §6) or as the records keep it (dicts)."""
    if isinstance(c, dict):
        return bool(c.get("ok"))
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return bool(c[1])
    return False


def _final(verdict: dict[str, Any] | None) -> bool:
    return bool(verdict) and str(verdict.get("status") or "").upper() in ("COMPLETE", "ERROR")


def prediction_history(manifest: Manifest, *, live: bool = False) -> list[dict[str, Any]]:
    """The History rows: the ledger's predict entries of the current project, newest first, each with
    its paired submission and Kaggle's verdict (stored, or read live by ref when ``live``), the
    public-score delta vs the previous Kaggle-scored submission, plus the latest predict record when it
    ended without a ledger entry (failed / interrupted / running)."""
    from kaggle_classification import predictor

    entries = ledger.read()
    submits_by_pid: dict[str, dict[str, Any]] = {}
    for s in entries:
        if s.get("kind") == "submit" and s.get("predict_job_id"):
            submits_by_pid[str(s["predict_job_id"])] = s   # file order: the newest attempt wins
    rows: list[dict[str, Any]] = []
    api_holder: dict[str, Any] = {}
    budget = {"left": LIVE_CALLS_MAX}
    for e in entries:
        if e.get("kind") != "predict" or not _in_project(e.get("run_url")):
            continue
        csv = e.get("csv") or {}
        score = e.get("local_score") or {}
        checks = e.get("checks") or []
        row: dict[str, Any] = {
            "job_id": e.get("job_id"),
            "ts": e.get("ts"),
            "run_name": e.get("run_name") or "",
            "run_folder": e.get("run_folder") or _run_folder(str(e.get("run_url") or ""), str(e.get("run_name") or "")),
            "run_url": e.get("run_url"),
            "train_job_id": e.get("train_job_id"),
            "status": "completed",
            "error": "",
            "csv_path": csv.get("path") or "",
            "csv_sha256": csv.get("sha256") or "",
            "rows": csv.get("rows"),
            "val_accuracy": score.get("value") if isinstance(score, dict) else None,
            "val_recorded": score.get("recorded") if isinstance(score, dict) else None,
            "checks_ok": bool(checks) and all(_check_ok(c) for c in checks),
            "checkpoint_sha256": (e.get("checkpoint") or {}).get("sha256_recorded") or "",
            "submission": None,
            "public_score": None,
            "delta": None,
        }
        s = submits_by_pid.get(str(e.get("job_id")))
        if s:
            k = s.get("kaggle") or {}
            sub: dict[str, Any] = {
                "job_id": s.get("job_id"), "ref": k.get("ref") or "", "status": k.get("status") or "",
                "reason": k.get("reason") or "", "message": s.get("message") or "", "ts": s.get("ts"),
                "csv_sha256": s.get("csv_sha256") or "", "kaggle": _verdict_from_submit_entry(s),
            }
            if sub["status"] == "submitted" and sub["ref"] and not _final(sub["kaggle"]) and live:
                fresh = _live_verdict(str(sub["ref"]), api_holder, budget)
                if fresh:
                    sub["kaggle"] = fresh
            v = sub["kaggle"] or {}
            if v.get("status") == "COMPLETE" and isinstance(v.get("public_score"), (int, float)):
                row["public_score"] = float(v["public_score"])
            row["submission"] = sub
        rows.append(row)
    # Δ on the public score vs the previous Kaggle-scored submission, oldest -> newest (ExDark's walk).
    last: float | None = None
    for row in rows:
        if isinstance(row["public_score"], float):
            row["delta"] = None if last is None else round(row["public_score"] - last, 5)
            last = row["public_score"]
    rows.reverse()
    # The latest predict record when it ended without a ledger entry: ExDark lists failed / interrupted /
    # running predictions too (its records are jobs, ours are the ledger + the one durable record).
    ps = predictor.read_predict_record()
    if ps and ps.get("status") != "completed" and not any(r["job_id"] == ps.get("id") for r in rows):
        facts = ps.get("facts") or {}
        if _in_project(facts.get("run_url"), facts.get("project_name")):
            status = predictor._orphaned(dict(ps), "inference").get("status")
            rows.insert(0, {
                "job_id": ps.get("id"), "ts": ps.get("created_at"), "run_name": facts.get("run_name") or "",
                "run_folder": facts.get("run_folder") or "", "run_url": facts.get("run_url"),
                "train_job_id": facts.get("train_job_id"), "status": status, "error": str(ps.get("error") or ""),
                "csv_path": "", "csv_sha256": "", "rows": None, "val_accuracy": None, "val_recorded": None,
                "checks_ok": False, "checkpoint_sha256": "", "submission": None, "public_score": None, "delta": None,
            })
    return rows


def best_public_score(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    for r in rows:
        s = r.get("public_score")
        if isinstance(s, float) and (best is None or s > best["score"]):
            best = {"score": s, "run_name": r.get("run_name"), "ref": (r.get("submission") or {}).get("ref")}
    return best


# ── Kaggle live (#9) ────────────────────────────────────────────────────────────


def _is_forbidden(text: str) -> bool:
    low = str(text).lower()
    return "403" in low or "forbidden" in low


def kaggle_live(manifest: Manifest) -> dict[str, Any]:
    """ExDark's ``kaggle_live_status`` with every call fenced so partial data still renders: the
    submissions list (403 while unlaunched), the leaderboard (empty while unlaunched) and the rank."""
    slug = manifest.competition.slug
    out: dict[str, Any] = {"slug": slug, "competition_url": kaggle_client.competition_url(slug)}
    if not kaggle_client.credentials_present():
        return {**out, "connected": False, "reason": "Connect your Kaggle account. " + kaggle_client.credentials_help()}
    api, reason = kaggle_client.authenticated_api()
    if api is None:
        return {**out, "connected": False, "reason": reason}
    out.update({"connected": True, "configured": True, "username": kaggle_client.api_username(api)})
    best: float | None = None
    try:
        subs = kaggle_client.list_submissions(api, slug, page_size=10)
        out["submissions"] = [
            {"ref": s["ref"], "date": s["date"], "status": s["status"], "public_score": s["public_score"],
             "message": s["description"]} for s in subs
        ]
        scores = [s["public_score"] for s in subs if isinstance(s["public_score"], float)]
        best = max(scores) if scores else None
    except Exception as exc:
        out["submissions_error"] = f"{type(exc).__name__}: {exc}"
    board: list[Any] = []
    try:
        board = list(api.competition_leaderboard_view(slug) or [])
        username = str(out.get("username") or "").lower()
        top = []
        rank = None
        for i, entry in enumerate(board[:50], start=1):
            team = str(getattr(entry, "team_name", "") or getattr(entry, "teamName", "") or "")
            score = getattr(entry, "score", None)
            if i <= LEADERBOARD_TOP:
                top.append({"rank": i, "team": team, "score": str(score) if score is not None else ""})
            if username and team.lower() == username:
                rank = {"rank": i, "team": team, "score": str(score) if score is not None else ""}
        out["leaderboard_top"] = top
        out["leaderboard_size"] = len(board)
        out["my_rank"] = rank
    except Exception as exc:
        out["leaderboard_error"] = f"{type(exc).__name__}: {exc}"
    if best is None:
        ledger_best = best_public_score(prediction_history(manifest, live=False))
        best = ledger_best["score"] if ledger_best else None
    out["best_public_score"] = best
    # The unlaunched event competition: the list is refused and the board is empty (probed 2026-10-05).
    out["launched"] = not (_is_forbidden(out.get("submissions_error", "")) and not board)
    return out


# ── Doctor (#11) ───────────────────────────────────────────────────────────────


def _dist_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def plugin_commit() -> str:
    """The installed dist's git commit (``direct_url.json``), else the checkout's HEAD for a folder
    source, else ``unknown``."""
    try:
        from importlib.metadata import distribution

        import kaggle_classification

        raw = distribution(kaggle_classification.DIST_NAME).read_text("direct_url.json")
        if raw:
            info = json.loads(raw)
            commit = str(((info.get("vcs_info") or {}).get("commit_id")) or "")
            if commit:
                return commit
            url = str(info.get("url") or "")
            if url.startswith("file://") and (info.get("dir_info") or {}).get("editable"):
                head = _git_head(Path(url[7:].lstrip("/")) if sys.platform == "win32" else Path(url[7:]))
                if head:
                    return head
    except Exception:
        pass
    head = _git_head(Path(__file__).resolve().parents[2])
    return head or "unknown"


def _git_head(path: Path) -> str:
    try:
        if not (path / ".git").exists():
            return ""
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _free_space(path: Path) -> dict[str, Any]:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        usage = shutil.disk_usage(str(p))
        return {"free_bytes": int(usage.free), "total_bytes": int(usage.total), "measured_at": str(p)}
    except Exception as exc:
        return {"free_bytes": None, "total_bytes": None, "error": f"{type(exc).__name__}: {exc}"}


def doctor(manifest: Manifest, *, kaggle: bool = True) -> dict[str, Any]:
    """The Doctor panel's facts (torch-free: versions from metadata, the device from the worker's probe).
    ``kaggle=False`` skips the network (the bundle's ``plugin.json`` and tests)."""
    import kaggle_classification
    from kaggle_classification import manifest as manifest_mod

    trainer.probe_device_async()
    probe = dict(trainer._device_probe)
    home = storage.describe()
    record = importer.read_record() or {}
    # The no-network resolution is cache / bundled by design; the background refresh (kicked here, as
    # GET /config kicks it) says whether the remote document was fetched. The row reports the effective
    # source, the fetched-at stamp and a failed fetch's error — not the stale local label (session 5 §3).
    refresh = manifest_mod.refresh_in_background()
    try:
        resolution = manifest_mod.resolve(network=False)
        prov = dict(resolution.manifest.provenance)
    except Exception as exc:
        prov = {"error": f"{type(exc).__name__}: {exc}"}
    cache_info = manifest_mod.cache_meta(manifest.competition.id) or {}
    prov["manifest_source_local"] = prov.get("manifest_source")
    if refresh.get("state") == "done" and refresh.get("source"):
        prov["manifest_source"] = refresh["source"]
    prov["manifest_fetched_at"] = prov.get("manifest_fetched_at") or cache_info.get("fetched_at")
    prov["refresh_state"] = refresh.get("state")
    prov["refresh_error"] = refresh.get("error")
    try:
        kit_state = kit.download_state(manifest)
    except Exception as exc:
        kit_state = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
    runs = trainer.read_state().get("runs") or []
    out: dict[str, Any] = {
        "plugin": {
            "id": storage.PLUGIN_ID, "version": kaggle_classification.__version__, "commit": plugin_commit(),
            "repository_url": kaggle_classification.REPOSITORY_URL,
        },
        "versions": {
            "sdk": _dist_version("3lc-compute-plugin-sdk"), "tlc": _dist_version("3lc"),
            "torch": _dist_version("torch"), "torchvision": _dist_version("torchvision"),
            "kaggle": _dist_version("kaggle"), "python": platform.python_version(),
            "platform": f"{sys.platform} · {platform.platform()}",
        },
        "device": {
            "probe": probe.get("state", "idle"), "device_class": probe.get("device_class"),
            "cuda_available": (probe.get("device_class") == "cuda") if probe.get("state") == "done" else None,
            "workers_default": trainer.default_workers(),
        },
        "manifest": {**prov, "refresh": manifest_mod.refresh_status(), "cache": cache_info or None},
        "kit": kit_state,
        "import": {
            "project_name": record.get("project_name"), "table_name": record.get("table_name"),
            "completed_at": record.get("completed_at"), "kit_version": record.get("kit_version"),
            "lineage_root": record.get("lineage_root"), "val_locked": (record.get("val_locked") or {}).get("url"),
        },
        "records": {
            "runs": len(runs), "project_runs": len(trainer.runs_in_project(runs)),
            "ledger_predict": len(ledger.read("predict")), "ledger_submit": len(ledger.read("submit")),
            "ledger_path": str(ledger.ledger_path()),
        },
        "plugin_home": {**home, "disk": _free_space(Path(home["path"])), "migrated_from": storage.migration_record()},
        "time": time.time(),
    }
    if kaggle:
        try:
            conn = kaggle_client.connection(manifest.competition.slug, int(manifest.submission.daily_limit))
            out["kaggle"] = {k: conn.get(k) for k in (
                "state", "username", "slug", "competition_url", "daily_limit", "kaggle_daily_limit",
                "submissions_used_today", "probe_error", "competition_title",
            )}
        except Exception as exc:
            out["kaggle"] = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}
    return out


# ── The verification bundle (#10; session 6: the tables, the runs and the selected checkpoints) ──


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str, sort_keys=False)


def _norm(url: Any) -> str:
    return str(url or "").replace("\\", "/").rstrip("/").lower()


def scan_for_secrets(name: str, text: str) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            msg = f"The export was refused: member {name!r} matches the {label} pattern. Nothing was exported."
            raise BundleRefused(msg)


def parquet_text(path: Path) -> str:
    """The string columns of a parquet file as one text (what a secret scan can read in a table's row
    cache or a run's metrics table); the raw bytes decoded as latin-1 when pyarrow cannot read it."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        parts: list[str] = []
        for name, col in zip(table.column_names, table.columns, strict=True):
            t = col.type
            if pa.types.is_string(t) or pa.types.is_large_string(t):
                parts.append(name)
                parts.extend(str(v) for v in col.to_pylist() if v)
        return "\n".join(parts)
    except Exception:
        try:
            return path.read_bytes().decode("latin-1", errors="ignore")
        except Exception:
            return ""


def _run_records(project: str) -> list[dict[str, Any]]:
    from kaggle_classification import predictor

    state = trainer.read_state()
    runs = [r for r in (state.get("runs") or []) if isinstance(r, dict) and r.get("id")]
    cur = state.get("current") if isinstance(state.get("current"), dict) else None
    out = []
    for r in trainer.runs_in_project(runs, project):
        weights = str(r.get("weights") or "")
        rec: dict[str, Any] = {
            "summary": r,
            "run_parameters_on_disk": trainer._run_parameters_on_disk(str(r.get("run_url") or "")),
            "best_checkpoint_sha256_recorded": r.get("best_checkpoint_sha256") or "",
            "best_checkpoint_sha256_on_disk_now": (
                predictor.sha256_of(weights) if weights and Path(weights).is_file() else ""
            ),
        }
        if cur and cur.get("id") == r.get("id"):
            rec["provenance_checks"] = cur.get("checks") or []
            rec["contract"] = (cur.get("result") or {}).get("contract")
            rec["params"] = cur.get("params")
        out.append(rec)
    return out


def _revision_chain(manifest: Manifest, project: str, seed_url: str) -> dict[str, Any]:
    try:
        used: dict[str, int] = {}
        for r in trainer.read_state().get("runs") or []:
            u = str(r.get("train_table_url") or "")
            if u:
                used[u] = used.get(u, 0) + 1
        listing = importer.list_project_tables(manifest, project, seed_url=seed_url, runs_used=used)
        train_ds = manifest.dataset_name("train")
        listing["datasets"] = [d for d in listing.get("datasets") or [] if d.get("name") == train_ds]
        return listing
    except Exception as exc:
        return {"project": project, "seed_url": seed_url, "error": f"{type(exc).__name__}: {exc}"}


def _local_dir(url: Any) -> Path | None:
    """The local folder a table / run URL names (None for a cloud URL or no URL)."""
    if not url:
        return None
    try:
        return trainer.run_local_dir(str(url))
    except Exception:
        raw = str(url)
        return None if "://" in raw and not raw.lower().startswith("file://") else Path(raw.replace("file://", ""))


def _rel_in_project(folder: Path, kind: str, fallback: str) -> str:
    """Where a table / run folder sits inside its project (``datasets/<ds>/tables/<rev>`` or
    ``runs/<run>``), read from the path's own tail; ``fallback`` when the layout is not the standard one."""
    parts = [p for p in folder.as_posix().split("/") if p]
    if kind == "table" and len(parts) >= 4 and parts[-4] == "datasets" and parts[-2] == "tables":
        return "/".join(parts[-4:])
    if kind == "run" and len(parts) >= 2 and parts[-2] == "runs":
        return "/".join(parts[-2:])
    return fallback


def _table_ancestry(url: str, seed_url: str) -> list[str]:
    """``url`` and its ancestors (``input_table_url`` / ``input_tables``, relative to the table folder)
    up to the seed — a torch-free JSON walk, at most 50 steps."""
    out: list[str] = []
    seen: set[str] = set()
    cur = str(url or "")
    while cur and _norm(cur) not in seen and len(out) < 50:
        seen.add(_norm(cur))
        out.append(cur)
        if seed_url and _norm(cur) == _norm(seed_url):
            break
        folder = _local_dir(cur)
        if folder is None:
            break
        try:
            d = json.loads((folder / "object.3lc.json").read_text(encoding="utf-8"))
        except Exception:
            break
        parent = d.get("input_table_url") or next(iter(d.get("input_tables") or []), None)
        if isinstance(parent, dict):
            parent = parent.get("url")
        if not parent:
            break
        parent = str(parent)
        cur = parent if ("://" in parent or Path(parent).is_absolute()) else (folder / parent).resolve().as_posix()
    return out


def _table_folder_files(folder: Path) -> list[Path]:
    """A table's 3LC records: its object file and the parquet files beside it — never an image."""
    out = [p for p in [folder / "object.3lc.json"] if p.is_file()]
    out += sorted(p for p in folder.glob("*.parquet") if p.is_file())
    return out


def _run_folder_files(folder: Path) -> list[Path]:
    """A run's 3LC records: the object file and the metrics tables (object + parquet); checkpoints apart."""
    out = [p for p in [folder / "object.3lc.json"] if p.is_file()]
    for m in sorted(p for p in folder.glob("metrics_*") if p.is_dir()):
        out += [p for p in [m / "object.3lc.json"] if p.is_file()]
        out += sorted(p for p in m.glob("*.parquet") if p.is_file())
    return out


def bundle_tables(manifest: Manifest, project: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    """The tables the bundle copies: every train revision in the seed lineage (the project listing when
    it is readable, plus the revisions the project's runs trained on and their ancestry) and the locked
    val table. Each: ``{url, dataset, name, role, folder}``; a table without a readable folder is listed
    with ``folder: None`` (nothing to copy, the record still names it)."""
    seed = str(((record.get("lineage_root") or {}).get("train_url")) or "")
    val = str(((record.get("val_locked") or {}).get("url"))
              or ((record.get("tables") or {}).get("val") or {}).get("url") or "")
    train_ds = manifest.dataset_name("train")
    tables: dict[str, dict[str, Any]] = {}

    def add(url: str, role: str, dataset: str = "", name: str = "") -> None:
        if not url or _norm(url) in tables:
            return
        folder = _local_dir(url)
        parts = [p for p in str(url).replace("\\", "/").rstrip("/").split("/") if p]
        tables[_norm(url)] = {
            "url": str(url), "role": role,
            "dataset": dataset or (parts[-3] if len(parts) >= 3 else ""), "name": name or (parts[-1] if parts else ""),
            "folder": str(folder) if folder is not None and folder.is_dir() else None,
        }

    add(seed, "seed", train_ds, "")
    chain = _revision_chain(manifest, project, seed) if record else {}
    for ds in chain.get("datasets") or []:
        for t in ds.get("tables") or []:
            if t.get("in_lineage"):
                add(str(t.get("url") or ""), "train revision", str(ds.get("name") or ""), str(t.get("name") or ""))
    runs = trainer.runs_in_project([r for r in trainer.read_state().get("runs") or [] if isinstance(r, dict)], project)
    for r in runs:
        for u in _table_ancestry(str(r.get("train_table_url") or ""), seed):
            add(u, "train revision", train_ds)
    add(val, "locked val", manifest.dataset_name("val"))
    return list(tables.values())


def eligible_checkpoint_runs(manifest: Manifest) -> list[dict[str, Any]]:
    """The project's SUBMITTED runs (a ``submit`` ledger entry with a Kaggle ref), newest submission first,
    each with its best public score, its latest submission's ref and time, the ledger's checkpoint sha256
    and the run record's best.pt — what the Status tab's checklist offers."""
    rows = prediction_history(manifest, live=False)
    runs_by_id = {str(r.get("id")): r for r in trainer.runs_in_project(
        [r for r in trainer.read_state().get("runs") or [] if isinstance(r, dict)])}
    by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        sub = row.get("submission") or {}
        if sub.get("status") != "submitted" or not sub.get("ref"):
            continue
        key = str(row.get("train_job_id") or row.get("run_url") or "")
        e = by_run.setdefault(key, {
            "train_job_id": str(row.get("train_job_id") or ""), "run_name": row.get("run_name") or "",
            "run_folder": row.get("run_folder") or "", "run_url": row.get("run_url"), "public_score": None,
            "submitted_at": None, "ref": "", "ledger_sha256": "", "submissions": 0,
        })
        e["submissions"] += 1
        ts = sub.get("ts") or row.get("ts")
        if ts and (e["submitted_at"] is None or float(ts) > float(e["submitted_at"])):
            e["submitted_at"] = float(ts)
            e["ref"] = str(sub.get("ref") or "")
            e["ledger_sha256"] = str(row.get("checkpoint_sha256") or e["ledger_sha256"])
        if not e["ledger_sha256"] and row.get("checkpoint_sha256"):
            e["ledger_sha256"] = str(row["checkpoint_sha256"])
        score = row.get("public_score")
        if isinstance(score, float) and (e["public_score"] is None or score > e["public_score"]):
            e["public_score"] = score
    for e in by_run.values():
        rec = runs_by_id.get(e["train_job_id"]) or {}
        e["weights"] = str(rec.get("weights") or "")
        e["record_sha256"] = str(rec.get("best_checkpoint_sha256") or "")
        p = Path(e["weights"]) if e["weights"] else None
        e["checkpoint_bytes"] = p.stat().st_size if p is not None and p.is_file() else None
    return sorted(by_run.values(), key=lambda e: -(e["submitted_at"] or 0))


def default_checkpoint_runs(eligible: list[dict[str, Any]]) -> list[str]:
    """The checkpoint rule: the run behind the best public score and the run behind the most recent
    submission — one id when they are the same run; empty when nothing was submitted."""
    picks: list[str] = []
    scored = [e for e in eligible if isinstance(e.get("public_score"), float)]
    if scored:
        best = max(scored, key=lambda e: (e["public_score"], e.get("submitted_at") or 0))
        picks.append(best["train_job_id"])
    if eligible:
        recent = max(eligible, key=lambda e: e.get("submitted_at") or 0)
        if recent["train_job_id"] not in picks:
            picks.append(recent["train_job_id"])
    return picks


def _verify_checkpoint(weights: str, expectations: dict[str, str]) -> dict[str, Any]:
    """best.pt on disk against every recorded sha256 handed in (``{"the ledger": …, "the run record": …,
    "the Run": …}``, empty values skipped): ``{ok, sha256, bytes, reason}``."""
    from kaggle_classification import predictor

    p = Path(weights) if weights else None
    if p is None or not p.is_file():
        return {"ok": False, "sha256": "", "bytes": None, "reason": "best.pt is not on disk"}
    sha = predictor.sha256_of(p)
    known = {k: v for k, v in expectations.items() if v}
    if not known:
        return {"ok": False, "sha256": sha, "bytes": p.stat().st_size,
                "reason": "no recorded checkpoint sha256 to verify against"}
    for who, expect in known.items():
        if sha != expect:
            return {"ok": False, "sha256": sha, "bytes": p.stat().st_size,
                    "reason": f"best.pt on disk ({sha[:12]}…) differs from {who}'s checkpoint sha256 ({expect[:12]}…)"}
    return {"ok": True, "sha256": sha, "bytes": p.stat().st_size, "reason": ""}


def _parse_runs(runs: Any) -> list[str]:
    if runs is None:
        return []
    if isinstance(runs, str):
        return [x.strip() for x in runs.split(",") if x.strip()]
    return [str(x).strip() for x in runs if str(x).strip()]


def bundle_plan(manifest: Manifest, *, checkpoints: str = "default", runs: Any = None) -> dict[str, Any]:
    """Everything the export will contain, before any byte is read: the text members, the files to copy
    (``{name, source, bytes, kind}``), the checkpoint decision and the checklist the Status tab renders.
    ``checkpoints``: ``default`` (the rule), ``selected`` (``runs`` = at most two submitted runs' train job
    ids), ``none`` / ``all`` (organizers). Raises ``ValueError`` on a bad mode or selection."""
    from kaggle_classification import manifest as manifest_mod

    mode = str(checkpoints or "default").strip().lower()
    if mode not in CHECKPOINT_MODES:
        msg = f"checkpoints must be one of {', '.join(CHECKPOINT_MODES)}, got {mode!r}"
        raise ValueError(msg)
    requested = _parse_runs(runs)
    record = importer.read_record() or {}
    project = str(record.get("project_name") or trainer.current_project() or manifest.default_project)
    seed_url = str(((record.get("lineage_root") or {}).get("train_url")) or "")
    doc = doctor(manifest, kaggle=False)
    exported_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text: dict[str, str] = {"README.txt": BUNDLE_README}
    text["plugin.json"] = _json({
        "plugin": doc["plugin"], "versions": doc["versions"], "device": doc["device"],
        "plugin_home": doc["plugin_home"], "exported_at": exported_at,
    })
    try:
        current_prov = dict(manifest_mod.resolve(network=False).manifest.provenance)
    except Exception as exc:
        current_prov = {"error": f"{type(exc).__name__}: {exc}"}
    text["manifest_provenance.json"] = _json({
        "current": current_prov, "cache": manifest_mod.cache_meta(manifest.competition.id),
        "import_record": record.get("manifest_provenance"),
    })
    text["import_record.json"] = _json(record)
    text["train_revisions.json"] = _json(_revision_chain(manifest, project, seed_url) if record else
                                         {"project": project, "error": "no import record"})
    run_records = _run_records(project)
    for rec in run_records:
        text[f"runs/{rec['summary'].get('id')}.json"] = _json(rec)
    for e in ledger.read("predict"):
        if _in_project(e.get("run_url")):
            text[f"predictions/{e.get('job_id')}.json"] = _json(e)
    for e in ledger.read("submit"):
        if _in_project(e.get("run_url")):
            text[f"submissions/{e.get('job_id')}.json"] = _json(e)
    path = ledger.ledger_path()
    text["ledger.jsonl"] = path.read_text(encoding="utf-8") if path.is_file() else ""

    # ── project/<project>/: the tables and the runs, in the project's own layout ──
    prefix = f"project/{project}/"
    files: list[dict[str, Any]] = []
    project_dir: str = ""
    tables = bundle_tables(manifest, project, record) if record else []
    for i, t in enumerate(tables):
        if not t.get("folder"):
            continue
        folder = Path(t["folder"])
        project_dir = project_dir or str(folder.parent.parent.parent.parent)
        rel = _rel_in_project(folder, "table", f"datasets/{t['dataset'] or 'dataset'}/tables/{t['name'] or f'table-{i}'}")
        for f in _table_folder_files(folder):
            files.append({"name": f"{prefix}{rel}/{f.name}", "source": str(f), "bytes": f.stat().st_size, "kind": "table"})
    runs_seen: list[dict[str, Any]] = []
    for rec in run_records:
        summary = rec["summary"]
        folder = _local_dir(summary.get("run_url"))
        if folder is None or not folder.is_dir():
            runs_seen.append({"id": summary.get("id"), "run_url": summary.get("run_url"), "folder": None})
            continue
        project_dir = project_dir or str(folder.parent.parent)
        rel = _rel_in_project(folder, "run", f"runs/{folder.name}")
        runs_seen.append({"id": summary.get("id"), "run_url": summary.get("run_url"), "folder": str(folder), "rel": rel})
        for f in _run_folder_files(folder):
            sub = f.relative_to(folder).as_posix()
            files.append({"name": f"{prefix}{rel}/{sub}", "source": str(f), "bytes": f.stat().st_size, "kind": "run"})

    # ── the checkpoints ──
    eligible = eligible_checkpoint_runs(manifest)
    default_ids = default_checkpoint_runs(eligible)
    by_id = {e["train_job_id"]: e for e in eligible}
    if mode == "default":
        chosen = list(default_ids)
    elif mode == "selected":
        if len(requested) > CHECKPOINT_MAX:
            msg = f"At most {CHECKPOINT_MAX} runs' checkpoints can be included (got {len(requested)})."
            raise ValueError(msg)
        unknown = [r for r in requested if r not in by_id]
        if unknown:
            msg = f"Not a submitted run of this project: {', '.join(unknown)}."
            raise ValueError(msg)
        chosen = list(dict.fromkeys(requested))
    elif mode == "all":
        chosen = [str(rec["summary"].get("id")) for rec in run_records if rec["summary"].get("weights")]
    else:
        chosen = []
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_sha: dict[str, str] = {}
    for run_id in chosen:
        e = by_id.get(run_id)
        rec = next((r for r in run_records if str(r["summary"].get("id")) == run_id), None)
        summary = (rec or {}).get("summary") or {}
        weights = (e or {}).get("weights") or str(summary.get("weights") or "")
        on_run = str(((rec or {}).get("run_parameters_on_disk") or {}).get("best_checkpoint_sha256") or "")
        expectations = {"the run record": str(summary.get("best_checkpoint_sha256") or ""), "the Run": on_run}
        if e is not None:
            expectations = {"the ledger": e.get("ledger_sha256") or "", **expectations}
        elif mode in ("default", "selected"):
            skipped.append({"train_job_id": run_id, "run_name": summary.get("run_name"), "reason": "not a submitted run"})
            continue
        v = _verify_checkpoint(weights, expectations)
        entry = {
            "train_job_id": run_id, "run_name": (e or {}).get("run_name") or summary.get("run_name"),
            "run_url": (e or {}).get("run_url") or summary.get("run_url"), "sha256": v["sha256"], "bytes": v["bytes"],
            "public_score": (e or {}).get("public_score"), "ref": (e or {}).get("ref"),
            "verified_against": [k for k, val in expectations.items() if val],
        }
        if not v["ok"]:
            skipped.append({**entry, "reason": v["reason"]})
            continue
        if v["sha256"] in seen_sha:
            included.append({**entry, "same_as": seen_sha[v["sha256"]], "name": None})
            continue
        folder = _local_dir(entry["run_url"])
        rel = _rel_in_project(folder, "run", f"runs/{folder.name}") if folder is not None else f"runs/{run_id}"
        name = f"{prefix}{rel}/model/best.pt"
        seen_sha[v["sha256"]] = name
        files.append({"name": name, "source": weights, "bytes": v["bytes"], "kind": "checkpoint", "sha256": v["sha256"]})
        included.append({**entry, "name": name})
    checklist = [{
        "train_job_id": e["train_job_id"], "run_name": e["run_name"], "run_folder": e["run_folder"],
        "public_score": e["public_score"], "submitted_at": e["submitted_at"], "ref": e["ref"],
        "submissions": e["submissions"], "checkpoint_bytes": e["checkpoint_bytes"],
        "available": bool(e["weights"]) and e["checkpoint_bytes"] is not None,
        "default": e["train_job_id"] in default_ids, "selected": e["train_job_id"] in chosen,
        "skipped_reason": next((s.get("reason") for s in skipped if s.get("train_job_id") == e["train_job_id"]), ""),
    } for e in eligible]
    text_bytes = sum(len(v.encode("utf-8")) for v in text.values())
    return {
        "project": project, "project_dir": project_dir, "exported_at": exported_at,
        "text_members": text, "files": files,
        "checkpoints": {"mode": mode, "requested": requested, "default_runs": default_ids, "included": included,
                        "skipped": skipped, "max": CHECKPOINT_MAX},
        "eligible_runs": checklist,
        "tables": [{k: v for k, v in t.items() if k != "folder"} | {"copied": bool(t.get("folder"))} for t in tables],
        "runs": runs_seen,
        "bytes": text_bytes + sum(int(f["bytes"] or 0) for f in files),
        "bytes_checkpoints": sum(int(f["bytes"] or 0) for f in files if f["kind"] == "checkpoint"),
        "counts": {
            "members": len(text) + len(files) + 1,   # + files.json
            "tables": sum(1 for t in tables if t.get("folder")),
            "runs": sum(1 for r in runs_seen if r.get("folder")),
            "metrics_tables": sum(1 for f in files if f["kind"] == "run" and f["name"].endswith(".parquet")),
            "checkpoints": sum(1 for f in files if f["kind"] == "checkpoint"),
        },
    }


def bundle_preview(manifest: Manifest, *, checkpoints: str = "default", runs: Any = None) -> dict[str, Any]:
    """``GET /status/bundle/preview``: the plan without its member texts — the size line and the checklist."""
    plan = bundle_plan(manifest, checkpoints=checkpoints, runs=runs)
    out = {k: v for k, v in plan.items() if k not in ("text_members", "files")}
    out["members"] = sorted(list(plan["text_members"]) + [f["name"] for f in plan["files"]]
                            + [f"project/{plan['project']}/files.json"])
    return out


def _bundles_dir() -> Path:
    d = storage.plugin_home() / "bundles"
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in d.glob("verification-bundle_*.zip"):
        try:
            if now - old.stat().st_mtime > BUNDLE_KEEP_S:
                old.unlink()
        except Exception:
            pass
    return d


def verification_bundle_file(manifest: Manifest, *, checkpoints: str = "default", runs: Any = None) -> tuple[Path, str]:
    """Write the bundle under ``<plugin home>/bundles/`` and return ``(path, file name)``. Every text member
    and every parquet member's string columns are scanned against ``SECRET_PATTERNS`` first (a match
    raises ``BundleRefused`` before any byte is zipped); checkpoints are verified by sha256 in the plan."""
    from kaggle_classification import predictor

    plan = bundle_plan(manifest, checkpoints=checkpoints, runs=runs)
    text = dict(plan["text_members"])
    for name, body in text.items():
        scan_for_secrets(name, body)
    for f in plan["files"]:
        scan_for_secrets(f["name"], f["name"])
        if f["name"].lower().endswith(".parquet"):
            scan_for_secrets(f["name"], parquet_text(Path(f["source"])))
        elif f["kind"] != "checkpoint":
            try:
                scan_for_secrets(f["name"], Path(f["source"]).read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    index = {
        "schema_version": 1, "project": plan["project"], "project_dir": plan["project_dir"],
        "exported_at": plan["exported_at"], "checkpoints": plan["checkpoints"], "tables": plan["tables"],
        "runs": plan["runs"],
        "files": [{"path": f["name"], "bytes": f["bytes"], "kind": f["kind"],
                   "sha256": f.get("sha256") or predictor.sha256_of(f["source"]), "source": f["source"]} for f in plan["files"]],
    }
    index_name = f"project/{plan['project']}/files.json"
    index_text = _json(index)
    scan_for_secrets(index_name, index_text)
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    name = f"verification-bundle_{manifest.competition.id}_{stamp}.zip"
    out = _bundles_dir() / name
    tmp = out.with_name(name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for member, body in text.items():
            zf.writestr(member, body.encode("utf-8"))
        zf.writestr(index_name, index_text.encode("utf-8"))
        for f in plan["files"]:
            zf.write(f["source"], arcname=f["name"],
                     compress_type=zipfile.ZIP_STORED if f["kind"] == "checkpoint" else zipfile.ZIP_DEFLATED)
    tmp.replace(out)
    return out, name


def verification_bundle(manifest: Manifest, *, checkpoints: str = "default", runs: Any = None) -> tuple[bytes, str]:
    """``(zip bytes, file name)`` — :func:`verification_bundle_file` read back (tests, small exports)."""
    path, name = verification_bundle_file(manifest, checkpoints=checkpoints, runs=runs)
    return path.read_bytes(), name


__all__ = [
    "CHECKPOINT_MAX",
    "CHECKPOINT_MODES",
    "SECRET_PATTERNS",
    "BundleRefused",
    "best_public_score",
    "bundle_plan",
    "bundle_preview",
    "bundle_tables",
    "default_checkpoint_runs",
    "doctor",
    "eligible_checkpoint_runs",
    "kaggle_live",
    "parquet_text",
    "plugin_commit",
    "prediction_history",
    "run_history",
    "scan_for_secrets",
    "verification_bundle",
    "verification_bundle_file",
]
