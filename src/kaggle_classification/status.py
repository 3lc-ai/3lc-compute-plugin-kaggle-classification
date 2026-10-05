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
    "ledger.jsonl (verbatim).\n"
    "\n"
    "Never included: images, table rows or any table data, checkpoints, prediction CSVs, Kaggle or 3LC\n"
    "credentials, the competition's answer keys.\n"
)

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
    try:
        resolution = manifest_mod.resolve(network=False)
        prov = dict(resolution.manifest.provenance)
    except Exception as exc:
        prov = {"error": f"{type(exc).__name__}: {exc}"}
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
        "manifest": {**prov, "refresh": manifest_mod.refresh_status(),
                     "cache": manifest_mod.cache_meta(manifest.competition.id)},
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
        "plugin_home": {**home, "disk": _free_space(Path(home["path"]))},
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


# ── The verification bundle (#10) ──────────────────────────────────────────────


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str, sort_keys=False)


def scan_for_secrets(name: str, text: str) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            msg = f"The export was refused: member {name!r} matches the {label} pattern. Nothing was exported."
            raise BundleRefused(msg)


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


def verification_bundle(manifest: Manifest) -> tuple[bytes, str]:
    """``(zip bytes, file name)``: the members of docs/STATUS_MIRROR.md §2, every text member scanned
    against ``SECRET_PATTERNS`` first (a match raises ``BundleRefused`` before any byte is zipped)."""
    from kaggle_classification import manifest as manifest_mod

    record = importer.read_record() or {}
    project = str(record.get("project_name") or trainer.current_project() or manifest.default_project)
    seed_url = str(((record.get("lineage_root") or {}).get("train_url")) or "")
    doc = doctor(manifest, kaggle=False)
    members: dict[str, str] = {"README.txt": BUNDLE_README}
    members["plugin.json"] = _json({
        "plugin": doc["plugin"], "versions": doc["versions"], "device": doc["device"],
        "plugin_home": doc["plugin_home"], "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    try:
        current_prov = dict(manifest_mod.resolve(network=False).manifest.provenance)
    except Exception as exc:
        current_prov = {"error": f"{type(exc).__name__}: {exc}"}
    members["manifest_provenance.json"] = _json({
        "current": current_prov, "cache": manifest_mod.cache_meta(manifest.competition.id),
        "import_record": record.get("manifest_provenance"),
    })
    members["import_record.json"] = _json(record)
    members["train_revisions.json"] = _json(_revision_chain(manifest, project, seed_url) if record else
                                            {"project": project, "error": "no import record"})
    for rec in _run_records(project):
        members[f"runs/{rec['summary'].get('id')}.json"] = _json(rec)
    for e in ledger.read("predict"):
        if _in_project(e.get("run_url")):
            members[f"predictions/{e.get('job_id')}.json"] = _json(e)
    for e in ledger.read("submit"):
        if _in_project(e.get("run_url")):
            members[f"submissions/{e.get('job_id')}.json"] = _json(e)
    path = ledger.ledger_path()
    members["ledger.jsonl"] = path.read_text(encoding="utf-8") if path.is_file() else ""
    for name, text in members.items():
        scan_for_secrets(name, text)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text.encode("utf-8"))
    stamp = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())
    return buf.getvalue(), f"verification-bundle_{manifest.competition.id}_{stamp}.zip"


__all__ = [
    "SECRET_PATTERNS",
    "BundleRefused",
    "best_public_score",
    "doctor",
    "kaggle_live",
    "plugin_commit",
    "prediction_history",
    "run_history",
    "scan_for_secrets",
    "verification_bundle",
]
