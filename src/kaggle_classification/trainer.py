# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Train stage (job kind ``train``): the locked from-scratch baseline on the imported tables.

Contract (docs/PLAN.md §B and §C "Trainer", docs/TRAIN_MIRROR.md, decisions D1–D14 of 2026-09-29):

* Model: the Intel kit's, exactly (part A of the 2026-09-29 review, which reversed the timm
  decision): torchvision ``resnet18(weights=None)`` with ``fc = Identity`` and the kit's MLP head
  (512 → 256 → ReLU → Dropout 0.3 → 128 → ReLU → Dropout 0.3 → N), torchvision's default init.
  ``manifest.model.backbone`` / ``head`` name it and are allowlisted (``manifest.BACKBONES`` /
  ``HEADS``); ``pretrained`` is never a parameter. Image size, the optimizer (Adam, D3) and the LR
  schedule (StepLR 5 / 0.1, D2) are locked; the Run records every locked fact.
* RNG parity with the kit: ``set_seed`` (random, numpy, torch, cuda, cudnn deterministic, no
  benchmark, PYTHONHASHSEED) first, then tables → transforms → sampler → loaders → model →
  criterion → optimizer → scheduler → the Run, in the kit's order; the model and the optimizer are
  created from the freshly seeded state and nothing draws from torch's RNG before the first
  training batch. A CPU retry re-seeds and starts over, so the stream a participant's run follows
  is the kit's for the same seed.
* Data: ``tlc.Table.from_url`` (``.latest()`` when asked); the train revision must descend from the
  import record's seed; the val table is the import record's LOCKED URL. Undefined rows
  (``label == manifest.undefined_label_id``) are treated as weight 0 regardless of their weight
  column; weights are read into memory and never written back (no cap, D14). The sampler is tlc's
  ``WeightedRandomSampler`` semantics (draw ∝ weight, weight 0 never drawn, epoch = non-zero rows),
  built here from the in-memory effective weights. Zero usable rows is a refusal, never a fallback.
* Params: ``manifest.training.defaults`` ⊕ the form, bounded on the merged kwargs by
  ``manifest.training.bounds`` (seed and workers bounds are plugin constants when the manifest has
  none); the augmentation is the Intel kit's torchvision recipe (D11); the seed and cudnn
  determinism follow the kit (D6).
* Per epoch: train loss, val loss, val accuracy (``tlc.log``, the progress channel); best by val
  accuracy (strict ``>``) saved atomically as ``<run>/model/best.pt``, ``last.pt`` every epoch (D12).
* End of training only: per-sample metrics on ALL train rows and all val rows — ``predicted``
  (class map), ``confidence``, ``prob_<class>``, ``loss`` (NaN for undefined rows, never fabricated),
  3-D ``embeddings`` (UMAP fit on train, val transformed; PCA fallback) — via ``run.add_metrics``.
* Robustness: a durable train record in the session store (``train_state``), worker-pid orphan
  detection (a compute restart mid-run reads back as ``stale``), sleep gaps logged, cooperative
  cancel that keeps the best-so-far checkpoint (D10), CPU retry when the accelerator fails at
  model or first-batch time, a duplicate-start guard (running record + client token). Tables are
  never modified.

Import-light: torch / torchvision / tlc / numpy / PIL are imported inside functions. Everything the host
request path calls (``training_facts``, ``build_train_kwargs``, ``train_state``) stays torch-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaggle_classification import benchmark, importer, session

if TYPE_CHECKING:
    from kaggle_classification.manifest import Manifest

# ── Locked baseline facts (plugin constants the manifest does not carry; D2, D3) ────────────
OPTIMIZER = "adam"
SCHEDULE: dict[str, Any] = {"kind": "step", "step_size": 5, "gamma": 0.1}
INFERENCE = "single forward pass"
# ImageNet normalisation, spelled out like the kit.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AFFINE_SHEAR = 10
AFFINE_SCALE = (0.8, 1.2)

# Bounds the manifest may not carry: plugin facts, served with the manifest's in ``_meta``.
SEED_BOUNDS_FALLBACK: tuple[int, int] = (0, 2**31 - 1)
WORKERS_BOUNDS: tuple[int, int] = (0, 16)
WORKERS_MAX_AUTO = 4
BOUND_MESSAGE = "{key} must be between {lo} and {hi} (got {val})."
# The bounded manifest fields, in form order; ``seed`` and ``workers`` join them at validation.
MANIFEST_FIELDS = ("epochs", "batch_size", "lr", "weight_decay")
INT_FIELDS = frozenset({"epochs", "batch_size", "seed", "workers"})

CHECKPOINT_BEST = "best.pt"
CHECKPOINT_LAST = "last.pt"
MODEL_DIR = "model"

HEARTBEAT_S = 10.0          # the record's heartbeat cadence during an epoch
SLEEP_GAP_S = 120.0         # a heartbeat gap longer than this is logged as a probable sleep
STALE_HEARTBEAT_S = 900.0   # a same-pid running record older than this is treated as dead
BATCH_FLUSH_S = 1.0         # ExDark: batch progress lands at most ~1/s
LOG_KEEP = 300              # log lines kept in the durable record
RUNS_KEEP = 20              # finished runs kept for the ETA history and the run list
LINEAGE_MAX_STEPS = 200

_DEVICE_ERROR_RE = re.compile(r"cuda|cudnn|mps|out of memory|device", re.IGNORECASE)


class TrainRefused(RuntimeError):
    """A participant-facing refusal (bad params, wrong table, nothing to train on). The job fails
    with the message verbatim; no Run is created."""


# ── Facts served to the fragment (torch-free) ──────────────────────────────


def default_workers() -> int:
    """Device-aware default: 0 on Windows (the safe default), ``min(4, cpu_count)`` elsewhere."""
    if sys.platform == "win32":
        return 0
    return max(0, min(WORKERS_MAX_AUTO, os.cpu_count() or 1))


def effective_bounds(manifest: Manifest) -> dict[str, tuple[float, float]]:
    """The manifest's bounds plus the plugin's seed / workers bounds where the manifest has none."""
    bounds = {k: (float(v[0]), float(v[1])) for k, v in manifest.training.bounds.items()}
    bounds.setdefault("seed", (float(SEED_BOUNDS_FALLBACK[0]), float(SEED_BOUNDS_FALLBACK[1])))
    bounds.setdefault("workers", (float(WORKERS_BOUNDS[0]), float(WORKERS_BOUNDS[1])))
    return bounds


def effective_defaults(manifest: Manifest) -> dict[str, Any]:
    out = dict(manifest.training.defaults)
    out.setdefault("seed", SEED_BOUNDS_FALLBACK[0])
    out["workers"] = default_workers()
    out["optimizer"] = OPTIMIZER
    return out


def _dist_version(name: str) -> str:
    """The installed version of a heavy dependency, "" outside the heavy venv (metadata only)."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return ""


def framework_versions() -> dict[str, str]:
    """The framework the model is built with; recorded on the Run, served to the fragment."""
    return {"torch_version": _dist_version("torch"), "torchvision_version": _dist_version("torchvision")}


HEAD_LABELS = {
    "kit_mlp_512_256_128_d03": "the kit's MLP head (512 → 256 → 128, dropout 0.3)",
    "linear": "a linear head",
}


def model_facts(manifest: Manifest) -> dict[str, Any]:
    """Display strings for the locked model rows (the fragment renders, never defines)."""
    m = manifest.model
    return {
        "backbone": m.backbone,
        "head": m.head,
        "arch": m.arch,
        "backbone_label": f"torchvision {m.arch}",
        "head_label": HEAD_LABELS.get(m.head, m.head),
        "init_label": "random (pretrained = false, torchvision default init)",
        **framework_versions(),
    }


def training_facts(manifest: Manifest) -> dict[str, Any]:
    """What ``GET /config`` serves under ``_meta.training``: the fragment renders these and defines
    nothing (defaults, bounds, the locked optimizer and schedule, the benchmark for the ETA)."""
    bounds = effective_bounds(manifest)
    return {
        "defaults": effective_defaults(manifest),
        "bounds": {k: [v[0], v[1]] for k, v in bounds.items()},
        "optimizer": OPTIMIZER,
        "schedule": dict(SCHEDULE),
        "inference": INFERENCE,
        "model": model_facts(manifest),
        "workers_default": default_workers(),
        "platform": sys.platform,
        "benchmark": benchmark.facts(),
        "max_rows": {"train": manifest.expected_rows("train")},
        "embeddings": {
            "method": manifest.training.embeddings.method,
            "fallback": manifest.training.embeddings.fallback,
            "n_components": manifest.training.embeddings.n_components,
        },
    }


# ── Params ─────────────────────────────────────────────────────────────────


def _coerce(key: str, raw: Any, default: Any) -> Any:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = default
    try:
        if key in INT_FIELDS:
            f = float(raw)
            if f != int(f):
                msg = f"{key} must be a whole number (got {raw!r})."
                raise TrainRefused(msg)
            return int(f)
        return float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"Invalid value for {key}: {raw!r} (expected a number)."
        raise TrainRefused(msg) from exc


def _check_bound(key: str, val: float, bounds: dict[str, tuple[float, float]]) -> None:
    lo, hi = bounds.get(key, (None, None))
    if lo is None:
        return
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not (lo <= val <= hi):
        shown = (int(lo) if float(lo).is_integer() else lo, int(hi) if float(hi).is_integer() else hi)
        raise TrainRefused(BOUND_MESSAGE.format(key=key, lo=shown[0], hi=shown[1], val=val))


def default_run_name(manifest: Manifest) -> str:
    return f"{manifest.competition.id}_run_{time.strftime('%Y%m%d_%H%M%S')}"


def build_train_kwargs(params: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """Manifest defaults ⊕ the form, bounded on the MERGED values; the locked facts are added last
    so nothing a participant sends can change them. Raises ``TrainRefused`` (participant-facing)."""
    defaults = effective_defaults(manifest)
    bounds = effective_bounds(manifest)
    kwargs: dict[str, Any] = {}
    for key in (*MANIFEST_FIELDS, "seed", "workers"):
        kwargs[key] = _coerce(key, params.get(key), defaults.get(key))
    for key, val in kwargs.items():
        _check_bound(key, val, bounds)
    optimizer = str(params.get("optimizer") or OPTIMIZER).strip().lower()
    if optimizer != OPTIMIZER:
        msg = f"optimizer is locked to {OPTIMIZER} for this competition (got {optimizer!r})."
        raise TrainRefused(msg)
    device = str(params.get("device") or "").strip()
    run_name = str(params.get("run_name") or "").strip() or default_run_name(manifest)
    project = str(params.get("project_name") or session.populated_session(manifest)["project_name"]).strip()
    for label, value in (("run name", run_name), ("project name", project)):
        if not value or "/" in value or "\\" in value or value in (".", ".."):
            msg = f"The {label} must be a plain name (no slashes), got {value!r}."
            raise TrainRefused(msg)
    use_latest = params.get("use_latest", True)
    if isinstance(use_latest, str):
        use_latest = use_latest.strip().lower() not in ("false", "0", "no", "")
    return {
        **kwargs,
        "optimizer": OPTIMIZER,
        "schedule": dict(SCHEDULE),
        "device": device,
        "run_name": run_name,
        "project_name": project,
        "use_latest": bool(use_latest),
        "train_table_url": str(params.get("train_table_url") or "").strip().strip('"'),
        "client_token": str(params.get("client_token") or "").strip(),
        # Locked, from the manifest — merged last, never from the form.
        "backbone": manifest.model.backbone,
        "head": manifest.model.head,
        "arch": manifest.model.arch,
        "image_size": int(manifest.model.image_size),
        "pretrained": False,
    }


def validate_train_url(url: str, manifest: Manifest) -> None:
    """Split identity (DP-11): the train slot must hold a table of the train dataset."""
    if not url:
        msg = "Missing the train table URL. Run Import (tab 1) first."
        raise TrainRefused(msg)
    expected = manifest.dataset_name("train")
    got = session.url_dataset(url)
    if got != expected:
        msg = (
            f"Train table URL points at {got or 'an unrecognized dataset'}; expected {expected}. Pick the "
            f"{expected} table (the revision picker only offers the matching split), or clear the field to "
            "restore the default."
        )
        raise TrainRefused(msg)


# ── Tables, rows, lineage (tlc) ─────────────────────────────────────────────


def _url_exists(url: str) -> bool:
    import tlc

    try:
        return bool(tlc.Url(url).exists())
    except Exception:
        return False


def resolve_train_table(url: str, use_latest: bool) -> tuple[Any, str, bool]:
    """``(table, resolved_url, followed_latest)``: the revision this run trains on."""
    import tlc

    table = tlc.Table.from_url(tlc.Url(str(url).strip().strip('"')))
    followed = False
    if use_latest:
        newest = importer.latest_url(str(table.url))
        if not _same_url(newest, str(table.url)):
            followed = True
            table = tlc.Table.from_url(tlc.Url(newest))
    return table, str(table.url), followed


def _parent_url(table: Any) -> str | None:
    """The table this revision was derived from (tlc's lineage: ``input_table_url`` on FromTable
    subclasses such as the Dashboard's EditedTable, else the first ``input_tables`` entry)."""
    import tlc

    raw = getattr(table, "input_table_url", None)
    if not raw:
        inputs = getattr(table, "input_tables", None) or []
        raw = inputs[0] if inputs else None
    if not raw:
        return None
    try:
        return str(tlc.Url(str(raw)).to_absolute(table.url))
    except Exception:
        return str(raw)


def _same_url(a: str, b: str) -> bool:
    return importer._norm(a) == importer._norm(b)


def descends_from_seed(table: Any, seed_url: str) -> bool | None:
    """Walk the lineage from ``table`` back to ``seed_url``. True when reached, False when the walk
    ends elsewhere, None when a step cannot be read (the gate names it, never refuses on it)."""
    import tlc

    cur = table
    for _ in range(LINEAGE_MAX_STEPS):
        if _same_url(str(cur.url), seed_url):
            return True
        parent = _parent_url(cur)
        if not parent:
            return False
        try:
            cur = tlc.Table.from_url(tlc.Url(parent))
        except Exception:
            return None
    return None


def lineage_steps(table: Any, seed_url: str) -> int | None:
    """Revision steps from ``table`` back to the seed (0 = the seed itself)."""
    import tlc

    cur = table
    for steps in range(LINEAGE_MAX_STEPS):
        if _same_url(str(cur.url), seed_url):
            return steps
        parent = _parent_url(cur)
        if not parent:
            return None
        try:
            cur = tlc.Table.from_url(tlc.Url(parent))
        except Exception:
            return None
    return None


def scan_rows(table: Any, manifest: Manifest) -> tuple[list[int], list[float]]:
    """``(labels, weights)`` read once from ``table.table_rows`` — the row view, no image decoding."""
    labels: list[int] = []
    weights: list[float] = []
    for row in table.table_rows:
        labels.append(int(row["label"]))
        weights.append(float(row["weight"]))
    return labels, weights


def effective_weights(labels: list[int], weights: list[float], manifest: Manifest) -> list[float]:
    """The sampler's weights: the table's, with undefined rows forced to 0 (in memory only)."""
    undefined = int(manifest.undefined_label_id)
    return [0.0 if lab == undefined else max(0.0, w) for lab, w in zip(labels, weights, strict=True)]


def summarize_rows(labels: list[int], weights: list[float], manifest: Manifest) -> dict[str, Any]:
    """The usable-row summary the gate shows (docs/TRAIN_MIRROR.md #9–#10)."""
    undefined = int(manifest.undefined_label_id)
    eff = effective_weights(labels, weights, manifest)
    per_class = dict.fromkeys(manifest.class_names, 0)
    names = manifest.class_names
    for lab, w in zip(labels, eff, strict=True):
        if w > 0 and 0 <= lab < len(names):
            per_class[names[lab]] += 1
    labeled_in_use = sum(1 for w in eff if w > 0)
    return {
        "total": len(labels),
        "labeled_in_use": labeled_in_use,
        "excluded_undefined": sum(1 for lab in labels if lab == undefined),
        "excluded_zero_weight": sum(1 for lab, w in zip(labels, weights, strict=True) if lab != undefined and w <= 0),
        "undefined_with_weight": sum(1 for lab, w in zip(labels, weights, strict=True) if lab == undefined and w > 0),
        "per_class": per_class,
        "classes_without_rows": [n for n in names if per_class[n] == 0],
        "sum_effective_weight": float(sum(eff)),
    }


def _table_label_names(table: Any) -> list[str]:
    vm = table.get_value_map("label") or {}
    return [
        str(vm[k].get("internal_name") if isinstance(vm[k], dict) else getattr(vm[k], "internal_name", vm[k]))
        for k in sorted(vm)
    ]


def _revision_info(url: str, manifest: Manifest, seed_url: str) -> dict[str, Any]:
    import tlc

    table = tlc.Table.from_url(tlc.Url(url))
    labels, weights = scan_rows(table, manifest)
    return {
        "url": str(table.url),
        "rows": len(labels),
        "descends_from_seed": descends_from_seed(table, seed_url) if seed_url else None,
        "steps_from_seed": lineage_steps(table, seed_url) if seed_url else None,
        "summary": summarize_rows(labels, weights, manifest),
        "label_names": _table_label_names(table),
    }


def preflight(data: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """The Train form's read-only gate (``GET /train/preflight``): what ExDark's ``/import/revisions``
    returned (exists, row counts, latest URL) PLUS the seed lineage, the usable-row summary and the
    class coverage for BOTH the base revision and its latest, so the fragment re-evaluates the gate
    when the Use-latest checkbox flips without a second fetch. Never writes."""
    url = str(data.get("train_url") or "").strip().strip('"')
    record = importer.read_record() or {}
    seed_url = str(((record.get("lineage_root") or {}).get("train_url")) or "")
    out: dict[str, Any] = {
        "url": url,
        "import_state": "success" if record else "empty",
        "seed_url": seed_url,
        "val_locked_url": str(((record.get("val_locked") or {}).get("url")) or ""),
        "expected_dataset": manifest.dataset_name("train"),
        "max_rows": manifest.expected_rows("train"),
        "class_names": manifest.class_names,
    }
    if not url:
        out.update({"exists": False, "error": "train_url is required"})
        return out
    try:
        validate_train_url(url, manifest)
    except TrainRefused as exc:
        out.update({"exists": _url_exists(url), "split_ok": False, "error": str(exc)})
        return out
    out["split_ok"] = True
    if not _url_exists(url):
        out["exists"] = False
        return out
    out["exists"] = True
    try:
        base = _revision_info(url, manifest, seed_url)
    except Exception as exc:
        out.update({"error": f"Could not read the table: {type(exc).__name__}: {exc}"})
        return out
    out["base"] = base
    out["row_count"] = base["rows"]
    latest_url = importer.latest_url(base["url"], timeout=importer._LATEST_TIMEOUT_ROUTE_S)
    out["latest_url"] = latest_url
    if _same_url(latest_url, base["url"]):
        out["latest"] = base
    else:
        try:
            out["latest"] = _revision_info(latest_url, manifest, seed_url)
        except Exception as exc:
            out["latest"] = {"url": latest_url, "error": f"{type(exc).__name__}: {exc}"}
    out["latest_row_count"] = (out["latest"] or {}).get("rows")
    out["has_revisions"] = not _same_url(latest_url, base["url"])
    label_map = record.get("label_map") or {}
    expected_names = [label_map[k] for k in sorted(label_map, key=int)]
    out["label_map_ok"] = (base["label_names"] == expected_names) if expected_names else None
    return out


# ── The durable train record (session store ``train_state``) ────────────────


def read_state() -> dict[str, Any]:
    data = session.load().get("train_state")
    return data if isinstance(data, dict) else {}


def _write_state(state: dict[str, Any]) -> None:
    session.save({"train_state": state})


def read_record() -> dict[str, Any] | None:
    cur = read_state().get("current")
    return cur if isinstance(cur, dict) and cur.get("id") else None


def _new_record(job_id: str, kw: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "id": job_id,
        "kind": "train",
        "status": "running",
        "pid": os.getpid(),
        "created_at": now,
        "started_at": now,
        "heartbeat": now,
        "finished_at": None,
        "cancelled": False,
        "params": {k: v for k, v in kw.items() if k not in ("client_token",)},
        "client_token": kw.get("client_token") or "",
        "progress": {"epoch": 0, "total_epochs": int(kw["epochs"]), "history": [], "batch_i": 0, "batch_n": 0},
        "facts": {},
        "checks": [],
        "result": None,
        "error": None,
        "log": [],
        "gaps": [],
    }


class _Recorder:
    """Owns the durable record: batched writes, log tail, heartbeat, terminal states."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self._lock = threading.Lock()
        self._last_write = 0.0

    def log(self, line: str) -> None:
        with self._lock:
            self.record["log"].append(line)
            del self.record["log"][:-LOG_KEEP]

    def touch(self, *, force: bool = False) -> None:
        """Heartbeat + flush, at most every ``HEARTBEAT_S`` unless forced."""
        now = time.time()
        with self._lock:
            self.record["heartbeat"] = now
            if not force and now - self._last_write < HEARTBEAT_S:
                return
            self._last_write = now
            self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._last_write = time.time()
            self._flush_locked()

    def _flush_locked(self) -> None:
        state = read_state()
        state["current"] = self.record
        _write_state(state)

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
            state = read_state()
            state["current"] = self.record
            runs = [r for r in (state.get("runs") or []) if isinstance(r, dict) and r.get("id") != self.record["id"]]
            runs.insert(0, run_summary(self.record))
            state["runs"] = runs[:RUNS_KEEP]
            _write_state(state)


def run_summary(record: dict[str, Any]) -> dict[str, Any]:
    """The compact per-run entry kept for the ETA history and the Predict run list (session 4)."""
    facts = record.get("facts") or {}
    progress = record.get("progress") or {}
    result = record.get("result") or {}
    checks = record.get("checks") or []
    return {
        "id": record.get("id"),
        "run_name": facts.get("run_name") or (record.get("params") or {}).get("run_name") or "",
        "run_url": facts.get("run_url"),
        "project_name": facts.get("project_name"),
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "finished_at": record.get("finished_at"),
        "epochs_requested": progress.get("total_epochs"),
        "epochs_completed": progress.get("epoch"),
        "best_epoch": result.get("best_epoch"),
        "best_val_accuracy": result.get("best_val_accuracy"),
        "weights": facts.get("weights") or "",
        "best_checkpoint_sha256": facts.get("best_checkpoint_sha256") or "",
        "last_checkpoint_sha256": facts.get("last_checkpoint_sha256") or "",
        "device_class": facts.get("device_class"),
        "device": facts.get("device"),
        "usable_rows": (facts.get("usable") or {}).get("labeled_in_use"),
        "avg_epoch_s": progress.get("avg_epoch_s"),
        "epoch_s_per_row": result.get("epoch_s_per_row"),
        "collect_s": result.get("collect_s"),
        "collect_rows": result.get("collect_rows"),
        "provenance_ok": bool(checks) and all(c.get("ok") for c in checks),
        "train_table_url": facts.get("train_table_url"),
        "val_table_url": facts.get("val_table_url"),
        "contract": {k: (record.get("params") or {}).get(k)
                     for k in ("backbone", "head", "arch", "image_size", "pretrained", "seed")},
    }


def _mark_if_orphaned(record: dict[str, Any]) -> dict[str, Any]:
    """A ``running`` record owned by another worker process is PROOF the run died with that worker
    (ExDark's pid rule): the host kills its plugin workers on shutdown and never resumes a job. A
    same-pid record without a heartbeat for ``STALE_HEARTBEAT_S`` is treated the same way."""
    if record.get("status") != "running":
        return record
    same_pid = record.get("pid") == os.getpid()
    hb = float(record.get("heartbeat") or record.get("started_at") or 0)
    if same_pid and time.time() - hb < STALE_HEARTBEAT_S:
        return record
    record["status"] = "stale"
    record["error"] = (
        "Interrupted: the compute service restarted while this run was training."
        if not same_pid
        else "Interrupted: the training thread stopped reporting progress."
    )
    record.setdefault("log", []).append(
        f"Marked stale on read: recorded owner pid {record.get('pid')} is not the current worker "
        f"({os.getpid()})" if not same_pid else "Marked stale on read: no heartbeat for "
        f"{int(time.time() - hb)} s."
    )
    if record.get("finished_at") is None:
        record["finished_at"] = time.time()
    state = read_state()
    state["current"] = record
    runs = [r for r in (state.get("runs") or []) if isinstance(r, dict) and r.get("id") != record["id"]]
    runs.insert(0, run_summary(record))
    state["runs"] = runs[:RUNS_KEEP]
    _write_state(state)
    return record


def train_state() -> dict[str, Any]:
    """``GET /train/state``: the current record (orphan-checked), the finished-run history for the
    ETA and the run list, the best checkpoint re-verified on disk. Torch-free."""
    state = read_state()
    current = state.get("current") if isinstance(state.get("current"), dict) else None
    if current and current.get("id"):
        current = _mark_if_orphaned(dict(current))
        weights = str((current.get("facts") or {}).get("weights") or "")
        current["weights_on_disk"] = bool(weights) and Path(weights).is_file()
    runs = [r for r in (state.get("runs") or []) if isinstance(r, dict)]
    return {
        "state": (current or {}).get("status") or "empty",
        "current": current,
        "runs": runs,
        "device_class": _device_probe.get("device_class"),
        "device_probe": _device_probe.get("state", "idle"),
        "workers_default": default_workers(),
    }


# ── Device probe (off the request path) ─────────────────────────────────────

_device_probe: dict[str, Any] = {"state": "idle", "device_class": None}
_probe_lock = threading.Lock()


def resolve_device(raw: Any) -> str:
    """Resolve the participant's Device field to a ``torch.device`` string.

    Blank = auto: CUDA -> MPS -> CPU. A bare index (``"0"``) means that CUDA
    device; any other non-blank string passes through (``"cpu"``, ``"cuda:1"``).
    Must NOT be called from the host request path (validation stays torch-free).
    """
    s = str(raw).strip() if raw is not None else ""
    if s.isdigit():
        return f"cuda:{int(s)}"
    if s:
        return s
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def device_class(device: str) -> str:
    d = str(device).lower()
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


def probe_device_async() -> dict[str, Any]:
    """Resolve the auto device on a thread once per worker process (importing torch inside a
    route is the cold-worker timeout the ExDark plugin hit); routes serve the cached answer."""
    with _probe_lock:
        if _device_probe["state"] in ("running", "done"):
            return dict(_device_probe)
        _device_probe["state"] = "running"

    def _run() -> None:
        try:
            cls = device_class(resolve_device(""))
        except Exception:
            cls = "cpu"
        with _probe_lock:
            _device_probe.update({"state": "done", "device_class": cls})

    threading.Thread(target=_run, name="kaggle-classification-device-probe", daemon=True).start()
    return dict(_device_probe)


# ── Checkpoints ────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_local_dir(run_url: str) -> Path | None:
    """The run folder on the local filesystem (None for a cloud run folder)."""
    import tlc

    try:
        abs_url = str(tlc.Url(run_url).to_absolute())
    except Exception:
        abs_url = str(run_url)
    if "://" in abs_url and not abs_url.lower().startswith("file://"):
        return None
    return Path(abs_url[7:] if abs_url.lower().startswith("file://") else abs_url)


def save_checkpoint(run_url: str, state_dict: dict[str, Any], name: str) -> tuple[str, str]:
    """Write ``<run>/model/<name>`` ATOMICALLY (``.tmp`` + ``os.replace``) so a cancel or crash
    mid-write never leaves a half-written file under a name Predict trusts. Returns (path, sha256).
    A cloud run folder goes through the SDK helper (no atomic rename there)."""
    import torch

    local = run_local_dir(run_url)
    if local is None:
        from tlc_plugin_sdk.shared.model_storage import save_model_to_run

        rel = save_model_to_run(run_url, state_dict, filename=name)
        return rel, ""
    model_dir = local / MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / name
    tmp = model_dir / f"{name}.tmp"
    torch.save(state_dict, str(tmp))
    os.replace(tmp, dest)
    return str(dest), _sha256_file(dest)


# ── Data pipeline (the kit's recipe) ────────────────────────────────────────


class _SampleTransform:
    """Picklable ``TableView`` transform: ``{image, label, weight}`` row -> ``(tensor, label)``.
    Top-level so DataLoader workers > 0 can pickle it."""

    def __init__(self, transform: Any, image_column: str = "image", label_column: str = "label") -> None:
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, dict):
            image = _as_rgb_image(sample[self.image_column])
            label = int(sample.get(self.label_column, 0))
        else:
            image = _as_rgb_image(sample[0])
            label = int(sample[1]) if len(sample) > 1 else 0
        return self.transform(image), label


def _as_rgb_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = str(value)
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        from tlc_plugin_sdk.shared.images import load_image

        return load_image(path)


def build_transforms(image_size: int) -> tuple[Any, Any]:
    """The Intel kit's torchvision recipe, verbatim (D11): resize(shorter side) -> random crop ->
    horizontal flip -> affine(shear 10, scale 0.8–1.2) -> ImageNet normalize; val/collection:
    resize -> center crop -> normalize."""
    from torchvision import transforms as tv

    train = tv.Compose([
        tv.Resize(image_size),
        tv.RandomCrop(image_size),
        tv.RandomHorizontalFlip(),
        tv.RandomAffine(0, shear=AFFINE_SHEAR, scale=AFFINE_SCALE),
        tv.ToTensor(),
        tv.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = tv.Compose([
        tv.Resize(image_size),
        tv.CenterCrop(image_size),
        tv.ToTensor(),
        tv.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


def set_seed(seed: int) -> None:
    """The kit's determinism procedure (D6), in the kit's order."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init(worker_id: int) -> None:
    """DataLoader worker seeding for ``workers > 0`` (the kit never uses workers, so this is the one
    place the plugin goes beyond it): torch seeds each worker from the base seed itself; ``random``
    and numpy are seeded here from that per-worker torch seed so the PIL/numpy paths are reproducible."""
    import random

    import numpy as np
    import torch

    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_model(backbone: str, head: str, num_classes: int) -> Any:
    """The manifest's model, allowlisted: torchvision ``resnet18(weights=None)`` with either the Intel
    kit's MLP head (``fc = Identity`` + 512 → 256 → ReLU → Dropout 0.3 → 128 → ReLU → Dropout 0.3 → N,
    exactly ``intel-kit/train.py``'s ``ResNet18Classifier``) or torchvision's own linear ``fc``.
    Module creation order matches the kit so the seeded init draws the same numbers."""
    import torch
    from torch import nn
    from torchvision import models

    if backbone != "torchvision_resnet18":
        msg = f"backbone {backbone!r} is not allowed"
        raise TrainRefused(msg)
    if head == "kit_mlp_512_256_128_d03":

        class ResNet18Classifier(nn.Module):
            """The kit's model: ResNet-18 backbone, ``fc`` replaced by Identity, the MLP head on top."""

            def __init__(self, num_classes: int) -> None:
                super().__init__()
                self.resnet = models.resnet18(weights=None)
                resnet_features = self.resnet.fc.in_features
                self.resnet.fc = nn.Identity()
                self.classifier = nn.Sequential(
                    nn.Linear(resnet_features, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )

            def features(self, x: torch.Tensor) -> torch.Tensor:
                return self.resnet(x)

            def head(self, feats: torch.Tensor) -> torch.Tensor:
                return self.classifier(feats)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.resnet(x))

        return ResNet18Classifier(num_classes)
    if head == "linear":

        class ResNet18Linear(nn.Module):
            def __init__(self, num_classes: int) -> None:
                super().__init__()
                self.resnet = models.resnet18(weights=None)
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

            def features(self, x: torch.Tensor) -> torch.Tensor:
                m = self.resnet
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
                return torch.flatten(m.avgpool(x), 1)

            def head(self, feats: torch.Tensor) -> torch.Tensor:
                return self.resnet.fc(feats)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.resnet(x)

        return ResNet18Linear(num_classes)
    msg = f"head {head!r} is not allowed"
    raise TrainRefused(msg)


def build_sampler(eff: list[float]) -> Any:
    """tlc's ``create_weighted_sampler`` semantics from the in-memory effective weights: draw
    probability ∝ weight, weight 0 never drawn, epoch length = non-zero rows, with replacement.
    The table's own column is never touched. Raises ``TrainRefused`` on zero usable rows."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    n = sum(1 for w in eff if w > 0)
    if n == 0:
        msg = "No usable rows: every row is undefined or at weight 0. Label images in the Dashboard first."
        raise TrainRefused(msg)
    return WeightedRandomSampler(weights=torch.as_tensor(eff, dtype=torch.double), num_samples=n, replacement=True)


# ── Provenance ─────────────────────────────────────────────────────────────


def get_run_parameters(run: Any) -> dict[str, Any]:
    constants = getattr(run, "constants", None)
    if isinstance(constants, dict) and isinstance(constants.get("parameters"), dict):
        return constants["parameters"]
    params = getattr(run, "parameters", None)
    return params if isinstance(params, dict) else {}


def check_provenance(run_url: str, manifest: Manifest, expected: dict[str, Any]) -> list[dict[str, Any]]:
    """The Run's own record proves the locked contract — eight checks read back from the Run."""
    import tlc

    run = tlc.Run.from_url(tlc.Url(run_url))
    p = get_run_parameters(run)
    best_sha = str(p.get("best_checkpoint_sha256") or "")
    best_path = str(expected.get("weights") or "")
    on_disk = _sha256_file(Path(best_path)) if best_path and Path(best_path).is_file() else ""
    pretrained = p.get("pretrained")
    tv_version = expected.get("torchvision_version")
    train_ok = bool(p.get("train_table_url")) and _same_url(
        str(p.get("train_table_url")), str(expected.get("train_table_url"))
    )
    val_ok = bool(p.get("val_table_url")) and _same_url(str(p.get("val_table_url")), str(expected.get("val_table_url")))
    short = lambda s: (s[:12] + "...") if s else "(missing)"  # noqa: E731 - a two-use formatter
    sha_detail = f"best_checkpoint_sha256={short(best_sha)}" + (
        "" if best_sha == on_disk else f", on disk {short(on_disk)}"
    )
    return [
        {"label": f"run records backbone == {manifest.model.backbone}",
         "ok": p.get("backbone") == manifest.model.backbone, "detail": f"backbone={p.get('backbone')!r}"},
        {"label": f"run records head == {manifest.model.head}", "ok": p.get("head") == manifest.model.head,
         "detail": f"head={p.get('head')!r}"},
        {"label": f"run records image_size == {manifest.model.image_size}",
         "ok": p.get("image_size") == manifest.model.image_size, "detail": f"image_size={p.get('image_size')!r}"},
        {"label": "run records pretrained == False (random init)", "ok": pretrained in (False, "False", 0),
         "detail": f"pretrained={pretrained!r}"},
        {"label": f"run records torchvision_version == {tv_version}",
         "ok": bool(tv_version) and p.get("torchvision_version") == tv_version,
         "detail": f"torchvision_version={p.get('torchvision_version')!r}, torch_version={p.get('torch_version')!r}"},
        {"label": "run records the seed",
         "ok": isinstance(p.get("seed"), int) and p.get("seed") == expected.get("seed"),
         "detail": f"seed={p.get('seed')!r}"},
        {"label": "run records the train revision trained on", "ok": train_ok,
         "detail": f"train_table_url={str(p.get('train_table_url') or '')[-48:]!r}"},
        {"label": "run records val_table_url == the locked val", "ok": val_ok,
         "detail": f"val_table_url={str(p.get('val_table_url') or '')[-48:]!r}"},
        {"label": "run records the best checkpoint sha256, equal to the file on disk",
         "ok": bool(best_sha) and best_sha == on_disk, "detail": sha_detail},
    ]


# ── The job ────────────────────────────────────────────────────────────────


@dataclass
class _Loop:
    """Mutable training-loop state shared by the helpers of ``run_training``."""

    epoch: int = 0
    total: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    batch_i: int = 0
    batch_n: int = 0
    train_start: float = 0.0
    last_flush: float = 0.0
    last_hb: float = 0.0
    best_val: float = -1.0
    best_epoch: int = 0
    best_path: str = ""
    best_sha: str = ""
    last_path: str = ""
    last_sha: str = ""
    cancelled: bool = False


def _device_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and bool(_DEVICE_ERROR_RE.search(str(exc)))


def run_training(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """The Train job. Raises ``TrainRefused`` (participant-facing, no Run created) on a refusal;
    returns ``{"cancelled": True, ...}`` when stopped; records everything in ``train_state``."""
    import tlc

    log = ctx.log
    set_progress = getattr(ctx, "set_progress", lambda p: None)
    set_field = getattr(ctx, "set_field", lambda k, v: None)
    set_checks = getattr(ctx, "set_checks", lambda c: None)
    is_cancelled = getattr(ctx, "is_cancelled", lambda: False)
    job_id = str(getattr(ctx, "job_id", "") or "") or f"local-{int(time.time())}"

    kw = build_train_kwargs(params, manifest)

    # ── Duplicate-start guards (a double-clicked Start never trains twice) ────────────────
    existing = read_record()
    if existing:
        existing = _mark_if_orphaned(dict(existing))
        if existing.get("status") == "running":
            busy = (existing.get("facts") or {}).get("run_name") or existing.get("id")
            msg = f"A training run is already in progress ({busy}). Wait for it to finish or cancel it first."
            raise TrainRefused(msg)
        if kw["client_token"] and kw["client_token"] == str(existing.get("client_token") or ""):
            msg = "This Start request was already used by the previous run (a double click?). Press Start again."
            raise TrainRefused(msg)

    # ── The import record: the seed lineage, the locked val, the label map ─────────────
    record_imp = importer.read_record()
    if not record_imp:
        msg = "Import the kit first: no import record exists on this machine."
        raise TrainRefused(msg)
    seed_url = str((record_imp.get("lineage_root") or {}).get("train_url") or "")
    val_url = str((record_imp.get("val_locked") or {}).get("url") or "")
    label_map = record_imp.get("label_map") or {}
    train_url = kw["train_table_url"] or str((record_imp.get("tables") or {}).get("train", {}).get("url") or "")
    validate_train_url(train_url, manifest)
    if not _url_exists(train_url):
        msg = f"Train table not found on disk: {train_url}. Run Import (tab 1) first."
        raise TrainRefused(msg)
    if not val_url or not _url_exists(val_url):
        msg = f"The locked val table is missing on disk ({val_url or 'no URL recorded'}). Run Import (tab 1) again."
        raise TrainRefused(msg)

    train_table, resolved_url, followed = resolve_train_table(train_url, kw["use_latest"])
    if followed:
        log(f"train: following latest revision {resolved_url}")
    lineage = descends_from_seed(train_table, seed_url) if seed_url else None
    if lineage is False:
        msg = (
            f"The train table revision {resolved_url} does not descend from the imported seed "
            f"({seed_url}). Pick a revision of the imported train table."
        )
        raise TrainRefused(msg)
    expected_names = [label_map[k] for k in sorted(label_map, key=int)] if label_map else []
    names = _table_label_names(train_table)
    if expected_names and names != expected_names:
        msg = f"The train table's label map ({', '.join(names)}) is not the imported one ({', '.join(expected_names)})."
        raise TrainRefused(msg)
    val_table = tlc.Table.from_url(tlc.Url(val_url))
    log(f"train: {resolved_url} ({train_table.row_count} rows)")
    log(f"val (locked): {val_url} ({val_table.row_count} rows)")

    labels, weights = scan_rows(train_table, manifest)
    summary = summarize_rows(labels, weights, manifest)
    eff = effective_weights(labels, weights, manifest)
    if summary["labeled_in_use"] == 0:
        msg = "No usable rows: every row is undefined or at weight 0. Label images in the Dashboard first."
        raise TrainRefused(msg)
    log(
        f"rows: {summary['labeled_in_use']:,} labeled in use · {summary['excluded_undefined']:,} excluded as "
        f"undefined · {summary['excluded_zero_weight']:,} excluded at weight 0"
    )
    if summary["undefined_with_weight"]:
        log(f"{summary['undefined_with_weight']:,} unlabeled rows have weight > 0 and are skipped until labeled.")
    if summary["classes_without_rows"]:
        log("No usable rows for: " + ", ".join(summary["classes_without_rows"]) + ". Those classes cannot be learned.")

    # ── The record: written before any Run exists (a restart from here on reads back as stale) ──
    rec = _Recorder(_new_record(job_id, kw))
    rec.record["facts"].update({
        "run_name": kw["run_name"], "project_name": kw["project_name"], "train_table_url": resolved_url,
        "train_table_requested": train_url, "val_table_url": val_url, "usable": summary,
        "use_latest": kw["use_latest"], "device_requested": kw["device"],
    })
    rec.flush()
    for k in ("run_name", "project_name", "train_table_url", "val_table_url"):
        set_field(k, rec.record["facts"][k])
    set_field("usable", summary)

    def rlog(line: str) -> None:
        rec.log(line)
        log(line)

    try:
        result = _train_and_collect(kw, rec, rlog, set_progress, set_field, set_checks, is_cancelled,
                                    manifest, train_table, val_table, resolved_url, val_url, eff, summary)
    except TrainRefused as exc:
        rec.finish("failed", error=str(exc))
        raise
    except Exception as exc:
        rec.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return result


def _train_and_collect(
    kw: dict[str, Any], rec: _Recorder, log: Any, set_progress: Any, set_field: Any, set_checks: Any,
    is_cancelled: Any, manifest: Manifest, train_table: Any, val_table: Any, resolved_url: str,
    val_url: str, eff: list[float], summary: dict[str, Any],
) -> dict[str, Any]:
    import tlc
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    backbone, head, arch = kw["backbone"], kw["head"], kw["arch"]
    image_size, seed = kw["image_size"], int(kw["seed"])
    epochs, batch_size, workers = int(kw["epochs"]), int(kw["batch_size"]), int(kw["workers"])
    fw = framework_versions()
    # The kit's order from here: seed → tables → transforms → sampler → loaders → model → criterion →
    # optimizer → scheduler → the Run → the loop. Nothing between set_seed and the model draws RNG.
    set_seed(seed)
    device_requested = kw["device"]
    device = resolve_device(device_requested)
    st = _Loop(total=epochs)

    train_tf, val_tf = build_transforms(image_size)
    train_view = train_table.with_transform(_SampleTransform(train_tf))
    train_eval_view = train_table.with_transform(_SampleTransform(val_tf))
    val_view = val_table.with_transform(_SampleTransform(val_tf))
    sampler = build_sampler(eff)   # raises TrainRefused; never falls back to plain shuffling
    n_usable = len(sampler)

    def loaders() -> tuple[Any, Any]:
        winit = worker_init if workers > 0 else None
        tl = DataLoader(train_view, batch_size=batch_size, sampler=sampler, num_workers=workers,
                        drop_last=False, pin_memory=False, worker_init_fn=winit)
        vl = DataLoader(val_view, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False,
                        worker_init_fn=winit)
        return tl, vl


    criterion = nn.CrossEntropyLoss()
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    train_loader: Any = None
    val_loader: Any = None

    rng_after_init: list[Any] = []

    def build_stack(dev: str) -> None:
        """The kit's creation order from a freshly seeded state: loaders (no RNG), model (the seeded
        init), criterion, optimizer, scheduler. The RNG state after the init is captured so the
        first-batch probe below can hand the loop exactly the stream the kit's loop would see."""
        nonlocal model, optimizer, scheduler, train_loader, val_loader
        set_seed(seed)
        train_loader, val_loader = loaders()
        model = build_model(backbone, head, manifest.num_classes).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(kw["lr"]), weight_decay=float(kw["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(SCHEDULE["step_size"]), gamma=float(SCHEDULE["gamma"])
        )
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        rng_after_init[:] = [torch.get_rng_state(), cuda_state]

    def announce_device(dev: str, note: str = "") -> None:
        rec.record["facts"].update({"device": dev, "device_class": device_class(dev)})
        set_field("device", dev)
        set_field("device_class", device_class(dev))
        log(f"Device: {dev}{' (auto)' if not device_requested else ''}{note}")

    def flush_progress(*, force: bool = False, stage_note: str | None = None, stage: str | None = None) -> None:
        now = time.time()
        if not force and now - st.last_flush < BATCH_FLUSH_S:
            return
        st.last_flush = now
        within = (st.batch_i / st.batch_n) if st.batch_n else 0.0
        percent = 100.0 * min(1.0, (st.epoch + within) / max(st.total, 1))
        payload: dict[str, Any] = {
            "percent": round(percent, 2), "label": f"Epoch {st.epoch}/{st.total}" if st.epoch else "Training: starting",
            "phase": "train", "epoch": st.epoch, "total_epochs": st.total, "history": list(st.history),
            "batch_i": st.batch_i, "batch_n": st.batch_n,
        }
        if st.epoch:
            avg = (now - st.train_start) / max(st.epoch, 1)
            payload["avg_epoch_s"] = round(avg, 1)
            payload["eta_s"] = round(avg * max(st.total - st.epoch, 0))
        if stage:
            payload["stage"] = stage
            payload["label"] = stage_note or payload["label"]
        if stage_note:
            payload["stage_note"] = stage_note
        rec.record["progress"] = {k: v for k, v in payload.items() if k not in ("percent", "label", "phase")}
        set_progress(payload)

    def heartbeat() -> None:
        now = time.time()
        if st.last_hb and now - st.last_hb > SLEEP_GAP_S:
            gap = int(now - st.last_hb)
            rec.record["gaps"].append({"at": now, "seconds": gap})
            log(f"No progress for {gap // 60} min — the machine slept? Training continues.")
        st.last_hb = now
        rec.touch()

    # ── Model + first batch, with the CPU retry (accelerator failures at model / first-batch time) ──
    fallback_reason = ""
    attempt_device = device
    while True:
        try:
            build_stack(attempt_device)
            announce_device(attempt_device)
            model.train()
            images, labels_b = next(iter(train_loader))
            images, labels_b = images.to(attempt_device), labels_b.to(attempt_device)
            optimizer.zero_grad()
            criterion(model(images), labels_b).backward()
            optimizer.zero_grad()
            break
        except Exception as exc:
            if attempt_device != "cpu" and _device_error(exc):
                fallback_reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                log(f"Accelerator failed ({fallback_reason}). Retrying on CPU from epoch 1.")
                rec.record["facts"]["device_fallback_reason"] = fallback_reason
                set_field("device_fallback_reason", fallback_reason)
                try:
                    del model, optimizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                attempt_device = "cpu"
                continue
            raise
    device = attempt_device
    # The probe drew from the RNG (a sampler draw, the augmentation, dropout); put the generators back
    # to the state right after the init so the loop's stream is the kit's for this seed.
    torch.set_rng_state(rng_after_init[0])
    if rng_after_init[1] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_after_init[1])

    # ── The Run: under the same project root the importer writes to (the tests' seam) ────────
    run = tlc.init(
        project_name=kw["project_name"], run_name=kw["run_name"],
        description=f"{manifest.competition.display_name} — {arch} from scratch, kaggle-classification",
        root_url=importer.project_root_url(),
    )
    run_url = str(run.url)
    rec.record["facts"]["run_url"] = run_url
    set_field("run_url", run_url)
    contract = {
        "backbone": backbone, "head": head, "arch": arch, "image_size": image_size, "pretrained": False,
        "torch_version": fw["torch_version"], "torchvision_version": fw["torchvision_version"], "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "lr": float(kw["lr"]), "weight_decay": float(kw["weight_decay"]),
        "optimizer": OPTIMIZER, "schedule": f"step(step_size={SCHEDULE['step_size']}, gamma={SCHEDULE['gamma']})",
        "train_table_url": resolved_url, "val_table_url": val_url, "usable_rows": n_usable,
        "undefined_excluded": int(summary["excluded_undefined"]),
        "sampler": "weighted, exclude_zero_weights, undefined forced to 0 (in memory)",
        "augmentation": "resize/random-crop/hflip/affine(shear 10, scale 0.8-1.2)/imagenet-normalize",
        "plugin": "kaggle-classification", "job_id": rec.record["id"],
        **{k: v for k, v in (manifest.provenance or {}).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    run.set_parameters(contract)
    log(
        f"Locked: backbone={backbone} · head={head} · image_size={image_size} · pretrained=False · "
        f"torchvision {fw['torchvision_version']} · seed {seed}. "
        f"Training {epochs} epochs, batch {batch_size}, lr {kw['lr']}, weight decay {kw['weight_decay']}, "
        f"{OPTIMIZER}, StepLR({SCHEDULE['step_size']}, {SCHEDULE['gamma']}), workers {workers}."
    )
    log(f"Run: {run_url}")
    run.set_parameters(
        {"device": device, "device_requested": device_requested, "device_fallback_reason": fallback_reason}
    )
    rec.record["facts"]["workers"] = workers
    set_field("workers", workers)

    # ── Epochs ──────────────────────────────────────────────────────────────
    st.train_start = time.time()
    st.last_hb = st.train_start
    st.batch_n = len(train_loader)
    flush_progress(force=True)
    epoch_times: list[float] = []
    for epoch in range(1, epochs + 1):
        if is_cancelled():
            st.cancelled = True
            break
        t_epoch = time.time()
        model.train()
        st.batch_i = 0
        st.batch_n = len(train_loader)
        running_loss, seen = 0.0, 0
        for images, labels_b in train_loader:
            if is_cancelled():
                st.cancelled = True
                break
            images, labels_b = images.to(device), labels_b.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels_b)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(images.size(0))
            seen += int(images.size(0))
            st.batch_i += 1
            heartbeat()
            flush_progress()
        if st.cancelled:
            break
        model.eval()
        v_loss, v_correct, v_seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels_b in val_loader:
                images, labels_b = images.to(device), labels_b.to(device)
                out = model(images)
                v_loss += float(criterion(out, labels_b).item()) * int(images.size(0))
                v_correct += int((out.argmax(1) == labels_b).sum().item())
                v_seen += int(images.size(0))
                heartbeat()
        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_loss = v_loss / max(v_seen, 1)
        val_acc = 100.0 * v_correct / max(v_seen, 1)
        st.epoch = epoch
        st.batch_i = 0
        epoch_times.append(time.time() - t_epoch)
        st.history.append({"e": epoch, "tl": round(train_loss, 4), "vl": round(val_loss, 4), "va": round(val_acc, 2)})
        tlc.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc,
                 "lr": float(optimizer.param_groups[0]["lr"])})
        # last.pt every epoch; best.pt on a strict improvement — both atomic (D10, D12).
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        st.last_path, st.last_sha = save_checkpoint(run_url, state_cpu, CHECKPOINT_LAST)
        if val_acc > st.best_val:
            st.best_val, st.best_epoch = val_acc, epoch
            st.best_path, st.best_sha = save_checkpoint(run_url, state_cpu, CHECKPOINT_BEST)
            rec.record["facts"].update({"weights": st.best_path, "best_checkpoint_sha256": st.best_sha})
            set_field("weights", st.best_path)
            set_field("best_checkpoint_sha256", st.best_sha)
        rec.record["facts"]["last_checkpoint_sha256"] = st.last_sha
        log(
            f"epoch {epoch}/{epochs} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_accuracy={val_acc:.2f}%" + (" (new best)" if st.best_epoch == epoch else "")
        )
        flush_progress(force=True)
        rec.touch(force=True)

    avg_epoch_s = (time.time() - st.train_start) / max(st.epoch, 1) if st.epoch else None
    checkpoints = {
        "best_checkpoint_sha256": st.best_sha, "last_checkpoint_sha256": st.last_sha,
        "best_checkpoint": f"{MODEL_DIR}/{CHECKPOINT_BEST}" if st.best_sha else "",
        "last_checkpoint": f"{MODEL_DIR}/{CHECKPOINT_LAST}" if st.last_sha else "",
        "best_epoch": st.best_epoch, "best_val_accuracy": round(st.best_val, 2) if st.best_epoch else None,
        "epochs_completed": st.epoch,
    }
    run.set_parameters(checkpoints)
    expected = {"torchvision_version": fw["torchvision_version"], "seed": seed, "train_table_url": resolved_url,
                "val_table_url": val_url, "weights": st.best_path}

    def finish_result(status: str, collect_s: float | None, collect_rows: int | None, reducer: str | None) -> dict:
        per_row_collect = (
            round(collect_s / max(collect_rows or 1, 1), 5) if collect_s is not None and collect_rows else None
        )
        return {
            "run_url": run_url, "run_name": kw["run_name"], "weights": st.best_path,
            "weights_exists": bool(st.best_path) and Path(st.best_path).is_file(),
            "cancelled": status == "cancelled", "epochs_completed": st.epoch, "best_epoch": st.best_epoch,
            "best_val_accuracy": checkpoints["best_val_accuracy"], "best_checkpoint_sha256": st.best_sha,
            "last_checkpoint_sha256": st.last_sha, "device": device, "device_class": device_class(device),
            "device_fallback_reason": fallback_reason, "usable_rows": n_usable,
            "avg_epoch_s": round(avg_epoch_s, 1) if avg_epoch_s else None,
            "epoch_s_per_row": round(avg_epoch_s / max(n_usable, 1), 5) if avg_epoch_s else None,
            "collect_s": round(collect_s, 1) if collect_s is not None else None, "collect_rows": collect_rows,
            "collect_s_per_row": per_row_collect,
            "reducer": reducer, "contract": contract, "checks": rec.record["checks"],
        }

    if st.cancelled:
        log(f"Training stopped by cancellation request after {st.epoch} epoch(s). The best checkpoint so far is kept.")
        run.set_status_cancelled()
        checks = check_provenance(run_url, manifest, expected) if st.best_sha else []
        rec.record["checks"] = checks
        set_checks(checks)
        for c in checks:
            log(("PASS " if c["ok"] else "FAIL ") + c["label"] + f" — {c['detail']}")
        result = finish_result("cancelled", None, None, None)
        rec.finish("cancelled", result=result)
        _release(model, optimizer)
        return result

    # ── Final pass: per-sample metrics + embeddings on the best model (both splits) ───────
    if st.best_path:
        model.load_state_dict(torch.load(st.best_path, map_location=device, weights_only=True))
        log(f"Restored best model from epoch {st.best_epoch} (val_accuracy={st.best_val:.2f}%)")
    model.eval()
    run.set_status_collecting()
    collect_rows = int(train_table.row_count) + int(val_table.row_count)
    note = f"Collecting per-sample metrics and embeddings on {collect_rows:,} rows…"
    log(note)
    flush_progress(force=True, stage="collect", stage_note=note)
    t_collect = time.time()
    reducer_used = _collect(model, device, manifest, run, train_table, train_eval_view, val_table, val_view,
                            resolved_url, val_url, batch_size, workers, st.best_epoch, log, heartbeat, is_cancelled)
    collect_s = time.time() - t_collect
    run.set_parameters({"reducer": reducer_used, "collect_s": round(collect_s, 1), "collect_rows": collect_rows})
    log(f"Per-sample metrics written for train and val ({collect_rows:,} rows, {collect_s:.1f} s, {reducer_used}).")

    checks = check_provenance(run_url, manifest, expected)
    rec.record["checks"] = checks
    set_checks(checks)
    for c in checks:
        log(("PASS " if c["ok"] else "FAIL ") + c["label"] + f" — {c['detail']}")
    run.set_status_completed()
    log(f"best.pt: {st.best_path} (exists: {Path(st.best_path).is_file() if st.best_path else False})")
    result = finish_result("completed", collect_s, collect_rows, reducer_used)
    rec.finish("completed", result=result)
    set_progress({"percent": 100.0, "label": "Done", "phase": "done", "epoch": st.epoch, "total_epochs": st.total,
                  "history": list(st.history), "batch_i": 0, "batch_n": st.batch_n})
    _release(model, optimizer)
    return result


def _release(model: Any, optimizer: Any) -> None:
    import gc

    try:
        del model, optimizer
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _collect(
    model: Any, device: str, manifest: Manifest, run: Any, train_table: Any, train_view: Any, val_table: Any,
    val_view: Any, train_url: str, val_url: str, batch_size: int, workers: int, epoch: int, log: Any,
    heartbeat: Any, is_cancelled: Any,
) -> str:
    """PLAN §B: predicted, confidence, prob_<class>, loss (NaN for undefined), 3-D embeddings — UMAP
    fit on train (labeled + undefined together), val transformed into the same space, PCA fallback.
    Returns the reducer that produced the coordinates."""
    import copy

    import numpy as np
    import tlc
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    n_classes = manifest.num_classes
    names = manifest.class_names
    n_comp = int(manifest.training.embeddings.n_components)
    ce = nn.CrossEntropyLoss(reduction="none")
    splits: list[dict[str, Any]] = []
    targets = (("train", train_table, train_view, train_url), ("val", val_table, val_view, val_url))
    for split, table, view, url in targets:
        loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
        embs, preds, confs, probs, losses = [], [], [], [], []
        with torch.no_grad():
            for images, labels_b in loader:
                images, labels_b = images.to(device), labels_b.to(device)
                feats = model.features(images)   # the 512-d backbone output: the kit's embedding layer (fc = Identity)
                emb = feats
                logits = model.head(feats)
                p = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                conf = torch.gather(p, 1, pred.unsqueeze(1)).squeeze(1)
                loss = torch.full((int(labels_b.shape[0]),), float("nan"), dtype=torch.float32, device=device)
                mask = labels_b < n_classes
                if bool(mask.any()):
                    loss[mask] = ce(logits[mask], labels_b[mask]).float()
                embs.append(emb.float().cpu().numpy())
                preds.append(pred.cpu().numpy())
                confs.append(conf.float().cpu().numpy())
                probs.append(p.float().cpu().numpy())
                losses.append(loss.cpu().numpy())
                heartbeat()
        splits.append({
            "split": split, "url": url, "table": table,
            "emb": np.vstack(embs), "pred": np.concatenate(preds).astype(np.int64),
            "conf": np.concatenate(confs).astype(np.float32), "prob": np.vstack(probs).astype(np.float32),
            "loss": np.concatenate(losses).astype(np.float32),
        })
        log(f"collected {split}: {len(splits[-1]['pred']):,} rows")

    train_emb = splits[0]["emb"]
    reducer_used, fitted = _fit_reducer(train_emb, n_comp, manifest.training.embeddings.method,
                                        manifest.training.embeddings.fallback, log)
    for s in splits:
        s["reduced"] = _transform(fitted, s["emb"], n_comp)

    for s in splits:
        metrics: dict[str, Any] = {
            "predicted": [int(v) for v in s["pred"]],
            "confidence": s["conf"],
            "loss": s["loss"],
            "embeddings": [row.astype(np.float32) for row in s["reduced"]],
        }
        for i, name in enumerate(names):
            metrics[f"prob_{name}"] = s["prob"][:, i]
        schema: dict[str, Any] = {
            "embeddings": tlc.schemas.Float32Schema(shape=(n_comp,), display_name=f"Embedding ({n_comp}D)"),
        }
        try:
            label_schema = s["table"].rows_schema.values.get("label")
            if label_schema is not None:
                schema["predicted"] = copy.deepcopy(label_schema)
        except Exception:
            pass
        run.add_metrics(metrics, foreign_table_url=s["url"], schema=schema, constants={"epoch": int(epoch)})
        log(f"metrics table written for {s['split']} ({len(metrics)} columns)")
    return reducer_used


def _fit_reducer(train_emb: Any, n_comp: int, method: str, fallback: str, log: Any) -> tuple[str, Any]:
    n = len(train_emb)
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=n_comp, n_neighbors=min(15, max(2, n - 1)), min_dist=0.1, random_state=42)
            reducer.fit(train_emb)
            return "umap", reducer
        except Exception as exc:
            log(f"UMAP failed ({type(exc).__name__}: {str(exc)[:120]}); falling back to {fallback}.")
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=n_comp, random_state=42)
    reducer.fit(train_emb)
    return fallback if fallback else "pca", reducer


def _transform(reducer: Any, emb: Any, n_comp: int) -> Any:
    import numpy as np

    out = reducer.transform(emb)
    return np.asarray(out, dtype=np.float32).reshape(len(emb), n_comp)


def state_for_json(obj: Any) -> Any:
    """Tests / diagnostics: a JSON-safe copy of a record."""
    return json.loads(json.dumps(obj, default=str))


__all__ = [
    "INFERENCE", "OPTIMIZER", "SCHEDULE", "TrainRefused", "build_train_kwargs", "check_provenance",
    "default_workers", "descends_from_seed", "effective_bounds", "effective_weights", "preflight",
    "probe_device_async", "read_record", "resolve_device", "run_summary", "run_training", "scan_rows",
    "summarize_rows", "train_state", "training_facts", "validate_train_url",
]
