# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import stage (job kind ``import``): the kit's ``train`` and ``val`` splits become 3LC tables.

Contract (docs/PLAN.md §C "Importer", session 2 decisions):

* Input: a verified kit (``session.kit_dir``, see ``kit.py``) and the manifest. The kit is
  validated AGAINST the manifest before any table is written: class directories equal the
  manifest classes (same names, manifest order), per-class labeled counts, the undefined pool
  count, per-class val counts, the test image count against BOTH ``splits.test.count`` and the
  rows of ``sample_submission.csv`` (and the ids themselves), and every image decodes. Any
  mismatch fails the job with one message that names every problem, and no table exists.
* ``train``: one row per labeled image with ``label`` in manifest order and ``weight = 1.0``,
  plus one row per pool image with ``label = manifest.undefined_label_id`` (the LAST map entry,
  named ``undefined`` so the Dashboard can filter ``label == undefined``) and ``weight = 0.0``.
  ``val``: labeled rows only, all weights ``1.0``; its URL is recorded as the locked revision.
  ``test`` is NEVER registered — the predictor reads ``data/test/`` directly.
* Tables are written with ``tlc.Table.from_dict`` and an explicit schema (``ImageSchema(url)``,
  ``CategoricalLabelSchema(classes + [undefined])``, ``SampleWeightSchema``) under
  ``<tlc.config.project_root_url>/<project>/datasets/<manifest.dataset_name(split)>/tables/<table>``.
  ``if_exists="raise"``: the importer never reuses and never overwrites.
* Existing tables (the ExDark mirror, docs/EXDARK_MIRROR.md #25): a table already at a split's
  target URL is REUSED, never rewritten, and the post-write checks run on it. ``mode =
  "reimport"`` writes FRESH tables under the next free name (``<table>-2``, ``<table>-3``, …) for
  both splits together and is kept for the re-import decision (#26). Existing tables are never
  touched — a participant's edited revisions stay intact.
* No partial tables: the second table failing, a verification failing, or a cancel after the
  first write deletes what this job created before it reports.
* After writing, each table is re-read and checked: row count == ``manifest.expected_rows``,
  value map == class names + ``undefined``, zero-weight rows == the pool size.
* The import record (``import_state`` in the session store) carries the manifest provenance
  (``resolve_manifest_for_job``), the kit facts, both table URLs as the lineage root, the
  locked val URL, every check, the per-stage timings and the job id. ``import_state()``
  re-verifies it against disk for the revisit view.
* ``num_workers=0`` everywhere data loads (the import loads nothing through torch; PIL decodes
  images one at a time in this process).

The ``ctx`` is the duck-typed job context ``kit.py`` uses (``log``, ``set_checks``,
``set_progress``, ``set_field``, ``is_cancelled``). tlc and PIL are imported inside functions.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaggle_classification import kit, session
from kaggle_classification.kit import DATA_DIR_NAME, FILES_INDEX_NAME, UNDEFINED_DIR_NAME

if TYPE_CHECKING:
    from kaggle_classification.manifest import Manifest

# The Dashboard-visible name of the unlabeled pool's label value (the LAST map entry). Not a
# competition fact: every classification hackathon this plugin serves has an undefined pool.
UNDEFINED_LABEL_NAME = UNDEFINED_DIR_NAME
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
REGISTERED_SPLITS = ("train", "val")
MODES = ("import", "reimport")
# How many fresh names to try before giving up (``initial-2`` … ``initial-99``).
_MAX_FRESH_SUFFIX = 99


class ImportRefused(RuntimeError):
    """A participant-facing refusal (kit defect, collision, bad params). The job fails with the
    message verbatim; nothing was written."""


class _Cancelled(Exception):
    """Internal: unwinds the register loop on a cooperative cancel."""


# ── Kit scan ───────────────────────────────────────────────────────────────


@dataclass
class KitScan:
    """What the kit tree contains, gathered once and checked against the manifest."""

    kit_root: Path
    train_labeled: dict[str, list[Path]] = field(default_factory=dict)  # class dir -> files
    train_undefined: list[Path] = field(default_factory=list)
    val: dict[str, list[Path]] = field(default_factory=dict)
    test: list[Path] = field(default_factory=list)
    submission_ids: list[str] = field(default_factory=list)
    submission_header: list[str] = field(default_factory=list)
    kit_version: str = ""

    @property
    def train_class_dirs(self) -> list[str]:
        return sorted(self.train_labeled)

    @property
    def val_class_dirs(self) -> list[str]:
        return sorted(self.val)

    @property
    def labeled_count(self) -> int:
        return sum(len(v) for v in self.train_labeled.values())

    @property
    def val_count(self) -> int:
        return sum(len(v) for v in self.val.values())


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def scan_kit(kit_root: Path, manifest: Manifest) -> KitScan:
    """Walk ``<kit_root>/data`` exactly as the builder lays it out. Never raises on a missing
    piece — the validators report what is missing, in one message."""
    scan = KitScan(kit_root=kit_root)
    data = kit_root / DATA_DIR_NAME
    train = data / "train"
    if train.is_dir():
        for class_dir in sorted(p for p in train.iterdir() if p.is_dir()):
            if class_dir.name == UNDEFINED_DIR_NAME:
                scan.train_undefined = _images_in(class_dir)
            else:
                scan.train_labeled[class_dir.name] = _images_in(class_dir)
    val = data / "val"
    if val.is_dir():
        for class_dir in sorted(p for p in val.iterdir() if p.is_dir()):
            scan.val[class_dir.name] = _images_in(class_dir)
    scan.test = _images_in(data / "test")
    ids_from = kit_root / manifest.splits.test.ids_from
    if ids_from.is_file():
        with ids_from.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            scan.submission_header = next(reader, []) or []
            scan.submission_ids = [row[0].strip() for row in reader if row and row[0].strip()]
    index_path = kit_root / FILES_INDEX_NAME
    if index_path.is_file():
        try:
            scan.kit_version = str(kit.load_files_index(kit_root).get("kit_version") or "")
        except (OSError, ValueError, RuntimeError):
            scan.kit_version = ""
    return scan


def validate_kit(scan: KitScan, manifest: Manifest) -> list[dict[str, Any]]:
    """Every structural check of the kit against the manifest, all evaluated (not first-fail),
    so one report names every defect. Image decoding is separate (``decode_images``)."""
    checks: list[dict[str, Any]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        checks.append({"label": label, "ok": bool(ok), "detail": detail})

    names = manifest.class_names
    check(
        f"{FILES_INDEX_NAME} present and the kit version matches the manifest",
        scan.kit_version == manifest.kit.version,
        f"{scan.kit_version or '(none)'} vs {manifest.kit.version}",
    )
    check(
        "train class directories equal the manifest classes",
        scan.train_class_dirs == sorted(names),
        f"found {scan.train_class_dirs}, manifest {sorted(names)}",
    )
    check(
        "val class directories equal the manifest classes",
        scan.val_class_dirs == sorted(names),
        f"found {scan.val_class_dirs}, manifest {sorted(names)}",
    )
    want = manifest.splits.train.labeled_per_class
    bad = {c: len(scan.train_labeled.get(c, [])) for c in names if len(scan.train_labeled.get(c, [])) != want}
    check(
        f"train labeled images per class == {want:,}",
        not bad,
        f"{scan.labeled_count:,} labeled images across {len(names)} classes" if not bad else f"off: {bad}",
    )
    check(
        f"train undefined pool == {manifest.splits.train.undefined:,}",
        len(scan.train_undefined) == manifest.splits.train.undefined,
        f"found {len(scan.train_undefined):,}",
    )
    want_val = manifest.splits.val.per_class
    bad_val = {c: len(scan.val.get(c, [])) for c in names if len(scan.val.get(c, [])) != want_val}
    check(
        f"val images per class == {want_val:,}",
        not bad_val,
        f"{scan.val_count:,} val images" if not bad_val else f"off: {bad_val}",
    )
    check(
        f"test images == {manifest.splits.test.count:,}",
        len(scan.test) == manifest.splits.test.count,
        f"found {len(scan.test):,}",
    )
    ids_from = manifest.splits.test.ids_from
    check(
        f"{ids_from} rows == test images",
        len(scan.submission_ids) == len(scan.test) > 0,
        f"{len(scan.submission_ids):,} rows vs {len(scan.test):,} images",
    )
    stems = sorted(p.stem for p in scan.test)
    ids = sorted(scan.submission_ids)
    dup = len(ids) != len(set(ids))
    check(
        f"{ids_from} ids are the test image ids",
        stems == ids and not dup,
        "every id has exactly one image"
        if stems == ids and not dup
        else f"{len(set(ids) - set(stems))} ids without an image, {len(set(stems) - set(ids))} images without an id"
        + (", duplicate ids" if dup else ""),
    )
    columns = list(manifest.submission.columns)
    check(
        f"{ids_from} header matches the submission columns",
        scan.submission_header == columns,
        f"found {scan.submission_header}, manifest {columns}",
    )
    return checks


def decode_images(paths: list[Path], report: Any = None) -> tuple[int, list[str]]:
    """Open and verify every image (PIL). Returns ``(ok_count, first_failures)``. A kit whose
    images do not decode is a kit defect — the import stops and reports rather than registering
    rows the trainer would choke on."""
    from PIL import Image

    failures: list[str] = []
    ok = 0
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                im.verify()
            ok += 1
        except Exception as exc:
            if len(failures) < 20:
                failures.append(f"{path.name}: {type(exc).__name__}")
        if report is not None and i % 500 == 0:
            report(i, len(paths))
    return ok, failures


# ── Table locations and collisions ─────────────────────────────────────────


def project_root_url() -> str:
    """The RESOLVED 3LC project root (``tlc.config`` is the public accessor)."""
    import tlc

    return str(tlc.config.project_root_url)


def table_url(project: str, dataset: str, table: str) -> str:
    import tlc

    return str(tlc.Url(project_root_url()) / project / "datasets" / dataset / "tables" / table)


def _url_exists(url: str) -> bool:
    import tlc

    try:
        return bool(tlc.Url(url).exists())
    except Exception:
        return False


def _row_count(url: str) -> int | None:
    import tlc

    try:
        return int(tlc.Table.from_url(tlc.Url(url)).row_count)
    except Exception:
        return None


def existing_tables(manifest: Manifest, project: str, table_name: str) -> dict[str, dict[str, Any]]:
    """Per registered split: the deterministic URL and whether a table is already there."""
    out: dict[str, dict[str, Any]] = {}
    for split in REGISTERED_SPLITS:
        url = table_url(project, manifest.dataset_name(split), table_name)
        exists = _url_exists(url)
        out[split] = {"url": url, "exists": exists, "rows": _row_count(url) if exists else None}
    return out


def fresh_table_name(manifest: Manifest, project: str, table_name: str) -> str:
    """``table_name`` if neither split has a table under it, else the first ``<name>-N`` (N from 2)
    free for BOTH splits — one name for the pair, so the lineage roots stay siblings."""
    candidates = [table_name] + [f"{table_name}-{n}" for n in range(2, _MAX_FRESH_SUFFIX + 1)]
    for name in candidates:
        if not any(_url_exists(table_url(project, manifest.dataset_name(s), name)) for s in REGISTERED_SPLITS):
            return name
    msg = f"No free table name under {table_name!r} in project {project!r} (tried {_MAX_FRESH_SUFFIX} suffixes)."
    raise ImportRefused(msg)


# ── Params ─────────────────────────────────────────────────────────────────


def resolve_params(data: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """Resolve project / table name / kit dir / mode from the request over the session. Raises
    ``ImportRefused`` with a participant-facing message; shared by the preflight route and the job."""
    from kaggle_classification.manifest import DEFAULT_TABLE_NAME

    current = session.populated_session(manifest)
    project = str(data.get("project_name") or current.get("project_name") or manifest.default_project).strip()
    table_name = str(data.get("table_name") or current.get("table_name") or DEFAULT_TABLE_NAME).strip()
    kit_dir = str(data.get("kit_dir") or current.get("kit_dir") or "").strip().strip('"')
    mode = str(data.get("mode") or "import").strip().lower()
    if mode not in MODES:
        msg = f"mode must be one of {MODES}, got {mode!r}"
        raise ImportRefused(msg)
    for label, value in (("project name", project), ("table name", table_name)):
        if not value or "/" in value or "\\" in value or value in (".", ".."):
            msg = f"The {label} must be a plain name (no slashes), got {value!r}."
            raise ImportRefused(msg)
    if not kit_dir:
        msg = "No starter kit on this machine yet. Download the starter kit first."
        raise ImportRefused(msg)
    kit_root = Path(kit_dir).expanduser()
    if not kit_root.is_absolute() or not (kit_root / FILES_INDEX_NAME).is_file():
        msg = f"No kit found at {kit_root} ({FILES_INDEX_NAME} missing). Download the starter kit first."
        raise ImportRefused(msg)
    return {"project_name": project, "table_name": table_name, "kit_dir": str(kit_root), "mode": mode}


# ── Rows ───────────────────────────────────────────────────────────────────


def build_rows(scan: KitScan, manifest: Manifest) -> dict[str, dict[str, list[Any]]]:
    """Column data per split, in a deterministic order: classes in manifest order, files sorted
    within a class, the undefined pool last. Image URLs are absolute posix paths."""
    ids = {name: c.id for name, c in zip(manifest.class_names, manifest.classes, strict=True)}
    train: dict[str, list[Any]] = {"image": [], "label": [], "weight": []}
    for name in manifest.class_names:
        for p in scan.train_labeled.get(name, []):
            train["image"].append(p.resolve().as_posix())
            train["label"].append(int(ids[name]))
            train["weight"].append(1.0)
    for p in scan.train_undefined:
        train["image"].append(p.resolve().as_posix())
        train["label"].append(int(manifest.undefined_label_id))
        train["weight"].append(0.0)
    val: dict[str, list[Any]] = {"image": [], "label": [], "weight": []}
    for name in manifest.class_names:
        for p in scan.val.get(name, []):
            val["image"].append(p.resolve().as_posix())
            val["label"].append(int(ids[name]))
            val["weight"].append(1.0)
    return {"train": train, "val": val}


def label_map(manifest: Manifest) -> dict[int, str]:
    """The value map both tables carry: the manifest classes by id plus ``undefined`` last."""
    out = {int(c.id): c.name for c in manifest.classes}
    out[int(manifest.undefined_label_id)] = UNDEFINED_LABEL_NAME
    return out


def _schema(manifest: Manifest) -> dict[str, Any]:
    from tlc.schemas import CategoricalLabelSchema, ImageSchema, SampleWeightSchema

    classes = {float(k): v for k, v in label_map(manifest).items()}
    return {
        "image": ImageSchema(sample_type="url"),
        "label": CategoricalLabelSchema(classes=classes),
        "weight": SampleWeightSchema(),
    }


def register_split(
    split: str, rows: dict[str, list[Any]], manifest: Manifest, project: str, table_name: str, description: str
) -> str:
    """Write one split's table at the deterministic URL (``table_url``), so what the collision
    check probed and what gets written are one and the same path. ``if_exists="raise"``: never
    reuse, never overwrite."""
    import tlc

    table = tlc.Table.from_dict(
        rows,
        schema=_schema(manifest),
        table_url=tlc.Url(table_url(project, manifest.dataset_name(split), table_name)),
        if_exists="raise",
        add_weight_column=False,
        description=description,
    )
    return str(table.url)


def verify_table(url: str, split: str, manifest: Manifest, undefined_rows: int) -> list[dict[str, Any]]:
    """Re-read a written table and check it against the manifest."""
    import tlc

    table = tlc.Table.from_url(tlc.Url(url))
    checks: list[dict[str, Any]] = []
    rows = int(table.row_count)
    want = manifest.expected_rows(split)
    checks.append({"label": f"{split} row count == {want:,}", "ok": rows == want, "detail": f"{rows:,} rows"})
    vm = table.get_value_map("label") or {}
    names = [str(vm[k].get("internal_name") if isinstance(vm[k], dict) else getattr(vm[k], "internal_name", vm[k]))
             for k in sorted(vm)]
    want_names = [label_map(manifest)[k] for k in sorted(label_map(manifest))]
    checks.append({
        "label": f"{split} label map is the manifest classes + {UNDEFINED_LABEL_NAME}",
        "ok": names == want_names,
        "detail": ", ".join(names),
    })
    zero = sum(1 for r in table.table_rows if float(r["weight"]) == 0.0)
    checks.append({
        "label": f"{split} zero-weight rows == {undefined_rows:,}",
        "ok": zero == undefined_rows,
        "detail": f"{zero:,} rows at weight 0",
    })
    return checks


def delete_tables(urls: list[str]) -> list[str]:
    """Best-effort removal of tables THIS job created (the no-partial-tables rule)."""
    import tlc

    removed = []
    for url in urls:
        try:
            u = tlc.Url(url)
            if u.exists():
                u.delete()
                removed.append(url)
        except Exception:
            continue
    return removed


# ── Table listing for the revision picker and the derived defaults (ExDark's, ported) ─────


def table_defaults(manifest: Manifest, project: str, table_name: str) -> dict[str, Any]:
    """``GET /tables/defaults``: the canonical URL per registered split for the given project /
    table pair, with an exists flag — what the Train field derives when no override is stored."""
    out: dict[str, Any] = {"project": project, "table": table_name}
    for split in REGISTERED_SPLITS:
        url = table_url(project, manifest.dataset_name(split), table_name)
        out[split] = {"url": url, "exists": _url_exists(url)}
    return out


def _dataset_lineage(tables_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Every readable table under ``<dataset>/tables`` keyed by normalised URL, with its parent
    (``input_table_url`` on FromTable subclasses such as the Dashboard's EditedTable, else the first
    ``input_tables`` entry) and the children map. Read-only; unreadable folders are skipped."""
    import tlc

    entries: dict[str, dict[str, Any]] = {}
    for tdir in tables_dir.iterdir():
        if not tdir.is_dir():
            continue
        try:
            table = tlc.Table.from_url(tlc.Url(tdir.as_posix()))
            raw_parent = getattr(table, "input_table_url", None)
            if not raw_parent:
                inputs = getattr(table, "input_tables", None) or []
                raw_parent = inputs[0] if inputs else None
            parent = str(tlc.Url(str(raw_parent)).to_absolute(table.url)) if raw_parent else ""
            entries[_norm(str(table.url))] = {
                "name": tdir.name,
                "url": str(table.url),
                "rows": int(table.row_count),
                "_parent": _norm(parent),
                "_mtime": tdir.stat().st_mtime,
            }
        except Exception:
            continue
    children: dict[str, list[str]] = {}
    for key, e in entries.items():
        if e["_parent"] and e["_parent"] in entries:
            children.setdefault(e["_parent"], []).append(key)
    return entries, children


def newest_descendant(url: str) -> str:
    """The newest revision derived from ``url`` by the directory walk: follow the newest child at
    each step (the picker's chain order). ``url`` itself when it has no children."""
    entries, children = _dataset_lineage(Path(url).parent)
    cur = _norm(url)
    seen: set[str] = set()
    while cur in children and cur not in seen:
        seen.add(cur)
        kids = sorted(children[cur], key=lambda k: entries[k]["_mtime"])
        cur = kids[-1]
    return entries[cur]["url"] if cur in entries else url


def latest_url(url: str, *, timeout: float = 30.0) -> str:
    """The newest revision descending from ``url``: tlc's lineage index first (what the kit's
    ``.latest()`` follows), waiting at most ``timeout`` seconds for an indexing cycle; the
    directory walk when the index cannot answer (a root no scan URL covers, the tests' tmp root)."""
    import tlc

    try:
        return str(tlc.Table.from_url(tlc.Url(url)).latest(timeout=timeout).url)
    except Exception:
        try:
            return newest_descendant(url)
        except Exception:
            return url


def list_project_tables(manifest: Manifest, project: str) -> dict[str, Any]:
    """``GET /tables/list``: datasets -> lineage-ordered revision chains for the revision picker.

    Layout-derived: ``table_url`` rebuilds ``<root>/<project>/datasets/<dataset>/tables/<table>``,
    so the datasets root is walked directly. Chain order is lineage, root first, following the
    newest child at each step; off-chain branches append in mtime order. ``latest`` comes from
    ``latest_url`` on the chain root — the resolution ``use_latest`` training follows.
    Read-only; an unreadable table folder is skipped, never a failure."""
    probe = table_url(project, "__probe__", "initial")
    datasets_root = Path(probe).parent.parent.parent
    out: dict[str, Any] = {"project": project, "datasets": []}
    if not datasets_root.is_dir():
        return out
    for ds_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        tables_dir = ds_dir / "tables"
        if not tables_dir.is_dir():
            continue
        entries, children = _dataset_lineage(tables_dir)
        if not entries:
            continue
        roots = [k for k, e in entries.items() if not (e["_parent"] and e["_parent"] in entries)]
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for root in sorted(roots, key=lambda k: entries[k]["_mtime"]):
            cur: str | None = root
            while cur and cur not in seen:
                seen.add(cur)
                ordered.append(entries[cur])
                kids = sorted(children.get(cur, []), key=lambda k: entries[k]["_mtime"])
                cur = kids[-1] if kids else None
        for key, e in sorted(entries.items(), key=lambda kv: kv[1]["_mtime"]):
            if key not in seen:
                ordered.append(e)
        latest = latest_url(ordered[0]["url"], timeout=_LATEST_TIMEOUT_ROUTE_S)
        rows = [{"name": e["name"], "url": e["url"], "rows": e["rows"], "latest": _norm(e["url"]) == _norm(latest)}
                for e in ordered]
        out["datasets"].append({"name": ds_dir.name, "tables": rows, "latest_url": latest})
    return out


# A route must not sit on the indexer's 30 s default: the fast path answers at once for revisions
# this process has seen, and a few seconds covers one scheduler cycle for the rest.
_LATEST_TIMEOUT_ROUTE_S = 5.0


def _norm(url: str) -> str:
    return str(url).replace("\\", "/").rstrip("/").lower()


# ── Record, revisit and preflight ──────────────────────────────────────────


def write_record(record: dict[str, Any]) -> None:
    session.save({"import_state": record})


def read_record() -> dict[str, Any] | None:
    data = session.load().get("import_state")
    return data if isinstance(data, dict) and data.get("tables") else None


def import_state() -> dict[str, Any]:
    """The last successful import, re-verified against disk (table existence decides).

    ``empty`` — no record; ``success`` — both tables still exist; ``stale`` — a recorded table
    is gone (the record is shown so the participant knows what was there)."""
    record = read_record()
    if not record:
        return {"state": "empty"}
    verified: dict[str, bool] = {}
    latest: dict[str, str] = {}
    for split in REGISTERED_SPLITS:
        url = str(((record.get("tables") or {}).get(split) or {}).get("url") or "")
        verified[split] = bool(url) and _url_exists(url)
        if verified[split]:
            latest[split] = _latest_url(url)
    state = "success" if all(verified.values()) else "stale"
    # ``latest``: the newest revision descending from each seed table (tlc's own ``latest()``),
    # so the Loop's Dashboard step opens what the participant is actually labeling. Equal to the
    # seed URL until a revision exists.
    return {"state": state, "verified": verified, "latest": latest, "record": record}


def _latest_url(url: str) -> str:
    return latest_url(url, timeout=_LATEST_TIMEOUT_ROUTE_S)


def preflight(data: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    """The Import form's read-only gate, in the shape ExDark's ``/import/preflight`` answers so
    the ported fragment renders it unchanged (docs/EXDARK_MIRROR.md #20). Never writes.

    ``error`` — the kit folder is missing or unreadable (parse failure / path not found).
    ``all_ok`` — every kit-vs-manifest structural check passes; ``splits`` carries per-split
    ``{found, expected, ok}`` (train counts labeled + pool), ``unlabeled`` the pool,
    ``classes`` ``{names, count, canonical}`` (the class directories in manifest order), and
    ``problems`` the failing checks as ``{label, detail, remedy}`` rows for the mismatch view.
    ``existing`` names the tables already at the target URLs (they will be REUSED).
    """
    import kaggle_classification

    out: dict[str, Any] = {"plugin_version": kaggle_classification.__version__, "kit": kit.download_state(manifest)}
    try:
        params = resolve_params(data, manifest)
    except ImportRefused as exc:
        out.update({"all_ok": False, "error": str(exc)})
        return out
    kit_root = Path(params["kit_dir"])
    out["kit_dir"] = str(kit_root)
    out["params"] = params
    scan = scan_kit(kit_root, manifest)
    checks = validate_kit(scan, manifest)
    failed = [c for c in checks if not c["ok"]]
    names = manifest.class_names
    out["splits"] = {
        "train": {
            "found": scan.labeled_count + len(scan.train_undefined),
            "expected": manifest.expected_rows("train"),
            "labeled": scan.labeled_count,
            "ok": scan.labeled_count == manifest.splits.train.labeled_per_class * manifest.num_classes
            and len(scan.train_undefined) == manifest.splits.train.undefined,
        },
        "val": {"found": scan.val_count, "expected": manifest.expected_rows("val"), "ok": scan.val_count == manifest.expected_rows("val")},
    }
    out["unlabeled"] = {"found": len(scan.train_undefined), "expected": manifest.splits.train.undefined}
    out["classes"] = {
        "names": names,
        "count": len(scan.train_class_dirs),
        "canonical": scan.train_class_dirs == sorted(names) and scan.val_class_dirs == sorted(names),
    }
    out["problems"] = [{"label": c["label"], "detail": c.get("detail", "")} for c in failed]
    out["all_ok"] = not failed
    out["existing"] = existing_tables(manifest, params["project_name"], params["table_name"])
    out["project_root"] = project_root_url()
    return out


# ── The job ────────────────────────────────────────────────────────────────


def run_import(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """The import job. Raises ``ImportRefused`` / ``RuntimeError`` with a participant-facing
    message on failure (no tables left behind); returns ``{"cancelled": True}`` when stopped."""
    log = ctx.log
    set_checks = ctx.set_checks
    set_progress = getattr(ctx, "set_progress", lambda p: None)
    set_field = getattr(ctx, "set_field", lambda k, v: None)
    is_cancelled = getattr(ctx, "is_cancelled", lambda: False)
    job_id = str(getattr(ctx, "job_id", "") or "")

    checks: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    started = time.time()

    def add_checks(new: list[dict[str, Any]]) -> bool:
        checks.extend(new)
        set_checks(checks)
        for c in new:
            log(("PASS " if c["ok"] else "FAIL ") + c["label"] + (f": {c['detail']}" if c.get("detail") else ""))
        return all(c["ok"] for c in new)

    def check(label: str, ok: bool, detail: str = "") -> bool:
        return add_checks([{"label": label, "ok": bool(ok), "detail": detail}])

    def stage(name: str, percent: float, label: str) -> None:
        set_progress({"percent": percent, "label": label, "phase": name})

    resolved = resolve_params(params, manifest)
    project, table_name, kit_root = resolved["project_name"], resolved["table_name"], Path(resolved["kit_dir"])
    mode = resolved["mode"]
    for k, v in resolved.items():
        set_field(k, v)

    # ── Validate the kit against the manifest ───────────────────────────
    stage("validate", 2.0, "Checking the kit against the manifest")
    t0 = time.time()
    scan = scan_kit(kit_root, manifest)
    structural = validate_kit(scan, manifest)
    timings["validate_s"] = round(time.time() - t0, 2)
    if not add_checks(structural):
        failed = [c for c in structural if not c["ok"]]
        msg = (
            f"The starter kit at {kit_root} does not match the competition manifest ({len(failed)} "
            + ("check" if len(failed) == 1 else "checks")
            + " failed): "
            + "; ".join(f"{c['label']} ({c['detail']})" for c in failed)
            + ". Nothing was imported. This is a kit defect: report it to the organizers."
        )
        raise ImportRefused(msg)
    if is_cancelled():
        return {"cancelled": True}

    # ── Decode every image ───────────────────────────────────────────────
    stage("decode", 15.0, "Decoding every image")
    t0 = time.time()
    all_images = (
        [p for name in manifest.class_names for p in scan.train_labeled[name]]
        + scan.train_undefined
        + [p for name in manifest.class_names for p in scan.val[name]]
        + scan.test
    )

    def decode_report(i: int, n: int) -> None:
        stage("decode", 15.0 + 15.0 * i / max(n, 1), f"Decoding image {i + 1:,}/{n:,}")

    ok_count, failures = decode_images(all_images, decode_report)
    timings["decode_s"] = round(time.time() - t0, 2)
    if not check("every kit image decodes", not failures, f"{ok_count:,}/{len(all_images):,} decoded"
                 + (f"; first failures: {', '.join(failures[:5])}" if failures else "")):
        msg = (
            f"{len(all_images) - ok_count} of {len(all_images)} kit images do not decode (first: "
            f"{', '.join(failures[:5])}). Nothing was imported. This is a kit defect: report it to the organizers."
        )
        raise ImportRefused(msg)
    if is_cancelled():
        return {"cancelled": True}

    # ── Existing tables (the ExDark mirror: an existing table at the target URL is REUSED) ──
    # ``if_exists="reuse"`` semantics as ExDark's ``from_yolo_url``: a table already at the
    # deterministic URL is not rewritten; the post-write checks run on it exactly as on a
    # created one, so a stale or edited table cannot pass unnoticed. ``mode="reimport"`` (fresh
    # ``<table>-N`` names beside the old ones) stays available for the re-import decision
    # (docs/EXDARK_MIRROR.md #26) and is never chosen by the fragment today.
    stage("collision", 31.0, "Checking for existing tables")
    existing = existing_tables(manifest, project, table_name)
    collision = any(v["exists"] for v in existing.values())
    actual_name = fresh_table_name(manifest, project, table_name) if (collision and mode == "reimport") else table_name
    reused = {s: bool(existing[s]["exists"]) and actual_name == table_name for s in REGISTERED_SPLITS}
    if any(reused.values()):
        listing = ", ".join(f"{s} ({existing[s]['rows']} rows)" for s in REGISTERED_SPLITS if reused[s])
        check("existing tables reused", True, listing)
    else:
        check(
            "target table names are free",
            True,
            f"{actual_name!r}" + (f" (fresh, {table_name!r} is taken)" if actual_name != table_name else ""),
        )
    set_field("actual_table_name", actual_name)

    # ── Register train, then val; verify; no partial tables ────────────
    rows = build_rows(scan, manifest)
    created: list[str] = []
    urls: dict[str, str] = {}
    description = (
        f"{manifest.competition.display_name} — kit {manifest.kit.version}, imported by kaggle-classification"
    )
    undefined_rows = {"train": len(scan.train_undefined), "val": 0}
    try:
        for i, split in enumerate(REGISTERED_SPLITS):
            if is_cancelled():
                raise _Cancelled()
            if reused[split]:
                stage("register", 35.0 + 25.0 * i, f"Reusing the existing {split} table")
                urls[split] = existing[split]["url"]
                set_field(f"{split}_table_url", urls[split])
                log(f"{split}: reused {urls[split]} ({existing[split]['rows']} rows)")
                continue
            stage("register", 35.0 + 25.0 * i, f"Registering the {split} table ({len(rows[split]['image']):,} rows)")
            t0 = time.time()
            url = register_split(split, rows[split], manifest, project, actual_name, description)
            timings[f"register_{split}_s"] = round(time.time() - t0, 2)
            created.append(url)
            urls[split] = url
            set_field(f"{split}_table_url", url)
            log(f"{split}: created {url} ({len(rows[split]['image']):,} rows, {timings[f'register_{split}_s']}s)")
        stage("verify", 86.0, "Verifying the written tables")
        t0 = time.time()
        for split in REGISTERED_SPLITS:
            if not add_checks(verify_table(urls[split], split, manifest, undefined_rows[split])):
                msg = f"The {split} table did not verify after writing (see the checks). It was removed."
                raise RuntimeError(msg)
        timings["verify_s"] = round(time.time() - t0, 2)
    except _Cancelled:
        removed = delete_tables(created)
        log(f"Cancelled. Removed {len(removed)} table(s) this job had written; nothing partial remains.")
        return {"cancelled": True, "removed": removed}
    except Exception as exc:
        removed = delete_tables(created)
        left = [u for u in created if u not in removed]
        note = f" Removed {len(removed)} table(s) this job had written." if created else ""
        if left:
            note += f" Could not remove: {', '.join(left)}."
        msg = f"{exc}{note}"
        raise RuntimeError(msg) from exc

    # ── Record ──────────────────────────────────────────────────────────
    stage("record", 96.0, "Recording the import")
    timings["total_s"] = round(time.time() - started, 2)
    record = {
        "project_name": project,
        "table_name": actual_name,
        "requested_table_name": table_name,
        "mode": mode,
        "kit_dir": str(kit_root),
        "kit_version": manifest.kit.version,
        "tables": {
            split: {
                "url": urls[split],
                "rows": int(existing[split]["rows"] or 0) if reused[split] else len(rows[split]["image"]),
                "undefined_rows": undefined_rows[split],
                # ExDark's per-split outcome: REUSED (an existing table at the URL, validated) or CREATED.
                "reused": reused[split],
            }
            for split in REGISTERED_SPLITS
        },
        # The lineage root every later revision descends from, and the locked val revision.
        "lineage_root": {"train_url": urls["train"], "val_url": urls["val"]},
        "val_locked": {"url": urls["val"], "editable": bool(manifest.splits.val.editable)},
        "label_map": {str(k): v for k, v in label_map(manifest).items()},
        "checks": checks,
        "timings": timings,
        "manifest_provenance": manifest.provenance,
        "job_id": job_id,
        "completed_at": time.time(),
    }
    write_record(record)
    if actual_name != table_name:
        current = session.populated_session(manifest)
        current["table_name"] = actual_name
        session.save({"session": current})
        log(f"Session table name is now {actual_name!r}")
    set_field("train_table_url", urls["train"])
    set_field("run_url", urls["train"])  # the Open button opens the seed train table
    stage("done", 100.0, "Complete")
    return {"cancelled": False, **{k: v for k, v in record.items() if k != "checks"}}
