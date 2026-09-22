# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The competition manifest — schema v1, validation, and resolution.

Everything competition-specific the plugin knows comes from one manifest
document: classes, split sizes, the locked model, training defaults and
bounds, the submission format, the kit shards, and the UI copy. No module in
this package carries a competition constant; they ask the ``Manifest``.

Resolution (``resolve``): **remote wins whenever it is reachable and the
fetched document validates**; the cache is the last remote document that
validated; the bundled ``manifests/<id>-v1.yaml`` is the last resort. There is
no "newer than" comparison — a hotfix or a rollback on the CDN takes effect on
the next load whatever the version fields say. A remote document that fetches
but fails validation is logged, falls to the cache, and surfaces a warning the
UI shows; a bad hotfix must never brick a participant.

The CDN layout is fixed under either tier's base URL (docs/PLAN.md §A3):
``kaggle/classification-index.json`` (mutable) lists competitions as
``{schema_version, competitions: [{id, display_name, manifest_url, active}]}``
with ``manifest_url`` relative to the index; ``kaggle/<id>/manifest.json``
(mutable, hotfixable) carries ``kit.path`` relative to ITSELF (``starter-kit/v1/``),
and the shards under that prefix are immutable. Shard URLs are resolved against
the URL the manifest was actually fetched from, so dev and prod serve
byte-identical documents. The plugin uses the single active competition; with
more than one active the UI shows a picker (``select_competition``); with none
it falls to the bundled manifest with a visible warning.

Fetching is server-side only (routes / worker), never from the browser:
connect+read budget ``FETCH_BUDGET_S`` in total with one retry. The UI renders
at once from cache/bundled (``resolve(network=False)``) and picks up the remote
result from ``refresh_in_background``.

Import-light: stdlib only at module level. PyYAML is imported inside the
loaders so ``import kaggle_classification`` stays cheap in a bare SDK venv.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaggle_classification import storage

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1

# Two CDN tiers serving byte-identical objects (docs/PLAN.md §A3). Prod is the release default,
# a CODE CONSTANT; the dev tier (and a local mock) is reached ONLY through the env override.
MANIFEST_BASE_URL = "https://competitions.3lc.ai"
DEV_MANIFEST_BASE_URL = "https://competitions.dev.3lc.ai"
MANIFEST_BASE_URL_ENV = "KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL"
# The layout under a base URL. Never query strings (the CDN's cache key ignores them).
INDEX_PATH = "kaggle/classification-index.json"
MANIFEST_NAME = "manifest.json"


def manifest_path(competition_id: str) -> str:
    return f"kaggle/{competition_id}/{MANIFEST_NAME}"


# Hosts a manifest may be served from — and therefore point the kit at, since shard URLs
# resolve against the manifest's own URL. The release default is the prod CDN alone; the dev CDN
# and loopback are allowed ONLY while the base-URL override is set (tests/test_packaging.py
# fails the release if the dev host is reachable without it). Loopback may use plain http.
RELEASE_HOSTS = frozenset({"competitions.3lc.ai"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEV_HOSTS = frozenset({"competitions.dev.3lc.ai"}) | _LOOPBACK_HOSTS


def override_active() -> bool:
    return bool(os.environ.get(MANIFEST_BASE_URL_ENV, "").strip())


def allowed_hosts() -> frozenset[str]:
    """The manifest/kit host allowlist in force: prod only, plus dev and loopback under the override."""
    return RELEASE_HOSTS | DEV_HOSTS if override_active() else RELEASE_HOSTS


# Which competition this build serves when the index does not decide (offline, or as the
# preferred one among several active). The manifest for that id is what gets resolved.
DEFAULT_COMPETITION_ID = "intel-scene"

BUNDLED_DIR = Path(__file__).resolve().parent / "manifests"
CACHE_DIR_NAME = "manifest-cache"

# The table revision a fresh session starts from (the ExDark convention).
DEFAULT_TABLE_NAME = "initial"

# Connect+read budget for the whole remote resolution of one document, retry included.
FETCH_BUDGET_S = 5.0
FETCH_RETRIES = 1
# How long a background refresh result is considered fresh before GET /config triggers another.
REFRESH_TTL_S = 600.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHARD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MARKUP_RE = re.compile(r"[<>\x00-\x1f\x7f]")


class ManifestError(ValueError):
    """A manifest or index document that fails validation. The message names the field."""


class FetchError(RuntimeError):
    """The remote could not be reached within the budget."""


# ── Schema v1 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Competition:
    id: str
    slug: str
    display_name: str
    deadline_utc: str


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str


@dataclass(frozen=True)
class Shard:
    name: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class Kit:
    # Relative to the manifest's own URL, always ending in "/" (e.g. "starter-kit/v1/").
    path: str
    version: str
    shards: tuple[Shard, ...]

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes for s in self.shards)

    @property
    def published(self) -> bool:
        """False until the kit builder's ``kit{}`` block has been pasted into the manifest."""
        return bool(self.shards)


@dataclass(frozen=True)
class TrainSplit:
    labeled_per_class: int
    undefined: int


@dataclass(frozen=True)
class ValSplit:
    per_class: int
    editable: bool


@dataclass(frozen=True)
class TestSplit:
    count: int
    ids_from: str


@dataclass(frozen=True)
class Splits:
    train: TrainSplit
    val: ValSplit
    test: TestSplit


@dataclass(frozen=True)
class Model:
    arch: str
    pretrained: bool
    image_size: int


@dataclass(frozen=True)
class Embeddings:
    method: str
    fallback: str
    n_components: int


@dataclass(frozen=True)
class Training:
    defaults: dict[str, Any]
    bounds: dict[str, tuple[float, float]]
    presets: dict[str, dict[str, Any]]
    embeddings: Embeddings


@dataclass(frozen=True)
class Submission:
    columns: tuple[str, ...]
    metric: str
    daily_limit: int


@dataclass(frozen=True)
class HelpLink:
    label: str
    url: str


@dataclass(frozen=True)
class Ui:
    loop_banner_text: str
    help_links: tuple[HelpLink, ...]


@dataclass(frozen=True)
class Manifest:
    """One validated competition manifest plus where it came from."""

    schema_version: int
    competition: Competition
    classes: tuple[ClassDef, ...]
    kit: Kit
    splits: Splits
    model: Model
    training: Training
    submission: Submission
    ui: Ui
    # Provenance: "bundled" | "cache" | "remote" (tests use "test"), a human-readable detail
    # (path or URL), the sha256 of the document bytes, and when a remote copy was fetched.
    source: str = "bundled"
    source_detail: str = ""
    sha256: str = ""
    fetched_at: str | None = None
    # The URL this document lives at on the CDN tier in force: the fetched URL for a remote
    # copy, the recorded one for a cached copy, the canonical layout URL for the bundled copy.
    # Every kit URL resolves against it.
    document_url: str = ""
    # Unknown fields seen while parsing (warned, never fatal).
    warnings: tuple[str, ...] = field(default=())

    @property
    def kit_base_url(self) -> str:
        """Absolute prefix the shards live under: ``document_url`` joined with ``kit.path``."""
        return urllib.parse.urljoin(self.document_url, self.kit.path)

    def shard_url(self, name: str) -> str:
        return urllib.parse.urljoin(self.kit_base_url, name)

    # ── derived facts every other module asks for ────────────────────────
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    @property
    def undefined_label_id(self) -> int:
        """The label id of the unlabeled pool rows: one past the last real class (the Intel
        convention — ``undefined`` is the LAST map entry, so ``label < num_classes`` is the
        "has a real label" test the trainer and the metrics masking rely on)."""
        return self.num_classes

    @property
    def default_project(self) -> str:
        return self.competition.id

    def dataset_name(self, split: str) -> str:
        """The dataset a split's tables live under: one definition, used by the importer, the
        gates, the server asserts and the override classifier (the ExDark DP-11 lesson)."""
        return f"{self.competition.id}_{split}"

    def expected_rows(self, split: str) -> int:
        """Row count a correct import of ``split`` produces (an upper bound at train time)."""
        if split == "train":
            return self.splits.train.labeled_per_class * self.num_classes + self.splits.train.undefined
        if split == "val":
            return self.splits.val.per_class * self.num_classes
        if split == "test":
            return self.splits.test.count
        msg = f"unknown split {split!r}"
        raise KeyError(msg)

    @property
    def provenance(self) -> dict[str, Any]:
        """What a job records about the manifest it ran under (the ledger's input)."""
        return {
            "manifest_sha256": self.sha256,
            "manifest_source": self.source,
            "manifest_source_detail": self.source_detail,
            "manifest_fetched_at": self.fetched_at,
            "manifest_document_url": self.document_url,
            "competition_id": self.competition.id,
            "kit_version": self.kit.version,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready copy (served on ``GET /config`` under ``_meta.manifest``)."""
        return asdict(self)


# ── Validation ─────────────────────────────────────────────────────────────


def _require(section: Any, key: str, where: str) -> Any:
    if not isinstance(section, dict):
        msg = f"{where}: expected a mapping"
        raise ManifestError(msg)
    if key not in section:
        msg = f"{where}.{key}: missing"
        raise ManifestError(msg)
    return section[key]


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{where}: expected an integer, got {value!r}"
        raise ManifestError(msg)
    if minimum is not None and value < minimum:
        msg = f"{where}: must be >= {minimum}, got {value}"
        raise ManifestError(msg)
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{where}: expected a non-empty string, got {value!r}"
        raise ManifestError(msg)
    return value.strip()


def _display(value: Any, where: str) -> str:
    """A string the fragment renders. Markup and control characters are refused here, and the
    fragment only ever assigns these through ``textContent`` — escaped at both ends."""
    text = _str(value, where)
    if _MARKUP_RE.search(text):
        msg = f"{where}: must not contain markup or control characters"
        raise ManifestError(msg)
    return text


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{where}: expected true/false, got {value!r}"
        raise ManifestError(msg)
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{where}: expected a number, got {value!r}"
        raise ManifestError(msg)
    return value


def _warn_unknown(section: dict[str, Any], known: set[str], where: str, warnings: list[str]) -> None:
    for key in section:
        if key not in known:
            note = f"{where}.{key}: unknown field ignored"
            warnings.append(note)
            _log.warning("manifest: %s", note)


def _check_document_url(url: str, hosts: frozenset[str] | set[str], where: str) -> str:
    """An absolute https URL on an allowed host (plain http only for loopback), no query/fragment."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        msg = f"{where}: expected an absolute URL, got {url!r}"
        raise ManifestError(msg)
    if parts.scheme != "https" and not (parts.scheme == "http" and host in _LOOPBACK_HOSTS):
        msg = f"{where}: must use https (got {parts.scheme!r})"
        raise ManifestError(msg)
    if host not in {h.lower() for h in hosts}:
        msg = f"{where}: host {host!r} is not an allowed host ({', '.join(sorted(hosts))})"
        raise ManifestError(msg)
    if parts.query or parts.fragment:
        msg = f"{where}: must not carry a query or fragment"
        raise ManifestError(msg)
    return url


def _check_kit_path(path: str, where: str) -> str:
    """A RELATIVE prefix under the manifest's own directory: no scheme, no leading slash, no
    parent steps, no query. Normalised to end with "/"."""
    if "://" in path or path.startswith(("/", "\\")) or ":" in path.split("/")[0]:
        msg = f"{where}: must be relative to the manifest (got {path!r}); absolute URLs are not allowed"
        raise ManifestError(msg)
    if any(seg in ("..", "") for seg in path.strip("/").split("/")) or "?" in path or "#" in path:
        msg = f"{where}: must be a plain relative prefix like 'starter-kit/v1/', got {path!r}"
        raise ManifestError(msg)
    return path.rstrip("/") + "/"


def parse_manifest(
    data: Any,
    *,
    source: str = "bundled",
    source_detail: str = "",
    sha256: str = "",
    fetched_at: str | None = None,
    document_url: str = "",
    hosts: frozenset[str] | set[str] | None = None,
) -> Manifest:
    """Validate a decoded manifest document and return the typed ``Manifest``.

    ``document_url`` is where this document lives (defaults to the canonical layout URL under
    the base in force); its host must be allowed, and every kit URL resolves against it.
    Raises ``ManifestError`` naming the offending field. Unknown fields are recorded on
    ``Manifest.warnings`` and logged, never fatal — a newer manifest must still load on an
    older plugin.
    """
    hosts = allowed_hosts() if hosts is None else hosts
    warnings: list[str] = []
    if not isinstance(data, dict):
        msg = "manifest: top level must be a mapping"
        raise ManifestError(msg)

    version = _int(_require(data, "schema_version", "manifest"), "schema_version")
    if version != SCHEMA_VERSION:
        msg = f"schema_version: this plugin understands {SCHEMA_VERSION}, got {version}"
        raise ManifestError(msg)

    top_known = {
        "schema_version",
        "competition",
        "classes",
        "kit",
        "splits",
        "model",
        "training",
        "submission",
        "ui",
    }
    _warn_unknown(data, top_known, "manifest", warnings)

    # competition
    comp = _require(data, "competition", "manifest")
    _warn_unknown(comp, {"id", "slug", "display_name", "deadline_utc"}, "competition", warnings)
    competition = Competition(
        id=_str(_require(comp, "id", "competition"), "competition.id"),
        slug=_str(_require(comp, "slug", "competition"), "competition.slug"),
        display_name=_display(_require(comp, "display_name", "competition"), "competition.display_name"),
        deadline_utc=_str(_require(comp, "deadline_utc", "competition"), "competition.deadline_utc"),
    )
    if not _ID_RE.fullmatch(competition.id):
        msg = f"competition.id: must be lowercase letters, digits and hyphens, got {competition.id!r}"
        raise ManifestError(msg)

    # classes — ids contiguous 0..N-1, names unique and renderable
    raw_classes = _require(data, "classes", "manifest")
    if not isinstance(raw_classes, list) or not raw_classes:
        msg = "classes: expected a non-empty list"
        raise ManifestError(msg)
    classes: list[ClassDef] = []
    for i, entry in enumerate(raw_classes):
        where = f"classes[{i}]"
        _warn_unknown(entry if isinstance(entry, dict) else {}, {"id", "name"}, where, warnings)
        classes.append(
            ClassDef(
                id=_int(_require(entry, "id", where), f"{where}.id"),
                name=_display(_require(entry, "name", where), f"{where}.name"),
            )
        )
    ids = sorted(c.id for c in classes)
    if ids != list(range(len(classes))):
        msg = f"classes: ids must be contiguous 0..{len(classes) - 1}, got {ids}"
        raise ManifestError(msg)
    names = [c.name for c in classes]
    if len(set(names)) != len(names):
        msg = f"classes: names must be unique, got {names}"
        raise ManifestError(msg)
    classes.sort(key=lambda c: c.id)

    # The document's own URL: where the kit resolves from, and the host check.
    doc_url = document_url or f"{base_url()}/{manifest_path(competition.id)}"
    doc_url = _check_document_url(doc_url, hosts, "manifest url")

    # kit — a relative prefix, shard names plain filenames
    raw_kit = _require(data, "kit", "manifest")
    if "base_url" in raw_kit:
        msg = "kit.base_url: absolute kit URLs are not allowed; use kit.path relative to the manifest"
        raise ManifestError(msg)
    _warn_unknown(raw_kit, {"path", "version", "shards"}, "kit", warnings)
    raw_shards = _require(raw_kit, "shards", "kit")
    if not isinstance(raw_shards, list):
        msg = "kit.shards: expected a list"
        raise ManifestError(msg)
    shards: list[Shard] = []
    for i, entry in enumerate(raw_shards):
        where = f"kit.shards[{i}]"
        _warn_unknown(entry if isinstance(entry, dict) else {}, {"name", "sha256", "bytes"}, where, warnings)
        sha = _str(_require(entry, "sha256", where), f"{where}.sha256").lower()
        if not _SHA256_RE.match(sha):
            msg = f"{where}.sha256: expected 64 hex characters"
            raise ManifestError(msg)
        name = _str(_require(entry, "name", where), f"{where}.name")
        if not _SHARD_NAME_RE.match(name) or ".." in name:
            msg = f"{where}.name: must be a plain file name, got {name!r}"
            raise ManifestError(msg)
        shards.append(
            Shard(name=name, sha256=sha, bytes=_int(_require(entry, "bytes", where), f"{where}.bytes", minimum=0))
        )
    if len({s.name for s in shards}) != len(shards):
        msg = "kit.shards: shard names must be unique"
        raise ManifestError(msg)
    kit = Kit(
        path=_check_kit_path(_str(_require(raw_kit, "path", "kit"), "kit.path"), "kit.path"),
        version=_str(_require(raw_kit, "version", "kit"), "kit.version"),
        shards=tuple(shards),
    )

    # splits
    raw_splits = _require(data, "splits", "manifest")
    _warn_unknown(raw_splits, {"train", "val", "test"}, "splits", warnings)
    tr, va, te = (_require(raw_splits, k, "splits") for k in ("train", "val", "test"))
    _warn_unknown(tr, {"labeled_per_class", "undefined"}, "splits.train", warnings)
    _warn_unknown(va, {"per_class", "editable"}, "splits.val", warnings)
    _warn_unknown(te, {"count", "ids_from"}, "splits.test", warnings)
    splits = Splits(
        train=TrainSplit(
            labeled_per_class=_int(
                _require(tr, "labeled_per_class", "splits.train"), "splits.train.labeled_per_class", minimum=0
            ),
            undefined=_int(_require(tr, "undefined", "splits.train"), "splits.train.undefined", minimum=0),
        ),
        val=ValSplit(
            per_class=_int(_require(va, "per_class", "splits.val"), "splits.val.per_class", minimum=0),
            editable=_bool(_require(va, "editable", "splits.val"), "splits.val.editable"),
        ),
        test=TestSplit(
            count=_int(_require(te, "count", "splits.test"), "splits.test.count", minimum=0),
            ids_from=_str(_require(te, "ids_from", "splits.test"), "splits.test.ids_from"),
        ),
    )

    # model — pretrained must be false (the fairness contract), arch required
    raw_model = _require(data, "model", "manifest")
    _warn_unknown(raw_model, {"arch", "pretrained", "image_size"}, "model", warnings)
    pretrained = _bool(_require(raw_model, "pretrained", "model"), "model.pretrained")
    if pretrained:
        msg = "model.pretrained: must be false — every participant trains from random init"
        raise ManifestError(msg)
    model = Model(
        arch=_str(_require(raw_model, "arch", "model"), "model.arch"),
        pretrained=False,
        image_size=_int(_require(raw_model, "image_size", "model"), "model.image_size", minimum=1),
    )

    # training — bounds are [lo, hi] with lo <= hi; defaults and presets must sit inside them
    raw_training = _require(data, "training", "manifest")
    _warn_unknown(raw_training, {"defaults", "bounds", "presets", "embeddings"}, "training", warnings)
    defaults = _require(raw_training, "defaults", "training")
    if not isinstance(defaults, dict):
        msg = "training.defaults: expected a mapping"
        raise ManifestError(msg)
    raw_bounds = _require(raw_training, "bounds", "training")
    if not isinstance(raw_bounds, dict):
        msg = "training.bounds: expected a mapping"
        raise ManifestError(msg)
    bounds: dict[str, tuple[float, float]] = {}
    for key, pair in raw_bounds.items():
        where = f"training.bounds.{key}"
        if not isinstance(pair, list) or len(pair) != 2:
            msg = f"{where}: expected [min, max]"
            raise ManifestError(msg)
        lo, hi = _number(pair[0], f"{where}[0]"), _number(pair[1], f"{where}[1]")
        if lo > hi:
            msg = f"{where}: min {lo} > max {hi}"
            raise ManifestError(msg)
        bounds[key] = (lo, hi)
    for key, value in defaults.items():
        if key in bounds and isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = bounds[key]
            if not lo <= value <= hi:
                msg = f"training.defaults.{key}: {value} outside bounds [{lo}, {hi}]"
                raise ManifestError(msg)
    raw_presets = _require(raw_training, "presets", "training")
    if not isinstance(raw_presets, dict):
        msg = "training.presets: expected a mapping"
        raise ManifestError(msg)
    presets: dict[str, dict[str, Any]] = {}
    for name, values in raw_presets.items():
        if not isinstance(values, dict):
            msg = f"training.presets.{name}: expected a mapping"
            raise ManifestError(msg)
        for key, value in values.items():
            if key in bounds and isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = bounds[key]
                if not lo <= value <= hi:
                    msg = f"training.presets.{name}.{key}: {value} outside bounds [{lo}, {hi}]"
                    raise ManifestError(msg)
        presets[str(name)] = dict(values)
    raw_emb = _require(raw_training, "embeddings", "training")
    _warn_unknown(raw_emb, {"method", "fallback", "n_components"}, "training.embeddings", warnings)
    embeddings = Embeddings(
        method=_str(_require(raw_emb, "method", "training.embeddings"), "training.embeddings.method"),
        fallback=_str(_require(raw_emb, "fallback", "training.embeddings"), "training.embeddings.fallback"),
        n_components=_int(
            _require(raw_emb, "n_components", "training.embeddings"), "training.embeddings.n_components", minimum=2
        ),
    )
    if embeddings.n_components > 3:
        msg = f"training.embeddings.n_components: the Dashboard plots 2 or 3, got {embeddings.n_components}"
        raise ManifestError(msg)
    training = Training(defaults=dict(defaults), bounds=bounds, presets=presets, embeddings=embeddings)

    # submission
    raw_sub = _require(data, "submission", "manifest")
    _warn_unknown(raw_sub, {"columns", "metric", "daily_limit"}, "submission", warnings)
    raw_cols = _require(raw_sub, "columns", "submission")
    if not isinstance(raw_cols, list) or not raw_cols:
        msg = "submission.columns: expected a non-empty list"
        raise ManifestError(msg)
    submission = Submission(
        columns=tuple(_str(c, f"submission.columns[{i}]") for i, c in enumerate(raw_cols)),
        metric=_str(_require(raw_sub, "metric", "submission"), "submission.metric"),
        daily_limit=_int(_require(raw_sub, "daily_limit", "submission"), "submission.daily_limit", minimum=1),
    )

    # ui — display strings renderable, help links https only
    raw_ui = _require(data, "ui", "manifest")
    _warn_unknown(raw_ui, {"loop_banner_text", "help_links"}, "ui", warnings)
    raw_links = raw_ui.get("help_links", [])
    if not isinstance(raw_links, list):
        msg = "ui.help_links: expected a list"
        raise ManifestError(msg)
    links: list[HelpLink] = []
    for i, entry in enumerate(raw_links):
        where = f"ui.help_links[{i}]"
        _warn_unknown(entry if isinstance(entry, dict) else {}, {"label", "url"}, where, warnings)
        url = _str(_require(entry, "url", where), f"{where}.url")
        if not url.startswith("https://"):
            msg = f"{where}.url: help links must be https"
            raise ManifestError(msg)
        links.append(HelpLink(label=_display(_require(entry, "label", where), f"{where}.label"), url=url))
    ui = Ui(
        loop_banner_text=_display(_require(raw_ui, "loop_banner_text", "ui"), "ui.loop_banner_text"),
        help_links=tuple(links),
    )

    return Manifest(
        schema_version=version,
        competition=competition,
        classes=tuple(classes),
        kit=kit,
        splits=splits,
        model=model,
        training=training,
        submission=submission,
        ui=ui,
        source=source,
        source_detail=source_detail,
        sha256=sha256,
        fetched_at=fetched_at,
        document_url=doc_url,
        warnings=tuple(warnings),
    )


def parse_index(data: Any, index_url_: str) -> list[dict[str, Any]]:
    """Validate the index and return its competition entries with absolute ``manifest_url``.

    ``manifest_url`` is relative to the INDEX's own URL (``intel-scene/manifest.json`` beside
    ``kaggle/classification-index.json`` resolves to ``kaggle/intel-scene/manifest.json``); an
    absolute value is accepted only on the index's host — the index must not send the plugin
    elsewhere.
    """
    base = index_url_
    if not isinstance(data, dict):
        msg = "index: top level must be a mapping"
        raise ManifestError(msg)
    version = _int(_require(data, "schema_version", "index"), "index.schema_version")
    if version != INDEX_SCHEMA_VERSION:
        msg = f"index.schema_version: this plugin understands {INDEX_SCHEMA_VERSION}, got {version}"
        raise ManifestError(msg)
    raw = _require(data, "competitions", "index")
    if not isinstance(raw, list):
        msg = "index.competitions: expected a list"
        raise ManifestError(msg)
    base_host = (urllib.parse.urlsplit(base).hostname or "").lower()
    entries: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        where = f"index.competitions[{i}]"
        cid = _str(_require(entry, "id", where), f"{where}.id")
        if not _ID_RE.fullmatch(cid):
            msg = f"{where}.id: must be lowercase letters, digits and hyphens, got {cid!r}"
            raise ManifestError(msg)
        url = urllib.parse.urljoin(base, _str(_require(entry, "manifest_url", where), f"{where}.manifest_url"))
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or (parts.hostname or "").lower() != base_host:
            msg = f"{where}.manifest_url: must stay on {base_host!r}, got {url!r}"
            raise ManifestError(msg)
        entries.append({
            "id": cid,
            "display_name": _display(_require(entry, "display_name", where), f"{where}.display_name"),
            "manifest_url": url,
            "active": _bool(_require(entry, "active", where), f"{where}.active"),
        })
    if len({e["id"] for e in entries}) != len(entries):
        msg = "index.competitions: ids must be unique"
        raise ManifestError(msg)
    return entries


# ── Loaders: bundled, remote, cache ────────────────────────────────────────


def bundled_path(competition_id: str = DEFAULT_COMPETITION_ID) -> Path:
    return BUNDLED_DIR / f"{competition_id}-v{SCHEMA_VERSION}.yaml"


def load_yaml_text(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def _decode_document(raw: bytes) -> Any:
    """``manifest.json`` is JSON; YAML is accepted too (JSON is a YAML subset, so one parser)."""
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except ValueError:
        return load_yaml_text(text)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_bundled(competition_id: str = DEFAULT_COMPETITION_ID) -> Manifest:
    """The manifest shipped inside the wheel — always available, possibly stale. Its kit resolves
    against the canonical layout URL under the base in force (prod, or the override)."""
    path = bundled_path(competition_id)
    raw = path.read_bytes()
    return parse_manifest(
        load_yaml_text(raw.decode("utf-8")),
        source="bundled",
        source_detail=str(path),
        sha256=_sha256_bytes(raw),
        document_url=f"{base_url()}/{manifest_path(competition_id)}",
    )


def base_url() -> str:
    """The CDN base URL: the env override for dev/local, else the prod constant."""
    return (os.environ.get(MANIFEST_BASE_URL_ENV) or MANIFEST_BASE_URL).rstrip("/")


def index_url() -> str:
    return f"{base_url()}/{INDEX_PATH}"


def _http_get(url: str, timeout: float) -> bytes:
    """One thin seam over urllib (the tests' stub point)."""
    req = urllib.request.Request(url, headers={"User-Agent": "3lc-compute-plugin-kaggle-classification"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_bytes(url: str, *, budget_s: float | None = None) -> bytes:
    """GET ``url`` within one total budget (connect + read), one retry inside that budget."""
    budget = FETCH_BUDGET_S if budget_s is None else budget_s
    deadline = time.monotonic() + budget
    last: Exception | None = None
    for _attempt in range(1 + FETCH_RETRIES):
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            break
        try:
            return _http_get(url, timeout=remaining)
        except (TimeoutError, urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
    msg = f"{url}: {last if last is not None else 'budget exhausted'}"
    raise FetchError(msg)


def cache_dir() -> Path:
    return storage.plugin_home() / CACHE_DIR_NAME


def _cache_paths(competition_id: str) -> tuple[Path, Path]:
    d = cache_dir()
    return d / f"{competition_id}.manifest.json", d / f"{competition_id}.meta.json"


def write_cache(competition_id: str, data: Any, *, raw: bytes, source_url: str, fetched_at: str) -> dict[str, Any]:
    """Store the validated document plus the sidecar ``{fetched_at, source_url, sha256}``."""
    doc_path, meta_path = _cache_paths(competition_id)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fetched_at": fetched_at,
        "source_url": source_url,
        "sha256": _sha256_bytes(raw),
        "competition_id": competition_id,
        "bytes": len(raw),
    }
    for path, payload in ((doc_path, data), (meta_path, meta)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        tmp.replace(path)
    return meta


def read_cache(competition_id: str, *, hosts: frozenset[str] | set[str] | None = None) -> Manifest | None:
    """The last remote document that validated, or None (missing, unreadable, no longer valid, or
    fetched from a host the allowlist in force no longer admits — a dev-tier cache is not served
    once the override is gone)."""
    doc_path, meta_path = _cache_paths(competition_id)
    if not (doc_path.is_file() and meta_path.is_file()):
        return None
    try:
        data = json.loads(doc_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return parse_manifest(
            data,
            source="cache",
            source_detail=str(doc_path),
            sha256=str(meta.get("sha256") or ""),
            fetched_at=meta.get("fetched_at"),
            document_url=str(meta.get("source_url") or ""),
            hosts=hosts,
        )
    except (OSError, ValueError) as exc:  # ManifestError is a ValueError
        _log.warning("manifest cache for %s unusable: %s", competition_id, exc)
        return None


def cache_meta(competition_id: str) -> dict[str, Any] | None:
    _doc, meta_path = _cache_paths(competition_id)
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── Competition selection (more than one active) ──────────────────────────


def selected_competition_id() -> str | None:
    from kaggle_classification import session

    chosen = session.load().get("competition")
    cid = str((chosen or {}).get("id") or "").strip() if isinstance(chosen, dict) else ""
    # Defensive: the store validates on save, but a hand-edited file must not reach a path.
    return cid if cid and _ID_RE.fullmatch(cid) else None


def select_competition(competition_id: str) -> None:
    from kaggle_classification import session

    if not _ID_RE.fullmatch(competition_id):
        msg = f"competition id must be lowercase letters, digits and hyphens, got {competition_id!r}"
        raise ValueError(msg)
    session.save({"competition": {"id": competition_id}})


# ── Resolution ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    manifest: Manifest
    warnings: tuple[str, ...] = ()
    # The active competitions the index listed when more than one was active (the picker).
    candidates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "provenance": self.manifest.provenance,
            "warnings": list(self.warnings),
            "candidates": list(self.candidates),
        }


def _local(competition_id: str, warnings: list[str], candidates: tuple[dict[str, Any], ...], *, allowed) -> Resolution:
    cached = read_cache(competition_id, hosts=allowed)
    if cached is not None:
        return Resolution(cached, tuple(warnings), candidates)
    if bundled_path(competition_id).is_file():
        return Resolution(load_bundled(competition_id), tuple(warnings), candidates)
    warnings.append(
        f"No bundled manifest for competition {competition_id!r}; using the bundled {DEFAULT_COMPETITION_ID} manifest."
    )
    return Resolution(load_bundled(DEFAULT_COMPETITION_ID), tuple(warnings), candidates)


def _fallback_note(competition_id: str) -> str:
    meta = cache_meta(competition_id)
    if meta and read_cache(competition_id) is not None:
        return f"using the cached copy from {meta.get('fetched_at')}"
    return "using the bundled copy"


def resolve(
    *,
    network: bool = True,
    competition_id: str | None = None,
    hosts: frozenset[str] | set[str] | None = None,
) -> Resolution:
    """Resolve the competition manifest: remote (when reachable and valid) -> cache -> bundled.

    ``network=False`` is the fragment's first render: cache or bundled, no fetch. Warnings
    are participant-facing and the UI shows them; every fallback says why it happened.
    """
    warnings: list[str] = []
    candidates: tuple[dict[str, Any], ...] = ()
    cid = competition_id or selected_competition_id() or DEFAULT_COMPETITION_ID

    if network:
        chosen: dict[str, Any] | None = None
        try:
            entries = parse_index(json.loads(fetch_bytes(index_url()).decode("utf-8")), index_url())
        except FetchError as exc:
            _log.info("manifest index unreachable: %s", exc)
            warnings.append(f"The competition index could not be reached; {_fallback_note(cid)}.")
        except (ManifestError, ValueError) as exc:
            _log.warning("manifest index invalid: %s", exc)
            warnings.append(f"The competition index on the CDN is invalid ({exc}); {_fallback_note(cid)}.")
        else:
            active = [e for e in entries if e["active"]]
            if not active:
                warnings.append("The competition index lists no active competition; using the bundled manifest.")
                return Resolution(load_bundled(DEFAULT_COMPETITION_ID), tuple(warnings), ())
            if len(active) == 1:
                chosen = active[0]
            else:
                candidates = tuple(active)
                preferred = competition_id or selected_competition_id() or DEFAULT_COMPETITION_ID
                chosen = next((e for e in active if e["id"] == preferred), None)
                if chosen is None:
                    warnings.append(
                        "More than one competition is active and none is selected; choose one. "
                        f"Showing the {_fallback_note(cid).replace('using ', '')} meanwhile."
                    )
        if chosen is not None:
            cid = chosen["id"]
            url = chosen["manifest_url"]
            try:
                raw = fetch_bytes(url)
            except FetchError as exc:
                _log.info("manifest unreachable: %s", exc)
                warnings.append(f"The manifest on the CDN could not be reached; {_fallback_note(cid)}.")
            else:
                fetched_at = _now_iso()
                try:
                    data = _decode_document(raw)
                    manifest = parse_manifest(
                        data,
                        source="remote",
                        source_detail=url,
                        sha256=_sha256_bytes(raw),
                        fetched_at=fetched_at,
                        document_url=url,
                        hosts=hosts,
                    )
                except (ManifestError, ValueError, UnicodeDecodeError) as exc:
                    _log.warning("remote manifest %s failed validation: %s", url, exc)
                    warnings.append(f"The manifest on the CDN is invalid ({exc}); {_fallback_note(cid)}.")
                else:
                    write_cache(cid, data, raw=raw, source_url=url, fetched_at=fetched_at)
                    warnings.extend(manifest.warnings)
                    _log.info(
                        "manifest: remote %s (kit %s, sha256 %s)", url, manifest.kit.version, manifest.sha256[:12]
                    )
                    return Resolution(manifest, tuple(warnings), candidates)

    resolution = _local(cid, warnings, candidates, allowed=hosts)
    m = resolution.manifest
    _log.info("manifest: %s source (%s), kit %s", m.source, m.source_detail, m.kit.version)
    return resolution


def load_manifest(*, network: bool = True) -> Manifest:
    """The resolved manifest only (callers that do not need warnings or candidates)."""
    return resolve(network=network).manifest


def resolve_manifest_for_job() -> tuple[Manifest, dict[str, Any]]:
    """Re-resolve at job start and return the document plus the provenance record every job
    writes into its outputs (sha256, source, competition id, kit version)."""
    resolution = resolve(network=True)
    for note in resolution.warnings:
        _log.warning("manifest (job start): %s", note)
    return resolution.manifest, resolution.manifest.provenance


# ── Background refresh (the fragment never waits on the network) ──────────

_refresh_lock = threading.Lock()
_refresh: dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None, "error": None, "source": None}


def refresh_status() -> dict[str, Any]:
    with _refresh_lock:
        return dict(_refresh)


def refresh_in_background(*, force: bool = False) -> dict[str, Any]:
    """Start a remote resolution on a thread unless one is running or a recent one finished."""
    with _refresh_lock:
        if _refresh["state"] == "running":
            return dict(_refresh)
        finished = _refresh.get("finished_at")
        if not force and finished is not None and time.monotonic() - float(finished) < REFRESH_TTL_S:
            return dict(_refresh)
        _refresh.update({"state": "running", "started_at": time.monotonic(), "finished_at": None, "error": None})

    def _run() -> None:
        try:
            resolution = resolve(network=True)
            with _refresh_lock:
                _refresh.update({"state": "done", "source": resolution.manifest.source, "error": None})
        except Exception as exc:  # a refresh failure must never take the routes down
            _log.exception("manifest refresh failed")
            with _refresh_lock:
                _refresh.update({"state": "failed", "error": str(exc)})
        finally:
            with _refresh_lock:
                _refresh["finished_at"] = time.monotonic()

    threading.Thread(target=_run, name="kaggle-classification-manifest-refresh", daemon=True).start()
    return refresh_status()


def reset_refresh_state() -> None:
    """Tests only."""
    with _refresh_lock:
        _refresh.update({"state": "idle", "started_at": None, "finished_at": None, "error": None, "source": None})
