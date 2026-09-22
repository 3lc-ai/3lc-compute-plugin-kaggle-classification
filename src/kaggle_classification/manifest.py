# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The competition manifest — schema v1, validation, and the bundled fallback.

Everything competition-specific the plugin knows comes from one manifest
document: classes, split sizes, the locked model, training defaults and
bounds, the submission format, the kit shards, and the UI copy. No module in
this package carries a competition constant; they ask the ``Manifest``.

Resolution order (``load_manifest``): remote ``<base_url>/index.json`` +
``<id>/manifest.json`` -> on-disk cache (last good, with a fetched-at stamp)
-> the bundled ``manifests/<id>-v1.yaml``. Phase 1 ships the schema, the
validation and the bundled loader; the remote and cache legs land in Phase 2
(docs/PLAN.md, session 1 map).

Import-light: stdlib only at module level. PyYAML is imported inside the
loaders so ``import kaggle_classification`` stays cheap in a bare SDK venv.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# The remote base URL is a CODE CONSTANT (locked decision). The exact CDN prefix is still to be
# confirmed with the Hub team; the env override exists for dev and for the Phase 2 mock server.
MANIFEST_BASE_URL = "https://competitions.3lc.ai/hackathon"
MANIFEST_BASE_URL_ENV = "KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL"

# Which competition this build serves. One plugin build, one competition id; the manifest for
# that id is what gets resolved. (A future multi-competition fork reads this from settings.)
DEFAULT_COMPETITION_ID = "intel-scene"

BUNDLED_DIR = Path(__file__).resolve().parent / "manifests"

# The table revision a fresh session starts from (the ExDark convention).
DEFAULT_TABLE_NAME = "initial"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """A manifest document that fails schema v1 validation. The message names the field."""


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
    base_url: str
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
    # Provenance: "bundled" | "cache" | "remote", and a human-readable detail (path or URL).
    source: str = "bundled"
    source_detail: str = ""
    # Unknown fields seen while parsing (warned, never fatal).
    warnings: tuple[str, ...] = field(default=())

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


def parse_manifest(data: Any, *, source: str = "bundled", source_detail: str = "") -> Manifest:
    """Validate a decoded manifest document and return the typed ``Manifest``.

    Raises ``ManifestError`` naming the offending field. Unknown fields are
    recorded on ``Manifest.warnings`` and logged, never fatal — a newer
    manifest must still load on an older plugin.
    """
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
        display_name=_str(_require(comp, "display_name", "competition"), "competition.display_name"),
        deadline_utc=_str(_require(comp, "deadline_utc", "competition"), "competition.deadline_utc"),
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", competition.id):
        msg = f"competition.id: must be lowercase letters, digits and hyphens, got {competition.id!r}"
        raise ManifestError(msg)

    # classes — ids contiguous 0..N-1, names unique
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
                name=_str(_require(entry, "name", where), f"{where}.name"),
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

    # kit
    raw_kit = _require(data, "kit", "manifest")
    _warn_unknown(raw_kit, {"base_url", "version", "shards"}, "kit", warnings)
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
        shards.append(
            Shard(
                name=_str(_require(entry, "name", where), f"{where}.name"),
                sha256=sha,
                bytes=_int(_require(entry, "bytes", where), f"{where}.bytes", minimum=0),
            )
        )
    if len({s.name for s in shards}) != len(shards):
        msg = "kit.shards: shard names must be unique"
        raise ManifestError(msg)
    kit = Kit(
        base_url=_str(_require(raw_kit, "base_url", "kit"), "kit.base_url").rstrip("/"),
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

    # ui
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
        links.append(
            HelpLink(
                label=_str(_require(entry, "label", where), f"{where}.label"),
                url=_str(_require(entry, "url", where), f"{where}.url"),
            )
        )
    ui = Ui(
        loop_banner_text=_str(_require(raw_ui, "loop_banner_text", "ui"), "ui.loop_banner_text"),
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
        warnings=tuple(warnings),
    )


# ── Loaders ────────────────────────────────────────────────────────────────


def bundled_path(competition_id: str = DEFAULT_COMPETITION_ID) -> Path:
    return BUNDLED_DIR / f"{competition_id}-v{SCHEMA_VERSION}.yaml"


def load_yaml_text(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def load_bundled(competition_id: str = DEFAULT_COMPETITION_ID) -> Manifest:
    """The manifest shipped inside the wheel — always available, possibly stale."""
    path = bundled_path(competition_id)
    data = load_yaml_text(path.read_text(encoding="utf-8"))
    return parse_manifest(data, source="bundled", source_detail=str(path))


def base_url() -> str:
    """The remote base URL: the env override for dev, else the code constant."""
    return (os.environ.get(MANIFEST_BASE_URL_ENV) or MANIFEST_BASE_URL).rstrip("/")


def load_manifest(competition_id: str = DEFAULT_COMPETITION_ID) -> Manifest:
    """Resolve the competition manifest.

    Phase 1: the bundled document only. Phase 2 adds the remote and cache legs
    in front of it (remote newer-than-cache wins; unreachable falls through)
    and logs which source won and its version.
    """
    manifest = load_bundled(competition_id)
    _log.info("manifest: using %s source (%s), kit %s", manifest.source, manifest.source_detail, manifest.kit.version)
    return manifest
