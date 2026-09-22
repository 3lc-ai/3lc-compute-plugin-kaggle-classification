# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Packaging invariants: the version strings agree, the wheel is complete and carries nothing
dev-only, the catalog is internally consistent, the SDK pin resolves on the hosts that exist,
the import stays light, and the license lineage is clean.

Every other test imports straight from ``src/``; this one builds the real wheel and asserts
against the artifact (an editable install never reads its own metadata back).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DIST_NAME = "3lc-compute-plugin-kaggle-classification"
IMPORT_NAME = "kaggle_classification"
FIXTURES = Path(__file__).parent / "fixtures"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _plugin_toml() -> dict:
    return tomllib.loads((ROOT / "src" / IMPORT_NAME / "plugin.toml").read_text(encoding="utf-8"))


def _catalog() -> dict:
    return json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> Path:
    """The real wheel, built with hatchling directly (offline, no isolated env)."""
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_the_version_strings_agree(wheel):
    """pyproject, plugin.toml, and the built metadata (what the footer reads back)."""
    versions = {
        "pyproject [project]": _pyproject()["project"]["version"],
        "plugin.toml": _plugin_toml()["version"],
    }
    with zipfile.ZipFile(wheel) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        text = zf.read(meta).decode("utf-8")
    match = re.search(r"^Version: (.+)$", text, re.M)
    assert match, "built wheel has no Version in its METADATA"
    versions["wheel METADATA"] = match.group(1).strip()
    assert len(set(versions.values())) == 1, f"version strings disagree: {versions}"


def test_the_description_is_the_same_on_every_surface():
    """plugin.toml and the newest catalog manifest (the Available card reads the catalog, the
    installed card reads plugin.toml). Older catalog entries record what shipped and are not
    checked; ``[project] description`` is the PyPI Summary, deliberately its own string."""
    newest = _catalog()["plugins"][0]["versions"][0]
    descriptions = {
        "plugin.toml": _plugin_toml()["description"],
        f"catalog {newest['version']} manifest": newest["manifest"]["description"],
    }
    assert len(set(descriptions.values())) == 1, f"descriptions disagree: {descriptions}"


def test_wheel_carries_the_data_files_and_nothing_dev_only(wheel):
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    for required in (
        f"{IMPORT_NAME}/plugin.toml",
        f"{IMPORT_NAME}/ui/ui.html",
        f"{IMPORT_NAME}/manifests/intel-scene-v1.yaml",
    ):
        assert required in names, f"{required} missing from the wheel"
    leaked = [n for n in names if n.startswith(("tools/", "tests/", "scripts/", "docs/"))]
    assert not leaked, f"dev-only files in the wheel: {leaked}"


def test_catalog_entries_are_internally_consistent():
    plugins = _catalog()["plugins"]
    assert len(plugins) == 1, "catalog carries more than one plugin block"
    for entry in plugins[0]["versions"]:
        version = entry["version"]
        assert entry["manifest"]["version"] == version
        tag = re.search(r"@v([^\s@]+)$", entry["source"])
        assert tag and tag.group(1) == version, f"catalog {version}: source does not pin @v{version}"
        assert entry["source"].startswith(f"{DIST_NAME}[{_plugin_toml()['runtime']['provision_extra']}] @ git+")


def test_catalog_ids_and_entry_point_match_the_plugin_id():
    plugin_id = _plugin_toml()["id"]
    plugins = _catalog()["plugins"]
    assert plugins[0]["id"] == plugin_id
    for entry in plugins[0]["versions"]:
        assert entry["manifest"]["id"] == plugin_id
        for key in ("min_service_version", "entrypoint", "provision_extra"):
            src = _plugin_toml() if key == "min_service_version" else _plugin_toml()["runtime"]
            dst = entry["manifest"] if key == "min_service_version" else entry["manifest"]["runtime"]
            assert src[key] == dst[key], f"catalog {entry['version']}: {key} differs from plugin.toml"
    entry_points = _pyproject()["project"]["entry-points"]["tlc_compute.plugins"]
    assert list(entry_points) == [plugin_id], "entry-point key must equal the plugin id"
    assert entry_points[plugin_id] == IMPORT_NAME


def test_catalog_versions_are_unique_and_newest_first():
    versions = [e["version"] for e in _catalog()["plugins"][0]["versions"]]
    assert len(versions) == len(set(versions))
    key = lambda v: tuple(int(p) for p in v.split("."))  # noqa: E731
    assert versions == sorted(versions, key=key, reverse=True)


def test_min_service_version_is_the_1_1_floor():
    """1.1.0 carries the torch-backend fix a clean Windows install needs (Gate 0 decision)."""
    assert _plugin_toml()["min_service_version"] == "1.1.0"


# ── The SDK pin resolves on the hosts that exist ─────────────────────────


def _our_sdk_specifier():
    from packaging.requirements import Requirement

    deps = [Requirement(d) for d in _pyproject()["project"]["dependencies"]]
    (sdk,) = [d for d in deps if d.name == "3lc-compute-plugin-sdk"]
    return sdk.specifier


def _overlap(ours, theirs, versions) -> list[str]:
    from packaging.version import Version

    return [v for v in versions if ours.contains(Version(v)) and theirs.contains(Version(v))]


def test_sdk_pin_overlaps_latest_compute_snapshot():
    """Offline half: against the committed snapshot of 3lc-compute's latest release metadata."""
    from packaging.requirements import Requirement

    snap = json.loads((FIXTURES / "compute_latest_requires.json").read_text(encoding="utf-8"))
    theirs = Requirement(snap["sdk_requirement"]).specifier
    both = _overlap(_our_sdk_specifier(), theirs, snap["sdk_released_versions"])
    assert both, (
        f"our pin {_our_sdk_specifier()} shares no released SDK version "
        f"with 3lc-compute {snap['compute_version']} ({theirs})"
    )


def test_sdk_pin_overlaps_latest_compute_live():
    """Network half: the same check against PyPI right now. Skips offline; when the live
    metadata differs from the snapshot it prints the new values so the fixture can be refreshed."""
    import urllib.request

    from packaging.requirements import Requirement

    try:
        with urllib.request.urlopen("https://pypi.org/pypi/3lc-compute/json", timeout=10) as r:
            compute = json.load(r)
        with urllib.request.urlopen("https://pypi.org/pypi/3lc-compute-plugin-sdk/json", timeout=10) as r:
            sdk = json.load(r)
    except Exception as exc:  # offline CI
        pytest.skip(f"PyPI unreachable: {exc}")
    reqs = [Requirement(r) for r in compute["info"]["requires_dist"] or []]
    (their_req,) = [r for r in reqs if r.name == "3lc-compute-plugin-sdk"]
    released = sorted(sdk["releases"], key=lambda v: tuple(int(p) for p in v.split(".")))
    both = _overlap(_our_sdk_specifier(), their_req.specifier, released)
    snap = json.loads((FIXTURES / "compute_latest_requires.json").read_text(encoding="utf-8"))
    drift = compute["info"]["version"] != snap["compute_version"] or str(their_req) != snap["sdk_requirement"]
    print(f"\n3lc-compute {compute['info']['version']} requires {their_req}; released SDKs {released}; overlap {both}")
    assert both, (
        f"our pin {_our_sdk_specifier()} shares no released SDK version "
        f"with 3lc-compute {compute['info']['version']} ({their_req})"
    )
    if drift:
        print("SNAPSHOT DRIFT: refresh tests/fixtures/compute_latest_requires.json with the values above")


# ── Import weight and license lineage ────────────────────────────────────


def test_package_import_is_light():
    """`import kaggle_classification` must not pull torch, timm, yaml, tlc or litestar: the host
    imports the entrypoint in a fresh worker and validation routes must stay cheap. ``routes``
    is the one module that imports litestar at module level (its handlers' annotations must
    resolve in its globals); ``__init__`` loads it lazily, so it is deliberately not in this list."""
    code = (
        "import sys; sys.path.insert(0, 'src'); import kaggle_classification, kaggle_classification.manifest, "
        "kaggle_classification.session, kaggle_classification.kit, kaggle_classification.storage; "
        "heavy = [m for m in ('torch', 'timm', 'yaml', 'tlc', 'litestar', 'PIL', 'numpy') if m in sys.modules]; "
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"heavy modules imported at package import: {proc.stdout} {proc.stderr}"


def test_license_headers_and_no_agpl_text():
    """Apache-2.0 on every shipped module; adapted files carry the relicensing header; the
    sibling plugin's copyleft licence identifier appears nowhere in code."""
    copyleft = "A" + "GPL"  # spelled apart so this file passes its own check
    py_files = [
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "tests").rglob("*.py"),
        *(ROOT / "tools").rglob("*.py"),
        *(ROOT / "scripts").rglob("*.py"),
    ]
    assert py_files
    for path in py_files:
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
        assert "SPDX-License-Identifier: Apache-2.0" in head, f"{path.relative_to(ROOT)} lacks the Apache header"
        assert copyleft not in path.read_text(encoding="utf-8"), f"{path.relative_to(ROOT)} mentions {copyleft}"
    assert _pyproject()["project"]["license"] == "Apache-2.0"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")


def test_adapted_modules_are_listed_for_sign_off():
    """Every file carrying the relicensing header is named in docs/STUDY.md G-5."""
    study = (ROOT / "docs" / "STUDY.md").read_text(encoding="utf-8")
    marker = "relicensed by the copyright holder"
    adapted = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for folder in ("src", "tests", "tools", "scripts")
        for p in (ROOT / folder).rglob("*.py")
        if marker in p.read_text(encoding="utf-8")
    )
    assert adapted, "expected at least one adapted module"
    missing = [p for p in adapted if f"`{p}`" not in study]
    assert not missing, f"adapted modules not listed in docs/STUDY.md G-5: {missing}"
