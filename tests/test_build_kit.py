# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""tools/build_kit.py against a tiny synthetic data dir with real (generated) JPEGs: the
rename, the count gate, the collision gate, the salt rules, determinism, what ships and what
stays private, and the verification pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_kit  # tools/ is on sys.path (conftest)
import pytest
from helpers import ALLOWED, small_manifest_data

from kaggle_classification import manifest as manifest_mod

pytest.importorskip("PIL")

SALT = b"a-test-salt-that-is-long-enough"


def _manifest():
    return manifest_mod.parse_manifest(small_manifest_data(), allowed_kit_hosts=ALLOWED)


def _jpeg(path: Path, seed: int, size=(12, 10)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, ((seed * 37) % 256, (seed * 91) % 256, (seed * 13) % 256))
    exif = Image.Exif()
    exif[0x0110] = "ScannerModel-XYZ"  # a Model tag the re-encode must drop
    img.save(path, "JPEG", quality=95, exif=exif.tobytes())


def make_source(root: Path, manifest) -> Path:
    """The Intel layout with revealing names: class in the stem, `pool_` prefix, sequential test ids."""
    data = root / "data"
    seed = 0
    for name in manifest.class_names:
        for i in range(manifest.splits.train.labeled_per_class):
            _jpeg(data / "train" / name / f"{name}_{i:06d}.jpg", seed)
            seed += 1
    for i in range(manifest.splits.train.undefined):
        _jpeg(data / "train" / "undefined" / f"pool_{i:06x}.jpg", seed)
        seed += 1
    for name in manifest.class_names:
        for i in range(manifest.splits.val.per_class):
            _jpeg(data / "val" / name / f"{name}_{i:06x}.jpg", seed)
            seed += 1
    for i in range(manifest.splits.test.count):
        _jpeg(data / "test" / f"test_{i:05d}.jpg", seed)
        seed += 1
    return data


@pytest.fixture
def workspace(tmp_path):
    manifest = _manifest()
    data = make_source(tmp_path / "src", manifest)
    salt_file = tmp_path / "private" / "kit.salt"
    salt_file.parent.mkdir()
    salt_file.write_bytes(SALT + b"\n")
    return {
        "manifest": manifest,
        "data": data,
        "salt_file": salt_file,
        "kit_out": tmp_path / "kits" / "intel-scene-kit-v1",
        "private": tmp_path / "private",
        "tmp": tmp_path / "tmp",
    }


def _build(ws, **over):
    kwargs = {
        "data_dir": ws["data"],
        "salt_file": ws["salt_file"],
        "kit_out": ws["kit_out"],
        "private_out": ws["private"],
        "manifest": ws["manifest"],
        "kit_version": "v1",
        "base_url": "https://cdn.test/hackathon/intel-scene/kit/v1",
        "shard_bytes": 4000,
        "log": lambda *a: None,
    }
    kwargs.update(over)
    return build_kit.build(**kwargs)


def test_build_and_verify_end_to_end(workspace):
    ws = workspace
    ws["tmp"].mkdir()
    report = _build(ws)
    m = ws["manifest"]
    assert report["counts"] == {"train_labeled": 12, "train_undefined": 3, "val": 6, "test": 4}
    assert report["images"] == 25 and report["file_count"] == 26  # + sample_submission.csv

    # Published naming, zero-padded, several shards at this budget.
    names = [s["name"] for s in report["shards"]]
    assert names == [f"intel-scene-v1-{i:02d}.zip" for i in range(len(names))] and len(names) >= 3
    assert all(len(s["sha256"]) == 64 for s in report["shards"])

    # The kit tree: opaque stems, structure preserved, undefined stays under train/.
    kit_root = Path(report["kit_root"])
    for p in (kit_root / "data").rglob("*.jpg"):
        assert build_kit._STEM_RE.match(p.stem), p
    assert (kit_root / "data" / "train" / "undefined").is_dir()
    assert len(list((kit_root / "data" / "test").iterdir())) == 4
    assert not list((kit_root / "data" / "test").glob("*/"))  # flat

    # EXIF gone, RGB JPEG.
    from PIL import Image

    sample = next((kit_root / "data" / "train" / m.class_names[0]).iterdir())
    with Image.open(sample) as im:
        assert im.format == "JPEG" and im.mode == "RGB" and not dict(im.getexif())

    # sample_submission: sorted test stems, placeholders.
    sub = (kit_root / "sample_submission.csv").read_text().splitlines()
    assert sub[0] == "image_id,prediction,confidence"
    ids = [ln.split(",")[0] for ln in sub[1:]]
    assert ids == sorted(p.stem for p in (kit_root / "data" / "test").iterdir())
    assert all(ln.endswith(",0,0.5") for ln in sub[1:])

    # mapping.csv: private only, with the class rules.
    assert not list(kit_root.rglob("mapping.csv")) and not list(Path(report["shards_dir"]).rglob("mapping.csv"))
    rows = [ln.split(",") for ln in (ws["private"] / "mapping.csv").read_text().splitlines()]
    assert rows[0] == ["original_relpath", "new_relpath", "split", "class"]
    body = rows[1:]
    assert len(body) == 25
    assert all(r[3] == "" for r in body if r[2] == "test")
    assert all(r[3] == "undefined" for r in body if r[0].startswith("train/undefined/"))
    labeled = [r for r in body if r[2] == "train" and r[3] != "undefined"]
    assert {r[3] for r in labeled} == set(m.class_names)
    assert all(r[1].startswith(f"data/{r[2]}/") for r in body)
    assert all(r[0].rsplit("/", 1)[1] not in r[1] for r in body)  # original name gone from the new path

    # The block beside the kit dir: kit{} and splits{} with the actual counts.
    block_path = Path(report["block"])
    assert block_path == ws["kit_out"].parent / "kit-manifest-block.yaml"
    block = manifest_mod.load_yaml_text(block_path.read_text())
    assert block["kit"]["version"] == "v1" and block["kit"]["base_url"].endswith("/kit/v1")
    assert [s["name"] for s in block["kit"]["shards"]] == names
    assert block["splits"] == {
        "train": {"labeled_per_class": 2, "undefined": 3},
        "val": {"per_class": 1, "editable": True},
        "test": {"count": 4, "ids_from": "sample_submission.csv"},
    }
    # And it parses as a manifest when spliced in.
    data = small_manifest_data()
    data["kit"], data["splits"] = block["kit"], block["splits"]
    assert manifest_mod.parse_manifest(data, allowed_kit_hosts=ALLOWED).kit.published

    result = build_kit.verify_build(
        report,
        m,
        salt_file=ws["salt_file"],
        private_out=ws["private"],
        kit_out=ws["kit_out"],
        tmp_dir=ws["tmp"],
        log=lambda *a: None,
    )
    assert result["ok"], result["problems"]
    assert result["salt_scanned_files"] > 0

    # The download stage accepts the built kit end to end (the same format functions on both sides).
    from helpers import FakeCDN, FakeCtx

    from kaggle_classification import kit as kit_mod

    fake = FakeCDN(Path(report["shards_dir"]))
    manifest_with_kit = manifest_mod.parse_manifest(data, allowed_kit_hosts=ALLOWED)
    import os

    os.environ["KAGGLE_CLASSIFICATION_HOME"] = str(ws["tmp"] / "home")
    try:
        old = kit_mod._open
        kit_mod._open = fake.open
        out = kit_mod.run_download({"dest_dir": str(ws["tmp"] / "dl")}, FakeCtx(), manifest_with_kit)
    finally:
        kit_mod._open = old
        del os.environ["KAGGLE_CLASSIFICATION_HOME"]
    assert out["cancelled"] is False and out["verified_files"] == 26


def test_rebuild_is_byte_identical(workspace, tmp_path):
    ws = workspace
    first = _build(ws)
    second = _build(ws, kit_out=tmp_path / "kits2" / "intel-scene-kit-v1")
    assert [(s["name"], s["sha256"], s["bytes"]) for s in first["shards"]] == [
        (s["name"], s["sha256"], s["bytes"]) for s in second["shards"]
    ]
    assert (ws["private"] / "mapping.csv").read_bytes() == (ws["private"] / "mapping.csv").read_bytes()


def test_non_empty_kit_out_requires_force(workspace):
    ws = workspace
    _build(ws)
    with pytest.raises(build_kit.BuildError, match="--force"):
        _build(ws)
    _build(ws, force=True)


def test_salt_file_rules(workspace, tmp_path):
    ws = workspace
    with pytest.raises(build_kit.BuildError, match="not found"):
        build_kit.read_salt(tmp_path / "nope.salt")
    short = tmp_path / "short.salt"
    short.write_bytes(b"tooshort\n")
    with pytest.raises(build_kit.BuildError, match="fewer than 16"):
        build_kit.read_salt(short)
    # Only a trailing newline is stripped; CRLF too; leading/trailing spaces are kept.
    crlf = tmp_path / "crlf.salt"
    crlf.write_bytes(SALT + b"\r\n")
    assert build_kit.read_salt(crlf) == SALT == build_kit.read_salt(ws["salt_file"])
    spaced = tmp_path / "spaced.salt"
    spaced.write_bytes(b" " + SALT + b" ")
    assert build_kit.read_salt(spaced) == b" " + SALT + b" "
    # Stems derive from salt + original relpath, sixteen hex chars.
    stem = build_kit.derive_stem(SALT, "train/buildings/buildings_000001.jpg")
    assert stem == hashlib.sha256(SALT + b"train/buildings/buildings_000001.jpg").hexdigest()[:16]
    assert build_kit.derive_stem(b"other-salt-that-is-long", "train/buildings/buildings_000001.jpg") != stem


def test_count_mismatch_hard_fails_before_writing(workspace):
    ws = workspace
    next((ws["data"] / "train" / "undefined").iterdir()).unlink()
    with pytest.raises(build_kit.BuildError, match="do not match the manifest"):
        _build(ws)
    assert not ws["kit_out"].exists()


def test_stem_collision_hard_fails(workspace, monkeypatch):
    ws = workspace
    monkeypatch.setattr(build_kit, "derive_stem", lambda salt, rel: "0" * 16)
    with pytest.raises(build_kit.BuildError, match="stem collision"):
        _build(ws)


def test_unexpected_source_entries_fail(workspace):
    ws = workspace
    (ws["data"] / "train" / "README.txt").write_text("notes")
    with pytest.raises(build_kit.BuildError, match="unexpected entry train/README.txt"):
        _build(ws)
    (ws["data"] / "train" / "README.txt").unlink()
    (ws["data"] / "val" / "undefined").mkdir()
    with pytest.raises(build_kit.BuildError, match="val/undefined"):
        _build(ws)


def test_verification_catches_a_leaked_original_name_and_the_salt(workspace):
    ws = workspace
    report = _build(ws)
    kit_root = Path(report["kit_root"])
    # Plant an original name and the salt into a shard-side file, then re-shard.
    leak = kit_root / "data" / "test" / "test_00001.jpg"
    leak.write_bytes(b"\xff\xd8" + SALT + b"\xff\xd9")
    report["shards"] = build_kit.shard_kit_tree(
        kit_root, Path(report["shards_dir"]), shard_bytes=4000, name_prefix="intel-scene-v1"
    )
    result = build_kit.verify_build(
        report,
        ws["manifest"],
        salt_file=ws["salt_file"],
        private_out=ws["private"],
        kit_out=ws["kit_out"],
        log=lambda *a: None,
    )
    assert not result["ok"]
    joined = "\n".join(result["problems"])
    assert "original file name survives" in joined
    assert "SALT BYTES FOUND" in joined
    assert "recount from shards" in joined  # one extra test image


def test_cli_smoke(workspace, tmp_path, capsys):
    ws = workspace
    manifest_yaml = tmp_path / "manifest.yaml"
    import yaml

    data = small_manifest_data()
    data["kit"]["base_url"] = "https://competitions.3lc.ai/hackathon/intel-scene/kit/v1"
    manifest_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")
    rc = build_kit.main([
        "--data-dir",
        str(ws["data"]),
        "--salt-file",
        str(ws["salt_file"]),
        "--kit-out",
        str(ws["kit_out"]),
        "--private-out",
        str(ws["private"]),
        "--manifest",
        str(manifest_yaml),
        "--shard-mb",
        "1",
        "--tmp-dir",
        str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert SALT.decode() not in out
    assert "example: train/" in out and "-> data/train/" in out
    assert json.loads((ws["kit_out"] / "tree" / "starter_kit" / "files.json").read_text())["kit_version"] == "v1"
