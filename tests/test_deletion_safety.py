# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deletion safety: a participant's edited revision survives every plugin action.

The plugin has exactly one code path that deletes a table — ``importer.delete_tables`` — and it
is reachable only from the import job's failure rollback and cancel branches, on the URLs THAT
job created. Start over is a fragment-only view change, re-import writes fresh names and never
touches existing tables, and a kit reset (running the download again) rewrites kit files, not
tables. This module pins all of that: a source census, then a real edited-label revision of a
train table (an ``EditedTable``, the shape the Dashboard writes) that must still exist, with its
edit intact, after Start over, re-import, kit reset, and a failure rollback during a re-import.

Needs the heavy extra (tlc + PIL); skips without it (CLAUDE.md §B).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from helpers import FakeCDN, FakeCtx, build_synthetic_kit, make_kit_tree, small_manifest_data

import build_kit
from kaggle_classification import importer, kit, session
from kaggle_classification import manifest as manifest_mod

tlc = pytest.importorskip("tlc")
pytest.importorskip("PIL")

SRC = Path(importer.__file__).parent


# ── Census: the only deletion path, and where it is reachable from ─────────────────────────


def test_the_only_table_deletion_is_the_import_rollback():
    text = {p.name: p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")}
    deleters = [(n, i + 1) for n, t in text.items() for i, line in enumerate(t.splitlines()) if ".delete(" in line]
    assert deleters == [("importer.py", next(i + 1 for i, l in enumerate(text["importer.py"].splitlines()) if "u.delete()" in l))]
    for n, t in text.items():
        assert 'if_exists="overwrite"' not in t and "if_exists='overwrite'" not in t, n
        assert "rmtree" not in t, n
    body = text["importer.py"]
    # delete_tables is defined once and CALLED exactly twice: the cancel branch and the failure branch.
    calls = [m.start() for m in re.finditer(r"delete_tables\(created\)", body)]
    assert len(calls) == 2
    run_import_start = body.index("def run_import(")
    assert all(c > run_import_start for c in calls)
    # kit.py deletes only shard archives and its own .part/probe files — never a table.
    kit_text = text["kit.py"]
    for line in kit_text.splitlines():
        if "unlink(" in line:
            assert any(w in line for w in ("probe", "final", "part", "entry.name")), line


def test_start_over_is_a_view_change_only():
    """The form state (kgEnterFormState) is a view change: the form comes back, the banners clear, the download
    section re-resolves from the config payload. No job starts, nothing is deleted. rc11: Start over is gone;
    the one Re-import… action confirms with the server-named table and then starts a job that writes beside."""
    html = (SRC / "ui" / "ui.html").read_text(encoding="utf-8")
    assert "function kgEnterFormState(animate)" in html and "el('kg-start-over')" not in html
    body = re.search(r"function kgEnterFormState\(animate\) \{(.*?)\n      \}", html, re.S).group(1)
    assert "el('kg-import-form').style.display = ''" in body and "dlInit()" in body
    for forbidden in ("PluginJobs.start", "kgStartJob(", "DELETE", "'reimport'"):
        assert forbidden not in body, forbidden
    # No plugin route deletes anything, and the fragment never issues a DELETE.
    assert "method: 'DELETE'" not in html and '"DELETE"' not in html
    # Re-import… (rc11, item 10): a read-only preflight names the table first, the participant confirms, and only
    # then a mode=reimport (or, for a stale record, mode=import) job starts. Nothing is overwritten anywhere.
    reimport = re.search(r"function kgConfirmReimport\(mode\) \{(.*?)\n      \}", html, re.S).group(1)
    assert "/import/preflight?kit_dir=" in reimport and "kgStartImport({ reimport: mode === 'reimport'" in reimport
    assert "Your existing tables and label edits are kept." in reimport
    for forbidden in ("DELETE", "overwrite"):
        assert forbidden not in reimport, forbidden
    # No overwrite request can leave the fragment: no such string literal, no force flag.
    assert "'overwrite'" not in html and '"overwrite"' not in html and "force_splits" not in html


# ── Behaviour: an edited-label revision survives ────────────────────────────────────────────


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    root = tmp_path / "3lc-root"
    root.mkdir()
    monkeypatch.setattr(importer, "project_root_url", lambda: root.as_posix())
    return root


@pytest.fixture
def served(tmp_path, home, monkeypatch):
    """A synthetic REAL-image kit behind the fake CDN, downloaded once (the kit reset re-runs it)."""
    srv = tmp_path / "srv"
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(srv / "tree", data, real_images=True)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    cdn = srv / "cdn" / "v1"
    shards = build_kit.shard_kit_tree(kit_root, cdn, shard_bytes=4000)
    data["kit"] = build_kit.kit_block("starter-kit/v1/", "v1", shards)["kit"]
    manifest = manifest_mod.parse_manifest(
        data, source="test", source_detail=str(cdn), document_url="https://cdn.test/kaggle/intel-scene/manifest.json",
        hosts=frozenset({"cdn.test"}),
    )
    fake = FakeCDN(cdn)
    monkeypatch.setattr(kit, "_open", fake.open)
    dest = tmp_path / "dest"
    kit.run_download({"dest_dir": str(dest)}, FakeCtx(), manifest)  # publishes session.kit_dir
    return manifest, dest


def _edit_labels(train_url: str, manifest) -> tuple[str, dict[int, int]]:
    """An EditedTable revision beside the train table: rows 0 and 1 relabeled (the Dashboard's shape)."""
    from tlc._core.objects.tables.from_table.edited_table import EditedTable

    parent = tlc.Table.from_url(tlc.Url(train_url))
    new_labels = {0: (manifest.num_classes - 1), 1: 0}
    runs_and_values: list = []
    for idx, value in new_labels.items():
        runs_and_values += [[idx], value]
    edited = EditedTable(
        input_table_url=parent,
        edits={"label": {"runs_and_values": runs_and_values}},
        url=parent.url.create_sibling("initial_edited"),
    )
    edited.ensure_fully_defined()  # resolves the schema and row count from the parent, as add_column does
    edited.write_to_url()
    return str(edited.url), new_labels


def _assert_revision_intact(url: str, expected: dict[int, int], rows: int) -> None:
    t = tlc.Table.from_url(tlc.Url(url))
    t.ensure_fully_defined()
    assert len(t) == rows
    got = list(t.table_rows)
    for idx, value in expected.items():
        assert got[idx]["label"] == value


def test_edited_revision_survives_start_over_reimport_kit_reset_and_rollback(project_root, served, monkeypatch):
    manifest, dest = served
    first = importer.run_import({}, FakeCtx(), manifest)
    train_url = first["tables"]["train"]["url"]
    rows = first["tables"]["train"]["rows"]
    rev_url, expected = _edit_labels(train_url, manifest)
    assert tlc.Url(rev_url).exists()
    _assert_revision_intact(rev_url, expected, rows)
    # The parent keeps its original labels: the edit lives only in the revision.
    assert list(tlc.Table.from_url(tlc.Url(train_url)).table_rows)[0]["label"] == 0

    # 1. Start over: nothing server-side runs (see test_start_over_is_a_view_change_only); the
    #    revisit record is untouched and re-verifies.
    assert importer.import_state()["state"] == "success"
    _assert_revision_intact(rev_url, expected, rows)

    # 2. Re-import: fresh names, both originals and the revision untouched.
    second = importer.run_import({"mode": "reimport"}, FakeCtx(), manifest)
    assert second["table_name"] == "initial-2"
    assert tlc.Url(train_url).exists() and tlc.Url(first["tables"]["val"]["url"]).exists()
    _assert_revision_intact(rev_url, expected, rows)

    # 3. Kit reset: the download runs again over the same destination (re-extracts every kit
    #    file, deletes only shard archives). Tables and the revision are untouched.
    kit.run_download({"dest_dir": str(dest)}, FakeCtx(), manifest)
    assert tlc.Url(train_url).exists()
    _assert_revision_intact(rev_url, expected, rows)
    assert all(Path(r["image"]).is_file() for r in tlc.Table.from_url(tlc.Url(rev_url)).table_rows)

    # 4. Failure rollback during a further re-import: only the tables THAT job created go.
    real = importer.register_split

    def fail_val(split, *a, **k):
        if split == "val":
            msg = "simulated failure"
            raise OSError(msg)
        return real(split, *a, **k)

    monkeypatch.setattr(importer, "register_split", fail_val)
    with pytest.raises(RuntimeError, match="simulated failure"):
        importer.run_import({"mode": "reimport"}, FakeCtx(), manifest)
    assert not tlc.Url(importer.table_url("intel-scene", manifest.dataset_name("train"), "initial-3")).exists()
    for url in (train_url, first["tables"]["val"]["url"], second["tables"]["train"]["url"], second["tables"]["val"]["url"]):
        assert tlc.Url(url).exists(), url
    _assert_revision_intact(rev_url, expected, rows)
    # The record still names the last SUCCESSFUL import.
    assert session.load()["import_state"]["table_name"] == "initial-2"
