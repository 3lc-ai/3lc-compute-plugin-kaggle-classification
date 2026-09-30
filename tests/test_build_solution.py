# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""tools/build_solution.py on SYNTHETIC fixtures only: a made-up mapping, a made-up original
key over made-up ids, a made-up sample_submission. Nothing here resembles the real answer key."""

from __future__ import annotations

import csv

import build_solution as bs  # tools/ is on sys.path (conftest)
import pytest

N_CLASSES = 3
N_TEST = 12  # 4 per class, 2 per class per Usage
ORIGINAL = [f"orig_{i:03d}" for i in range(1, N_TEST + 1)]
NEW = [f"{i * 2654435761 % 0xFFFFFFFF:08x}" for i in range(1, N_TEST + 1)]  # opaque, unsorted


def _write_csv(path, columns, rows, *, crlf=False, bom=False):
    text = "\n".join([",".join(columns), *(",".join(str(r[c]) for c in columns) for r in rows)]) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


def _mapping_rows(pairs):
    rows = [
        {"original_relpath": "train/a/x.jpg", "new_relpath": "data/train/a/aa.jpg", "split": "train", "class": "a"},
        {"original_relpath": "val/b/y.jpg", "new_relpath": "data/val/b/bb.jpg", "split": "val", "class": "b"},
    ]
    rows += [
        {"original_relpath": f"test/{o}.jpg", "new_relpath": f"data/test/{n}.jpg", "split": "test", "class": ""}
        for o, n in pairs
    ]
    return rows


def _old_rows(ids):
    rows = []
    for i, image_id in enumerate(ids):
        rows.append({"image_id": image_id, "Usage": "Public" if i % 2 == 0 else "Private", "label": i % N_CLASSES})
    return rows


@pytest.fixture
def files(tmp_path):
    mapping = _write_csv(
        tmp_path / "mapping.csv",
        ["original_relpath", "new_relpath", "split", "class"],
        _mapping_rows(zip(ORIGINAL, NEW)),
    )
    # The original key: CRLF + BOM like a file that went through Excel, and columns in a custom order.
    old = _write_csv(
        tmp_path / "solution.csv", ["image_id", "Usage", "label"], _old_rows(ORIGINAL), crlf=True, bom=True
    )
    sample = _write_csv(
        tmp_path / "sample_submission.csv",
        ["image_id", "prediction", "confidence"],
        [{"image_id": n, "prediction": 0, "confidence": 0.5} for n in sorted(NEW)],
    )
    return {"mapping": mapping, "old": old, "sample": sample, "out": tmp_path / "solution_kit_v1.csv"}


def _run(files, **overrides):
    mapping = bs.read_mapping_test(files["mapping"])
    columns, old_rows = bs.read_old_solution(files["old"])
    kwargs = {
        "mapping": mapping,
        "columns": columns,
        "old_rows": old_rows,
        "expected_rows": N_TEST,
        "num_classes": N_CLASSES,
    }
    kwargs.update(overrides)
    return bs.build(**kwargs)


def test_rekeys_every_row_and_keeps_columns_and_values(files):
    result = _run(files)
    assert result.ok, result.problems
    assert result.columns == ["image_id", "Usage", "label"]
    assert [r["image_id"] for r in result.rows] == sorted(NEW)
    # Same values under the new id: follow one row back through the mapping.
    by_new = {r["image_id"]: r for r in result.rows}
    old_by_id = {r["image_id"]: r for r in _old_rows(ORIGINAL)}
    for o, n in zip(ORIGINAL, NEW):
        assert by_new[n]["Usage"] == old_by_id[o]["Usage"]
        assert by_new[n]["label"] == str(old_by_id[o]["label"])
    assert not set(by_new) & set(ORIGINAL)


def test_counts_report_the_actual_split_and_the_balanced_flag(files):
    result = _run(files)
    c = result.counts
    assert c["rows"] == N_TEST and c["distinct_labels"] == N_CLASSES
    assert c["per_label"] == {"0": 4, "1": 4, "2": 4}
    assert c["per_usage"] == {"Private": 6, "Public": 6}
    assert c["expected_per_label"] == 4 and c["expected_per_label_per_usage"] == 2
    assert c["balanced"] is True
    # An unbalanced key is REPORTED, not a failure.
    skewed = _run(files, num_classes=2)
    assert skewed.ok and skewed.counts["balanced"] is False


def test_missing_and_extra_old_ids_fail_with_counts_only(files):
    old_rows = _old_rows([*ORIGINAL[:-2], "stray_1", "stray_2"])
    result = _run(files, old_rows=old_rows)
    assert not result.ok
    text = "\n".join(result.problems)
    assert "2 of 12 test image(s) have no row in the old solution" in text
    assert "2 of 12 old solution id(s) are not test images in the mapping" in text
    assert "10 row(s) built, expected exactly 12" in text
    assert "index range 11..12" in text
    for r in old_rows:  # never an id/label pair in the report
        assert f"{r['image_id']},{r['Usage']}" not in text


def test_row_count_must_equal_the_expectation(files):
    result = _run(files, expected_rows=N_TEST + 1)
    assert result.problems == [f"{N_TEST} row(s) built, expected exactly {N_TEST + 1}"]


def test_duplicate_mapping_stem_or_old_id_is_refused(files, tmp_path):
    dup = _write_csv(
        tmp_path / "dup_mapping.csv",
        ["original_relpath", "new_relpath", "split", "class"],
        _mapping_rows([*list(zip(ORIGINAL, NEW)), (ORIGINAL[0], "deadbeef")]),
    )
    with pytest.raises(bs.SolutionError, match="original test stem"):
        bs.read_mapping_test(dup)
    dup_old = _write_csv(tmp_path / "dup_old.csv", ["image_id", "Usage", "label"], _old_rows([*ORIGINAL, ORIGINAL[0]]))
    with pytest.raises(bs.SolutionError, match="appear more than once"):
        bs.read_old_solution(dup_old)


def test_missing_columns_are_refused(files, tmp_path):
    bad = _write_csv(tmp_path / "bad.csv", ["id", "label"], [{"id": "x", "label": 0}])
    with pytest.raises(bs.SolutionError, match="missing column"):
        bs.read_old_solution(bad)
    with pytest.raises(bs.SolutionError, match="missing column"):
        bs.read_mapping_test(bad)


def test_write_is_lf_utf8_and_never_overwrites(files):
    result = _run(files)
    bs.write_solution(files["out"], result.columns, result.rows)
    raw = files["out"].read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    with files["out"].open(newline="", encoding="utf-8") as fh:
        back = list(csv.DictReader(fh))
    assert back == result.rows
    with pytest.raises(bs.SolutionError, match="never overwrites"):
        bs.write_solution(files["out"], result.columns, result.rows)
    assert files["out"].read_bytes() == raw


def test_sample_submission_id_set_check(files):
    result = _run(files)
    bs.write_solution(files["out"], result.columns, result.rows)
    assert bs.check_ids_match_sample_submission(files["out"], files["sample"]) == []
    short = _write_csv(
        files["sample"].parent / "short.csv",
        ["image_id", "prediction", "confidence"],
        [{"image_id": n, "prediction": 0, "confidence": 0.5} for n in [*sorted(NEW)[:-1], "notinkit"]],
    )
    problems = bs.check_ids_match_sample_submission(files["out"], short)
    assert "1 solution id(s) missing from sample_submission" in problems
    assert "1 sample_submission id(s) missing from the solution" in problems


def test_cli_end_to_end(files, capsys):
    rc = bs.main(
        [
            "--mapping", str(files["mapping"]),
            "--old-solution", str(files["old"]),
            "--out", str(files["out"]),
            "--sample-submission", str(files["sample"]),
            "--expected-rows", str(N_TEST),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0, out
    assert files["out"].is_file()
    assert "image_id set equals sample_submission.csv: OK" in out
    for o in ORIGINAL:  # the report carries no id, let alone a row
        assert o not in out
    for n in NEW:
        assert n not in out
    # A second run refuses: the output exists.
    rc2 = bs.main(["--mapping", str(files["mapping"]), "--old-solution", str(files["old"]), "--out", str(files["out"])])
    assert rc2 == 1


def test_cli_refuses_the_original_as_out(files, capsys):
    rc = bs.main(["--mapping", str(files["mapping"]), "--old-solution", str(files["old"]), "--out", str(files["old"])])
    assert rc == 1
    assert "never written to" in capsys.readouterr().err
    assert files["old"].read_bytes().startswith(b"\xef\xbb\xbf")  # untouched


def test_cli_fails_and_writes_nothing_on_a_short_key(files, tmp_path, capsys):
    short_old = _write_csv(tmp_path / "short_old.csv", ["image_id", "Usage", "label"], _old_rows(ORIGINAL[:-3]))
    rc = bs.main([
        *("--mapping", str(files["mapping"])),
        *("--old-solution", str(short_old)),
        *("--out", str(files["out"])),
        *("--expected-rows", str(N_TEST)),
    ])
    captured = capsys.readouterr()
    assert rc == 1
    assert not files["out"].exists()
    assert "3 of 12 test image(s) have no row in the old solution" in captured.err
    assert "matched: 9 of 12 expected" in captured.out
