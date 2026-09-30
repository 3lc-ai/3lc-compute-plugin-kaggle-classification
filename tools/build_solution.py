# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-key the competition answer key onto the kit's opaque test ids — dev-only, judge-only.

    python tools/build_solution.py --mapping "<private>\\mapping.csv" --old-solution "<private>\\solution.csv"
        --out "<private>\\solution_kit_v1.csv" [--sample-submission "<kit>\\starter_kit\\sample_submission.csv"]

The kit builder renames every image to a salted opaque id and writes the judge-only
``mapping.csv`` (``original_relpath,new_relpath,split,class``). The original answer key is keyed by
the ORIGINAL test stems. This script joins the mapping's test rows to the old key on
``image_id`` and writes a new key with the SAME columns and values under the NEW ids, sorted by
the new id like ``sample_submission.csv``.

Hard checks — any failure means no file is written and exit 1:

* exactly ``expected_rows`` rows (the manifest's ``splits.test.count`` unless ``--expected-rows``);
* every new test stem appears exactly once; every old id matched exactly once (no test image
  without a label, no label without a test image, no duplicate on either side);
* no old id survives into the output;
* ``--out`` must not exist (never overwrite; the original key is never written to);
* with ``--sample-submission``: the written ``image_id`` set equals the kit's.

Reported, never enforced: the per-label and per-``Usage`` counts as they are in the file, beside
the balanced expectation (``rows / num_classes`` per label, half of that per label per Usage).

Answer-key material: this prints COUNTS ONLY — never an id/label pair, never a row. The inputs
and the output live in the private dir and nothing here is ever copied into the repo.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_classification import manifest as manifest_mod

ID_COLUMN = "image_id"
USAGE_COLUMN = "Usage"
LABEL_COLUMN = "label"
MAPPING_COLUMNS = ("original_relpath", "new_relpath", "split")


class SolutionError(ValueError):
    """A hard check failed; the message lists every failure, counts only."""


@dataclass
class Result:
    columns: list[str]
    rows: list[dict[str, str]]
    problems: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


# ── reading ────────────────────────────────────────────────────────────────


def _read_csv(path: Path, required: Iterable[str]) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        rows = [{k: (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    missing = [c for c in required if c not in columns]
    if missing:
        msg = f"{path.name}: missing column(s) {missing}; has {columns}"
        raise SolutionError(msg)
    return columns, rows


def read_mapping_test(path: Path) -> dict[str, str]:
    """``{original stem: new stem}`` over the mapping's test rows. Duplicates raise."""
    _, rows = _read_csv(path, MAPPING_COLUMNS)
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        msg = f"{path.name}: no rows with split == test"
        raise SolutionError(msg)
    originals = Counter(Path(r["original_relpath"]).stem for r in test)
    news = Counter(Path(r["new_relpath"]).stem for r in test)
    problems = []
    if dup := sum(1 for _, n in originals.items() if n > 1):
        problems.append(f"{dup} original test stem(s) appear more than once in the mapping")
    if dup := sum(1 for _, n in news.items() if n > 1):
        problems.append(f"{dup} new test stem(s) appear more than once in the mapping")
    if problems:
        raise SolutionError("; ".join(problems))
    return {Path(r["original_relpath"]).stem: Path(r["new_relpath"]).stem for r in test}


def read_old_solution(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    columns, rows = _read_csv(path, (ID_COLUMN,))
    ids = Counter(r[ID_COLUMN] for r in rows)
    if dup := sum(1 for _, n in ids.items() if n > 1):
        msg = f"{path.name}: {dup} image_id value(s) appear more than once"
        raise SolutionError(msg)
    return columns, rows


# ── the join ───────────────────────────────────────────────────────────────


def _index_range(ids: Iterable[str]) -> str:
    """A counts-only hint at WHICH ids are affected: the trailing-digit range, no label attached."""
    nums = []
    for i in ids:
        digits = "".join(ch for ch in i if ch.isdigit())
        if digits:
            nums.append(int(digits))
    return f"index range {min(nums)}..{max(nums)}" if nums else "no numeric index"


def build(
    *,
    mapping: dict[str, str],
    columns: list[str],
    old_rows: list[dict[str, str]],
    expected_rows: int,
    num_classes: int,
) -> Result:
    """Join, re-key and check. Never raises for a data failure — ``Result.problems`` lists them."""
    result = Result(columns=list(columns), rows=[])
    old_by_id = {r[ID_COLUMN]: r for r in old_rows}

    unmatched_test = sorted(set(mapping) - set(old_by_id))
    unmatched_old = sorted(set(old_by_id) - set(mapping))
    if unmatched_test:
        result.problems.append(
            f"{len(unmatched_test)} of {len(mapping)} test image(s) have no row in the old solution "
            f"({_index_range(unmatched_test)})"
        )
    if unmatched_old:
        result.problems.append(
            f"{len(unmatched_old)} of {len(old_by_id)} old solution id(s) are not test images in the mapping "
            f"({_index_range(unmatched_old)})"
        )

    new_rows = []
    for original, new in mapping.items():
        old = old_by_id.get(original)
        if old is None:
            continue
        new_rows.append({**old, ID_COLUMN: new})
    new_rows.sort(key=lambda r: r[ID_COLUMN])
    result.rows = new_rows

    if len(new_rows) != expected_rows:
        result.problems.append(f"{len(new_rows)} row(s) built, expected exactly {expected_rows}")
    new_ids = Counter(r[ID_COLUMN] for r in new_rows)
    if dup := sum(1 for _, n in new_ids.items() if n > 1):
        result.problems.append(f"{dup} new id(s) would appear more than once")
    if survivors := len(set(new_ids) & set(old_by_id)):
        result.problems.append(f"{survivors} old id(s) survive into the output")

    result.counts = summarize(new_rows, expected_rows=expected_rows, num_classes=num_classes)
    return result


def summarize(rows: list[dict[str, str]], *, expected_rows: int, num_classes: int) -> dict[str, Any]:
    """Counts as they are in the rows, beside the balanced expectation. Counts only."""
    per_label = Counter(r.get(LABEL_COLUMN, "") for r in rows)
    per_usage = Counter(r.get(USAGE_COLUMN, "") for r in rows)
    per_label_usage = Counter((r.get(LABEL_COLUMN, ""), r.get(USAGE_COLUMN, "")) for r in rows)
    expected_per_label = expected_rows // num_classes if num_classes else 0
    expected_per_label_usage = expected_per_label // 2
    balanced = (
        len(per_label) == num_classes
        and all(n == expected_per_label for n in per_label.values())
        and all(n == expected_per_label_usage for n in per_label_usage.values())
        and len(per_label_usage) == num_classes * 2
    )
    return {
        "rows": len(rows),
        "distinct_labels": len(per_label),
        "per_label": dict(sorted(per_label.items())),
        "per_usage": dict(sorted(per_usage.items())),
        "per_label_per_usage": {f"{label}/{usage}": n for (label, usage), n in sorted(per_label_usage.items())},
        "expected_per_label": expected_per_label,
        "expected_per_label_per_usage": expected_per_label_usage,
        "balanced": balanced,
    }


# ── writing + the sample_submission check ──────────────────────────────────


def write_solution(path: Path, columns: list[str], rows: list[dict[str, str]]) -> Path:
    """LF, UTF-8, no BOM, the input's column order. Refuses to overwrite anything."""
    if path.exists():
        msg = f"{path} already exists; this tool never overwrites (choose another --out)"
        raise SolutionError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def check_ids_match_sample_submission(solution: Path, sample_submission: Path) -> list[str]:
    """The written key's ``image_id`` set equals the kit's ``sample_submission.csv`` set. Counts only."""
    _, sol = _read_csv(solution, (ID_COLUMN,))
    _, sub = _read_csv(sample_submission, (ID_COLUMN,))
    sol_ids = {r[ID_COLUMN] for r in sol}
    sub_ids = {r[ID_COLUMN] for r in sub}
    problems = []
    if len(sol) != len(sol_ids):
        problems.append(f"{len(sol) - len(sol_ids)} duplicate id(s) in the solution")
    if len(sub) != len(sub_ids):
        problems.append(f"{len(sub) - len(sub_ids)} duplicate id(s) in sample_submission")
    if only_sol := len(sol_ids - sub_ids):
        problems.append(f"{only_sol} solution id(s) missing from sample_submission")
    if only_sub := len(sub_ids - sol_ids):
        problems.append(f"{only_sub} sample_submission id(s) missing from the solution")
    return problems


# ── CLI ────────────────────────────────────────────────────────────────────


def _print_counts(counts: dict[str, Any], log=print) -> None:
    log(f"rows: {counts['rows']}")
    log(f"distinct labels: {counts['distinct_labels']}")
    log(f"per label: {counts['per_label']}")
    log(f"per Usage: {counts['per_usage']}")
    log(f"per label per Usage: {counts['per_label_per_usage']}")
    if counts["balanced"]:
        log(
            f"split: balanced - {counts['expected_per_label']} per label, "
            f"{counts['expected_per_label_per_usage']} per label per Usage"
        )
    else:
        log(
            f"split: NOT the expected {counts['expected_per_label']} per label / "
            f"{counts['expected_per_label_per_usage']} per label per Usage - actual split reported above"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", required=True, type=Path, help="the judge-only mapping.csv from build_kit")
    parser.add_argument(
        "--old-solution", required=True, type=Path, help="the original answer key over the ORIGINAL ids (read-only)"
    )
    parser.add_argument("--out", required=True, type=Path, help="the new key; must not exist")
    parser.add_argument(
        "--sample-submission", type=Path, default=None, help="the kit's sample_submission.csv to compare id sets"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="manifest YAML for the expectations (default: bundled)"
    )
    parser.add_argument("--expected-rows", type=int, default=None, help="default: the manifest's splits.test.count")
    args = parser.parse_args(argv)

    if args.manifest:
        manifest = manifest_mod.parse_manifest(
            manifest_mod.load_yaml_text(args.manifest.read_text(encoding="utf-8")),
            source="file",
            source_detail=str(args.manifest),
        )
    else:
        manifest = manifest_mod.load_bundled()
    expected_rows = args.expected_rows if args.expected_rows is not None else manifest.splits.test.count

    if args.out.resolve() == args.old_solution.resolve():
        print("FAILED: --out is the original solution; it is never written to", file=sys.stderr)
        return 1
    if args.out.exists():
        print(f"FAILED: {args.out} already exists; this tool never overwrites", file=sys.stderr)
        return 1

    try:
        mapping = read_mapping_test(args.mapping)
        columns, old_rows = read_old_solution(args.old_solution)
    except SolutionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"mapping test rows: {len(mapping)}")
    print(f"old solution rows: {len(old_rows)}  columns: {columns}")
    result = build(
        mapping=mapping,
        columns=columns,
        old_rows=old_rows,
        expected_rows=expected_rows,
        num_classes=manifest.num_classes,
    )
    print(f"matched: {len(result.rows)} of {expected_rows} expected")
    _print_counts(result.counts)
    if not result.ok:
        print("FAILED - nothing written:", file=sys.stderr)
        for p in result.problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    try:
        write_solution(args.out, result.columns, result.rows)
    except SolutionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"written: {args.out}")

    if args.sample_submission:
        problems = check_ids_match_sample_submission(args.out, args.sample_submission)
        if problems:
            print(
                "FAILED - id set differs from sample_submission (file left in place for inspection):", file=sys.stderr
            )
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(f"image_id set equals {args.sample_submission.name}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
