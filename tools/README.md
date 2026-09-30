# tools/ — dev-only

Scripts that never ship in the wheel (`[tool.hatch.build.targets.wheel] packages` names only
`src/kaggle_classification`; `tests/test_packaging.py` asserts nothing from here is packaged).

- `build_kit.py` — builds the salted, re-encoded starter kit from the raw data directory:
  `files.json` inside the kit, deterministic shards `<competition>-<version>-NN.zip`, the
  `kit{}` + `splits{}` block beside the kit dir, and the judge-only `mapping.csv` in the private
  output dir; then verifies the result from the participant's side. Takes `--salt-file`, never a
  salt value. Its format functions (`write_files_index`, `shard_kit_tree`) are the same ones the
  tests build synthetic kits with, so the download stage and the builder cannot drift apart.

```powershell
uv run python tools/build_kit.py --data-dir "<raw data>" --salt-file "<private>\kit.salt" --kit-out "<kits>\intel-scene-kit-v1" --private-out "<private>" --shard-mb 50
```

- `build_solution.py` — re-keys the judge's answer key onto the kit's opaque test ids: joins
  `mapping.csv`'s test rows to the original `solution.csv` on `image_id`, writes a new key with the
  same columns and values under the new ids (sorted like `sample_submission.csv`), and refuses to
  write unless the row count equals the manifest's `splits.test.count`, every test image and every
  old id match exactly once, and no old id survives. Never overwrites; optionally checks the
  written `image_id` set against the kit's `sample_submission.csv`. Reports counts only (per label,
  per Usage) — never a row.

```powershell
uv run python tools/build_solution.py --mapping "<private>\mapping.csv" --old-solution "<private>\solution.csv" --out "<private>\solution_kit_v1.csv" --sample-submission "<kits>\intel-scene-kit-v1\tree\starter_kit\sample_submission.csv"
```
