# tools/ — dev-only

Scripts that never ship in the wheel (`[tool.hatch.build.targets.wheel] packages` names only
`src/kaggle_classification`; `tests/test_packaging.py` asserts nothing from here is packaged).

- `build_kit.py` (Phase 3 of session 1): builds the salted, re-encoded starter kit from the raw
  data directory, emits `files.json` inside the kit, the shard zips, the `kit{}` manifest block,
  and writes the judge-only `mapping.csv` to a PRIVATE output directory.
