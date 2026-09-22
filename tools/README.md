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
