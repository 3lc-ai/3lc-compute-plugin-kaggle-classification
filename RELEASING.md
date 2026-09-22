# Releasing a plugin version

Publishing `vX.Y.Z` is two moves, **in this order** — the catalog points at
the tag, so the tag must exist before the catalog advertises it.

## 0. Before the tag

```powershell
uv run pytest
uv run python scripts/release_version.py
```

The suite must be green in the **heavy** venv (`uv sync --extra kaggle-classification --group dev`);
`test_timm_model.py` skips in a light venv and a skip is not a pass. The version
script checks that `pyproject.toml` and `src/kaggle_classification/plugin.toml`
agree; `--stamp X.Y.Z` moves both together. Update `CHANGELOG.md`, run `uv lock`,
commit.

## 1. Push the tag

```powershell
git tag vX.Y.Z
git push origin vX.Y.Z
```

The catalog's install `source` is a PEP-508 git reference pinned to this tag;
the shop installs whatever the tag points at, never the working copy.

## 2. Update `catalog.json`

Add a new entry to the `versions` array (newest first): bump `version`, point
`source` at the new tag, paste a fresh copy of the manifest (it must match
`plugin.toml`, `version` and `description` included), bump `generated_at`.
Commit and push. `tests/test_packaging.py` checks the entry's internal
consistency, the id/entry-point pairing, description parity with `plugin.toml`
and newest-first ordering; it deliberately does NOT assert that the newest
entry equals the shipped version, because the suite runs before the tag.

**Version pins that ride the same commit** (the census — grep for the old
string before you believe this list is complete):

- `pyproject.toml` `[project] version` and `plugin.toml` `version` (the script).
- `uv.lock`'s own package entry (regenerates: `uv lock`).
- `CONTEXT.md` "Where we are".
- `CHANGELOG.md` section header.

## The catalog URL

```
https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/HEAD/catalog.json
```

`HEAD` follows the default branch. On a 1.x host the default
`plugin_install_policy = "catalog-only"` refuses a catalog added through the API,
so the URL must be in `TLC_COMPUTE_PLUGIN_CATALOG_URLS` on the service command.

## Kit data releases (separate from code releases)

The kit ships through the competition manifest, not through this repo's tags.

- `tools/build_kit.py` produces `<kit out>/<version>/part-*.zip`, prints the
  `kit{}` block, and writes the judge-only `mapping.csv` to the PRIVATE output
  dir. The kit dir and the private dir are never committed.
- **A staged kit version is immutable.** Updating the kit means a new
  `kit.version` in the manifest with a new prefix, never overwritten objects.
- The `kit{}` block is pasted into the REMOTE manifest first; the bundled
  `manifests/intel-scene-v1.yaml` follows in the next code release. The remote
  prefix must be live before either manifest names it: the download stage has
  no fallback and fails the job on a 404.
- Verify after staging: HEAD returns `Accept-Ranges: bytes`, a `-r 0-1023` GET
  returns 206, and every shard's sha256 matches the block.

## Install shapes

| | Dev Hub on a **folder source** | Tester on a **tag install** |
|---|---|---|
| Code lives | the checkout (`src/`) | a copy in the managed venv |
| Picks up an edit | yes, on worker reload | no — tag → catalog → reinstall |
| Dependency change | `uv sync` / re-provision | new tag |

Say which one a change needs when you finish it.
