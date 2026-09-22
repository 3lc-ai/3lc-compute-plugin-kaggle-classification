# STUDY.md — Phase 0 reference study (session 1, 2026-09-22)

Read-only study of the four reference plugins, the SDK, the HackNova
predecessor kit, and the Intel scene data, before any code is written for
`kaggle-classification`. Every claim below was checked against the files
named; versions were checked against the installed environments and public
PyPI on 2026-09-22.

Sections (a)–(e) follow the brief. §F is the reuse / adapt / write-new table.
§G lists the two circuit breakers that fired and the decisions Gate 0 needs.

---

## (a) `3lc-compute-plugin-timm` — the first-party classification trainer

Reference checkout: `main` at `b77f252` ("Align timm with SDK 0.5 and verified
private POC publishing"), 2 commits past tag `v0.2.6`. **`v0.2.6` is the latest
published release** (PyPI confirmed). The two are different stacks:

| | `v0.2.6` (PyPI, released 2026-09-11) | `main` @ `b77f252` (unreleased POC) |
|---|---|---|
| SDK pin | `3lc-compute-plugin-sdk>=0.4.0,<0.5.0` | `>=0.5.0,<0.6.0` |
| `uv.lock` | none committed | resolves SDK `0.5.0.20260916073124.19.1` and `3lc 3.4.0` — **neither exists on public PyPI** (SDK latest 0.4.0, 3lc latest 3.3.2); they come from the private CloudRepo POC index (`ci.yml` sets `UV_INDEX: poc=https://pypi.3lc.ai/repositories/prereleases/`) |
| Source diff | — | only CI/release workflow, `scripts/release_version.py`, `uv.lock`, pyproject pins. **Zero changes under `src/`** — the plugin code is identical. |

### Dependency pins (`pyproject.toml`)

- Base `dependencies`: the SDK only. The SDK brings `3lc[pandas]>=3.3.0,<4.0.0`, `litestar>=2.22,<3`, `uvicorn>=0.34`.
- Extra `timm = ["timm>=1.0", "torch", "torchvision", "3lc[pacmap,umap]>=3.3.0,<4.0.0"]`.
  **`timm` is floored, not pinned** (`>=1.0`); torch/torchvision are unpinned. The `main` lock resolves `timm 1.0.29`, `torch 2.14.0`, `torchvision 0.29.0`, `umap-learn 0.5.12`, `scikit-learn 1.7.2`, `pacmap 0.8.0`. PyPI's latest timm is also 1.0.29.
- `requires-python = ">=3.10"`, hatchling, ruff config identical to the SDK's, dev group `pytest>=8,<9`.
- `[[tool.uv.index]] pytorch-cu126` explicit, applied to torch/torchvision on linux + win32 (the kaggle plugin moved to **cu128** for Blackwell GPUs — see §c).
- Entry point `[project.entry-points."tlc_compute.plugins"] timm = "tlc_plugin_timm"`.

### `plugin.toml` shape

Top level: `id`, `name`, `description`, `version` (must equal pyproject's — the
release gate checks it), `min_service_version = "0.1.0"`, `icon`, `icon_svg`.
`[ui]`: `display_mode = "sidebar"`, `section = "AI Tools"`, `compatible_with = ["table"]`,
`input_types = ["table","table"]`, `output_types = ["run"]`, `min_input_count = 1`,
`action_param_names = ["train","val"]`, `quick_action*`.
`[runtime]`: `isolation = "venv"`, `entrypoint = "tlc_plugin_timm:TimmPlugin"`,
`provision_extra = "timm"`, `requires_gpu = true`, `training = true`.

### How it loads classification tables on tlc 3.x (`trainer.py`)

- Consumes tables only; never creates them. `tlc.Table.from_url(url)`, optional `.latest()`.
- `num_classes` = `len(train_table.get_value_map(label_column))`, falling back to
  `max(label column) + 1` via `get_column_as_pyarrow_array`, then to 2.
- Column names come from `tlc_plugin_sdk.shared.modality.detect_modality_from_table`
  (`task_detection.py`) — image column, label column, class names, project name.
- **Transforms attach with `table.with_transform(fn)` → `TableView`** (the 3.x API; the
  2.x `table.map()` is gone — see §G-1). The transform is a top-level picklable class
  `_SampleTransform(transform, image_column, label_column)` that accepts both dict
  samples (`sample[image_column]`, `sample.get(label_column, 0)`) and tuple samples, and
  decodes url-backed image columns through `tlc_plugin_sdk.shared.images.load_image`.
- Train loader: `DataLoader(view, batch_size, shuffle, sampler, num_workers=params["num_workers"] (default 8), pin_memory=True, drop_last=True)`.

### Sampler (weights)

Only when `sampling_weights` or `exclude_zero_weight_training` is set:
`from tlc.integration.torch.samplers import create_sampler` →
`create_sampler(train_table, exclude_zero_weights=..., weighted=...)`, `shuffle=False`.
Installed 3.3.0 signature: `create_sampler(table, exclude_zero_weights=True, weighted=True, shuffle=True, repeat_by_weight=False) -> Sampler[int]`.
Otherwise plain `shuffle=True` and weight-0 rows train like any other — our locked
decision ("undefined rows filtered from training regardless of weight") needs a
hard filter of our own, not just this sampler.

### Model

`timm.create_model(name, pretrained=True|False, num_classes=N)`; transforms via
`timm.data.resolve_data_config(model.pretrained_cfg)` with `input_size` overridden to
`(3, image_size, image_size)`, then `timm.data.create_transform(**cfg, is_training=…)`.
Our locked path is `pretrained=False` + a single allowlisted `arch`; `pretrained_cfg`
is still populated for `resnet18` (ImageNet mean/std, 224), so the same transform
recipe works with `image_size` from the manifest.

### Metrics collector — hand-rolled, NOT `EmbeddingsMetricsCollector`/`Predictor`

`_collect_metrics()` runs its own `DataLoader` over the val-transform view and, per
batch: `features = model.forward_features(x)`; `emb = model.forward_head(features, pre_logits=True)`;
`logits = model.forward_head(features)`; `softmax → predicted / confidence`;
`CrossEntropyLoss(reduction="none")` → per-sample `loss`. It then writes one metrics
table per split with

```python
run.add_metrics(
    {"embeddings": list(reduced), "predicted": ..., "confidence": ..., "loss": ...},
    foreign_table_url=table_url,
    schema={"embeddings": tlc.schemas.Float32Schema(shape=(dim,), display_name=...),
            "predicted": copy.deepcopy(source_table.rows_schema.values[label_column])},
    constants={"epoch": epoch},
)
```

(3.3.0 signature confirmed: `Run.add_metrics(metrics, *, schema=None, foreign_table_url=None, constants=None)`.)
`tlc.metrics.Predictor`, `EmbeddingsMetricsCollector`, `FunctionalMetricsCollector` and
`tlc.collect_metrics` are **not imported anywhere** in the timm plugin.

### Embedding reduction

Also hand-rolled: `_fit_reducer(train_emb, dim, "umap"|"pacmap")` fits
`umap.UMAP(n_components=dim, n_neighbors=min(15, n-1), min_dist=0.1, random_state=42)`
(or `pacmap.PaCMAP`) on the **train** embeddings, then `.transform()` val so both
splits share one coordinate space; the reduced vectors are written directly as the
`embeddings` column above. `run.reduce_embeddings_by_foreign_table_url` is never
called. Fallback when there is no train split: fit-transform per split. No PCA path.

### Checkpoints and run bookkeeping

- Best `state_dict` kept on CPU; after training `save_model_to_run(run_url, model_data=state, filename="best_model.pt", on_status)` and `store_model_info_in_run(run, model_name, model_path, source_url, on_status)` from `tlc_plugin_sdk.shared.model_storage` (writes under the run's `model/` folder — local runs only).
- `run = tlc.init(project_name, run_name, description)`; per epoch `tlc.log({...})`; `run.set_status_collecting()` / `set_status_completed()` / `set_status_cancelled()`.
- `run_name` defaults to `tlc_plugin_sdk.shared.naming.generate_name()`.

### Progress via `JobContext` (`__init__.py: run_job`)

- `ctx.params["project_config"]` (inline frozen config, preferred) else `config_id` →
  `PluginConfigStore` lookup; missing → `ctx.fail("config_id is required")`.
- Rich UI events: `ctx.emit("job_status", {job_id, status, message})` and
  `ctx.emit("epoch_progress", {job_id, epoch, total_epochs, metrics, timing})`.
- Generic panel: `tlc_plugin_sdk.shared.generic_job.epoch_progress(progress_raw, step_label="epoch")`
  → `ctx.progress(percent, label, timing)`; `ctx.log(msg)`; `ctx.result(run_url)`; final
  `ctx.progress(percent=100, label="Done")`. Cancellation = `ctx.cancelled` polled every
  batch through an `is_cancelled` callback.
- Alias overrides applied/restored around the job (`shared.aliases`), SDK ≥0.4 does this
  in the worker too.

### Release machinery (`main` only)

`scripts/release_version.py` — `check_sources` (pyproject version == every advertised
`plugin.toml` version), `--stamp X` (rewrites both in lockstep, refuses on drift),
`--wheel` (wheel METADATA + packaged manifests match). `tests/test_release_version.py`
covers stamp / drift / broken-wheel. `release.yml`: tag push → CI gate → `uv build --no-sources`
→ `uv publish --trusted-publishing` → GitHub Release from the CHANGELOG section.

---

## (b) `3lc-compute-plugin-template` — scaffold

Checkout `main` @ `c9c467e`. Files: `pyproject.toml`, `catalog.json`, `jsconfig.json`,
`.github/workflows/ci.yml` (ruff lint + format, standalone via `uvx`), `.gitignore`,
`LICENSE` (Apache-2.0), `README.md`, `src/tlc_plugin_template/{plugin.toml, __init__.py, ui.html}`,
`.claude/skills/create-plugin/`.

- **SDK pin `>=0.3.1,<0.4.0`** (same as the example plugin and our kaggle plugin).
- `pyproject`: hatchling; `license = { text = "Apache-2.0" }` + OSI classifier; base deps SDK only; extra `template = []`; entry point; `[tool.hatch.build.targets.wheel] packages`; ruff block.
- `plugin.toml` minimal: `id, name, description, version, icon, min_service_version="0.1.0"`, `[ui] display_mode/section/compatible_with`, `[runtime] isolation/entrypoint/provision_extra`.
- `__init__.py`: `class TemplatePlugin(ComputePlugin)` with `id: str` annotation, `get_ui_fragment()` reads `ui.html`, `compute(params)`.
- **UI fragment pattern**: one `<div class="<id>-plugin">` carrying `<style>`, markup and an IIFE `<script>`; reaches the host only through `window.PLUGIN_API` (`getConfig('compute_service_url')`, `authFetch`); theme via CSS vars (`--text`, `--bg-card`, `--border`, `--accent`, …). The example plugin shows the richer form: `inject_scripts(raw, data_source_ui_script(), job_tracker_script())` splices SDK client scripts after the first `<script>`, giving `window.PluginJobs` (start/track/abort on the generic channel).
- **Routes**: `get_route_handlers()` returns Litestar handlers built fresh per call, paths relative to `/api/plugins/<id>/`, `def` handlers with `sync_to_thread=True`. Reserved (host-owned): `/run /health /ui /compute /busy /reclaim /jobs/* /provision /reload /venv /worker/stop`.
- **Worker**: venv-isolated, out-of-process (`tlc_plugin_sdk.worker`), host reverse-proxies. Dev loop: `3lc-compute --plugin-dir <repo>/src` then `POST /api/admin/plugins/dirs/reload {"directory": "<repo>/src"}`.
- **Catalog entry shape** (`catalog.json`):
  `{"schema_version": 1, "generated_at": "", "plugins": [{"id", "versions": [{"version", "source": "<dist>[<extra>] @ git+<repo>.git@<ref>", "wheel_url": "", "manifest": {<copy of plugin.toml>}}]}]}`.
  Registered with `POST /api/admin/plugins/catalogs {"url": ..., "persist": true}` — on compute ≥1.0 the default `catalog-only` install policy refuses this; the URL must be in `TLC_COMPUTE_PLUGIN_CATALOG_URLS` on the service (kaggle CONTEXT.md, W6).
- `jsconfig.json` points TypeScript at `<site-packages>/tlc_plugin_sdk/contract/plugin-api.d.ts` for `PLUGIN_API` autocomplete in `ui.html`.

---

## (c) `3lc-compute-plugin-kaggle` (ExDark, detection) — `v1.2.15` on `port/0.2.x`

### Session object (`config_store.py`)

- One JSON file `~/.3lc-kaggle-plugin/ui_config.json`; `_ALLOWED_TABS = ("session","train","submit","import_state","predict_state","submit_state")`; anything else dropped on save.
- `session = {project_name, table_name, dataset_yaml, device, slug_override, overrides}`; `default_session()` fills from `constants.py` (the leaf, single definition site — the UI carries **no** default literals; `GET /config` always serves a populated session).
- Retired keys (`_RETIRED_TABS`, `_RETIRED_TAB_KEYS`) are **rejected on save → `ValueError` → 400**: the only writer that still sends them is a stale cached fragment.
- Migrations keyed by `_migrations` markers, fixed order (`device_blank_default` → `session_v1`), the one read-path write; `session_v1` records the deciding branch string.
- `classify_override(url, project, table_name, split) -> "drop"|"suppress"|"keep"` is the single predicate for table-URL overrides, mirrored in `ui.html` (`kgOverrideDisposition`) with `URL_SEG_PATTERNS` parsed **by position in the layout tail** (project root is configurable; `/projects/` is not a literal). `tests/test_url_regex_parity.py` fails on a one-sided edit.
- Reentrant lock; atomic `tmp` + `os.replace` write; `load()` returns `{}` on any read error.

### How locked fields and bounds reach the UI

`routes.py: GET /config` returns the stored config + populated `session` +
`_meta = {version, repository_url, host, default_slug, dataset_prefix, kit_dest, kit_version, contract{model, imgsz, pretrained, checkpoint_sha256, max_rows}}`.
The fragment **renders** these and defines none of them (`tests/test_contract_parity.py`:
served block == `trainer.LOCKED_TRAIN_ARGS` + sha + `importer.EXPECTED_ROWS`, and no
literal copy survives in `ui.html`). Bounds: `trainer._SETTINGS_BOUNDS` dict +
`BOUND_MESSAGE`, `_check_bound` over the **merged** kwargs so `extra_args` cannot
bypass them; `FORBIDDEN_LOCKED` keys rejected by name; `kwargs.update(LOCKED_TRAIN_ARGS)`
merged **last**. Validation routes (`/validate/*`) stay torch-free; `/run` re-validates
(defense in depth — the host dispatch never traverses `/validate`).

### Divergence paths documented (the "DP" register + later classes)

| Id | What diverged | Where fixed / pinned |
|---|---|---|
| DP-01 | cross-project table mixture (train/submit URLs in one project, session in another) | `session_v1` drops cross-project overrides; `test_session_migration` |
| DP-02 | non-default table name not followed | session `table_name` + derived URLs |
| DP-04 | persisted slug survived a `COMPETITION_SLUG` swap | `RETIRED_SLUGS` + `resolve_slug`; `test_slug_swap` (fired live 2026-09-14) |
| DP-06 | load sequencing (config vs fragment) | PRETAG 1.2.6 |
| DP-08 | revisit view keyed on CSV only; job record pruned at 50 | `test_predict_submit_state` |
| DP-10 | `epochs=999` hand-set → clamp | `_SETTINGS_BOUNDS` / `_check_bound` |
| DP-11 | a table URL valid on its own but from another **split** | `constants.split_dataset`, `classify_override`, server-side asserts in trainer/predictor, `test_dp11_cross_split` |
| v1.2.10 | project root read via string-keyed `ConfigStore.get(Option)` → always default root; URL parse needed literal `/projects/` | `tlc.config.project_root_url`; `ROOT_SHAPES` parametrization (default / relocated / bare) in `conftest.py`; `test_project_root` |
| v1.2.12 | kit version recorded vs shipped, never compared → `superseded` state | `download_state`; `test_downloader` two-version block |
| v1.2.13 | contract literals in `ui.html`; row budget unchecked at train time; host weights gate only on `/validate`; DOM node deleted by innerHTML | `test_contract_parity`, `test_row_budget`, `test_host_weights_gate`, `test_ui_node_lifetime` |
| v1.2.14 | trainer floated `ultralytics` 8.4.6→8.4.66, −0.035 mAP silently | exact pins `3lc-ultralytics==0.4.0` + `ultralytics==8.4.66` |

The standing rule (CLAUDE.md §B): **divergence tests are the suite's job** — every
one of these was one value read from two sources with nothing comparing them.

### `resolve_device()` (`predictor.py:367`)

Blank → `torch.cuda.is_available()` → `0`; `torch.backends.mps.is_available()` → `"mps"`;
else `"cpu"`. Digit string → `int`; any other non-blank string passes through.
**Worker-side only** — never called on the host request path (validation stays torch-free).
Returns an ultralytics-style arg (`0`); for us it becomes `torch.device("cuda"|"mps"|"cpu")`.

### Kit download / verify stage machine (`downloader.py`, 950 lines)

`run_download(params, ctx)`:
1. `resolve_params` — absolute writable `dest_dir` (probe write), `keep_archives=False`.
2. `fetch_manifest(version_dir, log)` — GET `<prefix>/manifest.json`; reject `schema_version != 1`; reject `kit_version != STARTER_KIT_VERSION` (staging error, not drift); persist beside the shards.
3. `_free_space` — `2 × total_bytes + 100 MB − already on disk`.
4. Per shard `_download_shard`: skip if final file exists with matching size+sha256; else resume `<name>.part` with `Range: bytes=N-` (206 → append; 200 → truncate and restart); `_CHUNK = 1 MiB`, progress/cancel poll every 4 chunks; `_RETRIES = 2`; sha256 verify then `part.replace(final)`.
5. Extract each zip with a zip-slip guard (`n.startswith("/")` or `..` segment → fail).
6. `verify_tree(manifest, version_dir)` — every `files[]` entry by size + sha256; `matched / mismatch / missing / extra`; extras are reported, never deleted.
7. `dataset.yaml` present at `<kit_dir_name>/`.
8. `_publish_session_yaml` (the one server-side session write, "A5"), record facts (`dest_dir`, `kit_version`, `kit_dir`, `dataset_yaml`), delete archives unless `keep_archives`.

Revisit states: `empty / success / superseded / stale` (`download_state`); `verify_now`
re-runs the full files[] check from the manifest kept beside the kit; `run_top_up`
updates in place (delta against disk, removals restricted to paths the old manifest
claimed, dataset.yaml load-bearing keys gate, whole-tree re-verify, restamp).

CDN manifest shape (`scripts/make_kit_manifest.py`, deterministic: zip epoch timestamps,
`0o644`, JPEGs stored, everything else deflated, sorted paths, size-cut shards):
`{schema_version, competition_id, kit_version, kit_dir_name, created_utc, total_bytes, file_count, archives[{name, bytes, sha256, file_count}], files[{path, bytes, sha256, archive}]}`.
**Note the delta vs our locked manifest**: our `kit{base_url, version, shards[{name, sha256, bytes}]}` has no `files[]`, so per-file verification (step 6) has nothing to check against — see §G-4.

### Jobs bridge (`jobs.py`)

Disk-backed job records `~/.3lc-kaggle-plugin/jobs/<id>.json` (log, checks, progress,
facts, result, error, pid stamp for orphan detection, prune at 50). `run_dispatch(kind,
params, target, sdk_ctx)` runs the target synchronously on the SDK dispatch thread with
`_BridgedJobCtx`, which mirrors `log/progress/metric/result` onto the `JobContext` and
unions cancellation (host `ctx.cancelled` OR the on-disk flag). `run_job(ctx)` dispatches
on `ctx.params["kind"]`. `tests/test_jobs_bridge.py` asserts against a fake ctx carrying
the **real 0.3.x signatures**, because the bridge swallows exceptions.

### Test layout (`tests/`, 16 files, pytest, `pythonpath = ["src"]`)

- `conftest.py`: SDK stub when `tlc_plugin_sdk` is absent; `store` fixture (monkeypatched `CONFIG_PATH`, fixture slug); `ROOT_SHAPES` + `root_shape` param; `fixture_config()` rehoming sanitized real configs (`<FIXTURE_ROOT>` token); `materialize_tables`; `tlc_stub` (fake `tlc.Url/Table/TableWriter` laying tables out under the configured root).
- Unit: `test_config_store`, `test_session_migration`, `test_config_coherence_invariant`, `test_slug_swap`, `test_dp11_cross_split`, `test_project_root`, `test_row_budget`, `test_host_weights_gate`, `test_predict_submit_state`.
- Divergence guards: `test_url_regex_parity`, `test_contract_parity`, `test_packaging` (builds the real wheel with `python -m hatchling build`; four version strings agree; description parity on the newest catalog entry only; wheel carries `ui.html` + `plugin.toml`; catalog entries internally consistent, ids match, newest-first, unique), `test_downloader` (FakeCDN with 206/200/tamper), `test_jobs_bridge`, `test_kit_scripts`, `test_ui_node_lifetime`.
- Dev group only: `pytest>=8`, `hatchling>=1.24` (a test input). `requires-python >= 3.11` because the `kaggle` client floors there.

### Release audit protocol (`RELEASING.md`)

Tag `vX.Y.Z` → update repo `catalog.json` (newest first, manifest pasted verbatim, `generated_at` bumped) → mirror to the gist (superseded by the repo raw `HEAD/catalog.json` once cut over). Version pin **census** (the list of files that carry the version and go stale on a bump — additive, never pruned). `docs/PRETAG_<ver>.md` frozen above "Post-tag verification". Kit data releases are separate: immutable CDN prefixes, `kit-*` tags, committed manifest as the verification anchor, `scripts/check_kit_parity.py`. Install-shape table (folder source vs tag install; "reload" only applies to the former). There is no single audit *script*; the audit is `RELEASING.md` + `test_packaging` + `check_kit_parity.py` + the PRETAG checklist. The timm plugin's `scripts/release_version.py` is the closest scriptable equivalent.

### Manifests

`pyproject.toml`: SDK `>=0.3.1,<0.4.0`; extra `kaggle` with exact trainer pins, `torch`, `torchvision`, `3lc[pacmap,umap]>=3.0.0,<4.0.0`, `kaggle>=2.2.3,<3`; `[tool.tlc-compute]` legacy mirror; **sdist allowlist** (an exclude list once leaked `.claude/settings.local.json`); `[[tool.uv.index]] pytorch-cu128` + `3lc-releases` (explicit); `.gitattributes` marks `kit/**/manifest.json` generated. `plugin.toml`: `min_service_version = "0.2.0"` deliberately (0.2.1 fallback host), `priority = 40`, `icon_svg`, `repository_url`. `catalog.json`: one entry per released version, `source = "3lc-compute-plugin-kaggle[kaggle] @ git+https://github.com/3lc-ai/3lc-compute-plugin-kaggle.git@v1.2.15"`.

---

## (d) `3lc-compute-plugin-sdk`

Checkout `main` @ `547549a` = `v0.4.0-7` ("Prepare SDK 0.5 POC staging on private
CloudRepo"); `pyproject version = "0.5.0"`, CHANGELOG lists `[0.5.0] - 2026-09-11`, but
**there is no `v0.5.0` tag and PyPI stops at 0.4.0**. 0.5.0 is a POC on the private index.

### `JobContext` (`job_context.py`, stdlib-only)

`JobContext(job_id, params, state_dir, *, sink, cancel_event, identity=None)`.
Plugin-facing: `job_id`, `params`, `state_dir` (writable scratch that survives
reinstall), `identity: JobIdentity(user_id, org_id, project_id)` (0.4+),
`cancelled` (property), `progress(*, percent, label="", timing=None)` (`-1` = indeterminate),
`metric(label, value)`, `log(message)`, `result(url)` (positional — 0.1.x was keyword
`run_url=`), `fail(message) -> NoReturn` (raises `JobFailed`, reported verbatim),
`emit(name, payload)` (`"job_update"` reserved → `ValueError`). Tests construct it with
their own `sink` list and `threading.Event`.

### Plugin contract (`contract.py`)

`HubPlugin(ABC)`: abstract `get_ui_fragment()`; optional `compute(params)`,
`shutdown_runtime()`, `get_route_handlers() -> list` (0.5 extracted this base; in 0.3/0.4
`ComputePlugin` is the root and carries the same members). `ComputePlugin(HubPlugin)`:
`initialise_runtime()`, `run_job(ctx)`. No metadata on the class; `id`/`name`/`icon` are
stamped by the host from the manifest.

### `SDK_CONTRACT_VERSION`

`tlc_plugin_sdk.SDK_CONTRACT_VERSION = importlib.metadata.version("3lc-compute-plugin-sdk")`
(`"0.0.0"` from a raw checkout). One axis since 0.3 (PY/JS markers removed). A plugin pins
the SDK dist `>=X,<Y`; the host compares **MAJOR.MINOR** of its own imported SDK against
the worker's and reports skew pre-flight (`tlc_compute/plugins/versioning.py:contract_skew_reason`,
`supervisor.py:679`, `routes/plugins.py:120`). `min_service_version` in the manifest is
the orthogonal *service* floor.

### Host facts that constrain our pin

| Host | compute | SDK in host venv | compute's declared SDK requirement |
|---|---|---|---|
| `3lc-hub-ga/` (current) | 1.0.1 | 0.3.2 | `3lc-compute-plugin-sdk>=0.3.1,<0.4.0` (dist METADATA) |
| `3lc-hub-next/` | 0.2.1 | 0.1.1 | (plugin venv separate; kaggle v1.2.x with the 0.3.1 pin installs and runs) |
| PyPI | 1.1.0 available | — | not inspected (no local install) |

### `tlc_plugin_sdk.shared.*` worth reusing (all stdlib + `tlc`, importable in the worker)

| Module | We use |
|---|---|
| `generic_job.epoch_progress(progress, step_label=)` | percent/label/timing for the generic panel |
| `model_storage.save_model_to_run / store_model_info_in_run` | checkpoint into the run folder |
| `config_store.PluginConfigStore` | optional; our session store is a different shape (one file, tab keys) — probably not |
| `images.load_image / get_image_column / read_image_from_table` | decoding url-backed image columns |
| `labels.get_label_map / find_label_column` | class map from a table |
| `modality.detect_modality_from_table` | column discovery (or we hardcode `image`/`label` since we create the tables) |
| `naming.generate_name` | run names |
| `url_utils.normalize_url` | `~` expansion on typed paths |
| `ui_inject.inject_scripts`, `job_tracker.job_tracker_script`, `data_source_ui/_routes` | fragment plumbing (session 2+) |
| `aliases.apply_alias_overrides / restore_aliases` | only if we support remote workers |

---

## (e) `hackathon_starter/` (Hack[AI]thon 2.0, chihuahua-vs-muffin) — and the Intel folder

**No kit-build script exists.** Contents (5,937 files, 229 MB): `kaggle_content/*.md`
(page copy), `metric-template-650e8c.ipynb` (Kaggle metric template: `score(solution,
submission, row_id_column_name)` over `image_id, prediction, confidence`, accuracy),
`solution.csv` (**the answer key — unversioned, sensitive; do not copy**), `Starter_Kit/`
(`config.yaml`, `register_tables.py`, `train.py`, `predict.py`, `README.md`,
`sample_submission.csv`, `data/{train,val,test}` = 3,733 / 1,000 / 1,184 images) and
three zips of it. `Dataset Description.md` points organizers to a "For organizers" section
of the Description tab that does not exist in the local `Description.md`. The kit scripts
target tlc 2.22.3 and use the retired 2.x API names (§G-1).

The Intel folder (`3-lc-hack-nova-scene-classification-challenge/`) is the same shape:
scripts + `config.yaml` (which cites a `host_manifests/dataset_stats.json` that is not
present) + data. Its builder was never archived. Behaviour worth carrying:
`register_tables.py` writes `train` (600 labeled weight 1.0 + 6,000 `undefined` label 6
weight 0.0) and `val` (1,200) only, test never registered; `train.py` filters with
`create_sampler(exclude_zero_weights=True)`, masks `undefined` (label 6 ≥ num_logits) out
of per-sample loss, collects on train, reduces with UMAP 3D, seed 42, `num_workers=0`.

Data facts (checked): **9,600 files, all `.jpg`, all 150×150 RGB JPEG, no EXIF.**
Counts match the locked manifest: train 100×6 + 6,000 undefined, val 200×6, test 1,800.
Filenames leak: `train/buildings/buildings_038426.jpg`, `val/sea/sea_002e4d.jpg` (class in
the stem), `train/undefined/pool_000982.jpg` (split membership in the prefix),
`test/test_00001.jpg` (sequential). The salted-opaque-id re-encode is justified.

Tooling on this machine: Python 3.12.3, uv 0.11.7, Pillow 12.2.0, PyYAML present,
**pydantic absent** (and not an SDK dependency) → manifest schema = dataclass + explicit
validation, no new dependency.

---

## F. Reuse / adapt / write new

| Component | Verdict | Source | Notes |
|---|---|---|---|
| Session object | **Adapt** | kaggle `config_store.py` | Keep one-file store, allowed tabs, retired-key 400, marker migrations, `classify_override` + `URL_SEG_PATTERNS`. Generalize: defaults from the manifest, drop `dataset_yaml`/`slug_override` → `{project_name, table_name, kit_dir, device, overrides}`; splits from `manifest.splits` |
| Tests | **Adapt** | kaggle `conftest.py`, `test_config_store`, `test_packaging`, `test_jobs_bridge` | Port the fixtures and the divergence-guard shape; drop ExDark fixtures. **New**: manifest loader tests, kit builder tests. `test_url_regex_parity` / `test_contract_parity` return when `ui.html` exists (session 2) |
| Release audit | **Adapt** | kaggle `RELEASING.md` + `test_packaging`; timm `scripts/release_version.py` (Apache) | Version parity script from timm (copy + origin header), catalog checks from kaggle, no gist step |
| Kit download stage | **Adapt** | kaggle `downloader.py` | Same machine (skip-verified / Range resume / sha256 / zip-slip / verify / states). Source of truth becomes the competition manifest's `kit{}`; per-file verification depends on §G-4 |
| Kit builder | **Write new** (borrow) | kaggle `make_kit_manifest.py` for deterministic zip writing | Salted-id rename, re-encode, `mapping.csv` to private dir, collision check, count gate, post-build verification — all new |
| Manifest loader | **Write new** | — | Schema v1 dataclasses, remote → cache → bundled, env override, unknown-field warnings |
| Importer | **Write new** | Intel `register_tables.py` (behaviour), tlc 3.3.0 `Table.from_image_folder` / `TableWriter` | 3.x API only (§G-1). Undefined rows weight 0; test unregistered |
| Trainer | **Write new** (borrow) | timm `trainer.py` loop, `_SampleTransform`, `_as_rgb_image`, optimizer/scheduler helpers (Apache) | `timm.create_model(arch, pretrained=False, num_classes=N)`, arch allowlisted to the manifest; hard filter of undefined rows; `workers=0`; run/`ctx` plumbing from timm `run_job` |
| Metrics | **Adapt** | timm `_collect_metrics` | `run.add_metrics` path with explicit schemas; mask undefined rows out of loss; recommend over `tlc.collect_metrics` (§G-1) |
| Embeddings | **Adapt + new** | timm `_fit_reducer / _transform_embeddings` | UMAP fit-on-train/transform-val; **new** sklearn PCA fallback per `training.embeddings{method, fallback, n_components}` |
| Device resolution | **Adapt** | kaggle `resolve_device` | Return `torch.device`; CUDA → MPS → CPU; worker-side only |
| Jobs bridge | **Adapt** (session 2) | kaggle `jobs.py` | Disk-backed records the tabs poll + `_BridgedJobCtx`; needed once the Import tab polls |
| UI fragment | **Adapt** (stubs now) | template `ui.html` + SDK `inject_scripts` / `job_tracker_script`; kaggle playbook for the tab/stepper/six-state pattern | Session 1 ships a four-tab shell only |
| Catalog | **Adapt** | template `catalog.json` shape + kaggle fields (`icon_svg`, `repository_url`, `priority`) | `test_packaging` catalog checks ported |
| Ledger + verification bundle | **Write new** (session 4/5) | — | — |

---

## G. Circuit breakers and Gate 0 decisions

### G-1. tlc 3.x API delta — FIRED, but not blocking

The **local** `3lc-examples` checkout (detached at `42798dd`, 2026-04-20) and both
predecessor kits use the 2.x surface. tlc 3.3.0 (installed in `3lc-hub-ga/`) **does not
export** any of: `tlc.Predictor`, `tlc.EmbeddingsMetricsCollector`,
`tlc.FunctionalMetricsCollector`, `tlc.PILImage`, `tlc.CategoricalLabel`, `tlc.ImagePath`,
`tlc.Int32Value`, `tlc.Float32Value`, `tlc.SampleWeightSchema`, `tlc.ImageSchema`,
`tlc.register_project_url_alias`; `Table.map` and `Table.create_sampler` are gone.
3.3.0's curated surface: `tlc.__all__` (32 names: `Table, TableView, TableWriter, Run,
Schema, Url, init, log, collect_metrics, metrics, schemas, integration, reduction, …`),
`tlc.metrics.{Predictor, PredictorOutput, EmbeddingsMetricsCollector, FunctionalMetricsCollector, collect_metrics}`,
`tlc.schemas.{CategoricalLabelSchema, SampleWeightSchema, Float32Schema, ImageSchema, MapElement, …}`,
`tlc.schemas.values.{Int32Value, Float32Value, …}`, `tlc.integration.torch.samplers.create_sampler`,
`Table.with_transform`, `Table.from_image_folder(root, *, …, add_weight_column=True, weight_column_value=None, label_overrides=…, if_exists="reuse")`.

The **upstream** `3lc-examples/main` notebook (fetched 2026-09-22) already uses the 3.x
names (`with_transform`, `tlc.metrics.Predictor`, `tlc.schemas.Float32Schema`,
`from tlc.integration.torch.samplers import create_sampler`, `tlc.collect_metrics`,
`run.reduce_embeddings_by_foreign_table_url`). So the timm plugin and current docs agree
on table loading; they **differ in metrics style**: docs = `tlc.collect_metrics` +
`Predictor(layers=[...])` + `EmbeddingsMetricsCollector` + `reduce_embeddings_by_foreign_table_url`;
timm = hand-rolled forward pass + own UMAP + `run.add_metrics`. Both paths exist in 3.3.0.

**Recommendation**: follow the timm path (`run.add_metrics` + own reducer). It gives us
explicit schemas, a fit-on-train shared coordinate space, a trivial PCA fallback, and
per-row loss masking for `undefined`; nothing in it depends on 2.x names. Design the
importer on `Table.from_image_folder` (3.x) with `label_overrides`/weight handling, or
`TableWriter` with `tlc.schemas` classes — decided in session 2.

### G-2. SDK pin range — FIRED (template excludes timm)

| Pins | Range |
|---|---|
| template / example / kaggle v1.2.15 / **compute 1.0.1 host requirement** | `>=0.3.1,<0.4.0` |
| timm **v0.2.6 (PyPI, "today")** | `>=0.4.0,<0.5.0` |
| timm `main` (unreleased POC, private index) | `>=0.5.0,<0.6.0` |

The brief says "SDK pin range matching what the timm plugin pins today". Taken literally
that is `>=0.4.0,<0.5.0`, which **compute 1.0.1 — our only 1.0.x host — flags as contract
skew** (host imports SDK 0.3.2, compares MAJOR.MINOR). Nothing this plugin needs is 0.4-only
(0.4 added `ctx.identity` and worker hardening; 0.5 added `HubPlugin`/infra). **Recommend
`>=0.3.1,<0.4.0`**, the range every host we can test actually runs, and revisit when
`3lc-hub-ga/` moves to a compute that requires 0.4+. Needs your call.

### G-3. `timm==<the version the timm plugin pins>` is unsatisfiable as written

The timm plugin pins `timm>=1.0` (a floor). Its `main` lock resolves **1.0.29**, which is
also PyPI's latest. **Recommend `timm==1.0.29`** and leave torch/torchvision floating with
the cu128 index (the kaggle plugin's Blackwell finding), or pin them too if you want
byte-identical trainers across participants (the v1.2.14 lesson).

### G-4. Kit integrity depth vs the locked manifest

The kaggle downloader verifies **every file** against `files[]`; our locked manifest's
`kit{shards[{name, sha256, bytes}]}` verifies **shards only**. Options: (i) shard sha256 +
per-split counts (cheap, what the manifest supports as locked); (ii) have the kit builder
also emit `files.json` (path, bytes, sha256) uploaded beside the shards and referenced by an
optional `kit.files_url`. Recommend (ii) as an additive optional field; Phase 3 emits it
either way.

### G-5. Relicensing record — modules adapted from `3lc-compute-plugin-kaggle` (for sign-off)

Decision (Gate 0): no AGPL headers travel into this repo. The ExDark plugin's session store,
downloader, tests and release-audit code are 3LC-authored and contain no Ultralytics-derived
code; 3LC as copyright holder relicenses those modules under Apache-2.0 for this project.
Each adapted file carries, verbatim:

`# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder under Apache-2.0 for this project.`

Audit before porting: each source module below was grepped for `ultralytics` / `yolo`
(case-insensitive) and read for derived logic. `predictor.py` (the source of
`resolve_device`) imports the kaggle client and ultralytics elsewhere in the module; only the
15-line device cascade was carried, rewritten to return a `torch.device` string.

| Adapted file here | Source in 3lc-compute-plugin-kaggle | Ultralytics hits | What was carried |
|---|---|---|---|
| `src/kaggle_classification/session.py` | `src/tlc_plugin_kaggle/config_store.py` | 0 | one-file store, allowlist, retired-key 400, marker migrations, atomic write, `URL_SEG_PATTERNS` + `classify_override` (defaults now manifest-derived; slug / dataset_yaml dropped) |
| `src/kaggle_classification/kit.py` | `src/tlc_plugin_kaggle/downloader.py` | 0 | the download / extract / verify stage machine (manifest `kit{}` + `files.json` replace the CDN manifest; job record replaces the jobs store; top-up not carried) |
| `src/kaggle_classification/trainer.py` (`resolve_device` only) | `src/tlc_plugin_kaggle/predictor.py:367-390` | 0 in the function (module imports ultralytics elsewhere) | the CUDA → MPS → CPU cascade |
| `tools/build_kit.py` (sharding + `files.json`) | `scripts/make_kit_manifest.py` | 0 | deterministic zip writing (fixed timestamps/attrs, stored JPEGs, size-cut groups) |
| `tests/conftest.py` | `tests/conftest.py` | 0 (two `from_yolo_url` mentions in the dropped `tlc_stub` fixture) | SDK stub, isolated-home fixtures, `ROOT_SHAPES` |
| `tests/test_session.py` | `tests/test_config_store.py`, `tests/test_url_regex_parity.py` (idea) | 0 | store semantics, retired-key rejection, URL parse across root shapes |
| `tests/test_kit.py` | `tests/test_downloader.py` | 0 | FakeCDN / FakeCtx pattern and the network-behaviour cases |
| `tests/test_packaging.py` | `tests/test_packaging.py` | 0 | wheel build, version parity, catalog consistency (extended: SDK overlap, import weight, licence lineage) |
| `tests/test_ctx_adapter.py` | `tests/test_jobs_bridge.py` | 0 | real-signature fake ctx |

Copied from `3lc-compute-plugin-timm` (already Apache-2.0, header names the origin):
`scripts/release_version.py`, `tests/test_release_version.py`.

Not ported: `jobs.py` (disk-backed job store; session 2 decides), `importer.py`,
`trainer.py`, `predictor.py`, `routes.py`, `ui.html` (ExDark-specific or Ultralytics-linked).

### G-6. Smaller defaults I will take unless told otherwise

- `requires-python = ">=3.11"` (the `kaggle` client floors there — same as ExDark).
- `min_service_version = "1.0.0"` (we never target the 0.2.1 fallback host for this plugin); say so if you want `0.2.0` for parity.
- Torch index: **cu128** explicit for linux/win32 (kaggle plugin's Blackwell finding), not timm's cu126.
- `hackathon_starter/` and the Intel `solution.csv`/`submission*.csv` are never read by code or copied.
