# PLAN.md — kaggle-classification: locked decisions and the session map

The decisions below are settled. They are recorded so nobody relitigates them
in a later session; a change to any of them is a competition-design decision,
not a code task. Depth on the reference material is in `docs/STUDY.md`.

## A. Locked decisions (brief, session 1)

| Topic | Decision |
|---|---|
| Repo / plugin id | `kaggle-classification` (dist `3lc-compute-plugin-kaggle-classification`, package `kaggle_classification`) |
| License | Apache-2.0. No Ultralytics anywhere. |
| SDK contract | `3lc-compute-plugin-sdk>=0.3.1,<0.4.0` — resolves on 3lc-compute 1.0.1 (`>=0.3.1,<0.4.0`) and 1.1.0 (`>=0.3.3,<0.4.0`); `tests/test_packaging.py` asserts the overlap against the latest 3lc-compute release metadata. timm's `>=0.4.0,<0.5.0` is not mirrored: no 1.x host accepts 0.4 (STUDY G-2). |
| tlc | `3lc>=3.3,<4.0` (the Compute Service resolves 3LC >= 3.3.0). 3.x API only: `Table.with_transform`, `tlc.integration.torch.samplers.create_sampler`, `tlc.schemas.*`, `run.add_metrics` (STUDY G-1). |
| Model | `timm.create_model(arch, pretrained=False, num_classes=N)`; `arch` allowlisted to the single manifest value (this event `resnet18`); `timm==1.0.29` pinned; the timm plugin is NOT imported at runtime. `pretrained: true` in a manifest is rejected. |
| Competition manifest | Remote `<MANIFEST_BASE_URL>/index.json` + `<id>/manifest.json`; base URL is a code constant (`manifest.MANIFEST_BASE_URL`, exact CDN prefix TBC, placeholder in use); env override `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL`; on-disk cache with fetched-at stamp; bundled `manifests/intel-scene-v1.yaml` as last resort. Unknown fields warn, never fail. |
| Splits | Unchanged from the Intel kit: 600 seed (100 × 6) + 6,000 `undefined` at weight 0 in `train`; `val` 1,200 (200 × 6) locked; `test` 1,800, flat, never registered. |
| Labeling | No cap. Undefined rows are filtered out of training regardless of weight. |
| Kit build | `tools/build_kit.py --salt-file` (never `--salt`); the salt is read from a private file, never printed or written; shards `intel-scene-v1-NN.zip`, deterministic; `kit-manifest-block.yaml` beside the kit dir; `mapping.csv` (original_relpath, new_relpath, split, class with class empty for test and `undefined` for pool rows) to the private dir only. |
| Kit | Images renamed to salted opaque ids (`sha256(salt + original_relpath)[:16]`) and re-encoded (JPEG q92, RGB, EXIF stripped). `mapping.csv` (the judge's key) lives ONLY in the private output dir. |
| Kit integrity | `files.json` (relpath, sha256, bytes) INSIDE the kit beside the data; the download stage verifies **per file** against it, not shard-only, then checks per-split counts against `manifest.splits`. Shards are sha256-verified from the manifest's `kit{}` block. |
| Predict | Only from plugin-created runs. |
| Ledger | Append-only ledger + a verification bundle per run (session 5). |
| Teams | Per machine. No table sharing. |
| Tabs | Import, Train, Predict + Submit, Status. |
| Hosts | `min_service_version = "1.1.0"` (the 1.1.0 torch-backend fix is required for a clean Windows install). |
| Torch index | Mirrors the timm plugin exactly: `pytorch-cu126` explicit index, `torch`/`torchvision` sourced from it on `linux` and `win32`, PyPI on macOS. |
| Python | `requires-python = ">=3.11"`: the timm plugin says `>=3.10`, but `kaggle>=2.2.3` and `scikit-learn>=1.9` floor at 3.11 and uv locks the whole range. |
| DataLoader workers | `num_workers=0` everywhere data loads (Windows). Device-aware in session 3. |
| Paths | Windows host, every path may contain spaces: quote everything. |

## A2. Manifest resolution (Phase 2 decisions)

- **Policy.** Remote wins whenever it is reachable AND the fetched document validates. The
  cache is the last remote document that validated; bundled is the last resort. **No
  "newer than" comparison** on version fields: a hotfix or rollback on the CDN takes effect
  on the next load regardless of ordering. Remote fetched but invalid → log the validation
  error, fall to cache, surface "manifest on CDN is invalid, using cached copy from
  <fetched-at>" in the UI. A bad hotfix never bricks a participant.
- **Index.** `<base>/index.json` = `{schema_version, competitions: [{id, display_name,
  manifest_url, active}]}`. One active competition → used. More than one → the UI shows a
  picker (`POST /manifest/select`, stored under the session store's `competition` key; the
  default id is used meanwhile if it is among the active ones). None → bundled with a
  visible warning. `manifest_url` may be relative but must stay on the index's host.
- **Fetching.** Server-side only (routes and worker), never from the browser. Connect+read
  budget 5 s total with one retry inside it (`FETCH_BUDGET_S`). `GET /config` resolves
  without the network (cache → bundled) and kicks a background refresh; the fragment
  polls `GET /manifest` until it settles and re-renders.
- **Cache.** Under the plugin home resolved by `storage.py` (env override → SDK helper →
  the worker's state root → `<cwd>/.plugin-state/<id>` → `~`), never from HOME first. On
  compute 1.1.0 the host passes no `--state-root`, so the fourth rule fires and the home is
  `<home>/.3lc-compute/managed-plugins/<id>/.plugin-state/<id>` (verified live 2026-09-22).
  `manifest-cache/<id>.manifest.json` holds the validated document,
  `<id>.meta.json` the sidecar `{fetched_at, source_url, sha256}`.
- **Provenance.** Every job re-resolves at start (`resolve_manifest_for_job()`) and records
  `{manifest_sha256, manifest_source, manifest_source_detail, manifest_fetched_at,
  competition_id, kit_version, schema_version}` into its outputs — the ledger's input.
- **Hardening.** `kit.base_url` must be https on an allowlisted host (`ALLOWED_KIT_HOSTS`,
  a code constant, initially the placeholder CDN host; loopback over http for the local
  mock only) and shard names must be plain file names, or the manifest is rejected. Every
  string the fragment renders (display_name, class names, loop_banner_text, help-link
  labels) is refused if it carries markup or control characters, and the fragment assigns
  them only through `textContent`; help-link URLs must be https.

## B. The labeling-loop contract (per-sample metrics)

What every plugin-trained run writes back onto the train and val tables, so the Dashboard
loop (inspect, label, fix, retrain) has what it needs. Written with `run.add_metrics(...,
foreign_table_url=<table>, schema=..., constants={"epoch": best_epoch})`, one metrics table
per split, computed on the restored best-epoch model with the val (non-augmented) transform.

| Column | Type / schema | Undefined rows |
|---|---|---|
| `predicted` | categorical, same value map as the table's `label` column (class names) | yes |
| `confidence` | float32, max softmax | yes |
| `prob_<class>` × N | float32 per class (per-class probabilities) | yes |
| `loss` | float32 cross-entropy, `reduction="none"` | **masked** — no value fabricated; the column is written only for rows with `label < num_classes` (a NaN/absent value, never 0 or a placeholder) |
| `embeddings` | float32 vector, shape `(n_components,)` = 3 | yes |

Embeddings come from `model.forward_head(model.forward_features(x), pre_logits=True)`
(512-d for resnet18). The reducer (`manifest.training.embeddings.method`, UMAP) is **fit on the
train embeddings — the 600 labeled and the 6,000 undefined rows together** — and val is
`transform`ed into that same space, so both splits share one coordinate system and the
unlabeled pool sits among the labeled clusters. `fallback` (PCA, scikit-learn) is used when
UMAP is unavailable or fails; the run records which reducer produced the coordinates.

Training itself excludes `undefined` rows by a hard filter on `label == manifest.undefined_label_id`
(the LAST map entry), regardless of weight; a participant labels a pool image in the Dashboard by
giving it a real class, and it enters the next revision's training set.

## C. Component contracts (what each session builds against)

- **Manifest (`manifest.py`)** — schema v1 dataclasses; `Manifest.num_classes`,
  `class_names`, `undefined_label_id`, `dataset_name(split)`, `expected_rows(split)`,
  `default_project`, `provenance`; `resolve()` / `resolve_manifest_for_job()` per §A2.
- **Session (`session.py`)** — `{project_name, table_name, kit_dir, device, overrides}` in
  `~/.3lc-kaggle-classification/ui_config.json`; retired keys 400; `classify_override`
  (drop / suppress / keep) mirrored in the fragment once pickers exist.
- **Kit (`kit.py`)** — job kind `download_kit`: manifest `kit{}` → sha256-verified,
  Range-resumable shard download → zip-slip-guarded extract into
  `<dest>/<kit.version>/starter_kit/` → per-file verify against `files.json` → split counts
  vs manifest → `ids_from` present → record + `session.kit_dir` → shards deleted. Revisit
  states `empty / success / superseded / stale`; `verify_now`. In-place top-up is deferred
  (the kit is small enough to re-download).
- **Importer (session 2)** — `train` (labeled weight 1.0 + undefined label N weight 0.0) and
  `val` tables under `<root>/<project>/datasets/<manifest.dataset_name(split)>/tables/<table>`;
  `test` never registered; REUSED vs CREATED per split; row counts vs `expected_rows`.
- **Trainer (session 3)** — see §B; params = manifest defaults ⊕ form, bounded on the merged
  kwargs; presets are named partial overrides; run records the contract (arch, image_size,
  pretrained=false, timm version, seed, table revisions).
- **Predictor + Submit (sessions 4, 5)** — plugin-run-only weights; `submission.csv` with
  `manifest.submission.columns` in `ids_from` order; Kaggle API against `competition.slug`;
  budget `daily_limit`.
- **Ledger (session 5)** — JSON-lines under the plugin home; verification bundle per run.
- **Status (session 6)** — history, budget, best score; friendly Kaggle-failure degradation.

## D. Session map

| Session | Delivers | Needs from the organizer |
|---|---|---|
| 1 (this) | scaffold, manifest schema + loader, session, kit stage, kit builder, docs | the salt; the CDN prefix; the Kaggle slug and deadline |
| 2 | Import tab: kit download UI, table registration, revisit view, pickers | the published kit prefix (Phase 3 output staged) |
| 3 | Train tab: trainer per §B, bounds, presets, device-aware workers | a GPU box to record the reference trajectory |
| 4 | Predict tab: plugin-run-only inference, submission.csv | test-set answer key on the organizer machine (local scoring, optional) |
| 5 | Submit + ledger + verification bundle | Kaggle credentials on a test account; the competition in draft |
| 6 | Status tab, release audit run, catalog tag | catalog URL policy on the participants' hosts |

## E. Open items (TBC, placeholders in the bundled manifest)

- `MANIFEST_BASE_URL` exact CDN prefix (`https://competitions.3lc.ai/hackathon` assumed).
- `competition.slug` (folder name `3-lc-hack-nova-scene-classification-challenge` assumed) and `deadline_utc`.
- `kit.base_url` prefix: the `kit{}` block is pasted (v1 build of 2026-09-22, five shards, 113,741,154 bytes) but the prefix itself is still the placeholder; the shards under `datasets/intel-scene-kit-v1/shards/` must be staged there before any host resolves this manifest, or the download fails on a 404 (RELEASING.md).
- The bundled copy ships the shards block; a wheel built before a kit re-publish therefore names a superseded kit until the remote manifest overrides it (remote wins, PLAN §A2).
