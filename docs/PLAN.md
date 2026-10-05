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
| Model | **The Intel kit's model, exactly** (part A of the 2026-09-29 review, reversing the morning's timm decision): torchvision `resnet18(weights=None)` with `fc = Identity` and the kit's MLP head 512→256→ReLU→Dropout(0.3)→128→ReLU→Dropout(0.3)→N, torchvision's default init. Manifest-driven for reuse: `model.backbone` (`torchvision_resnet18`) and `model.head` (`kit_mlp_512_256_128_d03` \| `linear`), allowlisted in `manifest.py` (`BACKBONES` / `HEADS`); `arch: resnet18` is accepted as the legacy spelling of the kit's model. `pretrained: true` is rejected. timm is not a runtime dependency; torch and torchvision versions are recorded on every Run. Requirement: a participant using the plugin and one using `intel-kit/train.py` get similar outcomes (the parity gate in `docs/TRAIN_MIRROR.md` §10). |
| Competition manifest | Remote `<base>/kaggle/classification-index.json` + `kaggle/<id>/manifest.json` (layout in §A3); base URL is a code constant (`manifest.MANIFEST_BASE_URL` = prod); env override `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL` for dev/local; on-disk cache with fetched-at stamp; bundled `manifests/intel-scene-v1.yaml` as last resort. Unknown fields warn, never fail. |
| Splits | Unchanged from the Intel kit: 600 seed (100 × 6) + 6,000 `undefined` at weight 0 in `train`; `val` 1,200 (200 × 6) locked; `test` 1,800, flat, never registered. |
| Labeling | No cap (reaffirmed 2026-09-29, D14: the Intel kit's 3,000 weight-1 refusal is not adopted; the Train gate's usable-row summary is informational). Undefined rows are filtered out of training regardless of weight. |
| Training recipe | Locked beyond the manifest (session 3, D2/D3/D11): Adam, `StepLR(step_size=5, gamma=0.1)` stepped per epoch, the Intel kit's torchvision augmentation (resize → random crop → horizontal flip → affine shear 10 / scale 0.8–1.2 → ImageNet normalize; val = resize + center crop), seed + cudnn determinism as the kit. Shown as locked rows on the Train tab; the plugin constants live in `trainer.py` and are served, never restated in the fragment. |
| Checkpoints | `<run>/model/best.pt` (best val accuracy, strict `>`) and `<run>/model/last.pt`, written atomically; sha256 of each on the Run and in the train record. **Predict uses best.** A cancelled run keeps its best-so-far checkpoint and is usable in Predict (D10). |
| Kit build | `tools/build_kit.py --salt-file` (never `--salt`); the salt is read from a private file, never printed or written; shards `intel-scene-v1-NN.zip`, deterministic; `kit-manifest-block.yaml` beside the kit dir; `mapping.csv` (original_relpath, new_relpath, split, class with class empty for test and `undefined` for pool rows) to the private dir only. |
| Kit | Images renamed to salted opaque ids (`sha256(salt + original_relpath)[:16]`) and re-encoded (JPEG q92, RGB, EXIF stripped). `mapping.csv` (the judge's key) lives ONLY in the private output dir. |
| Kit integrity | `files.json` (relpath, sha256, bytes) INSIDE the kit beside the data; the download stage verifies **per file** against it, not shard-only, then checks per-split counts against `manifest.splits`. Shards are sha256-verified from the manifest's `kit{}` block. |
| Predict | Only from plugin-created runs. |
| Ledger | Append-only ledger + a verification bundle per run (session 5). |
| Teams | Per machine. No table sharing. |
| Tabs | Import, Train, Predict + Submit, Status. |
| Hosts | `min_service_version = "1.1.0"` (the 1.1.0 torch-backend fix is required for a clean Windows install). |
| Torch index | Mirrors the timm plugin exactly: `pytorch-cu126` explicit index, `torch`/`torchvision` sourced from it on `linux` and `win32`, PyPI on macOS. |
| Python | `requires-python = ">=3.11"`, **no upper bound** (verdict 2026-09-28). The host picks the plugin venv's interpreter, never uv's default: compute 1.1.0 `provisioning.resolve_provision_python` passes `--python <host major.minor>` (`sys.version_info` of the service process) to both `uv sync` (folder source) and `uv venv` (catalog/spec install), overridable only by a `[runtime] python` key in `plugin.toml`, which this plugin does not declare. The service itself runs on 3lc, whose wheels are cp310–cp313 with `Requires-Python <3.14` (3.3.0–3.3.2 checked on PyPI), so no participant host can be 3.14 and no plugin venv can be either. The one place a newer interpreter CAN sneak in is a bare `uv sync` in a checkout (uv picks the newest Python on the machine), which is why the dev loop passes `--python 3.12`. The floor stays at 3.11: the timm plugin says `>=3.10`, but `kaggle>=2.2.3` and `scikit-learn>=1.9` floor at 3.11 and uv locks the whole range. |
| DataLoader workers | Device-aware since session 3: the served default is 0 on Windows and `min(4, cpu_count)` elsewhere (`trainer.default_workers`), bounds 0–16 (a plugin constant); an Advanced field on the Train tab. The import stage still loads nothing through torch. |
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

## A3. CDN layout and tiers (session 2 decisions)

- **Two tiers, byte-identical objects.** Dev: bucket `3lc-competitions-dev` at
  `https://competitions.dev.3lc.ai` (Rishikesh uploads via the console). Prod:
  `https://competitions.3lc.ai` (Gudbrand promotes by copying keys, only after the plugin is
  complete). Testing stays on dev; nothing on prod changes in session 2. `COMPETITION_ID`
  (`intel-scene`) is a stable bucket id decoupled from the Kaggle slug (the ExDark convention).
- **Layout** under either base: `kaggle/classification-index.json` (mutable),
  `kaggle/intel-scene/manifest.json` (mutable, hotfixable),
  `kaggle/intel-scene/starter-kit/v1/<shards>` (immutable once staged; a changed kit is a new
  version prefix).
- **Relative URLs only.** The index's `manifest_url` is relative to the index URL; the manifest's
  `kit.path` (`starter-kit/v1/`) is relative to the manifest's own URL, and shard URLs resolve
  against the URL the manifest was actually fetched from (`Manifest.document_url`,
  `Manifest.shard_url`). `kit.base_url` is gone from schema v1 and a document carrying it is
  rejected. The bundled copy resolves under the base in force (prod, or the override).
- **Host allowlist.** Release default is `competitions.3lc.ai` only (`RELEASE_HOSTS`).
  `competitions.dev.3lc.ai` and loopback are admitted only while
  `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL` is set (`allowed_hosts()`); a cached copy fetched
  from a dev host is not served once the override is gone.
  `tests/test_manifest.py::test_host_allowlist_is_prod_only_without_the_override` is the
  release-audit rule.
- **Integrity** is sha256 from the manifest (shards) and `files.json` (files), never ETag
  (multipart uploads make ETags meaningless).
- **Caching.** The dev distribution is Managed-CachingOptimized with no origin request policy;
  object `Cache-Control` is honored (min TTL 1 s) and query strings are not in the cache key.
  Therefore: no query-string cache busting anywhere; mutable objects are uploaded with
  `Cache-Control: max-age=60`, shards with `public, max-age=31536000, immutable`; every mutable
  update is followed by a CloudFront invalidation of the two mutable paths (`docs/PROMOTION.md`).
  `tools/verify_cdn.py` reports served `Content-Type`/`Cache-Control`/`X-Cache` and fails a
  mutable object without `max-age <= 300`.
- **Job store.** No separate disk-backed job store. Live progress is the host job panel;
  durable outcomes are the session store's kit and import records (later the ledger and run
  history). Compute 1.1.0 keeps job records in memory only (`plugins/job_manager.py`,
  `_jobs` dict pruned by `_MAX_RECORDS`): they do NOT survive a service restart, which is
  why the durable record lives in the session store.
- **Two install paths on `3lc-hub-11`.** `--plugin-dir` for iteration; a private test catalog
  (`catalog-test.json`, source = this repo at a pushed ref) via `TLC_COMPUTE_PLUGIN_CATALOG_URLS`
  for gate tests. Compute 1.1.0 reads that variable as the whole operator tier: it REPLACES the
  baked-in default catalog unless the default URL is listed too, and it accepts `https://`,
  `file://`, local paths, and `http://` on loopback only (`plugins/catalog.py`).

## B. The labeling-loop contract (per-sample metrics)

What every plugin-trained run writes back onto the train and val tables, so the Dashboard
loop (inspect, label, fix, retrain) has what it needs. Written with `run.add_metrics(...,
foreign_table_url=<table>, schema=..., constants={"epoch": best_epoch})`, one metrics table
per split, computed on the restored best-epoch model with the val (non-augmented) transform.

| Column | Type / schema | Undefined rows |
|---|---|---|
| `predicted` | categorical, same value map as the table's `label` column (class names) | yes |
| `confidence` | float32, max softmax | yes |
| `loss` | float32 cross-entropy, `reduction="none"` | **masked** — no value fabricated; the column is written only for rows with `label < num_classes` (a NaN/absent value, never 0 or a placeholder) |
| `embeddings` | float32 vector, shape `(n_components,)` = 3 | yes |

Those five, beside the table's own `label` and `weight`, are the collected columns: no `prob_*`
(2026-09-29 review, part C) and no `accuracy` (the same day's re-check, item 8 — a 1 / 0 column
adds nothing the Dashboard cannot derive from `predicted` and `label`).

Embeddings are the 512-d backbone output (the kit's `fc = Identity` layer, `model.features(x)`)
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
- **Importer (session 2, shipped)** — `train` (labeled weight 1.0 + `undefined` = label N at
  weight 0.0) and `val` (all weights 1.0, URL recorded as the locked revision) via
  `Table.from_dict` with an explicit schema (`ImageSchema(url)`, `CategoricalLabelSchema(classes +
  undefined)`, `SampleWeightSchema`) at `<tlc.config.project_root_url>/<project>/datasets/<manifest.dataset_name(split)>/tables/<table>`,
  `if_exists="raise"`; `test` never registered. The kit is validated against the manifest first
  (class dirs, per-class and pool counts, val counts, test count == `sample_submission.csv` rows
  and ids, header, kit version, every image decodes) and any mismatch fails with one message
  naming every defect, before any table exists. Collisions REFUSE; an explicit `mode=reimport`
  writes fresh `<table>-N` tables for both splits and never touches the existing ones. The second
  table failing, a post-write verification failing, or a cancel deletes what the job wrote (no
  partial tables). The record (`import_state`) carries the lineage root, the locked val URL,
  checks, timings and the manifest provenance from `resolve_manifest_for_job`.
- **Trainer (session 3, shipped)** — see §B and `docs/TRAIN_MIRROR.md`; job kind `train`. Params =
  manifest defaults ⊕ form, bounded on the merged kwargs (`training.bounds`; seed and workers bounds
  are plugin fallbacks when the manifest has none); presets stay a manifest fact, not rendered (D13).
  The train revision must descend from the import record's seed (`GET /train/preflight` walks the
  lineage; the job refuses otherwise); val is the import record's locked URL. The sampler is tlc's
  weighted semantics built from in-memory effective weights (undefined → 0; zero usable rows is a
  refusal, never a fallback to shuffling). The Run records the contract (arch, image_size,
  pretrained=false, timm version, seed, both table revisions, the checkpoint sha256s, device) and
  eight provenance checks read it back. A durable record (`train_state` in the session store)
  carries status, params, facts, checks, result, the log tail, the worker pid and a heartbeat: a
  compute restart mid-run reads back as `stale` (pid rule), never as complete; a duplicate start
  (running record, or a consumed client token) is refused. The accelerator failing at model or
  first-batch time retries on CPU and says so. Tables are never modified.
- **Predictor + Submit (sessions 4, 5)** — plugin-run-only weights; `submission.csv` with
  `manifest.submission.columns` in `ids_from` order; Kaggle API against `competition.slug`;
  budget `daily_limit`.
- **Ledger (sessions 4–5, shipped)** — JSON-lines under the plugin home (`ledger.py`); the verification
  bundle (`status.verification_bundle`, `GET /status/bundle`): the import record, the train revision
  chain, every run's provenance + checkpoint sha256, the ledger, the manifest provenance, the plugin
  version — never data, tokens or answer keys (a secret-pattern scan refuses the export).
- **Status (session 5, shipped)** — `docs/STATUS_MIRROR.md`: the hero strip, Runs, History (Kaggle's
  public score read back by ref, the delta vs the previous scored submission), Kaggle live with the
  after-launch degradation, the Doctor; routes `GET /status/{history,kaggle,doctor,bundle}`.

## D. Session map

| Session | Delivers | Needs from the organizer |
|---|---|---|
| 1 (this) | scaffold, manifest schema + loader, session, kit stage, kit builder, docs | the salt; the CDN prefix; the Kaggle slug and deadline |
| 2 | Import tab: kit download UI, table registration, revisit view, pickers | the published kit prefix (Phase 3 output staged) |
| 3 | Train tab: trainer per §B, bounds, device-aware workers, the ETA benchmark (shipped 2026-09-29) | a GPU box to record the reference trajectory (the laptop's RTX 3070 Ti, gate G1) |
| 4 | Predict tab: plugin-run-only inference, submission.csv | test-set answer key on the organizer machine (local scoring, optional) |
| 5 | Submit (shipped in 4) · the Status tab · the verification bundle · post-demo fixes (torch pinned, val-edit warning, the foreign-rows gate) · release candidates **1.0.0rc1** and **1.0.0rc2** (the proof's fix) tagged · the tester kit proven (`TESTING.md`, `TESTING_PROOF.md`) — 2026-10-05 | the hosted test catalog URL for testers (RELEASING.md) |
| 6 | tester feedback → 1.0.0; the open items (version-scoped plugin state across updates, the orphaned-worker investigation, the cdn manifest re-upload) | the launch date; Gudbrand's promotion of the manifest to prod |

## E. Open items (TBC, placeholders in the bundled manifest)

- `competition.slug` (folder name `3-lc-hack-nova-scene-classification-challenge` assumed) and `deadline_utc`.
- The `kit{}` block is pasted (v1 build of 2026-09-22, five shards, 113,741,154 bytes) with `path: starter-kit/v1/`; the shards must be staged under `kaggle/intel-scene/starter-kit/v1/` on the tier in force before any host resolves this manifest, or the download fails on a 404 (docs/PROMOTION.md).
- The bundled copy ships the shards block; a wheel built before a kit re-publish therefore names a superseded kit until the remote manifest overrides it (remote wins, PLAN §A2).
