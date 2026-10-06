# Changelog

All notable changes to `3lc-compute-plugin-kaggle-classification` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [1.0.0rc7] — 2026-10-06 (session 6: Paul's demo feedback — the Projects page while training, the bundle with the data)

### Added
- **Watch run in Projects** in the Train tab's in-run header while a run is live (`/projects/<project>#runs`,
  new tab, Hub origin only); "Open Run in Projects" takes over at the terminal state. Fixture `train-state3`.
- The Hub's **Queue & Progress** card now shows the timing line (Elapsed | ETA | Per epoch): the job
  context forwards `timing = {elapsed_s, eta_s, avg_step_s, step_label}` with every progress flush, the
  shape the timm plugin sends; the card's label names the epoch in progress ("Epoch k/N").
- **The verification bundle carries the project's tables and runs** (reversing "never table data"; still
  never images, tokens or answer keys): `project/<project>/` with the seed-lineage train revisions and the
  locked val table (`object.3lc.json` + `row_cache.parquet`), every run's `object.3lc.json` and per-sample
  metrics tables, `files.json` (sha256, size and source path of every copied file) and `best.pt` by the
  **checkpoint rule**: at most two runs, the one behind the best public score and the one behind the most
  recent submission, each verified against the ledger's checkpoint sha256. The Export section shows the
  submitted runs as a checklist (max 2, the rule's picks checked) and the bundle's size before the
  export; `GET /status/bundle/preview` serves the plan; `?checkpoints=none|all` for organizers. The README
  explains the machine-specific paths; parquet string columns are scanned for secrets like text members.

### Changed
- The bundle is written under `<plugin home>/bundles/` and streamed as a file (old zips pruned after an hour).

## [1.0.0rc6] — 2026-10-05 (session 5: the rc5 fix also covers the rc3 / rc4 carry-forward markers)

### Fixed
- A carry-forward marker written by rc3 or rc4 has no `legacy_data_dirs`, so rc5 re-created nothing
  when the host removed the version dir those installs' tables point into (the tester's rc4 → rc5
  upgrade proof: the rc5 install garbage-collected `1.0.0rc2`, training failed on a missing image).
  The legacy kit tree is now derived from the marker's `source` when the field is absent.

## [1.0.0rc5] — 2026-10-05 (session 5: the old kit tree survives the host's garbage collection)

### Fixed
- Tables imported before 1.0.0rc3 keep their image paths into the previous version's kit tree, and the
  host removes old version dirs after three updates (`gc_old_versions`) — on the development Hub the
  rc4 install removed exactly the dir the existing projects' images live in. The carry-forward marker
  now records that data tree (`legacy_data_dirs`) and the plugin re-creates it from its own
  sha256-verified copy whenever it is missing (`storage.ensure_legacy_data_dirs`, on every process
  start). Nothing is deleted; tables imported since rc3 live in the shared home and need no copy.

## [1.0.0rc4] — 2026-10-05 (session 5: the upgrade proof's fix)

### Fixed
- Training after a plugin update refused every row ("6,600 rows whose images are not in the kit's
  train folder"): the carry-forward rewrites the import record's kit path to the new home while the
  existing tables' image paths stay in the previous version's kit tree. The foreign-rows gate now
  accepts a train image by the kit LAYOUT too (any `<kit>/data/train/` path); val and test rows and
  images from outside a kit stay foreign. Found by the rc2 → rc3 upgrade proof (`TESTING_PROOF.md`).

## [1.0.0rc3] — 2026-10-05 (session 5: the hand test's four findings)

### Fixed
- **A plugin update no longer loses the kit, the import record, the train / predict records or the
  ledger** (the tester blocker): the plugin's state now lives in `<compute home>/plugin-state/<id>`,
  outside the host's version dir (`storage.py` rule "compute-home"); the first resolution after an
  update carries the newest previous version's `.plugin-state` forward — a copy with the stored
  paths rewritten, the old copy kept, `migrated_from.json` written — and the Doctor's plugin-home
  row says so. Tested with an rc2 → rc3 upgrade with state present (`tests/test_storage.py`,
  `TESTING_PROOF.md` run 3).
- The Doctor reloads on tab enter, on Refresh, after every job completion and on the 15 s poll (the
  poll without the Kaggle calls, `GET /status/doctor?kaggle=0`); the hero's budget block falls back
  to the Doctor's Kaggle state; one wording everywhere: "N of M left today".
- The Doctor's manifest row reports the background refresh's result (`remote` when the fetch
  succeeded, "remote fetch failed: …" when not, the fetched-at stamp) instead of the no-network
  label "cache"; the route kicks the refresh like `GET /config` does.
- The Loop's inspect link opens the current project's latest Run (re-targeted after an import, a run
  and Predict's run list); with no run it is disabled with a tooltip instead of opening the
  Dashboard root.

## [1.0.0rc2] — 2026-10-05 (session 5: the tester proof's fix)

### Fixed
- The Status tab's History listed no prediction: the ledger stores a prediction's checks as
  `[label, ok]` pairs and `status.prediction_history` read them as dicts. Found by the tester proof on
  1.0.0rc1 (`TESTING_PROOF.md`); the unit fixture now uses the ledger's real shape.

## [1.0.0rc1] — 2026-10-05 (session 5: the release candidate)

The first tagged version. Everything listed under "Carried from 0.1.0" below shipped on `develop`
untagged between sessions 1 and 4 and is part of this release.

### Added (session 5, the Status tab — `docs/STATUS_MIRROR.md`)
- The Status tab, ExDark's control for control under decisions D1–D10: the hero strip (best public
  score, latest activity, Kaggle budget, live now), **Runs** (the current project's runs with the
  revision trained on, labeled rows, best val accuracy and epoch, device, elapsed, status incl.
  cancelled / interrupted, Dashboard and Projects links), **History** (every prediction with its
  submission, Kaggle's public score read back by ref — `GetSubmission`, since `ListSubmissions` 403s
  while the competition is unlaunched — and the score change vs the previous scored submission,
  ExDark's outcome vocabulary, Copy CSV path / Download CSV), **Kaggle live** (leaderboard top 5 and
  rank once the API answers; "available after the competition launches" while it does not), the
  15 s auto-refresh while visible, "Continue to Status" lands on it, `?kgdev=status-…` fixtures.
- **Verification**: Export verification bundle (`GET /status/bundle`), a zip of the import record,
  the train revision chain with lineage, every run's provenance record and checkpoint sha256, the
  ledger, the manifest provenance and the plugin version — never images, table data, tokens or
  answer keys; a secret-pattern scan refuses the whole export on a match.
- **Doctor** (collapsed): plugin version and commit, compute service version (from `/health`),
  SDK / 3lc / torch / torchvision versions, CUDA availability in the worker, device class, manifest
  source and sha256, kit version and verification state, Kaggle connection, plugin home and how it
  was resolved, free disk space, Python; Copy diagnostics with a `[doctor]` section.
- Routes `GET /status/history`, `GET /status/kaggle`, `GET /status/doctor`, `GET /status/bundle`;
  `status.py`; `tests/test_status.py`.

### Changed (session 5, post-demo fixes)
- `torch==2.14.0` and `torchvision==0.29.0` pinned exactly on every platform (the parity-gate build;
  the 2026-10-02 demo worker had drifted to 2.14.1 / 0.29.1 through the catalog install path).
- Train and Predict warn when the val table has revisions newer than the locked one: "Your edits to
  val in the Dashboard are ignored; every run is scored on <locked revision>." (`import_state.val_edited`).
- The Train gate refuses a revision containing rows whose images are outside the kit's train folder
  (a Hub merge / concatenate that pulls in val or test rows can descend from the seed by lineage;
  `trainer.foreign_rows`, `GET /train/preflight` serves the count, Start disabled, the job refuses).
- Versions may be PEP 440 pre-releases (`scripts/release_version.py --stamp 1.0.0rc1`); the catalog
  orders entries by PEP 440. The catalog's never-tagged `0.1.0` entry is replaced by this one.
- Tester kit: `TESTING.md`, `tester/start_tester.ps1`, `tester/start_tester.sh`, `catalog-test.json`
  on the release branch; the proof of the kit on a fresh environment in `TESTING_PROOF.md`.

### Carried from 0.1.0 (sessions 1 to 4, never tagged)

### Added (session 4, the Predict + Submit tab — 2026-10-01)
- The Predict + Submit tab, ExDark's control for control (`docs/PREDICT_MIRROR.md`, decisions D1–D12
  of 2026-09-30): the run picker fed by the train records (`GET /runs`, unusable runs disabled with
  the reason), the locked test-images row + gate verifying every test file against the kit's
  `files.json` (`GET /predict/preflight`), inference from the run's best checkpoint only after the
  three-way sha256 match (train record · Run · file on disk) and green provenance, the val check on
  the locked val revision (the hero stat; must reproduce the run's recorded best val accuracy), the
  CSV in `sample_submission.csv` order with six-decimal confidence, the seven format checks (the file
  kept as `.INVALID.csv` on failure), the predicted-class distribution card (warns below 5 % / above
  50 % per class), the Kaggle connection card (credentials detected on the compute host, never read;
  the manifest's daily limit, "N of 100 left today" from GetSubmissionLimits), the locked competition
  slug, Submit with one status read-back by ref (Kaggle's public score or error description on the
  banner; ListSubmissions 403s on the unlaunched competition, so it is the fallback only), the soft outcomes (limit reached /
  not joined / no credentials) and the CSV download fallback (`GET /submissions/{job}/download`).
- Job kinds `predict` and `kaggle_submit`; durable `predict_state` / `submit_state` records with the
  Train pid rule (`GET /submit/state`); the append-only ledger (`ledger.jsonl` under the plugin home)
  with an entry per prediction and per submission (run URL, checkpoint hashes, manifest provenance,
  CSV sha256, Kaggle ref and read-back status); `?kgdev=submit-…` fixtures for every Predict state.
- Train's Previous runs and Predict's run picker list only the runs of the import record's project
  (`GET /train/state` serves `project_runs`; the duration estimate keeps every run).
- The Train tab's opening banner, the Predict revisit and the tab bar's done marks follow the same
  project: a record from another project opens its tab on the form (re-checked after an import and on
  tab enter).
- The bundled manifest's `competition.slug` is `3lc-scene-classification-challenge` (the event
  competition); `cdn/` regenerated.

### Fixed (session 3 re-check, 2026-09-29 evening)
- Use these settings never copies a run's device; the in-run header and the log name why the run is
  on its device (`cuda (auto)` / `cpu (forced in Advanced)` / `cpu (fallback: …)`); a blank Device
  field is CUDA whenever the worker's torch sees a GPU. (A copied forced `cpu` had silently trained
  later runs on CPU.)
- The duration estimate's history and benchmark are keyed by the resolved device class; the in-run
  remaining time follows the run's class.
- The config load retries with backoff while the worker starts and never renders an empty form (the
  last successful load fills the form if the load truly fails; nothing cached keeps it hidden).
- Use latest revision is on for every new run; a pinned revision with newer ones warns.
- The previous-runs dropdown shows each Run folder's unique name and says "interrupted"; older runs
  are backfilled from their Runs (or disable Use these settings with a reason).
- The revision tree's connector sits inline and names never wrap.

### Changed (session 3 review, 2026-09-29 afternoon)
- The model is the Intel kit's exactly: torchvision resnet18 with the kit's MLP head, torchvision's
  init, the kit's seeding and creation order; `model.backbone` / `model.head` in schema v1
  (allowlisted; `arch: resnet18` still read as the kit's model); timm dropped. Parity gate against
  `intel-kit/train.py` (3lc 2.22.3, seeds 42-44): kit 70.53 +/- 1.59, plugin 70.53 +/- 2.08 val accuracy.
- `training.editable` + `training.options` in schema v1: a field renders only when the manifest opens
  it, else as a locked row and refused server-side; optimizers adam / adamw / sgd, schedules steplr /
  cosine / none. This event opens epochs, batch_size, lr, weight_decay, seed.
- Per-sample columns: `prob_*` removed; `accuracy` added and, in the same day's re-check, removed
  again — the collected columns are label, weight, predicted, confidence, loss and Embedding (3D).
- The table picker shows the seed lineage as a tree with labeled-row counts and runs-used, greys
  other imports, links the Hub project's Datasets view; a Previous runs panel (dropdown + card,
  Use these settings); the train URL prefills from the seed; banners name the checkpoint and lead
  with the best val accuracy; interrupted runs are marked on the Run; a persistent numba cache and
  a UMAP pre-warm; the duration estimate = setup + training + collection, hidden while a field is
  out of bounds.

### Added (session 3, the Train tab — 2026-09-29)
- The Train tab, ExDark's control for control (`docs/TRAIN_MIRROR.md`): the locked contract (resnet18
  from random init · 150 px · Adam · StepLR ×0.1 every 5 epochs · single forward pass) as a banner and
  locked rows; the train table URL with the revision picker and the locked val row; a gate that verifies
  existence, split identity, the seed lineage and the label map and shows the usable-row summary
  (labeled in use · excluded as undefined · excluded at weight 0) with warnings for undefined rows at
  weight > 0 and for classes without rows; Epochs / Batch size / Learning rate / Weight decay with the
  manifest's bounds, Device / Workers / Seed under Advanced, inline errors that block Start; the in-run
  view (three chips with sparklines, the resolved device, the ETA), the provenance panel (eight checks
  read back from the Run), the success / cancelled / failed / interrupted banners, revisit; the
  duration hint from this machine's history scaled by usable rows, or the bundled benchmark on a first
  run; `?kgdev=train-*` fixtures.
- Job kind `train` (`trainer.py`): the Intel kit's recipe on timm's standard-head resnet18; tlc's weighted
  sampler semantics from in-memory effective weights (undefined forced to 0, never written back; zero
  usable rows refuses); best (strict `>`) and last checkpoints written atomically under `<run>/model`
  with sha256 on the Run; end-of-training per-sample metrics on every train and val row (predicted,
  confidence, per-class probabilities, loss absent for undefined rows, 3-D UMAP fit on train / val
  transformed, PCA fallback); cooperative cancel keeps the best-so-far checkpoint; CPU retry when the
  accelerator fails at start; a durable train record with worker-pid orphan detection (a restart reads
  back as interrupted), a heartbeat and sleep-gap log, and a duplicate-start guard.
- Routes: `GET /train/preflight`, `GET /train/state`, `GET /tables/list`, `GET /tables/defaults`;
  `_meta.training` (defaults, effective bounds, the locked optimizer and schedule, the benchmark) and
  `_meta.train_state` on `GET /config`. Seed bounds `[0, 2147483647]` in the bundled manifest.

### Changed (session 2.5, the ExDark mirror)
- The plugin now mirrors 3lc-compute-plugin-kaggle v1.2.15 (`docs/EXDARK_MIRROR.md`): the shell and
  the Import tab are a port of its fragment (innerHTML + `esc()`, icons, motion, connection guard,
  diagnostics, preflight, progress rows, result and failure banners, revisit, download section,
  `?kgdev` fixtures) under the four allowed differences; existing tables are REUSED and re-validated;
  Re-import fresh writes fresh `initial-N` tables beside the old ones; the hero title is the
  manifest's display name; a stale result clears when the preflight turns amber or red.
  `GET /download/verify`; the kit record carries the manifest provenance of the download.

### Changed (session 2.5, superseded passes)
- Competition display name in the bundled manifest: "3LC Scene Classification Challenge"; `cdn/` regenerated
  (manifest sha256 `d9f34aed…db83c`, index `e0e9c4bc…f262d`), not yet uploaded.
- The fragment at the ExDark presentation standard (`docs/UI_PARITY.md`): hero with constraint chips
  and the Loop row, the tab bar as stepper with state glyphs and keyboard support, Hub card and
  form classes, one callout geometry, stepper rows with status badges and elapsed, checks as a
  collapsed verdict that auto-expands on failure with remedies, Technical details and Show log
  disclosures, truncated paths with Copy, "Open in Dashboard" and the Hub project link, the
  connection guard, Copy diagnostics, "Next: train your first model", gated later tabs, reduced
  motion honoured everywhere. `import_state` now carries `latest` (the newest revision per split).
- Progressive disclosure after the State 1 review (`docs/UI_PARITY.md` §3b): one status line and one
  primary action per tab by default; step rows under "Show steps" (auto-open + scroll on failure),
  one progress line with shard, bytes and ETA while running; the import form hidden until the kit
  is on disk, Project / Table name under "Advanced", the re-import toggle only on a collision; no
  `undefined` class tag (the pool is explained on the classes chip); the manifest source moved
  into Technical details; `submission.daily_limit` in the bundled manifest is 100 (was the
  session-1 placeholder 3), so the chip reads "100 submissions per day"; `cdn/` regenerated
  (manifest sha256 `6a10e44f…efed2`), not uploaded.

### Changed (session 2)
- Manifest schema v1: `kit.base_url` replaced by a relative `kit.path`; shard URLs resolve against the
  URL the manifest was fetched from; the index's `manifest_url` is relative to the index. Layout
  `kaggle/classification-index.json`, `kaggle/<id>/manifest.json`, `kaggle/<id>/starter-kit/<v>/`.
- Host allowlist: prod only by default; the dev CDN and loopback only under the base-URL override.

### Added (session 2)
- Importer (`importer.py`, job kind `import`): kit validated against the manifest (structure,
  counts, `sample_submission.csv` ids, every image decodes), `train`/`val` tables on tlc 3.3 with a
  distinct `undefined` label value and per-row weights, collision refusal with explicit re-import
  to fresh `<table>-N` tables, no partial tables on failure or cancel, the import record with
  lineage root, locked val, checks, timings and manifest provenance. Routes `GET /import/preflight`
  and `GET /import/state`.
- Import tab: one stepper for download + import, preflight gate with the collision callout,
  per-check pass/fail, table links into the Hub, revisit from the import record, first-run
  provisioning rendered as an expected state.
- Line endings normalized to LF (`.gitattributes`).
- `docs/PROMOTION.md` §6: the dev tier verified (served headers, sha256s of all seven objects) and
  the console upload steps as performed; `tests/test_deletion_safety.py` (the deletion audit).
- `tools/make_cdn_tree.py` (the bucket mirror + `upload-plan.json`), `tools/verify_cdn.py`
  (served headers and sha256 verification), `docs/PROMOTION.md`.


### Added
- Plugin scaffold from the 3LC template: `plugin.toml` (id `kaggle-classification`,
  `min_service_version` 1.1.0), Apache-2.0, SDK window `>=0.3.1,<0.4.0`, torch from the
  cu126 index as the timm plugin declares it, `timm==1.0.29`.
- Competition manifest schema v1 (`manifest.py`) with field-naming validation, unknown-field
  warnings, the bundled `manifests/intel-scene-v1.yaml`, and the derived facts
  (`num_classes`, `undefined_label_id`, `dataset_name`, `expected_rows`).
- Session store (`session.py`): one canonical session derived from the manifest, retired-key
  rejection, atomic writes, URL parsing by position in the layout tail.
- Kit download stage (`kit.py`): sha256 + Range-resumable shards, zip-slip guard, per-file
  verification against `files.json`, split counts checked against the manifest, revisit states.
- `tools/build_kit.py`: the kit build pipeline (salted opaque ids from `--salt-file`, RGB JPEG q92 re-encode with EXIF/ICC stripped, `files.json`, `sample_submission.csv`, deterministic `intel-scene-v1-NN.zip` shards, `kit-manifest-block.yaml`, judge-only `mapping.csv`) and a participant-side verification pass; the bundled manifest carries the v1 build's `kit{}` block.
- Manifest resolution: remote index + manifest (5 s budget, one retry, server-side only) →
  cache with `{fetched_at, source_url, sha256}` sidecar → bundled; invalid remote falls back
  with a visible warning; competition picker when several are active; job-start provenance
  (`resolve_manifest_for_job`); kit host allowlist, https-only help links, markup-free display
  strings; background refresh so the fragment never waits on the network.
- `storage.py`: the plugin home resolved env → SDK helper → worker state root →
  `<cwd>/.plugin-state/<id>` → `~`, reported on `GET /config`.
- Four-tab fragment shell, `GET/POST /config`, the `download_kit` job kind.
- Test suite: manifest, session, kit stage, packaging + SDK-window overlap + import weight +
  license lineage, the ctx adapter, the timm offline model check, the release-version script.
