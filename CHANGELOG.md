# Changelog

All notable changes to `3lc-compute-plugin-kaggle-classification` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased] (session 11, 2026-10-09: the post-rc12 quality-of-life pass)

The UI look is frozen at rc12: further UI changes are incremental only (no layout changes, no redesigns) and
participants keep every control they had in rc10.

### Changed
- **Collapsible checks on Train and Predict**, the Import tab's component: Train's provenance list collapses to
  "Provenance verified · 9/9 checks ▸" when every check passes (the "Verified provenance recorded" verdict and the
  hint box stay), Predict's to "10/10 checks passed ▸" (the predicted-class distribution and val accuracy cards stay).
  Both open once when the checks arrive live, collapse on later visits, and auto-expand with the failures first on
  any failure. Nothing else on either tab moves.
- **Accessibility, without a layout or copy change** (the `web-design-guidelines` audit): one visible keyboard focus
  ring (`:focus-visible`) on every interactive element; the grey tints that vanished on the Hub's dark theme take its
  dark tokens there (light mode untouched); card and section titles carry `role="heading"`; every new-tab link carries
  `rel="noopener"`; the "Connect in Step 2" link answers Enter and Space; every field carries `name`, `autocomplete`,
  `spellcheck` and, for numbers, `inputmode`; the four progress bars are `role="progressbar"` with a live value.
- **Typography** (the `writing-guidelines` audit): curly apostrophes and quotes in every displayed string and in
  TESTING.md; the table-name helper now says "Table name for the imported train and val tables."; the LR-schedule help
  no longer shows a literal backslash ("the kit's step decay"); the two untagged code fences in TESTING.md are
  tagged `text`. No instruction changed meaning.

- **Review-list decisions (session 12):** the spinner and the refresh icon pulse in opacity under
  `prefers-reduced-motion` instead of rotating; button labels are sentence case on every tab ("Import & validate",
  "Start training", "Re-run training", "Open run in Dashboard", "Open run in Projects"; tab names stay
  capitalised in "Continue to Train" and "Go to Import"); **Re-import fresh** asks first, with the Re-import… line
  naming the table the server would write and saying the existing tables and label edits are kept; TESTING.md
  lost its filler words and three passive or metaphor sentences.

### Added
- Vercel's `web-design-guidelines` and `writing-guidelines` agent skills under `.claude/skills/` (`skills-lock.json`).

## [1.0.0rc12] — 2026-10-09 (session 10: the Import tab back on the rc10 layout)

Decision (Rishikesh, 2026-10-09): the rc11 Import redesign was too minimal. The Import tab goes back to
the rc10 layout; the only visual addition kept is the collapsible checks list. rc11's behaviour stays.

### Changed
- **The Import tab is the rc10 layout again.** The form is always visible: the starter kit folder, the
  project name and the table name with their helper text, the "Detected: train / val · N classes
  (manifest order) · N / N images. Matches the competition manifest." line, the locked **Splits to
  import** block, the **Dataset at a glance** card (train / unlabeled / val and the class tags). The
  post-import banner carries **Continue to Train**, **Explore train** and **Explore val**; the table rows
  carry the REUSED / CREATED badge, the path, Copy, Explore and **Re-import fresh** on a reused row (no
  dialog: nothing is discarded); **Show log** follows. The rc11 facts line, the **Advanced** disclosure
  and the Re-import… confirmation on the Imported view are gone; the stale view keeps its Re-import…
  with the confirmation. The Kaggle Data-page notice sits above the kit folder.
- **Kept from rc11, unchanged in behaviour:** the tab and the stepper render from the `GET /config`
  payload; the revisit shows the full post-import content (rc10's thin view does not come back);
  "Found existing tables · validated" vs "Imported · validated"; the import lock while a job runs; the
  stale-record message; the CDN bundled-manifest notice; the Kaggle Data-page fallback when a download
  or a verification fails; the five-minute remote-down memory; the header collapse; every other-tab
  change (the in-app submit confirmation, Other ways collapsed, the "Run inference first" line, the
  Kaggle heading, the Doctor refresh, the Queue subtitle).
- **The checks list stays collapsible:** "N/N checks passed ▸" collapsed when all pass, auto-expanded
  with the failures first when any fails, where rc10's checks report sat.
- **No em dashes anywhere in the plugin:** the fragment, run and table descriptions, job subtitles, the
  Doctor's text, the bundle README, every refusal and log line, `plugin.toml`, TESTING.md and the
  catalog descriptions. A period, colon, comma or middle dot instead; the empty cells of the Runs and
  History tables use the en dash the History table already used for an absent public score.

### Fixed
- **Re-import naming:** after `initial-2` the next re-import writes `initial-3`, never `initial-2-2`
  (`importer.fresh_table_name` continues the series from a generated `-N` tail; `GET /import/preflight`
  names it the same way).

### Added
- `tests/test_copy_rules.py`: fails on an em dash in `ui.html`, in every module's text, `plugin.toml`,
  `catalog.json`, `TESTING.md` and the tester scripts, and in the served config payload and the bundle
  README. `test_importer.py` pins the fourth re-import at `initial-3`.
- `tools/fixture_harness/`: the static screenshot harness (the Hub's public CSS, a stubbed `PLUGIN_API`,
  a config payload the plugin generates against a scratch home, Playwright on Edge), checked in; the
  rc12 shots of every Import fixture are in `docs/ui-review/after-rc12/` (untracked).

## [1.0.0rc11] — 2026-10-09 (session 9: the Import tab and shell UI pass)

A deliberate divergence from the ExDark mirror on the Import tab and the shell (CONTEXT.md, decided
2026-10-09; `docs/EXDARK_MIRROR.md` §5) — **back-port to ExDark after the event**.

### Changed
- **The Import tab renders from server truth.** `GET /config` now carries everything the tab needs
  (`_meta.import_state`: the state `empty` / `found` / `success` / `stale`, both tables with existence,
  row count and table name, the missing split, the last validation result with its timestamp and checks,
  the kit folder and the job's log; `_meta.kit_state`: image count, verification stamp, download or
  manual source; `_meta.manifest_fallback`), and the tab plus the stepper read that payload and the live
  job list only. Import counts as done when both tables exist on disk, with or without a record (a reset
  plugin state over an existing project root shows **Found existing tables**).
- **One Imported view** for every entry path: project, table, when, validated, the kit folder, one
  **Re-import…** action, the rows (table name, row count, "imported" / "found existing" with the time,
  middle-truncated path with the full path on hover and Copy, Explore), the collapsed
  "N/N checks passed ▸" line and the log. A successful import transitions into it with the checks
  expanded once. The revisit bug is gone: the outcome area sits outside the form, so hiding the form no
  longer hides the checks and rows.
- **Honest banner:** "Imported · validated" for a fresh import, "Found existing tables · validated" for a
  reuse; the Explore buttons left the banner (the rows keep theirs).
- **Stale record:** a recorded table missing on disk is named (split, table, path, when it was recorded)
  with **Re-import…**, instead of the plain form.
- **Re-import…** replaces Start over and the per-row Re-import buttons: an in-app confirmation names the
  table a re-import would write (the server answers `reimport_name` on `GET /import/preflight` before
  anything runs) and says "Your existing tables and label edits are kept."
- **Locked while running:** the button is disabled, progress shows, and a double click or a tab switch
  can never start two imports.
- **Form cleanup:** the kit folder, project name and table name live under **Advanced**; the locked
  "Splits to import" block is gone; the test-images sentence and every count live once in **Dataset at a
  glance** (train / unlabeled / val / test + class tags); the preflight verdict reads "Matches
  competition manifest ✓"; paths show one separator style with the full path on hover and a Copy button.
- **Kit line:** "Starter kit ready · 9,600 images + manifest · verified <date>" with **Re-verify** (never
  "9,601 files"); a folder the participant supplied verifies the same way (`GET /download/verify?kit_dir=`),
  and a pass records it as a manual kit.
- **Checks:** "N/N checks passed ▸" collapsed when all pass; a failure auto-expands with the failing checks
  first, each with its fix hint.
- **CDN fallback:** when the manifest in use is the built-in or the cached copy, one quiet notice on the
  Import tab and the Doctor's Manifest row say so ("Using the built-in competition manifest; the server
  couldn't be reached") — surfaced through `_meta.manifest_fallback`, never only the worker log. After
  one unreachable index, job starts resolve locally at once for five minutes while the background refresh
  keeps retrying (no 5 s budget per job start). When a download or a verification fails and a resume
  will not help, Advanced opens with the manual route: "Couldn't download the starter kit. Download it
  from the competition's Data page on Kaggle, unzip it, and paste the folder path here", linking the
  competition's Data page.
- **Shell:** the header card collapses to one line from the second visit on (per browser, **Show
  details** / **Hide details**); the Loop's links are underlined links; Status sits apart, right-aligned
  from steps 1–3.
- **Other tabs:** the Queue card's subtitle opens as "<job> · <project>" (the title is the host's plugin
  name — `tlc_compute` sets it, not the plugin); Submit asks in an in-app confirmation naming the run,
  its val accuracy and the submissions left (no native `confirm()`); "Other ways to connect" lines never
  wrap (horizontal scroll + Copy); "Run inference first" never shows over an existing CSV; the Status
  tab's Kaggle heading reads "Kaggle (competition not launched yet)" until the competition is live; the
  Doctor's Kaggle block refreshes right after a successful Connect.

### Added
- `?kgdev` fixtures: `state1-bundled`, `state2-existing`, `state6-found`, `state6-stale`, `dl-verify-fail`,
  `dl-fail-checks`, `dl-fail-kaggle`.
- Tests: the fallback status and the remote-down memory (`test_manifest_resolution.py`), the kit line's
  facts and the manual kit's Verify (`test_kit.py`), the import state's tables / validation / log, the
  stale split, the record-less found state and the re-import name (`test_importer.py`); the fragment
  census follows the new templates.

## [1.0.0rc10] — 2026-10-08 (session 8: the guided Kaggle connect flow)

### Changed
- **Connecting Kaggle is a form, not a paragraph** — a deliberate divergence from the ExDark plugin
  (CONTEXT.md, decided 2026-10-08; `docs/PREDICT_MIRROR.md` §13 D13). The rc9 hand test found the
  "Kaggle account not connected" card far too detailed and wrong on a redirected-home host: it told the
  participant to write the token under `$env:USERPROFILE` / `~`, while the compute service — every
  `start_tester` environment, the hub-rc9-clean environment — runs with its home redirected, so the token
  landed where the service never looks. The not-connected state now shows "Kaggle isn't connected yet",
  one line on where the token comes from (kaggle.com › Settings › API › Create New Token), a masked token
  field and **Connect**. `POST /kaggle/connect` trims the paste, checks the `KGAT_` shape, writes the token
  to the exact file its Kaggle client reads (`os.path.expanduser("~/.kaggle/access_token")`, resolved on
  the service, so a redirected home is honoured by construction) as plain ASCII, no BOM, no trailing
  newline, user-only permissions on macOS / Linux, verifies it through the client's own authenticate call
  and answers **Connected to Kaggle as <username>**. One plain sentence on failure (not a token / Kaggle
  rejected it / no network / the service's `KAGGLE_API_TOKEN` would override it), the field kept; a token
  Kaggle rejects is removed again. The card sits outside Step 2's muted block, so Kaggle can be connected
  before anything is predicted.
- The token is never logged, never echoed in a response, a reason or a record, masked in the field and
  dropped from the browser after the connect; the Doctor and the connection card carry its PATH only.
- Under a collapsed **Other ways to connect**: the compute service's OS-specific one-line command with
  the resolved path filled in, the `KAGGLE_API_TOKEN` variable, the legacy `kaggle.json` — one line each,
  with Copy.
- The results panel always shows the saved `submission_<stamp>.csv`'s full path (never truncated) with
  the note that it can be uploaded by hand on Kaggle's Submit page.
- Kaggle's connection state is shown at the top of Predict + Submit before anything is predicted
  ("Kaggle: not connected · Connect in Step 2" / "connected as <user> · N of M left today"); the Doctor's
  Kaggle row points at Step 2 and copies the token's path; the Status tab's Kaggle live block and a
  skipped submit say where to connect instead of repeating instructions.
- `TESTING.md` §7.5b tests the Connect button (optional, with a real token; still no submit).

### Added
- `tests/test_kaggle_connect.py`: the write path under a redirected home (the kagglesdk path identity,
  the bytes, the POSIX modes), the "Other ways" lines for win32 / linux / darwin, the format validation
  (what a paste carries, what is refused), the connect outcomes (ok · rejected removes the file ·
  unreachable keeps it · bad format and env override write nothing · errors scrubbed), the route, and the
  no-token-leak scan over every served surface (the card, the Doctor with and without Kaggle, the Status
  tab's live block, the session store, `GET /config`, the verification bundle).

## [1.0.0rc9] — 2026-10-07 (session 7: the tester release — rc8 plus documentation, no code change)

### Changed
- `TESTING.md` tells testers to run the services on the Hub's default ports (object 5015, compute
  5020) and why: on non-default ports the Hub's project page does not show its Queue & Progress card
  while a job runs (a Hub frontend bug — the card polls the compute service on the default port and
  ignores the URL saved on Getting Started; the global Queue page and every plugin tab still work).
  Diagnosed live on 2026-10-07 by polling the host's job list during a run and comparing the record
  with a stock plugin's: identical shape, the right `project_name`, listed under `?project=` on every
  poll; the card appeared as soon as the same services ran on 5015 / 5020.
- `tester/start_tester.ps1` / `.sh` default to the 1.0.0rc9 test catalog.

No change under `src/`: the plugin is rc8's code with its version stamped.

## [1.0.0rc8] — 2026-10-06 (session 6: the Hub's job card during the collection pass)

### Added
- The Hub's Queue & Progress card moves during the final per-sample pass: "Collecting metrics
  3,900/7,800" with the pass's percent, Elapsed and ETA after every batch (1 s throttle), a pulsing bar
  "Reducing embeddings (UMAP)…" during the reducer, "Writing metrics tables…" during the table writes —
  the shape the yolo (collect mode) and sam3 plugins send.
- End-of-job cards on the Queue card, as the sam3 / image-metrics / importer plugins send them: rows
  collected, metrics tables written, best val accuracy "NN.NN % (epoch N)". No per-epoch training metric
  rides the generic card (the SDK guide's rule; TRAIN_MIRROR §15 S6-8).
- A cancel request during the collection pass now stops the pass (checked before every batch, before the
  reducer and before the table writes); the run ends cancelled with its best checkpoint and no metrics
  table.

### Fixed
- Nothing in the Export checklist: a test now pins that two verified rule picks give two checkpoints in
  both the preview and the export (the rc7 hand test's "two rule tags · 1 checkpoint" is the state when
  one pick fails verification or was unchecked).

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
