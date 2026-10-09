# STATUS_MIRROR.md — the Status tab mirrors 3lc-compute-plugin-kaggle v1.2.15

Session 5, Phase 0 (2026-10-05): study, then the implementation in the same session. The governing
rule is `docs/EXDARK_MIRROR.md`: the Status tab mirrors ExDark's Status tab **control for control, in
functionality and looks**. Allowed differences: (1) competition identity, (2) dataset, (3) competition
rules (accuracy, 100 submissions/day, the Kaggle public score), (4) task-inherent. Where the session-5
brief asks for something ExDark does not have (run history, the Doctor panel, the verification bundle,
the public-score delta, the unlaunched-leaderboard copy), the element is marked **+brief** and the
choice taken is recorded under **Decided in session 5 (review later)** — the brief said not to stop
for decisions, so every one of them was taken as the option closest to ExDark.

References (read-only): `../3lc-compute-plugin-kaggle/src/tlc_plugin_kaggle/ui.html` (Status CSS
448–480, markup 1190–1215, JS 6186–6549, fixtures 4238–4340), `routes.py` (`/kaggle/status` 373,
`/pipeline` 492), `predictor.py` (`kaggle_live_status` 705–763), `docs/ui-notes.md` §11 and the Status
fixture map (751–759). Ours: `trainer.py` (`train_state`, `run_summary`), `predictor.py`
(`predict_submit_state`, the ledger entries), `ledger.py`, `kaggle_client.py` (`get_submission`,
`list_submissions`, `connection`), `storage.py`, `kit.py` (`download_state`, `verify_now`),
`manifest.py` (`resolve`, `cache_meta`).

Legend: **V** ported verbatim · **A(n)** adapted under allowed difference n (how) · **X** dropped (why) ·
**+brief** required by the session-5 brief, no ExDark equivalent · **D** decided here (§3).

## 1. Status tab, in render order

| # | ExDark element (ui.html line) | Ours | Mark |
|---|---|---|---|
| 1 | Tab 4 in the stepper: "4 Status" · "History & leaderboard" · glyph `#kg-state-status` always empty (Status is not a pipeline step; `renderPipeline` marks import / train / submit only) (633) | already in the shell (EXDARK_MIRROR #5, fix b); the glyph stays empty | V |
| 2 | `.card-header`: `plugin-section-number` 4 · "Status"; subtitle "Your best score, latest activity, and submission history." (1194) | same; subtitle verbatim (the gate-era "Your runs, submissions and leaderboard standing." is replaced by ExDark's) | V |
| 3 | `#st-conn-banner` (1199): the connection guard's slot | verbatim (`conn` already owns a slot per tab) | V |
| 4 | Gate: ExDark has none on Status (its history is empty-state driven) | the ExDark Train gate card ("Tables not found." + Go to Import) that gated this tab since session 2.5 stays until an import record exists; once imported the body renders. Status is scoped to the import record's project like Train and Predict (D8) | A(plumbing) · D8 |
| 5 | Hero strip `#st-hero` (`stRenderHero` 6238): `stHeroBlock` × **Best local mAP@0.5** (big number or "no scores yet"; note = the run, or "Local scoring runs on the host machine only.") · **Latest activity** (`kgWhenSpan` + status badge; note "run · outcome") · **Kaggle budget** (only when the connection is ready: "2 of 3 today / Resets 00:00 UTC." or "3/day limit / Per-day counter unavailable on this competition.") · **Live now** (one running job: Training "Epoch n/N" / Predicting "Inference: n / N images" / Submitting + "Open tab") | same four blocks and renderer. The first block is **Best public score** (Kaggle's score of the best submission, "no scores yet" until one is scored; note = the run) — the local metric ExDark shows is host-only and dropped (PREDICT D1/D6), so the leaderboard number is the one worth leading with (D1). Budget from `psConnState` as ExDark (ours counts through GetSubmissionLimits). Live now reads the host's job list (`PluginJobs.list`) instead of five `GET /jobs?kind=` calls | A(3) · D1 |
| 6 | `.kg-updated-row`: `#st-updated` "Updated 12 seconds ago" (`kgFmtAgo`, 5 s ticker) + `#st-refresh-btn` ghost refresh (spins while busy) (1201) | verbatim | V |
| 7 | — | **Runs** section (`.kg-sec-head` "Runs", `#st-runs`): the current project's runs newest first in ExDark's `.kg-st-table` geometry — Run (Dashboard link when available) · Revision trained on (`kgTableRef`) · Labeled rows · Best val accuracy at epoch · Device · Elapsed · Status (badge; "interrupted" for `stale`, "cancelled") · Actions (Open in Dashboard, Open in Projects on the Hub origin). Fed by `GET /train/state` `project_runs` (+ the running current record). Empty state "No runs yet." + Go to Train | +brief · D2 |
| 8 | `.kg-sec-head` "History" · `#st-history` (`renderHistoryFrom` 6298): empty state `kg-callout info` "No predictions yet. Train a model and predict on the test set to see history here." + **Go to Train**; else `.kg-st-table` Run (Dashboard link to the originating Run) · When · Local mAP@0.5 · Δ (vs the previous SCORED row, ▲ +0.0031 / ▼ −0.0120 / – 0.0000, "–" on the first scored row) · Outcome (`stOutcome` vocabulary: "Submitted · #ref", "CSV generated (daily limit reached)", "CSV generated (not joined on Kaggle)", "Submission rejected", "CSV generated (not submitted)", "Validation failed", "Failed", "Interrupted", "Running…") · Actions (Copy CSV path, Download CSV) | same section, renderer and vocabulary; rows are the **predictions** (the ledger's `predict` entries, newest first, project-scoped), each with its paired submission. Columns: Run · When · **Val accuracy** (the predict entry's val check, A(3)) · **Public score** (Kaggle's, D3) · **Δ** (the public score vs the previous SUBMISSION that Kaggle scored — the brief's "score change vs the previous submission") · Outcome · Actions. Outcome adds "Submitted · #ref · scored 0.6567" / "Submitted · #ref · rejected by Kaggle" from the read-back (PREDICT D12). Download CSV goes through `GET /submissions/{job}/download` (the ledger resolves older jobs) | A(3) · +brief · D3 |
| 9 | `hr.divider` · `.kg-sec-head` "Kaggle live" · `#st-kaggle` (`renderKaggleFrom` 6392, fed by `GET /kaggle/status?slug=`): not connected → `kg-callout info` with the server reason; "Best public score so far: **x**"; "Leaderboard: rank **n** (score)" when the team name equals the username; a submissions table When · Status · Public score · Message (last 10) or "No Kaggle submissions yet."; on API errors the friendly degradation callout "Kaggle’s submission list isn’t available for this private competition (a known Kaggle API limitation). Your submissions still score normally on the web leaderboard." + **View leaderboard on Kaggle** + a "Show details" accordion with the raw errors | same section and renderer, fed by `GET /status/kaggle` (no slug parameter: the slug is locked from the manifest, PREDICT D8). While the competition is unlaunched Kaggle answers 403 to ListSubmissions and an EMPTY list to the leaderboard view (probed read-only 2026-10-05: leaderboard `[]`, submissions 403, GetSubmission by ref answers), so the degradation callout is the normal state until launch and its copy says so: "The leaderboard and Kaggle’s submission list become available after the competition launches. Your submissions still score normally on the web leaderboard." + View leaderboard on Kaggle + Show details (D4). The submissions table renders when ListSubmissions answers (after launch); until then the History section above is the submission history (every ref with its read-back score) | A(1,3) · D4 |
| 10 | — | **Verification** section (`.kg-sec-head` "Verification"): **Export verification bundle** (`btn-secondary`, download icon) + help "A zip of the records an organizer verifies a leaderboard entry against: the import record, the train revision chain, every run’s provenance and checkpoint sha256, the ledger, the manifest provenance and the plugin version — plus this project’s tables and runs (their 3LC records and per-sample metrics) and the best checkpoint of up to two submitted runs. Never images, prediction CSVs, tokens or answer keys." (session 6, D16: the old copy said "never table data"). Under the button row the **checkpoint checklist** `#st-bundle-runs` (the project's submitted runs, the rule's two picks checked, at most two, each with its public score · submitted when · #ref · size) and the size line "Bundle: N members · about X MB (…)" from `GET /status/bundle/preview`, re-planned on tab enter / Refresh and on every checkbox change. Streams `GET /status/bundle?checkpoints=…` the way Download CSV does; the note then reads "Exported <name> (X MB)" | +brief · D5 · D16 |
| 11 | — | **Doctor** disclosure (`kgl-acc-summary` "Doctor", collapsed; opens instantly when a check is red): ExDark's locked-row geometry, one row per fact from `GET /status/doctor` — plugin version · commit · compute service version (from the host's `/health`) · SDK version · 3lc version · torch / torchvision in the worker · CUDA available in the worker · device class · manifest source + sha256 · kit version + verification state · Kaggle connection (state + username, never the token) · plugin home + resolved_by · free disk space · Python; plus **Copy diagnostics** (the ```` ```3lc-kaggle-diagnostics ```` fence with a `[doctor]` section, `kgBuildDiagnosticsCore`) | +brief · D6 |
| 12 | `stFetchAll` (6454): one pass for hero + history (`/runs` + four `/jobs?kind=`), `renderKaggleLive` separately so the slow API round-trip never holds the local data back; `stBusy` guard; the refresh button spins | same shape: `GET /train/state` + `GET /status/history` + `PluginJobs.list` for the hero and the two tables, `GET /status/kaggle` separately; the Doctor loads on tab enter and on Refresh | A(plumbing) |
| 13 | `refreshStatus` = `renderPipeline()` + `stFetchAll(true)` on the button; auto-refresh every 15 s while the tab is visible (ticks skip while hidden), a 5 s ticker for "Updated …", `visibilitychange` refreshes a visible Status tab at once; `stStopAuto` when another tab shows; `stOnTabEnter` gated on `configLoaded` (6489–6530) | verbatim | V |
| 14 | `?kgdev` Status fixtures `stDevForce` (4238): `status-empty`, `status-history`, `status-live-running`, `status-kaggle-live`, `status-kaggle-403` — static records, `RUNSBYJOB`, `LIVE_TRAIN`, `KAGGLE_LIVE`, `KAGGLE_403`; `psConnState` set per fixture; `stStopAuto` so fixtures own the tab | ported with our names, numbers derived from the served manifest, plus the Doctor payload in every state and a run list; `status-kaggle-403` renders the unlaunched copy (D4) | A · D7 |
| 15 | Version footer `data-kg-footer` (1213) | verbatim (already rendered by `kgRenderFooters`) | V |
| 16 | `GET /pipeline` (routes 492): submit done = a `kaggle_submit` job with `facts.submission.status == "submitted"` | already ours (`renderPipeline` reads the submit record); the Status tab's Refresh re-runs it like ExDark | V |
| 17 | Predict's success banner **Continue to Status** → `showTab('status')` (PREDICT #35) | lands on the ungated tab now; `kgApplyTab` calls `stOnTabEnter` for `status` and `stStopAuto` otherwise | V |

## 2. Backend (plumbing, ours)

- **`status.py`** (new, import-light; tlc / the kaggle client inside functions):
  - `run_history()` — `trainer.train_state()`'s `project_runs` plus the running current record, newest
    first, each row carrying what #7 shows (name, folder, URL, revision trained on, usable rows, best val
    accuracy + epoch, device label, elapsed, status, project).
  - `prediction_history(manifest, *, live)` — the ledger's `predict` entries of the current project, newest
    first, each joined to its `submit` entry (by `predict_job_id`) and to the predict/submit records when
    they are the latest ones (status, error); the Kaggle verdict is the submit entry's stored read-back when
    it is final (`COMPLETE` / `ERROR`), else — with `live` — one `GetSubmission(ref)` per unresolved ref,
    cached per worker for `LIVE_CACHE_S` (D3); `delta` is the public score minus the previous scored
    submission's (chronological order). Never rewrites the ledger.
  - `kaggle_live(manifest)` — ExDark's `kaggle_live_status` with every call fenced: `connection` state,
    `list_submissions` (403 while unlaunched → `submissions_error`), the leaderboard view (`leaderboard_top`
    5 + `my_rank`; an empty board while unlaunched, an error → `leaderboard_error`), `best_public_score`
    from the ledger's read-backs when the list is unavailable, `launched: false` while the submissions list
    is refused and the board is empty (the unlaunched copy).
  - `doctor(manifest)` — the facts of #11; torch-free on the route (versions from `importlib.metadata`,
    CUDA / device class from the worker's device probe, free disk from `shutil.disk_usage`); the commit from
    the installed dist's `direct_url.json` (`vcs_info.commit_id`), else `git rev-parse HEAD` of a folder
    source, else `unknown`; the compute version is the host's and the fragment reads it from `/health`.
  - `bundle_plan(manifest, checkpoints=, runs=)` → `verification_bundle_file` / `verification_bundle` /
    `bundle_preview` (session 6, D16) — the zip of D5 **plus the project's data records**: `README.txt`
    (the member list, the checkpoint rule, the machine-specific-paths section, the GitHub note),
    `plugin.json`, `manifest_provenance.json`, `import_record.json`, `train_revisions.json`
    (`importer.list_project_tables` over the import record's project with the seed lineage),
    `runs/<job id>.json` (the run summary, the provenance checks, the contract, the Run's parameters on
    disk, the checkpoint sha256 recorded and on disk now), `predictions/<job id>.json` and
    `submissions/<job id>.json` (the ledger entries, grouped), `ledger.jsonl` (verbatim), and
    `project/<project>/…` in the project's own layout: `datasets/<ds>/tables/<revision>/object.3lc.json`
    (+ `row_cache.parquet` where one exists) for every train revision in the seed lineage (`bundle_tables`:
    the listing's `in_lineage` rows ∪ the revisions the project's runs trained on with their
    `input_table_url` ancestry ∪ the seed) and the locked val table; `runs/<run>/object.3lc.json` +
    `metrics_*/object.3lc.json` + `metrics_*/*.parquet` for every run of the project; `runs/<run>/model/best.pt`
    per the checkpoint rule; `files.json` (every copied file with sha256, bytes, kind, source path; the
    checkpoint decision with `included` / `skipped` and reasons; the tables with roles; the source project
    dir). Never an image, `last.pt`, a CSV. The zip is written under `<plugin home>/bundles/` (zips older
    than an hour pruned) and streamed as a file. Scans: every text member, every parquet member's string
    columns (`parquet_text`, pyarrow) and `files.json` against the secret patterns before any byte is zipped
    (a match aborts with the member's name, never a partial zip); checkpoints are verified by sha256
    (`_verify_checkpoint`) instead.
  - **The checkpoint rule** (`eligible_checkpoint_runs`, `default_checkpoint_runs`): eligible = the project's
    runs with a `submit` entry carrying a Kaggle ref, newest submission first, each with its best public
    score, latest ref and time, the ledger's checkpoint sha256 and the run record's best.pt. Default = the run
    behind the best public score + the run behind the most recent submission (one when they coincide; the
    most recent alone when nothing is scored; none when nothing was submitted). `?checkpoints=selected&runs=`
    = the participant's checklist (at most `CHECKPOINT_MAX` = 2 eligible ids, else 400). `none` / `all` are
    for organizers (`all` = every run's best.pt that matches its record and its Run; a submitted run must
    also match the ledger). A best.pt is included only when its sha256 equals every recorded value available
    (the ledger's for a submitted run, the run record's, the Run's parameters); otherwise it is skipped and
    named in `files.json` and in the checklist's row. Duplicates by sha256 ride once (`same_as`).
- **Routes**: `GET /status/history?live=1`, `GET /status/kaggle`, `GET /status/doctor`,
  `GET /status/bundle?checkpoints=default|selected|none|all&runs=<ids>` (a zip download with
  `Content-Disposition`, like the CSV route; 400 with `{error}` on a refused export or a bad selection),
  `GET /status/bundle/preview?checkpoints=&runs=` (the plan: members, bytes, counts, `eligible_runs`,
  `checkpoints.default_runs`; `{error}` on a bad selection).
- **Tests** (`tests/test_status.py`): the history join + delta + outcome fields on a synthetic ledger; the live
  read-back path through a fake client (one call per unresolved ref, cached); `kaggle_live` under 403s
  (`launched: false`, best score from the ledger); the doctor payload's keys; the bundle's file list and the
  data rule (parquet only under `project/…/tables/` and `…/metrics_*/`, the only `.pt` a run's `best.pt`,
  never an image / CSV / `last.pt`), no secret pattern (`KGAT_`, `kaggle.json` contents, `3lc_api_key`,
  `mapping.csv`, `solution`), a planted secret in a text member or in a table's row cache aborting it; on a
  fake project tree on disk: the lineage tables copied and a stray root table not, every run's object and
  metrics, the checkpoint rule (best + most recent, the ledger mismatch skipped and named), `files.json`
  with real sha256s, the preview's checklist, the `selected` / `none` / `all` modes and their refusals, the
  bundles dir and its pruning; the fragment needles (`test_routes.py::test_fragment_is_the_exdark_status_tab`).

## 3. Decided in session 5 (review later)

| # | Question | ExDark | Decided (closest to ExDark) |
|---|---|---|---|
| D1 | The hero's first block | Best local mAP@0.5 (host-only local scoring) | **Best public score** from the submissions Kaggle scored; "no scores yet" until one is. Our local metric (val accuracy) stays in the History column, not the hero: it is the split participants tune on, so it would mislead as "best" |
| D2 | A run history (+brief) | none (runs appear only as the History rows' origins) | a **Runs** section above History in the same table geometry, project-scoped, with the columns the brief lists; links follow the Train tab's rules (Dashboard always, Projects on the Hub origin) |
| D3 | The public score and its Δ (+brief) | Δ on the local score, between scored predictions; no public score locally | rows stay predictions (ExDark); **Public score** from the stored read-back, refreshed live by `GetSubmission(ref)` for refs Kaggle had not scored at submit time (ListSubmissions 403s while unlaunched); **Δ** on the public score vs the previous Kaggle-scored submission, ExDark's ▲ / ▼ / – rendering at 5 decimals (Kaggle reports 5) |
| D4 | The leaderboard while unlaunched | the 403 degradation callout (private competition) | the same callout with the copy "available after the competition launches", View leaderboard on Kaggle, Show details with the raw errors; shown while ListSubmissions is refused and the board is empty (what the unlaunched event competition answers); the live table and rank render by themselves once the API answers (no code change at launch) |
| D5 | Export verification bundle (+brief) | none (ExDark has no ledger) | a zip streamed by `GET /status/bundle`, named `verification-bundle_<competition id>_<UTC stamp>.zip`, the member list of §2; a secret-pattern scan aborts the export. **Superseded in part by D16** (session 6): the bundle now also carries the project's tables and runs |
| D6 | The Doctor panel (+brief) | the Copy diagnostics fence only | a collapsed **Doctor** disclosure at the bottom of the tab (locked-row geometry) with Copy diagnostics; the compute version read from the host's `/health` by the fragment (the worker cannot know it) |
| D7 | Fixtures | five `status-*` states | the same five names; each also renders the Runs section and the Doctor from fixture data; `status-kaggle-403` is the unlaunched state |
| D8 | Scope | one project (ExDark's `DEFAULT_PROJECT`) | the import record's project, as Train and Predict since `ee85dee`; the ledger entries are attributed by `trainer.run_project` (recorded project, else the URL) |
| D9 | Where submissions without a prediction row would show | n/a | not possible: every submit entry references a predict entry; an orphan is listed under its own row with the run name from the submit entry |
| D10 | Auto-refresh and the live read-back | 15 s poll | the 15 s poll passes `live=0`; only the manual Refresh (and the tab enter) passes `live=1`, so the auto-refresh never calls Kaggle |
| D11 (rc3) | The Doctor going stale after a job (the hand test of rc2: old counts and "100 of 100" while the hero said otherwise) | the Doctor has no ExDark equivalent | the Doctor reloads on tab enter and Refresh (with Kaggle), on every job completion (`kgOnRecord` terminal → `stOnJobDone`), and on the 15 s poll with `?kaggle=0` (counts only; the last Kaggle block is kept); the hero's budget block falls back to the Doctor's Kaggle state when Predict has not loaded the connection card |
| D12 (rc3) | One budget wording | "2 of 3 today" (hero) vs "2 of 3 submissions left today" (card) | **"N of M left today"** on the hero, the connection card and the Doctor |
| D13 (rc3) | The manifest row read "cache" after a fresh start | — | the no-network resolution IS cache by design; the Doctor now kicks the background refresh and reports its result (`remote` when the fetch succeeded, the local label plus "remote fetch failed: …" when not) with the fetched-at stamp; `manifest_source_local` keeps the raw label. The worker's fetch from `https://competitions.dev.3lc.ai` succeeds (the proof's refresh `done / remote`); only the label was stale |
| D14 (rc3) | The Loop's inspect link on first load | ExDark re-targets it only when the Predict tab loads its runs | `kgRenderLoopLinks` binds inspect to the current project's latest Run (the running record first, then `project_runs[0]`), re-rendered after config, after an import / a run and on Predict's run list; with no run it is **disabled** (`kg-loop-off`, `aria-disabled`, the tooltip "No run in this project yet. Train first; inspect then opens the latest run in the Dashboard.") instead of opening the Dashboard root |
| D16 (rc7, session 6) | **The bundle carries the tables and runs** — Paul's demo feedback item 3 reversed the session-5 "never table data" rule (still never images, tokens or answer keys) | none | §2: `project/<project>/` with the seed-lineage train revisions and the locked val (object + row cache), every run's object and metrics tables, `files.json` with sha256s, the README's machine-specific-paths section (the row caches' image column and the runs' `train_table_url` / `val_table_url` are the exporting machine's absolute paths; the chain is sha256 / string equality, so they are evidence, not instructions). Checkpoints by **Rishikesh's rule** (2026-10-06, replacing the Phase 0 proposal "best.pt of every submitted run"): best.pt of at most two runs — the best public score and the most recent submission — verified against the ledger's sha256 and deduplicated; a checklist in the Export section lets the participant pick other submitted runs (max 2) with the resulting size shown before the export; `?checkpoints=none|all` stay route-only for organizers. Sizes measured on `intel-scene-demo` 2026-10-06: tables 0.26 MB, a run without checkpoints ~0.26 MB, a resnet18 best.pt 45.44 MB (under GitHub's 100 MB per-file limit; the zip with two checkpoints is not, so the README says to unzip before committing) |
| D17 (rc7, session 6) | Where the bundle is built | in memory (`Response[bytes]`) | written under `<plugin home>/bundles/` and streamed as a `File` (an organizer's `all` export of a project with many runs would not fit a response buffer); zips older than an hour are pruned on the next export |
| D15 (rc3) | Version-scoped plugin state (the tester blocker) | ExDark's state is version-independent — `~/.3lc-kaggle-plugin/ui_config.json` (`config_store.py:51`, `Path.home()`) — which is also its redirected-home bug: it does NOT use the host's versioned layout at all | `storage.py` rule 3 "compute-home": inside the host's managed layout the state lives in `<compute home>/plugin-state/<id>`, outside the version dir; the first resolution after an update carries the newest previous version's `.plugin-state` forward (copy, the stored paths rewritten, the old copy kept, `migrated_from.json`); the Doctor's plugin-home row says so. Tables imported before rc3 keep their image paths into the old version's kit dir, which the host keeps for the next two updates (`gc_old_versions` keep=3); the Train gate accepts those rows by the kit layout (rc4); the carry-forward marker records that kit tree and the plugin re-creates it from its own copy when the host's GC removes it (rc5; rc6 derives the tree from an older marker's `source`) |

## rc11 (2026-10-09) — back-port to ExDark after the event

- The **Kaggle live** heading reads "Kaggle (competition not launched yet)" while `GET /status/kaggle`
  answers `launched: false` (`renderKaggleFrom` sets `#st-kaggle-head`).
- The Doctor's **Manifest** row appends the fallback notice when `manifest.fallback.active` ("Using the
  built-in competition manifest; the server couldn't be reached"); its **Kit** row reads
  "v1 · 9,600 images + manifest · verified <date>" (+ "folder supplied by hand" for a manual kit).
- The Doctor's Kaggle block refreshes right after a successful Connect (PREDICT_MIRROR §14 D17).
