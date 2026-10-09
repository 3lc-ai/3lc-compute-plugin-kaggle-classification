# EXDARK_MIRROR.md — the plugin mirrors 3lc-compute-plugin-kaggle v1.2.15

Governing rule (2026-09-28): this plugin **mirrors the ExDark plugin in functionality and looks**.
Its fragment and interaction code are ported as close to verbatim as possible, including the
`innerHTML` + `esc()` pattern. The only allowed differences are:

1. **Competition identity** — name, slug, competition id, links, deadline: all from the manifest.
2. **Dataset** — 6 classes from the manifest, the 6,000-image unlabeled pool (label `undefined`,
   weight 0), a labeled val table, no test table, the kit layout and counts.
3. **Competition rules** — metric accuracy, 100 submissions per day, resnet18 from scratch at
   150 px, the submission CSV schema.
4. **Task-inherent** — no Ultralytics or YOLO (no license band, no format banner, no label-format
   specifics); the trainer is timm resnet18, the Train tab mirrors ExDark control for control.

Anything else that would differ is **not designed here**: it is listed under "needs decision"
with ExDark's version and the reason it cannot be copied. Invisible plumbing stays ours: remote
manifest resolution, sha256 verification, deletion safety (the plugin never deletes or overwrites
tables or revisions, except rolling back what a failed job itself created), provenance records.

Scope of this pass: the shell (header, chips, Loop, tab bar, footer, connection guard) and the
Import tab. Train, Predict + Submit and Status keep ExDark's gate card until their sessions.

Reference: `../3lc-compute-plugin-kaggle/src/tlc_plugin_kaggle/ui.html` (CSS 3–568, markup
569–1215, JS 1216–6549) and its `docs/ui-notes.md`. Legend: **V** ported verbatim · **A(n)**
adapted under allowed difference n (how) · **D** needs decision.

## 1. Shell, in render order

| # | ExDark element (line) | Ours | Mark |
|---|---|---|---|
| 1 | `.kg-license-note` Ultralytics band (571) | dropped | A(4) |
| 2 | `.plugin-hero` › `.kg-id-row`: flag icon, `h1.kg-id-title` "Kaggle Competition", `p.kg-id-sub` "Import the dataset, train the fixed baseline, predict and submit. The whole competition loop without leaving the Hub." (586) | same row; icon = the plugin's own `icon_svg` (four tiles); title "Kaggle Classification" (plugin name); sub verbatim | A(1) |
| 3 | `.kg-constraints` micro-label "Competition constraints" + three `.kg-chip`s with `title`s: lock "YOLOv11n · COCO-pretrained · <imgsz>px", "12 classes", "Scored by mAP@0.5" (595) | micro-label verbatim; lock chip "resnet18 · from scratch · 150 px" (`model.*`), "6 classes · 6,000 unlabeled" (`classes`, `splits.train.undefined`), "Scored by accuracy" (`submission.metric`); tooltips reworded for the classification contract | A(2,3) |
| 4 | `.kg-loop`: "The Loop" · import › train › inspect ↗ › fix labels ↗ › retrain › submit; `data-goto` steps switch tabs, inspect/fix labels deep-link the Dashboard with `?object_service=` (605) | verbatim wording and mechanics; inspect → Dashboard root (the newest Run once Train exists), fix labels → the latest train revision (`import_state.latest.train`) once imported, Dashboard root until then | V |
| 5 | `.kg-tabs` 1 Import "Starter kit → 3LC tables" · 2 Train "YOLOv11n, pinned init" · 3 Predict + Submit "Predict → CSV → Kaggle" · 4 Status "History & leaderboard"; `.kg-tab-state` glyphs; keyboard Enter/Space; `localStorage` tab memory (622) | verbatim; Train sub "resnet18, from scratch" | A(3) |
| 6 | `renderPipeline()` glyphs from `GET /pipeline`: check-circle done, dot on the FIRST not-done step, circle pending (2255) | fed from `_meta.import_state` (import done) — Train/Submit stay pending until their sessions. **Deviation for your bug 2:** the dot marks the SELECTED tab when it is not done; ExDark puts it on the next pipeline step regardless of selection (what you saw after importing) | D |
| 7 | Version footer per tab: "3LC Kaggle Competition plugin v1.2.15 · GitHub · Docs" (2139) | "3LC Kaggle Classification plugin v0.1.0 · GitHub · Docs" | A(1) |
| 8 | Connection guard `kgConn`: warn banner in every finished tab's `*-conn-banner` slot, 2/5/10/15 s pings of `/import/state`, "Reconnected." hold + fade, resume callbacks (1601) | verbatim; pings our `/import/state` | V |
| 9 | Document-title progress nudge `kgSetTitleProgress` (1671) | verbatim | V |
| 10 | Motion tokens, icon system, class tint, `esc`, `fmtCount`, `midTrunc`, `kgBindCopy`, accordions, diagnostics core (`kgBuildDiagnosticsCore`, fence ```` ```3lc-kaggle-diagnostics ````) (1420–1795) | verbatim (the plugin line reads `3lc-compute-plugin-kaggle-classification`) | V |
| 11 | `?kgdev=<state>` fixtures for every state (3522–4339) | not ported in this pass (dev affordance, large) | D |
| 12 | Train / Predict tab gate: amber `.kg-callout.warn` "**Tables not found.** … Looking in project `x` — no import has produced these tables there yet. If you just changed the Project name, Import (tab 1) needs one run for that project." + **Go to Import** (4858) | the same callout is the gate card on Train, Predict + Submit and Status until an import record exists | V |

## 2. Import tab, in render order

| # | ExDark element (line) | Ours | Mark |
|---|---|---|---|
| 13 | `.card-header`: `plugin-section-number` 1 · "Import"; subtitle "Creates `exdark_train` / `exdark_val` / `exdark_test` tables from the starter kit and validates them against the competition dataset." (641) | subtitle "Creates `intel-scene_train` / `intel-scene_val` tables from the starter kit and validates them against the competition manifest." (dataset names from `manifest.dataset_name`) | A(1,2) |
| 14 | `.format-selected-banner` "YOLO format · Object Detection — Fixed for this competition…" (650) | dropped | A(4) |
| 15 | `#kg-conn-banner` slot (659) | verbatim | V |
| 16 | Starter-kit section `#kg-dl-section`: `.kg-sec-head`, `#kg-dl-banner`, offer with TWO toggled blurbs (download / top-up), `#kg-dl-dest` inside the download blurb, `#kg-dl-btn` + `#kg-dl-state`, `#kg-dl-progress`, "Show log" accordion, `hr.divider` (665) | verbatim structure and ids; download blurb "…the competition images from the 3LC content network…" (no labels); the top-up blurb kept in the markup (toggled, never rewritten, the node-lifetime test) but see #27 | V / A(4) |
| 17 | `#kg-import-banner` (success / failure banner slot) (708) | verbatim | V |
| 18 | Form left column: **Dataset YAML path** (required, `guide-next` pulse, hint "Point this at the starter kit's dataset.yaml (or the folder containing it)…", `#kg-preflight` slot) · **Project name** · **Table name** ("Revision name for the imported tables. `initial` is the convention for a first import.") (712) | **Starter kit folder** (required; the folder holding `files.json`, typically `…\starter_kit`), same hint shape, same preflight slot; Project name and Table name verbatim | A(2) |
| 19 | Form right column: **Splits to import** placeholder → locked Train · Val · Test rows with lock glyphs, "All three splits are mandatory for the competition." · `#kg-glance` "Dataset at a glance" (counts per split, class tags, GT-guard note) (735) | locked **Train · Val** ("Both splits are mandatory for the competition."); glance: train labeled / unlabeled / val counts + class tags; no GT-guard note (no test table) | A(2) |
| 20 | Preflight (`kgRunPreflight`, `kgRenderPreflight`): debounced GET `/import/preflight?yaml_path=`; states idle / checking / error / ok ("Detected: train / val / test · 12 classes (canonical order) · N / N / N images. Matches the competition dataset.") / mismatch (per-problem rows with remedies + Copy diagnostics) (2281) | same states and rendering against `/import/preflight?kit_dir=`; ok line "Detected: train / val · 6 classes (manifest order) · 6,600 / 1,200 images. Matches the competition manifest."; mismatch rows from the kit-vs-manifest checks | A(2) |
| 21 | `hr.divider` · **Import & Validate** `btn-primary btn-lg` (upload icon; refresh-cw + "Re-run Import & Validate" after a failure) · spinner · `#kg-import-state` live text (745) | verbatim | V |
| 22 | State 3 progress rows `kgRenderProgress`: `.kg-job-row` per stage (Import train / Import val / Import test / Validate (9 checks)) with `badge badge-status-queued|running|completed`, elapsed, indeterminate bar under the current row (2705) | stages Import train / Import val / Validate (18 checks); built from our job channel (stage_progress + checks) instead of a polled job record — same rows, same badges | A(2) |
| 23 | `renderChecks`: verdict line, two groups "Dataset structure" (first 5) / "Competition integrity" (rest), compress-on-pass, remedy on fail (`KG_REMEDIES`) (2636) | verbatim renderer; groups: "Dataset structure" = the kit-vs-manifest checks (11), "Competition integrity" = the table checks (6); remedies rewritten for our check labels | A(2) |
| 24 | State 4 `renderResult`: `alert alert-success` "Imported & validated: exdark_train · exdark_val · exdark_test" + **Continue to Train** + **Explore train/val/test** + (revisit) **Start over**; summary rows split · rows · CREATED/REUSED badge · truncated path · Copy · Explore · Re-import slot (2751) | verbatim with two rows (train, val); "Imported & validated: intel-scene_train · intel-scene_val"; Continue to Train switches the tab | V |
| 25 | **Collision handling**: `from_yolo_url(if_exists="reuse")` — an existing table at the target URL is REUSED (badge REUSED, tooltip "An identical table already existed and was reused. Validation still ran on it.") and the checks run on it (2751, importer.py) | mirrored: `importer.run_import` reuses a table that already exists at the target URL and runs the post-write checks on it; the session-2 refusal + `-2` naming is retired from the UI (the `mode=reimport` path stays in the backend for #26) | V |
| 26 | **Re-import fresh** per REUSED row (`kgReimportFresh`): confirm dialog with the revision count, then `force_splits` → `if_exists="overwrite"` (2841) | **not rendered**: ExDark overwrites the table (and its revisions), which deletion safety forbids. Options for you: (a) omit the action; (b) render the same button but write a fresh `<table>-2` beside the old one (our session-2 backend, already tested); (c) copy ExDark verbatim and allow the overwrite for tables the plugin itself created. The slot stays (`.kg-reimport-slot`) so rows align as in ExDark | D |
| 27 | Superseded kit → **top-up** offer (`dlRenderSuperseded`, `mode: 'top_up'`, in-place update of changed files) (3237) | our download stage has no top-up: a newer kit downloads BESIDE the old one. The section renders ExDark's info callout with the version facts, and the offer button reads "Download starter kit" (the top-up blurb is toggled off). Copy ExDark's top-up or keep "download beside"? | D |
| 28 | Quiet revisit line `dlRenderQuiet` "Starter kit downloaded 2 hours ago (14,004 files verified then)." + **Verify** → `/download/verify` (3165) | verbatim; Verify calls our `kit.verify_now` through a new `/download/verify` route | V |
| 29 | Download progress `dlRenderProgress`: `.kg-run-head` "Starter kit download" · running badge · label "Downloading shard 3/10 · 210 of 625 MB" · elapsed · Cancel · `plugin-progress-bar` · quartile announcements (3301) | verbatim, fed from our stage_progress events (label, phase, bytes) | V |
| 30 | Download terminal states: success callout ("Starter kit downloaded and verified. N files match the published manifest. The dataset YAML path below is filled in and ready to import."), fail callout with failed checks + Copy diagnostics + Resume offer, cancelled callout (3256) | verbatim; success copy ends "The starter kit folder below is filled in and ready to import." | A(2) |
| 31 | Two funnels: on download success the client re-reads `/config` and adopts `session.dataset_yaml` (`dlAdoptSessionYaml`) (3353) | verbatim with `session.kit_dir` | V |
| 32 | Tab-open resolution `kgInitImportTab`: running import → reconnect (State 3, "Importing… (safe to navigate away)"), else `/import/state` success → revisit (form hidden, dl section hidden except running download / superseded), else form + `dlInit` (3071) | verbatim; running jobs come from the host's job list (`PluginJobs.list`), reconnect through the job channel | V |
| 33 | Revisit `kgRenderRevisit`: result rows + checks from the snapshot; "Show log" from the surviving job record (3033) | rows + checks from our import record; **no log on revisit**: the host keeps no job log (PLAN §A3, no separate job store) | A(plumbing) |
| 34 | Failure `renderFailBanner`: `alert alert-error` "Import failed" + message + Copy diagnostics; the CTA relabels "Re-run Import & Validate" (2866) | verbatim | V |
| 35 | Session editor: the form persists as it settles (`kgImportCfgSettled`, 600 ms debounce + blur) into the session (project, table, yaml) (2525) | verbatim with `kit_dir` in place of `dataset_yaml` | V |
| 36 | Job start `kgStartJob`: POST `/validate/<kind>` then the host `/run`; abort translation and `kgAdoptFreshJob` (1282) | our routes have no `/validate`: the preflight is the gate and `run_import` re-validates; `/run` and the provisioning answer are handled the same way (a 201 `status: provisioning` renders ExDark's message) | A(plumbing) |
| 37 | `poll(jobId)` every second against the plugin's job store (2876) | the same terminal handling driven by `PluginJobs.run/track` events; the log accordion fills from a `log` event the worker adapter emits | A(plumbing) |
| 38 | Bug 4 (stale manifest): n/a in ExDark | every manifest fact renders from ONE source: the latest resolution (the refresh result once it lands, else the config's). The kit record now stores the manifest provenance the download job ran under, so "which manifest did the download use" is answerable from the record | plumbing |

## 3. Decisions (Rishikesh, 2026-09-28, after the visual review of the mirror)

- **#26 Re-import fresh** — ExDark's button and placement (the `.kg-reimport-slot` of a REUSED
  row). It starts a `mode=reimport` job that writes fresh `initial-N` tables beside the old pair;
  it never overwrites a table or its revisions. Consequence: ExDark's confirm dialog (a revision
  count, "re-importing fresh discards them") is not shown — nothing is discarded.
- **#27 Superseded kit** — download-beside stays (existing tables point at the v1 image files);
  the callout keeps ExDark's shape and wording where it is true, with the top-up sentence replaced
  by the download-beside fact.
- **#6 Stepper dot** — ExDark's exact behaviour: the dot marks the next pipeline step.
- **#11 `?kgdev` fixtures** — ported (kgDevForce / kgDevDisableActions, verbatim structure) for
  the states this plugin has: `state1`, `state2`, `state2-mismatch`, `state2-error`, `state3`,
  `state4`, `state5`, `state6`, `state6-superseded`, `dl-empty`, `dl-running`, `dl-verify`,
  `dl-success`, `dl-fail`, `dl-cancelled`, `dl-revisit`, `dl-superseded`. Fixture numbers, class
  names and shard names derive from the served manifest (the literal census stays clean); fixture
  pages disable both job-firing buttons and every start path guards on `kgDevMode`.

Fixes from the same review:

- **a. Competition name** — the hero title is the manifest's `display_name` (allowed difference 1).
- **b. Status glyph** — checked: ExDark's `#kg-state-status` is empty too (Status is not a
  pipeline step; `renderPipeline` marks import/train/submit only). Mirrored as is.
- **c. Stale result next to a new preflight verdict** — checked: ExDark clears the banner, checks
  and result only at import start and Start over, and its splits placeholder always says "Enter a
  YAML path above…". **Deviation:** `kgClearStaleResult()` runs when the preflight transitions to
  mismatch or error while a finished import's result is on screen (form state only; never during
  an import or on the revisit view), and the splits placeholder reads "Splits are detected once
  the kit folder matches the competition manifest." when the folder is filled.
- **d. (future)** — the gate says "no import has produced these tables" when the tables exist but
  the plugin's record was reset (e.g. after a `--reset-state` reinstall or a fresh machine sharing
  the project root). ExDark's `verified_import_state` synthesizes a snapshot from the canonical
  URLs when they all exist; port that when the Train tab needs the tables (session 3).

## 4. Tests

ExDark's own fragment safety tests, ported: `test_ui_node_lifetime.py` (the `#kg-dl-dest` node
lifetime and the toggled blurbs), `test_contract_parity.py`'s literal census (ours:
`test_no_competition_literal_outside_the_manifest` + the fragment needles), and the JS/Python
URL-regex parity (`test_url_regex_parity.py`, live once the fragment carries `kgUrlSeg` for the
Train fields). Added: an `esc()` census (every value interpolated into an `innerHTML` template
passes through `esc()` or a markup-producing helper) and the markup-in-manifest test (a class
name or display name of `<img src=x onerror=alert(1)>` renders as literal text: `esc` is run
through node on that payload; skipped when node is absent).

## 5. rc11 divergences (Rishikesh, 2026-10-09) — back-port to ExDark after the event

The Import tab and the shell leave the mirror on purpose. Each line names ExDark's version and ours;
`CHANGELOG.md` 1.0.0rc11 has the participant-facing wording. Plumbing stays ours as before.

| # | ExDark | rc11 (ours) |
|---|---|---|
| 1 | The tab resolves its state three ways: `/config` for the form, `/import/state` for the revisit, `/config` again for the download section | `GET /config` carries the whole truth (`_meta.import_state` with `found` / `success` / `stale`, the tables with existence · rows · table name, the missing split, the validation result + stamp + checks, the kit folder, the log; `_meta.kit_state` with image count · verification stamp · source; `_meta.manifest_fallback`); the tab and the stepper read it plus the live job list (`kgRenderImportFromState`) |
| 2 | Import done = the record exists | Done when both tables exist on disk, record or not (`importer._found_state`) |
| 3 | Form vs revisit (`kgRenderRevisit` renders into the form it hides) | One Imported view outside the form (`#kg-imported` + the outcome area after the form): project, table, when, validated, kit, Re-import…, rows, collapsed checks, log; a successful import transitions into it with the checks expanded once |
| 4 | "Imported & validated" always | "Imported · validated" for a fresh import, "Found existing tables · validated" for a reuse |
| 5 | A stale record falls back to the plain form | The stale view names the missing table (split, table, path, recorded when) and offers Re-import… |
| 6 | The button disables while a job runs | Plus a guard in `kgStartImport` (never two imports from a double click or a tab switch) |
| 7 | Dataset YAML / Project / Table in the open form; "Revision name for the imported tables" | The three fields under **Advanced**; "Table name" is the one term; paths middle-truncated with the full path on hover and Copy, one separator style (`kgDisplayPath`) |
| 8 | Locked "Splits to import" rows; "Detected: train / val / test · 12 classes (canonical order) · N / N / N images. Matches the competition dataset." | No splits block; "Matches competition manifest ✓"; every count once in Dataset at a glance (train / unlabeled / val / test + tags) with the test-images sentence |
| 9 | "Starter kit downloaded 2h ago (14,005 files verified then)" + Verify | "Starter kit ready · 9,600 images + manifest · verified <date>" + Re-verify; never a raw file count |
| 10 | Start over (view change) + per-row Re-import fresh | One **Re-import…** with an in-app confirmation naming the table the server would write (`reimport_name`) and "Your existing tables and label edits are kept." |
| 11 | Explore train / val / test in the success banner | Per-row Explore only |
| 12 | The checks grid always open | "N/N checks passed ▸" collapsed when all pass; a failure auto-expands with the failing checks first and their fix hints |
| 13 | A fallback to the cached / bundled manifest is logged only | One quiet notice on the Import tab and in the Doctor ("Using the built-in competition manifest; the server couldn't be reached"); the warning rides `_meta.manifest_fallback`; after one unreachable index, job starts resolve locally for five minutes (`remote_down_recently`) while the refresh retries |
| 14 | A failed download offers Resume | Resume first; when it cannot help (a failed check, or the second network failure) Advanced opens with the manual route and a link to the competition's Data page on Kaggle; a pasted folder goes through the full preflight and `GET /download/verify?kit_dir=` (a pass records it as a manual kit) |
| 16 | The hero card is always expanded; Loop links styled like text; Status in the same tab row | The card collapses to one line from the second visit (per browser, Show / Hide details); Loop links underlined; Status right-aligned behind a divider |
| 17–22 | — | The other tabs' items: the Queue subtitle "<job> · <project>" (the title is host-owned), the in-app submit confirmation (`docs/PREDICT_MIRROR.md` §14), "Other ways" lines never wrap, "Run inference first" never over a CSV, the "Kaggle (competition not launched yet)" heading, the Doctor refresh after Connect |

The `?kgdev` fixture map grows by `state1-bundled`, `state2-existing`, `state6-found`, `state6-stale`,
`dl-verify-fail`, `dl-fail-checks`, `dl-fail-kaggle`. The esc() census and the literal census stay green.

**rc12 (Rishikesh, 2026-10-09): the rc11 Import redesign was too minimal; the Import tab went back to the
rc10 layout.** Items 7, 8, 10 and 11 are ExDark's again (the open form with the three fields and their
helper text, the locked splits block, the detected line, the glance card without the test count, Explore
train / val in the banner, per-row Re-import fresh without a dialog); the facts line and the Advanced
disclosure are gone. Items 1 to 6, 9 and 12 to 22 stay as rc11 shipped them (item 3's view now renders
around the form, never instead of it; item 5's stale view keeps Re-import… with its confirmation; item
14's notice sits above the kit folder). The only visual addition rc12 keeps is item 12, the collapsible
checks list, in rc10's slot. rc12 also bans the em dash from every participant surface
(`tests/test_copy_rules.py`).
