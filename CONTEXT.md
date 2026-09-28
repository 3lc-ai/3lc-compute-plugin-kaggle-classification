# CONTEXT.md — shared language (kaggle-classification plugin)

One line per term. Depth: docs/PLAN.md (decisions), docs/STUDY.md (references).

## Resume point (2026-09-22, end of day)

- **Session 2, Gate B/C PASSED on 2026-09-28** on the laptop through the catalog install path (`../3lc-hub-11/SETUP.md` § Gate B/C): importer + Import tab shipped (`79f553f`, `3fb8e33`, `28e0034`); Phase D (dev-bucket verification) awaits Rishikesh's go after the `cdn/` upload.
- **Session 2, Gate A PASSED at `dcdf20b`** (`develop`, pushed): relative kit paths, tiered
  host allowlist, `tools/make_cdn_tree.py`, `tools/verify_cdn.py`, `docs/PROMOTION.md`,
  PLAN A3. 146 tests green. `cdn/` is built locally and gitignored.
- **Session 2.5 (2026-09-28): the plugin now MIRRORS ExDark** (`docs/EXDARK_MIRROR.md`, the governing rule; `docs/UI_PARITY.md` superseded). The fragment is a port of ExDark's shell + Import tab (innerHTML + `esc()`), the importer REUSES existing tables like ExDark, display name "3LC Scene Classification Challenge", `submission.daily_limit` 100. Open decisions for Rishikesh: Re-import fresh (#26), superseded-kit top-up (#27), the stepper dot (#6), `?kgdev` fixtures (#11). **Waiting for the visual review in the real Hub**; then Phases 3 (Hub display checks) and 4 (participant matrix).
- **Phase D PASSED on 2026-09-28**: `verify_cdn.py` PASS against dev (headers + sha256s in `docs/PROMOTION.md` §6), full Import on `3lc-hub-11` through the test catalog pinned to `4a42051` with the override pointed at dev (`../3lc-hub-11/SETUP.md` § Phase D). **Next:** Rishikesh's manual check in the real Hub (services and URLs in SETUP.md § Real Hub check), then session 3 (Train tab).
- **Deletion audit (2026-09-28):** the plugin's only table deletion is `importer.delete_tables`, reached from the import job's failure rollback and cancel, on the URLs that job created; Start over is a view change, re-import writes fresh names, a kit reset rewrites kit files and removes shard archives only. `tests/test_deletion_safety.py` pins it with an edited-label revision that survives all four. The gate's delete-and-reimport deletion was the test's own `tlc.Url.delete`.
- **`starter-kit/v1` is NOT uploaded to the dev bucket** and must not be until Gate B/C
  passes on `3lc-hub-11`.
- **The Phase B/C go carries two additions:** (1) on `3lc-hub-11`, list BOTH the default
  catalog (`https://3lc-public-examples-2-2.s3.amazonaws.com/hub/catalog.json`) and
  `catalog-test.json` in `TLC_COMPUTE_PLUGIN_CATALOG_URLS` (the variable replaces the default
  unless it is listed); (2) if the import reveals ANY kit defect, stop and report before
  changing anything: a fix before staging is a v1 rebuild and needs Rishikesh's go.
- **Open items:** relicensing sign-off for the modules in `docs/STUDY.md` G-5 pending from
  Paul / Gudbrand · `solution.csv` (Kaggle answer key: `image_id, Usage, label` over the new
  test ids) not started, and cannot start from this repo alone: `mapping.csv` maps the new
  test ids back to the original test filenames only, so the test ground truth must come from
  the organizer · Kaggle sandbox competition not yet created (slug and deadline in the
  bundled manifest are still TBC placeholders).
- **Hosts:** nothing is running on `3lc-hub-11` (:5023) or any other Hub port at close;
  start commands are in `../3lc-hub-11/SETUP.md`.

## Competition & contract

- **the manifest** — the remote competition document (schema v1) every competition fact comes from; resolved remote → cache → bundled; `manifest.py`. "Bundled" is the copy inside the wheel, "cache" the last good remote copy on disk.
- **the two tiers** — dev `https://competitions.dev.3lc.ai` (bucket `3lc-competitions-dev`, console uploads) and prod `https://competitions.3lc.ai` (promoted copy); byte-identical objects, relative URLs throughout (PLAN §A3).
- **the index** — `<base>/kaggle/classification-index.json` listing competitions with an `active` flag; one active → used, several → picker, none → bundled + warning.
- **resolution** — remote (reachable and valid) → cache (last valid remote) → bundled; no version ordering; `resolve(network=False)` is what the page renders first, the background refresh brings the remote result.
- **provenance** — `{manifest_sha256, manifest_source, …, competition_id, kit_version}` every job records at start (`resolve_manifest_for_job`); the ledger's input.
- **the plugin home** — resolved by `storage.py` (env → SDK helper → worker state root → `<cwd>/.plugin-state/<id>` → `~`), reported in `_meta.plugin_home`; holds `ui_config.json`, `kit/<id>.json`, `data/<id>/<kit version>/`, `manifest-cache/`.
- **competition id** — the stable CDN id (`intel-scene`), never the Kaggle slug; names the manifest path, the default project, the dataset prefix and the kit directory.
- **the slug** — the Kaggle URL slug (`competition.slug`); used only by Submit/Status.
- **the contract (locked)** — `arch` from the manifest (`resnet18`), `pretrained=false`, `image_size` from the manifest, `timm==1.0.29`; identical init for every participant; `pretrained: true` is rejected at manifest load.
- **undefined** — the unlabeled pool: label id `num_classes` (the LAST map entry), weight 0, filtered out of training regardless of weight, given predictions/confidence/embeddings but no loss.
- **the labeling loop / loop contract** — the per-sample metrics a run writes back (PLAN §B): predicted, confidence, per-class probabilities, loss (masked for undefined), 3D embeddings with UMAP fit on train (labeled + undefined), val transformed into the same space.
- **starter kit** — `starter_kit/data/{train/<class>,train/undefined,val/<class>,test}` + `sample_submission.csv` + `files.json`; images renamed to salted opaque ids and re-encoded; the judge's `mapping.csv` never ships.
- **files.json** — the per-file index inside the kit (relpath, sha256, bytes, kit_version); the download stage verifies every file against it.
- **kit{} block** — the manifest's `kit{path, version, shards[{name, sha256, bytes}]}`, `path` relative to the manifest's own URL; emitted by `tools/build_kit.py`, pasted into the manifest; an empty `shards` list means "not published" and the download refuses.
- **download_kit** — the job kind: shards (sha256, Range resume, .part files) → extract → per-file verify → split counts vs manifest → record + `session.kit_dir`; states `empty / success / superseded / stale`.
- **the session object** — `{project_name, table_name, kit_dir, device, overrides}` in `~/.3lc-kaggle-classification/ui_config.json`; tabs render projections; defaults derive from the manifest; retired keys 400.
- **REUSED vs CREATED** — per-split import outcome, the ExDark mirror (`docs/EXDARK_MIRROR.md` #25, decision 2026-09-28): a table already at the target URL is REUSED (never rewritten) and the post-write checks run on it; otherwise CREATED. The backend's `mode=reimport` (fresh `initial-2`, `-3` names) exists for the pending **Re-import fresh** decision (#26) and is not offered by the fragment.
- **the mirror rule** — the plugin mirrors ExDark (3lc-compute-plugin-kaggle v1.2.15) in functionality and looks; the fragment is a port of its `ui.html` (innerHTML + `esc()`), the four allowed differences and every needs-decision item are in `docs/EXDARK_MIRROR.md`. `docs/UI_PARITY.md` is superseded.
- **the import record** — `import_state` in the session store: project, actual table name, kit facts, `tables{train,val}` with row counts, the **lineage root** (`{train_url, val_url}`, the seed both later revisions descend from), the **locked val** URL, the label map, every check, per-stage timings, the manifest provenance and the job id. `importer.import_state()` re-verifies it against disk (`empty / success / stale`).
- **first-run provisioning** — on compute 1.1.0 the first plugin route builds the venv and answers `201 {status: "provisioning"}`; the fragment renders "Setting up the plugin environment. The first run takes a few minutes.", polls `/config` every 5 s and then continues — never an error. While the venv is missing the host itself serves a placeholder for `/ui`, so the fragment is not even shown in that window.
- **the test catalog** — `../3lc-hub-11/catalog-test.json`: this repo at a pinned commit (Phase D: `4a42051`), listed beside the default catalog in `TLC_COMPUTE_PLUGIN_CATALOG_URLS`; gates run through the catalog install path, not only `--plugin-dir`. Never register the folder source and the catalog install of the same id together.
- **dev host / harness** — `../3lc-hub-11/dev_host.py` (the in-process app on :5023 with auth bypass) and `../3lc-hub-11/harness/index.html` (the fragment with a stubbed `PLUGIN_API`) stand in for the Hub frontend on the laptop.
- **plugin-run-only** — participants predict only from runs this plugin trained (session 4).
- **the ledger / verification bundle** — the append-only record of every step and the per-run zip an organizer verifies a leaderboard entry against (session 5).

## Machines

- **office** — dev root `C:\Users\Owner\Desktop\3LC competitions\3LC Kaggle Competitions`; `3lc-hub-11` at `<root>\3lc-hub-11` (built 2026-09-22 on system Python 3.12.3, key byte-copied from `3lc-hub-ga`).
- **laptop** — dev root `C:\Users\rishi\Desktop\3LC Hackathons` (since 2026-09-28); `3lc-hub-11` at `<root>\3lc-hub-11`, rebuilt on uv-managed CPython 3.12.13 (`uv python install 3.12`; the machine's Anaconda Python/3lc is never used). Folder names under the two roots are identical.
- **path rule** — a path in any doc written before 2026-09-28 (STUDY, PLAN §A3 notes, the office section of `../3lc-hub-11/SETUP.md`, `config_probe.log` references) refers to the **office** root; substitute the laptop root, nothing else changes.
- **laptop key store** — 3lc 3.3.0 resolves its config dir and API-key file through `platformdirs`/native known-folder lookups, so the `LOCALAPPDATA` redirect does NOT isolate them (they stay under `C:\Users\rishi\AppData\Local\3LC\3LC`); only `<home>/.3lc-compute/` follows `USERPROFILE`. Decision 2026-09-28: no `TLC_CONFIG_FILE` isolation; the pre-existing `3LC\3LC` (Anaconda 3lc 2.19.1, 2025) was renamed `3LC-bak-2025-anaconda` before the login, so 3lc 3.3 started from a clean config like a new participant. Details and the login command: `../3lc-hub-11/SETUP.md` § Laptop.
- **line endings** — normalized to LF in one commit on 2026-09-28 (`.gitattributes`: `* text=auto eol=lf`, CRLF only for `*.ps1/*.bat/*.cmd`, binary for images/archives/wheels). The office checkout must run `git rm -rq --cached . && git reset --hard HEAD` after its next pull, or every renormalized file shows as modified.
- **laptop transfer archives** — `intel-data.zip`, `migrate-public.zip`, `migrate-private.7z` were deleted from the dev root on 2026-09-28; Google Drive keeps the backup.
- **solution.csv** — on the laptop the Intel answer key lives at `<root>\hackathon_private\intel-scene-v1\solution.csv` (beside `kit.salt` and `mapping.csv`); it is the `--old-solution` input of the solution-file task and is never read or copied by code.

## Ops & environments

- **hosts** — `../3lc-hub-11/` (compute **1.1.0** + 3lc 3.3.0 + SDK 0.3.3, compute :5023, object :5017 reserved, redirected home; recipe `../3lc-hub-11/SETUP.md`) is the plugin's host: the card reads `compatible: true` there and `GET /config` was verified through the host proxy on 2026-09-22. `../3lc-hub-ga/` (1.0.1, :5022) greys the card out (`min_service_version` 1.1.0) and is untouched.
- **the storage verdict on 1.1.0** — the host passes no `--state-root`, so `storage.py` rule four fires: the plugin home is `<home>/.3lc-compute/managed-plugins/kaggle-classification/.plugin-state/kaggle-classification` (the SDK worker's default under its cwd), independent of HOME. Verified live (`../3lc-hub-11/config_probe.log` P3/P4).
- **the SDK window** — `>=0.3.1,<0.4.0`; `test_packaging.py` checks it overlaps the latest 3lc-compute release's declared range (snapshot offline, PyPI live).
- **folder source vs tag install** — a dev Hub registers `src/` and picks up edits on worker reload; a tester's catalog install pins a tag and never sees the checkout.
- **the catalog** — `catalog.json`: one entry per released version, newest first, manifest pasted verbatim, `source` pinned to the tag; `HEAD/catalog.json` on the default branch is the URL hubs consume.
- **the four tabs** — Import · Train · Predict + Submit · Status; the tab bar is the stepper. The Import tab's own stepper is Manifest · Disk space · Download · Verify · Extract · Register · Validate, one list for the download_kit and import jobs.
- **?kgdev fixtures, six-state machine, motion tokens, glance card, verdict line** — the UI playbook vocabulary, inherited from the ExDark plugin's docs/ui-notes.md from session 2 onward.
- **the components (session 2.5)** — the fragment's reusable pieces every tab uses: `callout(kind, lines)` (ok/info/warn/err, one geometry), `pathCell`/`pathRow` (middle-truncated path + Copy), `accordion` (button + region, reduced-motion aware), `renderChecks` (verdict toggle, collapsed on pass, auto-open on failure, Kit/Tables groups + remedies), `gateCard` (`renderGates`: "Import the kit first." + Go to Import on Train/Predict/Status until an import record exists), `chip`/class tags from the manifest, the Loop row, the tab bar as stepper with state glyphs, `conn` (the connection guard), `diagBtn`/`buildDiagnostics`, `REMEDIES` (error text → one-line hint). All element builders + `textContent`; `docs/UI_PARITY.md` is the map.
- **Technical details** — the one disclosure on the Import card holding what a participant does not need by default: plugin home, kit destination and directory, manifest source + sha256 + URL, job id, timings, lineage root, locked val, each with Copy. The participant view above it carries no raw path, id or JSON.
- **the connection guard** — ExDark's `kgConn` in full: a network-level fetch failure (TypeError) shows "Compute service unreachable, retrying…" in every tab's slot, pings `/import/state` with 2/5/10/15 s backoff, shows "Reconnected." and runs the queued resume callbacks (config reloads, gates, polls). It never starts or repeats a job.
- **Open in Dashboard / Open project in the Hub** — the two links on a table row: the Dashboard deep link (`?table=` + `object_service=`, the Hub's own Open target) and, only when the fragment runs on `hub.3lc.ai` / `hub-beta.3lc.ai`, `<origin>/projects/<project>`; the Loop's Dashboard step opens `import_state.latest.train` (tlc's `latest()`), not the seed.

## Naming

- **project** — default `intel-scene` (the competition id); **tables** — `initial` by default; **datasets** — `<competition id>_<split>` (`intel-scene_train`, `intel-scene_val`).
- **run names** — `<competition id>_run_<YYYYMMDD_HHMMSS>` when blank (session 3).
- **tags** — `vX.Y.Z`; the version string is identical in pyproject, plugin.toml and the catalog entry (`scripts/release_version.py`, `test_packaging.py`). Kit data uses the manifest's `kit.version` (`v1`, `v2`, …), never a git tag.
- **plugin id** — `kaggle-classification`; the entry-point key equals it.
