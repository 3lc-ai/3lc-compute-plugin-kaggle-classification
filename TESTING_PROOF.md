# TESTING_PROOF.md — TESTING.md followed on a brand-new environment (2026-10-05)

The tester kit was proven the way the brief asked: a fresh environment, nothing copied from the
development Hub, every step of `TESTING.md` that does not need a browser driven through the plugin's
routes exactly as the Hub calls them, and the document corrected wherever reality differed. Runs 1–9
are recorded (run 9 is the rc9 install-path proof on a second fresh environment); the first two: the first against 1.0.0rc1 found a defect, the second against 1.0.0rc2 (the fix) passed
clean. Full paths throughout.

## The environment

| | value |
|---|---|
| Folder | `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester` (created 2026-10-05, empty before) |
| venv | `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester\.venv` — `uv venv --python 3.12`, then `uv pip install --index-url https://pypi.org/simple "3lc-compute==1.1.0" "3lc==3.3.0"` (TESTING.md §2, word for word) |
| Home | redirected to `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester\home` (USERPROFILE, LOCALAPPDATA, APPDATA, HOME), as `tester/start_tester.ps1` does |
| Project root | `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester\home\projects` (`TLC_PROJECT_ROOT_URL`, the script's optional `-ProjectRoot`), so the laptop's existing 3LC projects were never touched |
| Ports | compute 5024, object 5018 reserved (the development Hub uses 5017 / 5023); the proof drives the in-process host, so no port actually listened |
| Manifest tier | `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL=https://competitions.dev.3lc.ai` |
| Catalogs | `https://3lc-public-examples-2-2.s3.amazonaws.com/hub/catalog.json,https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc2/catalog-test.json` |
| 3LC login | the laptop's existing key (on Windows the key store is not moved by the redirect — TESTING.md §2 says so) |
| Kaggle | no token visible to the services (the home is redirected); asserted by the proof |
| uv cache | the machine's `C:\Users\rishi\AppData\Local\uv\cache` was reused (`UV_CACHE_DIR`, the script's optional reuse), so the torch download is not in the timings below |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB), Windows 11, CPython 3.12.13 |
| Driver | `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester\proof_tester.py` — the in-process compute 1.1.0 host (`create_app`, auth exemption, Litestar `TestClient`) calling the same admin and plugin routes the Hub calls; record `proof_tester.json`, logs `proof_tester.log` / `proof_tester.err.log` (the rc1 run: `proof_tester_rc1.*`) |

What a browser does that the proof could not: the Getting Started connection (§5), clicking in the
Hub (§6, the tab bar, the buttons), the Dashboard labeling (§7.3) and the downloads landing in a browser.
For §7.3 the proof wrote what the Dashboard writes — an `EditedTable` revision beside the seed with 20
pool rows set to a class and weight 1 — and checked that the Train tab's preflight sees it as the latest
revision with 620 labeled rows. Everything else is the route the button calls.

## Run 2 — 1.0.0rc2, fresh home — PASS (12:24:04 → 12:28:12, 4 min 8 s)

| TESTING.md step | Route(s) | Result |
|---|---|---|
| §4 start | `GET /health` | host up in 12.4 s; `{"status":"ok","service":"3lc-compute","version":"1.1.0","plugin_install_policy":"catalog-only"}`; the eight stock plugins seeded first (see the doc fix below) |
| §6.1 Available | `GET /api/admin/plugins/catalog` | card `kaggle-classification` **1.0.0rc2**, `compatible: true`, source `…@v1.0.0rc2` from the hosted test catalog |
| §6.2 Install | `POST /api/admin/plugins/install` + status poll | `succeeded` in **55 s** (venv built from the git tag; torch from the cache); first-use provisioning `ready` in 2.9 s; `GET /config` `_meta.version` = `1.0.0rc2`; the managed venv `…\home\.3lc-compute\managed-plugins\kaggle-classification\1.0.0rc2\.venv` |
| §6.3 the page | `GET /config`, `GET /manifest` | display name "3LC Scene Classification Challenge", slug `3lc-scene-classification-challenge`, manifest refresh `done / remote`, sha256 `ec0c60cf45f0…bdfcb`, kit state `empty`, plugin home `…\1.0.0rc2\.plugin-state\kaggle-classification` |
| §7.1 Download | `POST /run download_kit`, `GET /download/verify` | completed in **70.1 s**; kit v1, **9,601 files**, verify 9,601 / 9,601 matched, 0 missing, 0 mismatch |
| §7.2 Import | `GET /import/preflight`, `POST /run import`, `GET /import/state` | preflight all ok; completed in **6.0 s**; **18/18 checks**; train **6,600** rows (6,000 undefined), val **1,200**, both CREATED (`reused: false`); project `intel-scene`; `val_edited: false` |
| §7.3 Label + commit (stand-in) | an `EditedTable` `labels-batch-1`; `GET /train/preflight` | 20 pool rows → class 4 (sea) at weight 1; the preflight sees `has_revisions`, latest = the revision, **620 labeled rows in use** · 5,980 excluded as undefined · 0 at weight 0; `descends_from_seed: true`, `foreign_rows: 0`, `label_map_ok: true` |
| §7.4 Train 2 epochs | `POST /run train` (Use latest on, epochs 2), `GET /train/state` | completed in **76.1 s** (setup 11.6 s · 7.7 s/epoch · collection pass 47.0 s); run `intel-scene_run_20261005_122648` on `labels-batch-1`, 620 usable rows, `cuda (auto)`; best val accuracy **41.67 %** at epoch 2; provenance **9/9**; contract torch `2.14.0+cu126`, torchvision `0.29.0+cu126`, seed 42 |
| §7.5 Predict | `GET /runs`, `GET /predict/preflight`, `POST /run predict`, `GET /submit/state` | 1 run listed and usable; test images 1,800 / 1,800 verified; completed in **6.0 s**; state `predicted`; **10/10 checks**; val check 41.67 % recorded · 41.67 % now (delta 0.00); distribution buildings 432 · forest 376 · glacier 0 · mountain 631 · sea 329 · street 32, mean confidence 0.427, 1,456 below 0.5, the skew warning shown (expected after two epochs) |
| §7.6 Download CSV | `GET /submissions/{job}/download`, `GET /kaggle/connection` | HTTP 200, `Content-Disposition: attachment; filename="submission_20261005_122810.csv"`, 50,431 bytes, header `image_id,prediction,confidence`, **1,800 data rows**, predictions 0–5, confidences in [0, 1] with six decimals; Kaggle connection **`no_credentials`** (nothing can submit) |
| §7.7 Status | `GET /status/history?live=1`, `GET /status/kaggle`, `GET /status/doctor` | Runs: the one run (completed, 620 rows, 41.67 % at epoch 2, cuda (auto), 73.8 s). History: the one prediction (val 41.67 %, no public score, no submission → "CSV generated (not submitted)"). Best public score none. Kaggle live `connected: false` with the connect-your-account reason. Doctor: plugin 1.0.0rc2 · commit `37c7e11c…`, compute 1.1.0, SDK 0.3.3, 3lc **3.4.0**, torch 2.14.0+cu126 · torchvision 0.29.0+cu126, CUDA available (device class cuda), manifest cache · `ec0c60cf…` · intel-scene kit v1, kit success v1 · 9,601 files, Kaggle `no_credentials`, plugin home resolved by `worker`, 306 GB free, Python 3.12.13 |
| §7.8 Export bundle | `GET /status/bundle` | HTTP 200, `verification-bundle_intel-scene_20261005_182811Z.zip`, 8,870 bytes; members `README.txt, import_record.json, ledger.jsonl, manifest_provenance.json, plugin.json, predictions/<id>.json, runs/<id>.json, train_revisions.json`; no image / CSV / checkpoint member; the ledger has 1 line |
| §7.9 Stop | `POST /api/admin/plugins/kaggle-classification/worker/stop` + the folder's workers | stopped; no listener was opened by the proof |

Not reproduced live: the training set's tlc indexer logged its known transient while the Run's
`object.3lc.json` was being written ("invalid content in object file … EOF", then "indexer skipping
URL"); the run completed with 9/9 provenance checks, as on 2026-09-29 (CONTEXT.md open item).

## Runs 3–5 — the upgrade path (TESTING.md §8b): rc2 state present, a newer candidate installed

Same environment, the rc2 proof's state left in place (`…\managed-plugins\kaggle-classification\1.0.0rc2\.plugin-state\…`:
the kit, the import record, one run, one prediction, the ledger), the new candidate installed from its hosted
catalog, then `proof_tester.py --upgrade`: it asserts that the plugin home moved to
`C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-tester\home\.3lc-compute\plugin-state\kaggle-classification`
(`resolved_by: compute-home`, `migrated_from` = the rc2 state, `ui_config.json` / `ledger.jsonl` / `kit/intel-scene.json`
rewritten), the kit still `success` (download skipped), the import record `success`, the previous run and prediction
listed, the previous prediction's CSV still downloadable (HTTP 200), the rc2 version dir kept — and then runs the
checklist again on the carried-forward tables (the import REUSES them; a new labeled revision `labels-batch-N`).

| Run | Candidate | Result |
|---|---|---|
| 3 (13:15–13:17) | rc2 → **rc3** | state carried forward (all assertions above PASS), kit download skipped, import REUSED 18/18, revision `labels-batch-2` seen — **FAIL at Train**: "The train table revision … contains 6,600 row(s) whose images are not in the kit's train folder": the carry-forward rewrote the record's `kit_dir` to the new home while the tables' image paths stay in rc2's kit tree, and the rc2 foreign-rows gate compared against the new path only. Fixed in rc4 (the gate also accepts a train image by the kit layout `<kit>/data/train/`) |
| 4 (13:22:44 → 13:25:46) | rc2 → **rc4** (the shared home removed first, so the carry-forward ran afresh from the rc2 state) | **PASS** — carried forward (`rewritten: ui_config.json, ledger.jsonl, kit/intel-scene.json`; 1 run and 1 prediction before; old CSV HTTP 200; version dirs rc2 · rc3 · rc4); kit skipped; import REUSED 18/18 in 4 s; `labels-batch-3`; train 2 epochs **38.1 s** on the carried tables, 41.67 % at epoch 2, 9/9; predict 6 s, 10/10; CSV 1,800 rows; Status: **2 runs, 2 predictions**; bundle 10 members |
| 5 (13:31–13:34) | rc4 → **rc5** | state carried forward again (shared home, 2 runs and 2 predictions before, old CSV HTTP 200), kit skipped, import REUSED, `labels-batch-4` — **FAIL at Train**: `FileNotFoundError … 1.0.0rc2\…\starter_kit\data\train\sea\….jpg`: the rc5 install made rc2 the fourth version dir and the host removed it; the tester's carry-forward marker was written by rc4 (no `legacy_data_dirs`), so rc5's re-creation had nothing to act on. Fixed in rc6 (the legacy tree is derived from the marker's `source`). A first rc5 attempt had stopped earlier on a driver bug (the Start client token reused from run 4; the double-click guard refused it, correctly, across the upgrade) |
| 8 (2026-10-06 16:25 → 16:28:36) | rc7 → **rc8** (session 6: the job card during the collection pass) | **PASS** — the hosted rc8 catalog listed the candidate; install from the git tag **30.0 s**, provisioning 8.2 s, `_meta.version` `1.0.0rc8`; state carried (5 runs and 4 predictions before, the old CSV HTTP 200, version dirs rc2 · rc7 · rc8); kit skipped; import REUSED 18/18; `labels-batch-8`; train 2 epochs **68.1 s** (setup 21.6 · 9.2 s/epoch · collection 27.0) — **the host's job labels during the pass: "Collecting metrics 480/7,800" … "7,464/7,800" every 2 s poll, then "Reducing embeddings (UMAP)…"**, 41.67 % at epoch 2, 9/9; predict 8 s, 10/10; CSV 1,800 rows; Kaggle `no_credentials`; Status 6 runs · 5 predictions; bundle HTTP 200, `verification-bundle_intel-scene_20261006_222835Z.zip`, 1,202,328 bytes zipped, 55 members (10 tables, 5 run folders, 10 metrics tables, 37 files indexed, 0 checkpoints: no submission), data rule held. Record `proof_tester.json` (the rc7 run kept as `proof_tester_rc7_upgrade.json`), logs `proof_tester_rc8_upgrade.log` / `.err.log` |
| 7 (2026-10-06 15:38:28 → 15:42:47) | rc6 → **rc7** (session 6: the Projects-page progress, the bundle with the data) | **PASS** — the hosted rc7 catalog listed the candidate (`compatible: true`, source `…@v1.0.0rc7`); install from the git tag **75.1 s**, first-use provisioning 9.1 s, `_meta.version` `1.0.0rc7`; state carried (plugin home `…\3lc-hub-tester\home\.3lc-compute\plugin-state\kaggle-classification`, `resolved_by: compute-home`, marker source the rc2 state, 4 runs and 3 predictions before, the old CSV HTTP 200, version dirs rc2 · rc6 · rc7); kit skipped (9,601 / 9,601 verified); import REUSED 18/18 in 8 s; `labels-batch-7`; train 2 epochs **90.1 s** (setup 21.2 · 14.2 s/epoch · collection 38.1), 41.67 % at epoch 2, 9/9, the host's job labels now "Epoch 1/2" / "Epoch 2/2" for the epoch in progress; predict 8 s, 10/10; CSV 1,800 rows; Kaggle `no_credentials`; Status 5 runs · 4 predictions; **bundle `GET /status/bundle/preview` then `GET /status/bundle`: HTTP 200, `verification-bundle_intel-scene_20261006_214246Z.zip`, 988,927 bytes zipped (1,402,387 planned), 47 members = the 15 session-5 records + `project/intel-scene/files.json` (31 files indexed) + 9 tables (the seed `initial` with its row cache, `labels-batch-1` … `-7`, the locked val with its row cache) + 4 run folders (object + 8 metrics tables); 0 checkpoints because the tester environment never submitted (`eligible_runs: []`, the rule picks nothing), the data rule held (parquet only under `project/…`, no image / CSV / `.pt`)**. Record `proof_tester.json` (the rc6 run kept as `proof_tester_rc6_upgrade.json`), logs `proof_tester_rc7_upgrade.log` / `.err.log` |
| 6 (13:38:08 → 13:42:47) | rc5 → **rc6** | **PASS** — the rc6 worker's first route re-created the rc2 kit tree the host had removed (`migrated_from.json` → `restored` at 19:39:20Z, `…\1.0.0rc2\.plugin-state\kaggle-classification\data`; version dirs rc2 · rc4 · rc5 · rc6); state carried (3 runs and 2 predictions before, old CSV HTTP 200); kit skipped; import REUSED 18/18; `labels-batch-6`; train 2 epochs 144 s on the carried tables (the development Hub's services were starting at the same time), 41.67 % at epoch 2, 9/9; predict 10/10; CSV 1,800 rows; Status: 4 runs, 3 predictions; bundle 13 members |

What run 4 also showed, on the development Hub rather than the tester: installing a fourth version let the
host's `gc_old_versions` (keeps three) remove the `0.1.0` dir — the one hub-11's hand-copied records and its
existing projects' tables point into for their images. The kit tree was restored from the shared home's
sha256-verified copy, and rc5 makes the plugin do that by itself (`legacy_data_dirs` in the carry-forward
marker, re-created on every process start when missing). The tester environment never had that problem: its
records were rewritten to the shared home, and its tables' kit dir (rc2's) is still kept.

## Run 1 — 1.0.0rc1, fresh home — PASS on every step, one defect found (12:14:31 → 12:18:13)

Same environment, same steps, same numbers within seconds (install 30 s with the cache already warm,
download 70.1 s, import 8.0 s, train 80.1 s, predict 8.0 s, best val accuracy 41.67 %). The Status
step answered **no prediction rows** although the ledger held the entry: `status.prediction_history`
read a prediction's checks as dicts, while the ledger stores them as `[label, ok]` pairs
(`predictor.py`). Fixed in `status.py` (both shapes), the unit fixture now uses the ledger's shape,
released as **1.0.0rc2** (tag `v1.0.0rc2`, branch `release/1.0.0rc2`), and run 2 above shows the
History row. A first attempt before run 1 stopped at the catalog card because the proof driver read
the admin catalog's `sources` list instead of its `plugins` list (a driver bug, no plugin change).

## Run 9 — 1.0.0rc9 on a brand-new environment, the install path only — PASS (2026-10-07, 15:16:25 → 15:17:23)

The tester release (rc8's code, version stamped, documentation) proven the way a participant installs:
a fresh folder `C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-rc9-clean` (nothing copied from the
development Hub or the tester environment), `uv venv --python 3.12` + `uv pip install --index-url
https://pypi.org/simple "3lc-compute==1.1.0" "3lc==3.3.0"` (CPython 3.12.13, SDK 0.3.3), home redirected
to `...\3lc-hub-rc9-clean\home`, project root `...\home\projects`, ports 5015 / 5020 (the defaults —
TESTING.md §3 now insists on them, see CHANGELOG 1.0.0rc9), manifest tier dev, catalogs = the default
catalog + `https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc9/catalog-test.json`.
Driver `proof_clean_rc9.py` (the in-process host, TESTING.md §6 and the Doctor only; no kit, no
import, no Kaggle), record `proof_clean_rc9.json`:

| Step | Result |
|---|---|
| §6.1 Plugins → Available | the card lists `1.0.0rc9` (`latest_version` 1.0.0rc9, compatible), install source pinned to `@v1.0.0rc9` |
| §6.2 Install | `POST /api/admin/plugins/install` 202 → `succeeded` in 45.0 s (the machine's uv cache held torch); first-use provisioning ready 1.7 s later |
| §6.3 The page | `GET /config` `_meta.version` **1.0.0rc9**, plugin home `...\home\.3lc-compute\plugin-state\kaggle-classification` (`compute-home`), kit state `empty`, manifest `remote` from the dev tier (refresh `done`) |
| §7.7 Doctor | plugin 1.0.0rc9 at commit `bb3f33c` (the tag), SDK 0.3.3, torch 2.14.0+cu126, torchvision 0.29.0+cu126, kaggle 2.2.4, Python 3.12.13; kit empty; records 0 / 0 / 0 / 0; Kaggle `no_credentials` (expected: the redirected home has none); the device probe was still `running` 2 s after provisioning (asynchronous by design) |
| persisted | `settings.json` `installed_plugins` carries `kaggle-classification 1.0.0rc9` |

The services were then started with the participant's own script (`start_tester.ps1 -ProjectRoot
...\home\projects`, a copy of `tester/start_tester.ps1`) on 5015 / 5020 for the hand test in the Hub.
One observation, not a failure: the plugin venv resolved **3lc 3.4.0** (PyPI's latest since 2026-10-01;
the extra says `3lc>=3.3,<4.0` and the catalog install path is `uv pip install`, not the lock) while the
host and the object service run 3.3.0. Every catalog install since rc7 (hub-11's rc7 and rc8 venvs, the
tester environment's rc8 venv) resolved the same way, and the rc7 / rc8 hand tests ran on it.

## Run 10 — 1.0.0rc9 → 1.0.0rc10 on the clean environment, plus the connect flow without a Kaggle account — PASS (2026-10-08, 18:31:14 → 18:33:21)

The rc10 candidate (the guided Kaggle connect flow, CHANGELOG 1.0.0rc10) proven as TESTING.md §8b on
`C:\Users\rishi\Desktop\3LC Hackathons\3lc-hub-rc9-clean` — the run-9 environment after Rishikesh's rc9
hand test (one imported kit, one prediction, one ledger line) — with the services stopped, through the
in-process host and the same environment `start_tester.ps1` gives it (home redirected to `…\home`,
project root `…\home\projects`, dev manifest tier, catalogs = the default catalog + the rc10 test
catalog). Driver `proof_clean_rc10.py` (+ `run_proof_rc10.ps1`), record `proof_clean_rc10.json`:

| Step | Result |
|---|---|
| §8b Plugins → Available | the card reads `latest_version` **1.0.0rc10**, compatible, source pinned to `@v1.0.0rc10` |
| §8b Update | `POST /api/admin/plugins/install` 202 → `succeeded` in **105.1 s** (the rc10 venv built beside rc9's; version dirs after: `1.0.0rc10`, `1.0.0rc9`); first-use provisioning ready 11.8 s later |
| §8b carried forward | `_meta.version` 1.0.0rc10, plugin home unchanged (`…\home\.3lc-compute\plugin-state\kaggle-classification`, `compute-home`); the state snapshot identical before and after (`ui_config.json` 31,677 B, 1 ledger line, 1 prediction, the kit record, 9,602 kit files) |
| §7.5 connection card | `GET /kaggle/connection` → `no_credentials`, `token_path` = `…\3lc-hub-rc9-clean\home\.kaggle\access_token` (the redirected home, not the real profile), the "Other ways" block for PowerShell with that path filled in, one line each |
| §7.5b a wrong paste | `POST /kaggle/connect` with `""` → "Paste the token first."; with `abc` → "That doesn't look like a Kaggle access token: it should start with KGAT_ (…)"; no file written |
| §7.5b a well-formed fake token | `KGAT_proofclean…` (pasted with spaces and a newline) → the service wrote the file, Kaggle's introspect rejected it in 2.9 s → "Kaggle rejected the token (revoked, expired or mistyped); create a new one and try again.", **the file removed again** |
| §7.7 Doctor | plugin 1.0.0rc10 at commit `8dfd06f` (the tag), SDK 0.3.3, 3lc 3.4.0 (the known open item), torch 2.14.0+cu126, torchvision 0.29.0+cu126, kaggle 2.2.4, Python 3.12.13; Kaggle `no_credentials` with `token_path` |
| no leak | the fake token appears in none of: the connection card, the three connect answers, the Doctor, the Kaggle live block, `GET /config` |
| persisted | `settings.json` `installed_plugins` carries `kaggle-classification 1.0.0rc10` (installed 2026-10-09T00:33:01Z) |

The services were then started with the rc10 start script copied beside the rc9 one
(`start_tester_rc10.ps1 -ProjectRoot …\home\projects`, the defaults 5015 / 5020): both ports answer, the
compute log registers the `/kaggle-classification` namespace; left running for the hand test of Connect
and Submit in the Hub. The environment holds no Kaggle token (an empty `home\.kaggle` folder is the
only trace of the fake-token check). Not proven here: a real token's "Connected to Kaggle as <username>"
(the harness check in session 8 used a stubbed service; the real path is Rishikesh's hand test).

## What run 7 adds (session 6)

The driver (`proof_tester.py`, `EXPECTED_VERSION = "1.0.0rc7"`) now reads the bundle's preview first, keeps
the member list, checks the session-6 data rule (parquet only under `project/…/tables/` and `…/metrics_*/`,
the only `.pt` a run's `best.pt`, never an image / CSV / `last.pt`) and requires `files.json`; a checkpoint
can only appear once a submission exists, which the tester environment never makes. The checkpoint rule
itself was exercised on the development Hub's export (`../3lc-hub-11/export_bundle_s6.py`, 2026-10-06:
`intel-scene-demo`, two best.pt files of 45,442,563 bytes verified against the ledger, the run record and
the Run; 92,084,847 bytes zipped, 52 members) and in `tests/test_status.py` on a fake project tree.

## What the proof changed in TESTING.md

- §6: on the very first start the compute service seeds its eight stock plugins before the catalog
  lists ours (about a minute); reload the Available page if it is still empty.
- §7.1: the kit download took 70 s on this connection ("about 70 s on a fast connection").
- §7.3: a pool image enters training only with a real label AND weight 1 — the labeling step says so
  (the proof's first draft only set the label; the trainer rightly kept excluding weight-0 rows).
- §7.5: an amber "predicted-class distribution is skewed" note is normal after two epochs.
- §7.7: the Doctor may read the manifest as `cache` (the last fetched copy, the same sha256); the
  worker's 3lc is 3.4.x, not 3.3.x (`3lc>=3.3,<4.0` in the plugin's extra); the plugin home resolves
  by `worker` inside a job and by `cwd` on a plain route — both are the same folder.

## What is still only hand-testable

The browser steps: the Getting Started connection, the Hub's Install button and the plugin page's
rendering, the Dashboard labeling (filter, edit label and weight, Commit), the download prompts,
"Continue to Status", the Status tab's auto-refresh and Copy diagnostics, the Export button. Those are
the items in the session report's hand-test list.
