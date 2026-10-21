# HANDOVER.md: the Kaggle Classification plugin 1.0.0, handed to Gudbrand (2026-10-21)

Written for the promotion to prod and the launch. Everything here is current as of the 1.0.0 release;
the rationale behind the decisions is in docs/PLAN.md, the test evidence in TESTING_PROOF.md.

## 1. What 1.0.0 is

- **The release:** tag `v1.0.0` (commit `f4ce7cb`), the final release, no suffix. The catalog hubs consume
  is `https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/HEAD/catalog.json`
  (its newest entry is 1.0.0, pinned to the tag). The tester kit (TESTING.md, `tester/start_tester.ps1`
  and `.sh` on the default branch) defaults to that catalog.
- **What it does:** one plugin for the 3LC Hub that runs the image-classification hackathon end to end on a
  participant's own machine: Import (the starter kit from the CDN, verified per file, registered as the
  `intel-scene_train` and `intel-scene_val` tables with the project URL alias
  `<KAGGLE_INTEL_SCENE_V1>`), Train (torchvision resnet18 from random init with the kit's MLP head, the
  locked recipe, the per-sample metrics and embeddings written back to the Run for the labeling loop),
  Predict + Submit (the plugin-run-only checkpoint gate, the val check, the validated `submission.csv`, the
  Kaggle upload with the read-back), Status (Runs, History with Kaggle's scores, the Doctor, the
  verification bundle for the organizers).
- **Hosts and pins:** 3lc-compute 1.1.0 or newer (`min_service_version`), SDK window `>=0.3.1,<0.4.0`; the
  plugin's own venv pins `3lc==3.4.1`, `torch==2.14.0`, `torchvision==0.29.0` (the CUDA 13.0 index on
  Windows and Linux, PyPI on macOS), `umap-learn==0.5.12`. Participants keep the exact 3lc pin; a staff host
  on a 3lc dev build overrides it with `UV_OVERRIDE` (docs/PLAN.md §A, tlc row; `3lc-hub-11/SETUP.md`).
- **The competition manifest:** served from the CDN tier the compute service is pointed at
  (`KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL`; the start scripts default to the dev tier, production builds
  resolve `https://competitions.3lc.ai`), cached locally, bundled as the last resort. The 1.0.0 manifest
  (3,258 bytes, sha256 `383cab9a0e8630015a05bdd4532ae2e78f192d2835b7b89bab92e70a5604754e`) carries the
  kit fallback fields: `kit_fallback_url` (the Google Drive folder), `kit_fallback_file`
  (`intel-scene-v1-starter_kit.zip`), `kit_fallback_sha256`
  (`ae4e5937fb5e11c5773790677921f6cfcea1e725c20e1b7fe48e4b843d9b5015`), `kit_fallback_bytes`
  (112,270,148).
- **Where things live on a participant's machine:** the plugin state under
  `<compute home>/plugin-state/kaggle-classification` (the kit, the records, `ledger.jsonl`, the
  predictions, the bundles); tables and runs under the 3LC project root the services run with (the tester
  scripts use `<tester folder>\projects`).

## 2. What was tested

- **The unit suite:** 435 tests (`uv run pytest` in the heavy venv), a divergence guard first: version and
  description parity, the SDK window against the latest 3lc-compute, the wheel contents, import weight,
  licence headers, the manifest-literal census (no competition constant in code), the copy rules.
- **The hardening campaign (TESTING_PROOF.md run 14, 2026-10-12 to 13, docs/TEST_PLAN.md sections A to H,
  records in docs/campaign/):** the environment matrix (upgrade with state, same-version reinstall,
  uninstall and reinstall, service restart, CPU-only, Python 3.11, a host on 3lc 3.3.0, a folder with spaces
  and non-ASCII characters), the kit and import (wrong and nested folders, a file modified after verify, a
  kit deleted after import, an interrupted download resumed, a corrupted shard replaced, re-import fresh,
  two imports at once, a manual kit with the CDN blocked, a renamed project), training (every bound on every
  field, cancel, the service killed mid-run, the browser closed and reopened, two runs queued, Dashboard
  edits with Use latest, 1 and 50 epochs, unicode run names), predict and submit (cancelled, interrupted and
  running runs, a checkpoint modified on disk, a CSV edited by hand, bad and rejected Kaggle tokens, five
  back-to-back submissions), the bundle and the Doctor after every state, and the shared-resource load on
  the CDN (50 concurrent kit downloads, 5.69 GB in 51 s, every sha256 matched). Two plugin defects found and
  fixed; one host defect noted (below).
- **The memory soaks:** the 3-hour soak found a leak (about 15 MB per train-and-predict cycle, UMAP's
  per-fit compile and torch's per-thread state); after the two fixes the release gate ran 30 cycles and the
  evidence soak on the rc14 tag 100 cycles: 0.24 MB per cycle over cycles 15 to 100, 0 failures, training
  16 s and predict 4 s per cycle on the reference GPU.
- **The rc14 tester round:** 3LC staff on Windows (Gudbrand) and macOS (Paul); Gudbrand's round produced
  the kit URL alias and the cloud run folder's local checkpoint copy (merged for 1.0.0) and the first-open
  loading state (fixed for 1.0.0).
- **The error-visibility audit (session 16, for 1.0.0):** no job ends silently (a per-kind job note, failed
  records for refusals before a stage's own record, the Predict reconcile), every failure says what to do
  next, the Doctor's red rows are named on the Import tab at first open, every wait says how long it takes.
- **The 1.0.0 release check (TESTING_PROOF.md run 15, 2026-10-21):** `3lc-hub-fresh` rebuilt from scratch
  with the new start script and its default root; 1.0.0 installed from the hosted catalog (65 s), the kit
  from the dev CDN (44 s), import (10 s, 19/19 checks, aliased rows), train (2 epochs, 68 s on CUDA, 9/9
  provenance checks), predict (10 s, 10/10 checks, the val score reproduced), the Doctor with every row
  named (the manifest `383cab9a…` remote, the Drive link in Kit fallback, the GPU's kernels and driver),
  the bundle with best.pt verified against the record and the Run; the Dashboard and the Hub dataset
  overview render the images through the alias; one submission (#57489771, public 0.50666) with Status
  correct.
- **The reference machine:** Windows 11, NVIDIA GeForce RTX 5070 Ti (compute capability 12.0, sm_120
  kernels in the cu130 build), driver 617.42.

## 3. Known limitations

- **A plugin worker outlives its compute service** (3lc-compute 1.1.0, the Hub bug note's fourth finding):
  after a service is force-stopped, the orphaned worker holds the managed venv's `python.exe`, and the next
  Update or reinstall fails with "Access is denied" until the worker is stopped. The tester scripts' `-Stop`
  stops the folder's workers; TESTING.md tells testers to restart the service.
- **Path length on Windows:** a tester folder about 180 characters deep fails at service start (3lc's
  native module, the 260-character limit). TESTING.md prescribes `C:\3lc-tester` and says why.
- **Kaggle's submission list answers 403 until the competition launches**, so the organizers'
  ledger-to-Kaggle check (the anti-cheat review, docs/PREDICT_MIRROR.md section 15) is unexercised before
  launch day: docs/LAUNCH_CHECKLIST.md.
- **The submission message limit** is the plugin's default of 255 characters; Kaggle documents none (a
  manifest can set `submission.message_max_chars`).
- **Not tested on this side:** OneDrive-synced folders, macOS (mps), Safari, a second Windows account; a GPU
  outside the kernel list and a driver older than R580 are covered by unit simulation and the Doctor
  fixture only (the Doctor names both and the runs use CPU).
- **The alias on a cloud project root:** the import persists the kit's local path as the project alias;
  on an S3 root every reader of the project sees that path (Gudbrand's godfire setup). Participants are on
  local roots. The checkpoint mirror a cloud run folder keeps under the plugin home is never cleaned; the
  bundle carries no alias file (organizers see the token in the rows, never the images).
- **Remote compute nodes:** `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL` is not forwarded to a node worker
  (the host does not pass it, the node agent filters non-`TLC_*` names), so a remote node resolves the
  production tier; on a large remote root the index waits of the table picker and the Train preflight stay
  5 s (the settings load is capped at 1 s). Released SDKs (0.3.3, 0.4.0) call `Url.write_file` /
  `copy_to`, which staged 3lc 3.5 cores no longer have: an SDK release is needed before 3lc 3.5.
- **Copy left for later:** "shard" in the download copy; the expectation lines stay visible while a job
  runs; the Train result banner shows the raw checkpoint path; a Kaggle rejection gets the generic next
  step; the Hub's own "Loading plugin" text while a reaped worker restarts (the host reaps an idle worker
  after five minutes; the first open afterwards takes about 10 seconds and says so).

## 4. Promotion to prod (docs/PROMOTION.md section 5)

Two tiers serve byte-identical objects; promotion is a copy of keys from `3lc-competitions-dev` to the prod
bucket with their metadata, then an invalidation, then a verification.

1. Copy these keys from `3lc-competitions-dev` to the prod bucket **with their metadata**
   (`aws s3 cp --metadata-directive COPY`, or the console's copy, which preserves metadata):

   | Key | Content-Type | Cache-Control |
   |---|---|---|
   | `kaggle/classification-index.json` (203 B, `e0e9c4bc…f262d`) | `application/json` | `max-age=60` |
   | `kaggle/intel-scene/manifest.json` (3,258 B, `383cab9a0e86…`) | `application/json` | `max-age=60` |
   | `kaggle/intel-scene/starter-kit/v1/intel-scene-v1-00.zip` to `-04.zip` (five shards, sha256s in docs/PROMOTION.md) | `application/zip` | `public, max-age=31536000, immutable` |

   **The manifest to promote is the 1.0.0 document** (`383cab9a…`, on dev since 2026-10-21). The older
   `ec0c60cf…` document has no kit fallback fields and must not reach prod. Never overwrite an object under
   `starter-kit/v1/`.
2. Create a CloudFront invalidation on the prod distribution with exactly these paths:

   ```
   /kaggle/classification-index.json
   /kaggle/intel-scene/manifest.json
   ```

3. Verify what prod serves (from the plugin repo):

   ```powershell
   uv run python tools/verify_cdn.py https://competitions.3lc.ai
   ```

   The tool fetches the index, the manifest and every shard, checks each shard's sha256 and bytes against
   the manifest, and prints the served `Content-Type`, `Cache-Control` and `X-Cache` per object.
4. **The fallback-link check** (the one line that proves the right manifest is served):

   ```powershell
   (Invoke-RestMethod https://competitions.3lc.ai/kaggle/intel-scene/manifest.json).kit_fallback_url
   ```

   It must print `https://drive.google.com/drive/folders/1mUleAk1PZtxyCooUVH-n6JfGEKn7Uc8N?usp=sharing`.
   Anything else (empty, or an error) means the old manifest is still served: re-copy the 1.0.0 manifest
   and invalidate again. A plugin pointed at prod then shows the same link in the Doctor's Kit fallback row.

Timeline (Rishikesh, 2026-10-21): handover 2026-10-27, prod verified by 2026-10-31, the hackathon in the
second week of November 2026.

## 5. The launch checklist (docs/LAUNCH_CHECKLIST.md)

Before launch, on a tester environment with 1.0.0 from the hosted catalog pointed at prod: the production
manifest resolves (the Doctor's Manifest row `remote` with the production URL) and the Kit fallback row shows
the Drive link. On the day the competition opens, in order, recording each result with its date in
TESTING_PROOF.md: the Kaggle submission list check for the organizers (every `submit` entry's `kaggle.ref` in
a bundle's `ledger.jsonl` appears in Kaggle's list for the account, and one submission made by hand on
kaggle.com shows in Kaggle's list with no ledger entry: that unmatched row is what organizers look for),
the Status tab's live section (`connected: true`, submissions, the best public score and the rank), the
leaderboard read, the daily cap from Kaggle's side, the "competition not joined" card, the manifest tier
and the kit fallback on prod.

## 6. Open item for 1.0.1

- **A run name with a character tlc forbids** (`< > / \ | , . : " ' ? *`, a dot the likely one) was refused
  by tlc inside the job on 1.0.0, after the tables were opened; the record is failed with the reason and
  Predict refuses the run, so nothing is silent, but the participant loses a minute. Fixed on `develop`
  (commit `47e7625`): the name is checked before the run starts, with the characters named and the fix in
  the sentence. It ships with whatever the testers find next; a 1.0.1 tag follows RELEASING.md.
- **PR 1** (`3lc>=3.4.1,<4`) stays open and unmerged: participants keep the exact pin; the staff path is
  `UV_OVERRIDE`. Gudbrand's uv error text from the dev host would close it either way.

## 7. Where to look

| Need | Place |
|---|---|
| Every decision and contract | docs/PLAN.md |
| Each version's changes | CHANGELOG.md |
| The tester kit and its proof | TESTING.md, TESTING_PROOF.md (runs 1 to 15) |
| The campaign plan and records | docs/TEST_PLAN.md, docs/campaign/ |
| The CDN layout, upload and promotion | docs/PROMOTION.md |
| Launch day | docs/LAUNCH_CHECKLIST.md |
| The anti-cheat review | docs/PREDICT_MIRROR.md section 15 |
| Releasing a version | RELEASING.md |
| The shared vocabulary and the session history | CONTEXT.md |
