# PREDICT_MIRROR.md — the Predict + Submit tab mirrors 3lc-compute-plugin-kaggle v1.2.15

Session 4, Phase 0 (2026-09-30): study only, no code. The governing rule is
`docs/EXDARK_MIRROR.md`: the Predict + Submit tab mirrors ExDark's **control for control, in
functionality and looks**. Allowed differences: (1) competition identity, (2) dataset, (3) competition
rules (classification CSV `image_id, prediction, confidence`; accuracy; 100 submissions/day),
(4) task-inherent (classification inference instead of YOLO). Anything else that would differ is listed
under **§9 D (needs decision)** with ExDark's version and the reason it cannot be copied; nothing there
is designed here. The Kaggle sandbox slug is not known: `<sandbox-slug>` stands for it throughout.

References (read-only): `../3lc-compute-plugin-kaggle/src/tlc_plugin_kaggle/ui.html` (Predict + Submit
markup 1060–1188, JS 5259–6184, fixtures 4031–4236, shared helpers `kgStartJob` 1282, `kgConn` 1601,
`kgBindTablePicker` 1805, `kgApplyDerivedUrls` 1955, `renderPipeline` 2255), `predictor.py` (1008 lines),
`routes.py` (predict/submit/runs/kaggle/download routes 237–407), `jobs.py`, `config_store.py`,
`constants.py`, `docs/ui-notes.md` §1, §2, §5, §10, §11, §13, §14, §18; `tests/test_host_weights_gate.py`,
`tests/test_predict_submit_state.py`. Ours: `trainer.py` (`run_summary` 604, `train_state` 819,
`save_checkpoint` 943, `check_provenance` 1181, the Run parameters 1508–1616), `manifests/intel-scene-v1.yaml`,
`kit.py`, `importer.py` (the import record 852–867), `session.py` (`_ALLOWED_TABS`), `docs/TRAIN_MIRROR.md`
§7–§8 (the train record, D10, D12), `docs/PLAN.md` §A, §C, §D. The kit: `../hackathon_starter/intel-kit/
{predict.py, train.py, README.md}`, `../hackathon_starter/kaggle_content/{Evaluation, Rules, Overview, FAQs}.md`,
`../hackathon_starter/metric-template-650e8c.ipynb` (the HackNova metric). The installed Kaggle client:
`kaggle 2.2.4` + `kagglesdk 0.1.37` in this repo's `.venv` (`kaggle_api_extended.py`, `kaggle_env.py`,
`competition_api_service.py`). No answer-key file was opened for this study.

Legend: **V** ported verbatim · **A(n)** adapted under allowed difference n (how) · **X** dropped (why) ·
**D** needs decision (§9) · **+brief** required by the session-4 brief, no ExDark equivalent.

Three things ExDark's tab does **not** have, which the brief expected: a *Technical details* disclosure
(the only disclosure is *Show log*), a *cancel* control for predict or submit, and a *submission history*
table (history lives on the Status tab, session 6; §1 #46 records it for reference).

## 1. Predict + Submit tab, in render order

| # | ExDark element (ui.html line) | Ours | Mark |
|---|---|---|---|
| 1 | Tab 3 in the stepper: "3 Predict + Submit" · "Predict → CSV → Kaggle" · state glyph `#kg-state-submit` (629); `renderPipeline` marks submit done when a `kaggle_submit` record has `facts.submission.status == "submitted"` (2255; `GET /pipeline` routes 492) | already in the shell (EXDARK_MIRROR #5–6); done = a **submit record** with status `submitted` (from the session store, not a job store) | V / A(plumbing) |
| 2 | Loop banner's `submit` link (`data-goto="submit"`, 603); `psLoadRuns` rewrites the `inspect` href to the newest run with a `run_url` (5427) | verbatim; inspect → newest run in `train_state.runs` with a `run_url` | V |
| 3 | Header chips "YOLOv11n · COCO-pretrained · 640px" · "12 classes" · "Scored by mAP@0.5" (596) | already rendered from the manifest (EXDARK_MIRROR #3): "resnet18 · from scratch · 150px" · "6 classes" · "Scored by accuracy" | V (done) |
| 4 | `.card-header`: `plugin-section-number` 3 · "Predict + Submit"; subtitle "Inference on the test table (`<imgsz>`px, locked) becomes a validated `submission.csv`; submitting it to Kaggle spends one of your daily attempts." (`.kg-contract-imgsz` from `_meta.contract.imgsz`, no fallback literal) (1062) | same geometry; "Inference on the kit's test images (150px, locked) becomes a validated `submission.csv`; submitting it to Kaggle spends one of your daily attempts." — "test table" → "test images": the test split is never a table (PLAN §A Splits); the size comes from `model.image_size` | A(2) |
| 5 | `#ps-conn-banner` (1070): the connection guard's slot | verbatim (`conn` already owns a slot per tab) | V |
| 6 | `.kg-sec-head` "Step 1 · Predict" (1075) | verbatim | V |
| 7 | `#ps1-banner` failure slot (1077) | verbatim | V |
| 8 | **Weights source** segmented toggle `#ps-src-seg` "Plugin run" / "Weights file" — **host-only**: revealed only when `/config` `_meta.host` is true (`is_host()` = `HOST_DIR/metric_exdark.py` + `HOST_DIR/solution.csv` exist, `TLC_KAGGLE_HOST_DIR` or `~/.3lc-kaggle-plugin/host`); help "Predictions come from runs trained in this plugin, so every submission carries verified provenance." (1085) | help text verbatim. The toggle and the host mode itself are **D1**: our host has no `metric_*.py`/answer key pair to detect, and PLAN §A says "Only from plugin-created runs" without a host exception. If kept: `KAGGLE_CLASSIFICATION_HOST_DIR` holding the metric module + `solution_kit_v1.csv`, never shipped | D1 |
| 9 | **Run picker** `<select id="ps-run">` + refresh `#ps-run-refresh` ("Refresh runs", spins); `GET /runs` (routes 237): train jobs newest first; placeholder "Select a run…" / "No plugin runs yet. Train first (tab 2)."; option text `<run_name> · <summary>`; summary = [`from-scratch · legacy` \| `provenance unverified`] · "N epochs" · "best mAP50 0.7241" · "Sep 29"; unusable rows disabled with the server `reason` ("still training", "failed: …", "no best.pt saved", "best.pt missing on disk"); default = previous value if still usable, else first usable (1093, 5398) | same control, fed by the **train record's `runs[]`** (`run_summary`: `id, run_name, run_url, status, epochs_completed, best_epoch, best_val_accuracy, weights, best_checkpoint_sha256, provenance_ok, contract, device_label`) through a new `GET /runs` that adds `usable` + `reason` the way ExDark's route does. Summary "N epochs · best val accuracy 57.58 % · Sep 29"; `legacy` tag dropped (no legacy contract here). Reasons: "still training" · "failed: …" · "interrupted" (our `stale`) · "no best checkpoint saved" · "best checkpoint missing on disk" · **"best checkpoint changed on disk (sha256 mismatch)"** (§2, ours). A `cancelled` run with a best checkpoint is usable (D10 of TRAIN_MIRROR) | A(3)(4) |
| 10 | `#ps-run-note` (5380): ok "Verified provenance on this run’s record · init sha256 `<12>`…" / "Verified provenance on this run’s record." / legacy variant; warn "This run has no recorded provenance checks." | "Verified provenance on this run’s record · best checkpoint sha256 `<12>`…" — the hash shown is the trained checkpoint's (D12), not an init hash (we have none). The warn line verbatim; whether an unverified run may still predict is **D2** (ExDark: allowed, only the note warns) | A(4) · D2 |
| 11 | **Weights file** `#ps-src-weights` (host-only): text input placeholder `C:\...\weights\best.pt`, warn "External weights: provenance cannot be verified by the plugin. Path as seen by the compute service's machine." (1101) | part of D1; if kept, placeholder `C:\...\model\best.pt` and the warn verbatim; participants never see it and the server refuses it (§2) | D1 |
| 12 | **Test table URL** `#ps-test-url` + revision picker `#ps-test-pick` (`kgBindTablePicker`, `GET /tables/list`, filtered to `<prefix>_test`) + derived default (`GET /tables/defaults`, override via `kgSetUrlOverride`) + gate `#ps-table-gate` (`GET /import/revisions?url=`): idle "Run Import (tab 1) to create the test table, or paste its URL from the Dashboard." · "Checking table…" · ok "Table verified: exdark_test/initial" · wrong split (DP-11) · missing "This table isn’t on disk yet." + "Looking for the test table in project `x`…" + **Go to Import** (1110, 5462–5543) | **no test table exists** (PLAN §A: test is flat, never registered). The field, picker and derived-URL chain are dropped; the gate stays as a **locked row + gate** over the kit's test directory from the import record (`kit_dir`): idle "Run Import (tab 1) to download the starter kit." · "Checking test images…" · ok "Test images verified: 1,800 files match the kit’s files.json" · missing "The kit’s test images aren’t on disk." + "Looking under `<kit_dir>`. If you moved or deleted the kit, Import (tab 1) restores it." + Go to Import. The per-file sha256 check against `files.json` is what replaces "revision identity" here (the inputs a submission was predicted from are provable). Copy and the files.json check are proposals | A(2) · **D3** |
| 13 | **Conf threshold** `#ps-conf` (0.25, 0–1, step 0.01; blur validation "conf must be between 0 and 1 (got x)."; persisted as `submit.conf`) (1120, 5568) | dropped: a detection cutoff; classification has no threshold (argmax). Nothing replaces it; the two-column row keeps **Device** alone on the right unless D3's locked row takes the left column | X A(4) · D3 (geometry) |
| 14 | **Device** `#ps-device` (placeholder `auto`, help "Blank = auto (CUDA → `mps` → CPU).", the session device mirrored with Train's field on blur) (1127, 2558) | verbatim (`session.device`, `resolve_device` already ported) | V |
| 15 | Action row: `#ps-predict-btn` `btn-primary btn-lg` play + "Run inference" → refresh + "Re-run inference" after a terminal state · `#ps-spinner` · `#ps-state` live text (1135); enable rule `!predicting && gate.ok && conf valid && weightsChosen && !kgDevMode` (5590) | verbatim; rule = `!predicting && gate.ok && run chosen && !kgDevMode` | V |
| 16 | Start click (5797): validate conf → `saveTabConfig('submit')` (`POST /config {submit:{conf}}`) → body `{train_job_id \| weights_path, test_table_url, conf, device}` → `kgStartJob('predict')` = `POST /validate/predict` then host `/run` → placeholder progress → 1 s poll; abort/adopt copy verbatim from `kgStartJob` | same through our `kgStartJob` (no `/validate`, EXDARK_MIRROR #36: `run_predict` re-validates); body `{train_job_id, device}` only — the checkpoint path is **never posted** (§2), the test dir comes from the import record server-side; nothing to persist per tab (no conf) | A(plumbing) |
| 17 | Poll `psPollPredict` (5735) every 1 s on `GET /jobs/<id>`: running → progress + title "⟳ N% —" + "Predicting… (safe to navigate away)"; completed → results, "Re-run inference", basis set, scroll; failed/stale → partial checks + fail banner; network → "Connection lost. Inference continues on the host." | verbatim via the job channel (`PluginJobs.track` events → the record shape, as Import/Train); `stale` from the durable **predict record** (§6) with the Train pid rule | V / A(plumbing) |
| 18 | Progress block `psRenderProgress` (5606): `.kg-run-head` "Inference" · badge running · "Inference: 312 / 715 images" (or "Starting…") · "elapsed 41s" · determinate bar · quartile announcer "Inference 50% complete"; no ETA; server progress `{images, total_images}` flushed ≤ 1/s | verbatim; "Inference: n / 1,800 images"; no ETA (V — ExDark has none here) | V |
| 19 | **No cancel** for predict (the predictor never reads the cancel flag; no button) | mirror: no cancel button. Whether the job should at least honour a host-side cancel between batches is plumbing (ours would), the control stays absent | V · D4 |
| 20 | Results panel `psRenderResults` (5707) ① verdict + checks `psRenderChecks`: `.kg-verdict` "5/5 format checks passed" / "(k failed)", grid group "Submission format", ✓/✗ rows `label (detail)`, cascade on first live render, no per-check remedies (the server message is the remedy) | verbatim renderer; the check list is §3 (columns · exactly 1,800 rows · no duplicated image_id · image_id set and order equal to sample_submission.csv · prediction an integer class id 0–5 · confidence finite in [0, 1] · no missing values). On failure the CSV is kept as `submission_<stamp>.INVALID.csv` (V) | A(3) |
| 21 | ② `.kg-results-row` › **Prediction sanity** card (`psSanityHtml` 5657): stats "boxes 2,706" · "mean/image 3.78" · "empty images 41"; class tags `<Class> · <count>` for 12 classes; warn (mean < 1.0) "… boxes across … images is unusually low. Are these fully-trained weights? (Submitting is still fine.)" | same card and tag component; stats and heuristic are task-inherent and **not designed here**: candidate stats "images 1,800 · mean confidence 0.83 · low confidence (< 0.5) 212", tags = predicted count per class (6), warn when a class receives 0 predictions or one class exceeds half of all rows, same closing sentence | A(4) · **D5** |
| 22 | ② **Hero stat** `psHeroHtml` (5683), only when `local_score` is a number: "0.6402" · "Local mAP@0.5" · "Host-only preview, not the leaderboard." | the metric is accuracy (A(3)); *what* is scored locally and for *whom* is §5 / **D6**: ExDark scores the TEST split on organizer machines only; the brief proposes val accuracy of the chosen checkpoint on the locked val table, for everyone | A(3) · D6 |
| 23 | ③ CSV row `psCsvRowHtml` (5690): bold `submission_<stamp>.csv` · mid-truncated path (`title` = full) · ghost Copy ("Copy CSV path (a path on the compute service's machine — Download CSV works from anywhere)") · secondary **Download CSV** (`data-download=<job id>`) | verbatim (`pathCell` + Copy exist); path under `storage.plugin_home()/predictions/<run_name>/` (never `~`; ExDark writes `~/.3lc-kaggle-plugin/runs/<run>/submissions/`) | V / A(plumbing) |
| 24 | Failure banner `psRenderPredictFail` (5725): `alert-error` "Inference failed" + message + **Copy diagnostics** (`psBuildDiagnostics`: version/os/time, run, weights, init sha256, contract, conf, predict job, submit job, `[format checks]`, `[sanity]`, `[submission]`, `[log tail]` 40 lines); form stays, CTA "Re-run inference" | verbatim; diagnostics lines: run, run URL, checkpoint sha256 (recorded / on disk), contract, device, predict job, submit job, checks, sanity, submission, log tail | V |
| 25 | `<hr class="divider">` (1145) | verbatim | V |
| 26 | `.kg-sec-head` "Step 2 · Submit to Kaggle" (1150) | verbatim | V |
| 27 | `#ps-basis` `psRenderBasis` (5844): help "Run inference first. A validated CSV unlocks this step." / ok "Submitting: **run** · predicted 3 minutes ago · local mAP@0.5 0.6402 (from a previous session)" | verbatim; the score fragment follows D6 ("· val accuracy 57.58 %") | V · D6 |
| 28 | `#ps-step2-body` `.kg-step-muted` (opacity .55, no pointer events) + real `disabled` on slug and message until a basis exists (`psSetBasis` 5835) | verbatim | V |
| 29 | **Kaggle connection card** `#ps-kconn` (`renderConnection` 5884, `GET /kaggle/connection?slug=`): `no_credentials` err "**Kaggle account not connected.**" + server help (token instructions, PowerShell/POSIX blocks, `KAGGLE_API_TOKEN`, legacy kaggle.json, "The generated submission.csv is saved locally, so you can always upload it manually…") — **replaced in rc10 by the guided connect form, D13 (§13)** · `not_joined` warn "**Join the competition on Kaggle first.** Your account (**user**) has not accepted the competition rules yet; submissions would be rejected." + **Open competition page** · `ready` ok "**Connected to Kaggle** as **user** · 2 of 3 submissions left today" / "· 3 submissions/day" (+ "Competition probe failed: …") · exhausted → info "Daily limit reached, resets 00:00 UTC. Your CSV is saved and ready." · fetch error "Could not check the Kaggle connection: …". **The plugin never takes or stores a token**; credentials are detected on the compute host (§7) | verbatim, including never touching the token. Numbers: `daily_limit` comes live from Kaggle's `max_daily_submissions`; ours also has `manifest.submission.daily_limit` (100) — precedence and the fallback copy are **D7** | V · D7 |
| 30 | **Competition slug** `#ps-slug` (placeholder "loading default…"; value = `session.slug_override \|\| _meta.default_slug`; blur: blank → default, equal → stores null) (1160, 2574) | the slug is competition identity → from `manifest.competition.slug` (`<sandbox-slug>` until it exists). ExDark's field is an editable override; ours would be a locked row (the manifest is the single source, an override can submit to the wrong competition) — **D8** | A(1) · D8 |
| 31 | **Submission message** `#ps-message` (placeholder "`<run_name>` via 3LC plugin", help "Shows in your Kaggle submission history. Blank uses "<run name> via 3LC plugin".") (1165) | verbatim | V |
| 32 | Action row: `#ps-submit-btn` upload + "Submit to Kaggle" · spinner · `#ps-submit-state`; enable rule `!submitting && step2 && basis && !connBlocked && !budgetExhausted && !kgDevMode` (5876) | verbatim | V |
| 33 | Submit click (6042): `window.confirm('This spends 1 of your <limit\|\|3> daily submissions. Submit "<run>" now?')` → `saveTabConfig('submit')` → `kgStartJob('kaggle_submit', {predict_job_id, message, competition_slug})` (`POST /validate/submit`: "Missing required field 'predict_job_id'" / "No validated prediction CSV found for that job. Run inference first.") → 1.5 s poll | verbatim confirm with the D7 number ("1 of your 100 daily submissions"); body `{predict_job_id, message}` — the slug is never posted (D8) | V · A(1) |
| 34 | Poll `psPollSubmit` (5994): "Submitting…"; terminal reads `facts.submission`: `submitted` → success banner · other soft status → soft callout · else fail banner; then `renderPipeline()` + `renderConnection()` ("budget just changed"); network "Connection lost. The submission continues on the host." | verbatim via the job channel and the **submit record** | V |
| 35 | Success banner `psRenderSubmitSuccess` (5947): `alert-success` drawn check · "Submitted to Kaggle" · "Submission `<ref>`" · **Continue to Status** · **View on Kaggle** (`<competition_url>/submissions`) | verbatim; link from the manifest slug | V · A(1) |
| 36 | Soft outcomes `psRenderSubmitSoft` (5980): `limit_reached` info / `not_joined`, `skipped` warn; text = server `reason`; + **Download CSV**. Server reasons: "Daily submission limit reached (N/day), resets midnight UTC. Your CSV is saved and validated. Submit it tomorrow from here, or upload it manually on the competition's Submit page." · "Join the competition on Kaggle first (accept the rules on the competition page), then submit again: `<url>`" · "Kaggle rejected the submission because the competition rules are not accepted yet. Join here, then submit again: `<url>`" | verbatim | V |
| 37 | Fail banner `psRenderSubmitFail` (5969): `alert-error` "Submission failed" · "Kaggle rejected the submission: `<exc>`" · Copy diagnostics | verbatim | V |
| 38 | `#ps-banner` slot (1172) | verbatim | V |
| 39 | **Show log** accordion `#ps-log-details` (1174), hidden until the first poll, auto-scroll | verbatim, filled from `log_line`; no log on revisit (our host keeps none) | V / A(plumbing) |
| 40 | **Download CSV** `downloadCsv` (5289): `GET /submissions/<job_id>/download` via `authFetch` → blob → `<a download>`; filename from `Content-Disposition` (fallback `submission.csv`); `alert('Download failed: …')`; server 404s "No such job" / "has no submission CSV on disk." | verbatim; route ported; the job id resolves through the predict record | V |
| 41 | Revisit `psRenderRevisit` (6096): form hidden; static results; prepended ok "Last prediction: **run** · predicted 2 hours ago" + ghost **New prediction** (`psEnterFormState`: clears, shows the form with `kglEnter`, re-mutes step 2); `#ps-banner` ok "Submitted to Kaggle · submission `<ref>` · 2 hours ago" or info "Last submit attempt: `<status>`. `<reason>`"; step 2 unlocked with "(from a previous session)" | verbatim | V |
| 42 | Tab-open resolution `psInitSubmitTab` (6147): `GET /jobs?kind=predict` ∥ `?kind=kaggle_submit` → a running predict reconnects (1 s poll, returns) · a running submit reconnects (1.5 s, falls through) → `GET /submit/state` (`predict_submit_state`: `empty` unless the `predict_state` snapshot's CSV exists **and** its job record still exists in the 50-record store; `submission` attached when `submit_state.predict_job_id == predict_state.job_id`) → `predicted` / `submitted` → revisit, else form | same precedence from the durable **predict record + submit record** in the session store (`_ALLOWED_TABS` already lists `predict_state`, `submit_state`) plus `PluginJobs.list` for the running case; `GET /submit/state` re-verifies the CSV on disk (`empty / predicted / submitted / stale`) — no 50-record pruning to key on | A(plumbing) |
| 43 | Tab enter `psOnTabEnter` (5449): `psLoadRuns()` · `renderConnection()` · derived URLs · `psVerifyTable(true)` | `psLoadRuns()` · `renderConnection()` · the test-images gate (D3) | V |
| 44 | Session projection: `CFG_FIELDS.submit = [conf]`; `slug_override` and `device` are session keys (2060, 1930) | `device` only; `conf` gone (X #13), `slug_override` gone (D8): nothing per-tab to persist | A |
| 45 | `?kgdev` fixtures `psDevForce` (4034): `submit-state1`, `submit-participant`, `predict-legacy-run`, `submit-gated`, `submit-inference`, `submit-results`, `submit-results-low`, `submit-checks-fail`, `submit-nokaggle`, `submit-limit`, `submit-success`, `submit-fail`, `submit-revisit`; kgdev disables the two job buttons with "demo state — actions disabled" | ported with our names and numbers derived from the served manifest (1,800 rows; 6 classes); `predict-legacy-run` dropped; added `submit-checkpoint-changed` (the sha mismatch reason) and, per D6, `submit-results-val` | A · +D6 |
| 46 | *(Status tab, session 6, reference)* history `renderHistoryFrom` (6303): columns Run · When · Local mAP@0.5 · Δ · Outcome · Actions; outcome vocabulary "Submitted · #ref", "CSV generated (daily limit reached)", "CSV generated (not joined on Kaggle)", "Submission rejected", "CSV generated (not submitted)", "Validation failed", "Interrupted", "Running…"; budget hero "2 of 3 today / Resets 00:00 UTC." or "3/day limit / Per-day counter unavailable on this competition." | session 6; the predict and submit records this session writes must carry every column (§6) | plumbing |

## 2. Which checkpoint, and the plugin-run-only guarantee

**ExDark.** `predictor.resolve_weights(params)` (107–167) is "THE single definition of the plugin-run-only
policy — routes and the job both call it": (1) `train_job_id` **wins** — the weights path comes from that
job record's `facts.weights` and any supplied `weights_path` is *ignored*, not merely out-ranked (a
participant could otherwise POST a real id plus an arbitrary path straight to the host's `/run`);
(2) a bare `weights_path` without an id is **host-only** (`is_host()`), refused with "Direct weights files
are host-only. Select a run trained in this plugin — predictions must carry verified provenance.";
(3) the file must exist. The route turns a refusal into a 400, the job into `JobFailed` (the host's `/run`
never passes through `/validate`, so the job must re-check — the layer-3 lesson, ui-notes §13; pinned by
`tests/test_host_weights_gate.py`). `GET /runs` adds the display-side reasons (still training / failed /
no best.pt saved / best.pt missing on disk) but `resolve_weights` does not repeat them.

What ExDark does **not** check, verified in the source: the job **kind** (a predict record also sets
`facts.weights`, so a predict job id would resolve), the job **status** or `provenance_ok` (a cancelled
run passes; an unverified run predicts with only the note warning), the **run URL**, and — decisive for
us — **any hash of the trained weights**. The only sha256 in ExDark is `checkpoint_sha256`, the hash of the
official *init* `yolo11n.pt`, recorded as a train fact and on the Run's parameters and re-asserted by the
four provenance checks *at train time*. Predict time never re-verifies anything about `best.pt`. Nothing
about the prediction is hashed either (§6).

**Ours, mapped onto that.** Session 3 already produces everything the stronger rule needs (TRAIN_MIRROR
§7, D12):

| ExDark | Ours (exists today) |
|---|---|
| job record `facts.weights` (`<home>/runs/<name>/weights/best.pt`) | `train_state.runs[i].weights` = `<run>/model/best.pt` (`save_checkpoint`, atomic `.tmp` + `os.replace`), `last.pt` beside it |
| `facts.checkpoint_sha256` = init hash | `runs[i].best_checkpoint_sha256` (+ `last_checkpoint_sha256`), the hash of the file at the moment it became best |
| Run parameter `checkpoint_sha256` | Run parameters `best_checkpoint_sha256`, `best_checkpoint` (`model/best.pt`), `best_epoch`, `best_val_accuracy`, `epochs_completed`, plus the contract (`backbone, head, arch, image_size, pretrained, torch_version, torchvision_version, seed, epochs, batch_size, lr, weight_decay, optimizer, schedule, train_table_url, val_table_url, usable_rows, plugin, job_id`, the manifest provenance) |
| 4 train-time provenance checks | 9 (`check_provenance`), the ninth being "run records the best checkpoint sha256, equal to the file on disk" |
| `result.run_name`, `facts.run_url` | `runs[i].run_name`, `run_url`, `project_name`, `status`, `provenance_ok`, `contract` |

The predict stage therefore resolves and verifies as follows (plumbing, invisible, "ours" under the
mirror rule): a request carries **only** `train_job_id` (+ device); the checkpoint path is taken from the
train record entry with that id **and status in {completed, cancelled}** (kind is implied: only train
records live there) — a path in the request body is ignored exactly as ExDark ignores it, and a bare path
is refused for participants (host exception = D1); the file must exist; its sha256 is recomputed and must
equal **both** the record's `best_checkpoint_sha256` and the Run's `best_checkpoint_sha256` parameter
(`tlc.Run.from_url(run_url)`), else "The best checkpoint on disk no longer matches the run’s record
(sha256 differs). Re-train, or pick another run." — the third state the run picker shows as unusable;
`run_url` and both hashes are recorded on the predict record and in the ledger (§6). A run whose
`provenance_ok` is false renders ExDark's warn note; whether it may still predict is D2. Refusals are
enforced in the job (`run_predict`), the route being a convenience (EXDARK_MIRROR #36) — the same
two-layer rule ExDark has. The guarantee then reads: a submission CSV can be traced to a checkpoint file
whose hash was recorded by the training job that created it, on a Run whose parameters state the locked
contract and the exact table revisions trained on; no hand-supplied weights ever reach inference on a
participant host.

## 3. The submission format and Kaggle's metric

**From the manifest and the kit.** `submission.columns = [image_id, prediction, confidence]`,
`submission.metric = accuracy`, `splits.test.count = 1800`, `splits.test.ids_from = sample_submission.csv`,
`classes` ids 0–5 in order buildings, forest, glacier, mountain, sea, street. The kit's
`sample_submission.csv` (`tools/build_kit.py write_sample_submission`) is `image_id,prediction,confidence` +
1,800 rows of `<16-hex stem>,0,0.5`, **sorted by stem**; the ids are the salted opaque stems of
`starter_kit/data/test/<stem>.jpg`, verified per file by `files.json`. `solution_kit_v1.csv` (the judge's
copy, 2026-09-30) has exactly that id set (`tools/build_solution.py`). Hence our CSV: header exactly the
manifest columns in order; one row per sample_submission id **in sample_submission order** (so the file
is diffable against the sample); `prediction` = `argmax(softmax)` as the class id integer; `confidence` =
`max(softmax)` in [0, 1]; LF, UTF-8, no index column, no BOM. Precision of `confidence` is D9 (the kit
writes full float repr; ExDark writes its box values at 6 decimals).

**What Kaggle enforces** — the HackNova metric (`metric-template-650e8c.ipynb`, one `score(solution,
submission, row_id_column_name)` cell; Kaggle strips `Usage` before calling and passes `image_id` as the
id column). Checks in order; **PVE** = `ParticipantVisibleError`, which the participant sees as the
submission's error description; anything else is host-only:

| Condition | Outcome |
|---|---|
| id column missing from the submission | PVE "Submission must have a 'image_id' column" |
| duplicate ids | **silently** de-duplicated, first row kept (the Evaluation page says "rejected") |
| `prediction` column missing | PVE "Submission must have a 'prediction' column (0 or 1)" |
| `confidence` column missing | **silently** filled with 0.5 (the page says required) |
| prediction or confidence not a numeric dtype (class names, blanks mixed with numbers) | PVE "Submission column '…' must be numeric" |
| a prediction whose `int(round(float(p)))` is not in the allowed set | PVE "Predictions must be 0 or 1. Found invalid: […]" |
| confidence not finite / outside [0, 1] | PVE "Confidence must be finite" / "Confidence must be in [0, 1]. Found […]" |
| any NaN prediction / confidence | PVE "Missing predictions" / "Missing confidence values" |
| a solution id absent from the submission | PVE "Missing N predictions. First missing: […]" |
| **extra** submission ids | **silently** dropped by the inner merge (the page says "rejected") |
| any row order, extra columns | tolerated |
| float predictions (`3.0`) | tolerated; validation **rounds**, scoring **truncates** (`astype(int)`), so `0.6` passes as 1 and scores as 0 |
| column names in another case (`Prediction`) | rejected (missing column); ids compared exactly, no whitespace stripping |
| empty file | rejected (min/max NaN) |
| accuracy = mean(`label == prediction`) over the inner merge; `confidence` never used | the score |

Two organizer-side facts the plugin cannot fix and the sandbox depends on (listed for Rishikesh, not
decided here, **D10**): the metric template still hard-codes the allowed set `(0, 1)` at both the
submission and the solution check — with the Intel key every label 2–5 raises, so the notebook must
become `range(6)` (and its messages) before the sandbox is created; and `kaggle_content/*.md` are still
the two-class pages (1,184 ids, 592/592, "0 = chihuahua"), while `Rules.md` says **20** submissions per
day against the manifest's **100** (`submission.daily_limit`) — one number has to win before the
connection card's copy is right.

**Our preflight**, stricter than the metric on purpose so a participant never learns a format problem
from Kaggle: the seven checks of §1 #20. Order and detail mirror ExDark's `validate_submission_df`
(first failure raises; the CSV is kept as `.INVALID.csv`; a PASS/FAIL log line per check). The messages
name the remedy the way ExDark's do ("Re-run Import to rebuild the test images, then predict again." for a
row-count or id-set mismatch).

## 4. Inference parity with `intel-kit/predict.py`

| Step | The kit (`predict.py`) | Ours (session 4) | Difference · effect |
|---|---|---|---|
| Image discovery | flat `data/test`, `*.jpg *.jpeg *.png`, de-duplicated by lower-cased filename, sorted by name; `image_id = stem` | ids from the kit's `sample_submission.csv`; each `<kit_dir>/data/test/<id>.jpg` must exist and match `files.json` (D3) | none on the shipped kit (all `.jpg`, one file per id); ours refuses an incomplete kit instead of predicting on what is there |
| Missing / unreadable image | grey 150×150 placeholder, still predicted; ids without an image get `0, 0.5` | **D11**: fail the job (the kit is verified, so this is corruption) vs. mirror the kit's placeholder | a placeholder row is a silent wrong answer; a failure is loud |
| Preprocessing | `Resize(150)` → `CenterCrop(150)` → `ToTensor` → `Normalize(ImageNet)`; `.convert("RGB")` always | identical: `build_transforms(150)[1]` (the val transform, D11 of TRAIN_MIRROR) + RGB convert | none |
| Model | `ResNet18Classifier` redefined in the script: `resnet18(weights=None)`, `fc = Identity`, MLP 512→256→ReLU→Dropout .3→128→ReLU→Dropout .3→6 | `build_model("torchvision_resnet18", "kit_mlp_512_256_128_d03", 6)` — the same module, same creation order | none (parity gate, TRAIN_MIRROR §11) |
| Checkpoint | `torch.load(path)` raw state dict, `load_state_dict(strict=True)`; `weights_only` left to the torch default | `best.pt` is a raw state dict of CPU tensors (`save_checkpoint`); load with `weights_only=True`, strict | none in shape; ours refuses a pickled non-tensor payload |
| Mode | `eval()`, `no_grad()`, no autocast, no TTA, no seed, cudnn defaults | identical; single forward pass (the Train tab's locked "Inference: single forward pass" row) | none |
| Batch | `DataLoader(batch_size=32, shuffle=False, num_workers=0)` | 32 fixed (a plugin constant, not a competition fact); workers = `default_workers()` (0 on Windows) | none on outputs; batch composition does not change per-image logits in eval mode |
| Device | `cuda` if available else `cpu` | `resolve_device(session.device)` (auto → CUDA → MPS → CPU) | none in intent; CUDA vs CPU kernels differ at ~1e-6, which can flip a near-tie argmax on a handful of images — inherent, same for the kit |
| Output | `softmax(dim=1)`, `probs.max(1)` → `int(pred)`, `float(conf)` | identical | none |
| Ordering / fill | follows `sample_submission.csv` order when present, else sorted filenames; missing ids filled `0, 0.5`; extra images silently dropped | sample_submission order, every id required (no fill, no drop) | ours fails where the kit papers over |
| Writing | `csv.DictWriter`, header, float repr; `submission.csv` overwritten + `submissions/submission_<stamp>.csv` | `submission_<stamp>.csv` under the plugin home (never overwritten), 6-decimal confidence (D9) | rounding never changes the argmax or the validity |
| Which weights | `best_model.pth` written once at the end from `best_model_state = model.state_dict().copy()` — a **shallow** copy whose tensors alias the live parameters, so the saved file is in practice the **last** epoch's weights, not the best epoch's | `best.pt` written atomically at the epoch that improved val accuracy (D10/D12) | a kit-side defect, not ours to fix; it means a kit participant predicts from last-epoch weights while a plugin participant predicts from true best — the parity gate compared val accuracy, which the kit reports from its in-memory best, so this did not show there. Worth a note to the kit's owner |

## 5. Local scoring

**ExDark.** `try_local_score` runs only on machines where `HOST_DIR/metric_exdark.py` **and**
`HOST_DIR/solution.csv` exist (organizer machines; `TLC_KAGGLE_HOST_DIR` relocates them). It imports the
metric module and scores the fresh CSV against the hidden **test** key — the leaderboard number itself —
and stores it as `facts.local_score`; the hero reads "Local mAP@0.5 · Host-only preview, not the
leaderboard." and the Status history shows a Δ column. On a participant machine the function returns
`None` silently: there is **no local metric of any kind** for participants; the only local numbers are the
sanity counts, and the run picker's `best mAP50` comes from training, not from predict. The plugin ships
neither the metric nor the key; both are dropped into the host dir by hand.

**Ours (proposal, D6).** The plugin ships no test labels and never will (`solution_kit_v1.csv` stays in
`hackathon_private`). Two candidates, not exclusive:

- **(a) Val accuracy of the chosen checkpoint, for everyone** — the brief's proposal. At predict time,
  run the same single forward pass over the **locked val revision** (`import_state.locked_val`, the URL
  the Run's `val_table_url` parameter names; `val.editable: true` means the *latest* val may carry the
  participant's own label edits, so the locked seed revision is the reproducible choice), 1,200 images,
  a few seconds on GPU. It costs nothing new (the transform, model and table loading exist) and it is a
  **provenance check in disguise**: the recomputed number must equal the run's recorded
  `best_val_accuracy` (the checkpoint *behaves* like the file the record describes — "Checkpoint reproduces
  the recorded val accuracy: 57.58 % recorded · 57.58 % now"). Hero: "57.58 %" · "Val accuracy" · "Your
  locked validation split, not the leaderboard." Honest caveat for the copy: val is the split
  participants tune on, so it is an upper-bound hint, not a leaderboard preview.
- **(b) Host-only test accuracy, the ExDark mirror** — `KAGGLE_CLASSIFICATION_HOST_DIR` with the metric
  module and `solution_kit_v1.csv`; `_meta.host` true → hero "Local accuracy · Host-only preview, not the
  leaderboard." and the Weights-file source (#8/#11). This is PLAN §D's "test-set answer key on the
  organizer machine (local scoring, optional)". It reuses ExDark's exact mechanism and copy.

## 6. The ledger

**ExDark has no ledger.** Its history is the **job store**: one JSON per job under
`~/.3lc-kaggle-plugin/jobs/<id>.json`, *rewritten* atomically on every log/progress/field write,
**pruned to the newest 50** across all kinds, and even a finished predict record is rewritten when the
submit job writes `facts.submission` back onto it. Plus two overwritten snapshots in `ui_config.json`
(`predict_state`, `submit_state`) that key the revisit view, and the CSV files (never pruned).

Recorded per **prediction** (`kind=predict`): `params.train_job_id`, `weights_path` (resolved),
`run_name`, `test_table_url`, `conf`, `device`; facts `run_name, weights, conf, csv_path, sanity,
local_score, submission (written back later)`; `checks`; `progress {images, total_images}`; result
`run_name, weights, csv_path, rows, total_boxes, sanity, conf, local_score, checks`; `created_at`,
`finished_at`. Per **submission** (`kind=kaggle_submit`): `params.predict_job_id, message,
competition_slug`; facts `run_name, csv_path, predict_job_id, submission {status: submitted \| skipped \|
not_joined \| limit_reached \| failed, ref, response, reason, detail}`. **Absent**: a csv sha256, the
checkpoint hash (init hash lives only on the train record), the run URL, the manifest sha / plugin
version, Kaggle's own submission timestamp, the public score. The Status tab reconstructs history from
five `GET /jobs?kind=` calls — and loses it past 50 jobs.

**Ours.** PLAN §A/§C: an **append-only JSON-lines ledger** under `storage.plugin_home()` (session 5 owns
the file and the bundle; session 4 must write entries that already satisfy it), beside the durable
records the tabs read. Per prediction: `ts, kind: predict, job_id, competition_id, kit_version,
manifest_sha256 + manifest_source (the provenance block), plugin_version, train_job_id, run_url,
run_name, checkpoint {path, sha256_recorded, sha256_on_disk}, contract (from the Run), test_inputs
{count, files_json_sha256}, device, csv {path, sha256, rows}, checks, sanity, local_score {kind: val \|
host_test, value, table_url}`. Per submission: `ts, kind: submit, job_id, predict_job_id, csv_sha256,
slug, message, kaggle {status, ref, response, submitted_at, public_score (when read back), error_description
(when read back)}`. Every entry is also what the Status tab lists (§1 #46 columns) and what `GET /submit/state`
re-verifies.

**What the verification bundle (session 5) needs from these entries**: given a leaderboard row
(Kaggle submission `ref`, date, public score) → the submit entry with that `ref` → `csv_sha256`, which the
organizer can check against the file Kaggle lets competition hosts download → the predict entry → the
checkpoint hash (recorded and on disk), `run_url`, contract, `test_inputs.files_json_sha256` → the Run's
parameters (train revision URL, locked val URL, seed, torch/torchvision versions) → the table revisions.
Each link is a hash or a URL already produced by an earlier stage; nothing in the chain is a free-text
claim.

## 7. The Kaggle API

**Library and credentials (ExDark).** The Python `kaggle` package (2.x, `KaggleApi` + `kagglesdk`); no
subprocess CLI, no raw REST. `kaggle_credentials_present()` looks for `~/.kaggle/access_token`,
`~/.kaggle/kaggle.json`, `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`; `KaggleApi()` is
imported and `authenticate()`d **inside** the function (kaggle 2.x authenticates at import). Failure →
"Kaggle authentication failed: …". Since rc10 the plugin writes the token once (the Connect button,
`kaggle_client.connect`, to the one file the client reads — §13 D13) and still never reads it back, stores
it elsewhere or forwards it. `~` is the **worker's** home — on our redirected-home hosts that is
`3lc-hub-11/home/.kaggle/access_token`, not the user's real profile (workspace CLAUDE.md §A4); the copy
that says "on the machine running the compute service" is exactly right and must stay.

The installed client (2.2.4) resolves, in order: `KAGGLE_API_TOKEN` (a token, or a *path* to one) →
`~/.kaggle/access_token` → `~/.kaggle/access_token.txt` → legacy `kaggle.json` / `KAGGLE_USERNAME`+`KEY` →
OAuth cache. **Hazard for the worker**: when nothing is found, `authenticate()` prints help and calls
`exit(1)` — a `SystemExit` inside the plugin worker, which ExDark avoids by checking presence first; ours
must do the same and additionally catch `SystemExit` around `authenticate()`.

**Calls** (all through `KaggleApi`): `get_competition(slug)` → `user_has_entered`,
`max_daily_submissions`, `title` (the "entered" check and the daily limit); `competition_submissions(slug)`
→ the used-today counter, counting entries whose `date` starts with today's UTC date (returns `None` on
any error; **known to 403 on a private competition** — a LAUNCH-VERIFY item for the sandbox, in which
case the card falls back to "N submissions/day"); `competition_submit(file_name, message, competition)` →
`ApiCreateSubmissionResponse {message, ref}`; `competition_leaderboard_view(slug)` (Status tab). The
slug: `params.competition_slug` from the UI field, defaulting to the code constant (ours: the manifest,
D8). The message: `"<run_name> via 3LC plugin"` when blank.

**Score after submit.** The backend **does not poll**. `GET /kaggle/status?slug=` (Status tab, refreshed
every 15 s while visible) lists the last 10 submissions `{date, description, status, public_score}`,
`best_public_score`, top-5 leaderboard and `my_rank`. It reads **neither `privateScore` nor
`errorDescription`**; the SDK's `ApiSubmission` carries `status ∈ {PENDING, COMPLETE, ERROR}`,
`error_description`, `public_score`, `private_score`. So a metric rejection (a PVE from §3) is invisible
in ExDark's UI: the submit shows "Submitted to Kaggle · Submission <ref>" and the Status row shows a
status string. **D12**: read `status` + `error_description` back (one poll a few seconds after submit,
or on the Status refresh) so "Predictions must be 0–5. Found invalid: …" reaches the participant.

**Daily limit and errors.** Proactive: the card's budget (`daily_limit − submissions_used_today`), Submit
disabled at 0 with "Daily limit reached, resets 00:00 UTC. Your CSV is saved and ready."; the pre-submit
`confirm` names the price. Reactive: `classify_kaggle_error(exc)` matches substrings of the lowercased
exception text — `daily_limit` if ("daily" and ("limit" or "submission")) or "submission limit" or
("maximum" and ("per day" or "today")); `not_joined` if ("rules" and "accept") or "must accept" or "not
accepted"; else `error`. Soft outcomes (`limit_reached`, `not_joined`, `skipped`) **complete** the job with
the reasons of §1 #36; anything else fails it with "Kaggle rejected the submission: `<exc>`". No timeouts,
no retries, no special handling of network errors, 401/403/429 or a wrong slug — one generic branch.

**CSV download fallback.** `GET /submissions/{job_id}/download` streams the file with
`text/csv`, `Content-Disposition: attachment; filename="submission_<stamp>.csv"`, `Cache-Control:
no-store`; 404 "No such job" / "has no submission CSV on disk". Reachable from the results row, the soft
banner and the Status rows; the credentials help ends with "…you can always upload it manually on the
competition's Submit page." Everything above ports verbatim; `kaggle>=2.2.3,<3` is already in the extra.

## 8. Plumbing (invisible, ours) the port requires

- **Durable predict record + submit record** (`predict_state`, `submit_state`, already allowed in
  `session.py`): status, job id, params, facts (§6), checks, sanity, result, log tail, worker pid,
  heartbeat; `GET /submit/state` re-verifies the CSV on disk and applies the Train pid rule (`stale`).
  Replaces ExDark's job store and snapshots (compute 1.1.0 keeps job records in memory).
- **New routes**: `GET /runs` (from `train_state.runs`, with `usable`/`reason`), `GET /predict/preflight`
  (D3's test-images check), `GET /submit/state`, `GET /kaggle/connection`,
  `GET /submissions/{job_id}/download`; job kinds `predict`, `kaggle_submit` in `plugin.toml`. No
  `/validate/*` (EXDARK_MIRROR #36).
- **`predictor.py`** replaces the stub: resolve + verify (§2), test inputs from the import record's
  `kit_dir` verified against `files.json`, the val transform and `build_model` from `trainer.py`, batch 32,
  progress `{images, total_images}` ≤ 1/s, the seven checks, `.INVALID.csv` on failure, sanity, local score
  (D6), the CSV under `plugin_home()/predictions/<run_name>/`, the ledger entry, the record. Cooperative
  cancel between batches (no button, D4). Never writes a table.
- **Kaggle module** (`kaggle_client.py` or inside `predictor.py`, as ExDark): presence check, guarded
  `authenticate()` (`SystemExit`), the four calls, `classify_kaggle_error`, the counter; slug from the
  manifest.
- **Ledger writer** (`ledger.append`) goes live for predict and submit entries; the bundle stays session 5.
- **Tests**, ported from ExDark's two files and extended: the plugin-run-only gate (`train_job_id` wins,
  bare path refused, missing weights named, **sha mismatch refused**, cancelled-with-best allowed), the
  revisit resolution (record + CSV present / CSV missing / record missing), the seven format checks on
  synthetic CSVs, Kaggle error classification, the credentials-absent path never raising `SystemExit`,
  the fragment safety tests (`esc()` on every interpolation, no competition literal).

## 9. What the port would differ in, beyond the four allowed differences — decided 2026-09-30

**Taken (Rishikesh, 2026-09-30; no code that day):** D1 **drop host mode** — no Weights-file source,
no `_meta.host`, no host dir (§1 #8 and #11 are X; participants and organizers run the same build) ·
D2 **block** prediction when any provenance check fails — stricter than ExDark, for anti-cheat: the gate
is the **three-way sha256 match plus green provenance** (`provenance_ok`), so #10's warn line becomes an
unusable-run reason ("provenance check failed") rather than a note on a selectable run · D3 the locked
test-inputs row + the gate verifying the 1,800 test files against `files.json`, copy as written in #12 ·
D4 no cancel, mirror · D5 a **predicted-class distribution** card: tags per class with counts, warn when
any class takes **< 5 % or > 50 %** of the predictions ("… Are these fully-trained weights? (Submitting is
still fine.)") · D6 **val accuracy on the locked val table, for everyone**, also checked against the run's
recorded `best_val_accuracy`; **no host-only test scoring** (§5 (a) only; the hero reads "Val accuracy ·
Your locked validation split, not the leaderboard.", the basis line "· val accuracy 57.58 %") · D7 the UI
shows the **manifest's `daily_limit`**; Kaggle's refusal (`limit_reached`) is authoritative; `Rules.md`'s
20 is the old kit's and is ignored · D8 slug **locked from the manifest** (no field, nothing posted) ·
D9 **six decimals** · D10 not a plugin change: the laptop's metric notebook and `Rules.md` are the old
two-class kit's; the sandbox, cloned from the six-class HackNova competition, will prove the real metric
accepts 0–5 — kept as a **sandbox check**, not a blocker · D11 **fail the job** on an unreadable test image
· D12 **one status read-back** after submit, showing Kaggle's `error_description` (and the public score
when `COMPLETE`) on the submit banner and the record.

Consequences for the plan above: §1 #8/#11 → X; #10 → unusable reason; #21 → D5's card; #22/#27 → D6's
copy; #29 → the manifest number with Kaggle's refusal as the only override; #30 → locked row; §5 (b) and
D1 (a) are void; §8's predictor gains the val pass (1,200 images, the locked revision) and the read-back
call (`competition_submissions` once, a few seconds after `competition_submit`); the fixtures gain
`submit-provenance-blocked`, `submit-results-val`, `submit-kaggle-error`.

| # | Question | ExDark | Options / recommendation (superseded by the decisions above) |
|---|---|---|---|
| D1 | **Host mode**: the Weights-file source (#8, #11) and `_meta.host` | `is_host()` = metric + answer key present in `TLC_KAGGLE_HOST_DIR`; participants never see the toggle | (a) mirror with `KAGGLE_CLASSIFICATION_HOST_DIR` holding the metric module + `solution_kit_v1.csv` (enables §5 (b) too); (b) drop: PLAN §A "Only from plugin-created runs" with no exception, organizers verify through the bundle instead. **Recommend (a)** if §5 (b) is wanted, else (b) — the two go together |
| D2 | A run with `provenance_ok == false` | allowed to predict; the note warns "This run has no recorded provenance checks." | (a) mirror (warn, allow); (b) refuse. **Recommend (a)** — the hard gate is the sha match (§2), which is refused regardless; a failed *check* (e.g. a torchvision version drift) should not strand a participant |
| D3 | **Test inputs gate** replacing the test-table field/picker/gate (#12) and its copy; files.json verification of the 1,800 images before inference | test table URL + revision picker + `/import/revisions` gate | locked row "Test images · `<kit_dir>/data/test` · 1,800" + gate states as written in #12, verifying every file's sha256 against the kit's `files.json` (≈ 2 s). The check is additional functionality; the copy is a proposal. Geometry: the locked row takes the left column, Device the right (#13) |
| D4 | Cancel for predict | none (no button; the job never reads the flag) | mirror: no control. Plumbing may still honour a host-side cancel between batches |
| D5 | **Prediction sanity** card contents and warning heuristic (#21) | boxes / mean per image / empty images / per-class box tags; warn when < 1 box per image | candidate: images · mean confidence · low-confidence count (< 0.5) · predicted-per-class tags (6); warn when a class has 0 predictions or one class exceeds half the rows, same closing sentence "(Submitting is still fine.)". Thresholds are design, not mine |
| D6 | **Local scoring** and the hero stat (#22, #27, §5) | host-only test mAP@0.5; participants get nothing | (a) val accuracy on the locked val revision for everyone, doubling as the "checkpoint reproduces the recorded accuracy" check; (b) host-only test accuracy (needs D1 (a)); (c) both, hero shows (b) on hosts and (a) otherwise. **Recommend (c)**, with the val copy stating it is not a leaderboard preview |
| D7 | **Daily limit** source and copy | live `max_daily_submissions` from Kaggle; fallback copy "N submissions/day"; the confirm falls back to 3 | manifest `daily_limit` (100) as the number the UI shows until Kaggle answers, Kaggle's live value when it differs (log the difference), used-today from Kaggle when the list call works. Also: `Rules.md` says 20/day — one number must win before launch (organizer) |
| D8 | **Competition slug** control (#30) | editable field, session `slug_override` | (a) locked row from the manifest (competition identity is a manifest fact; the retired-key machinery in `session.py` already exists for `slug_override`); (b) mirror the editable field. **Recommend (a)** |
| D9 | `confidence` precision | box values at 6 decimals; the kit writes float repr | 6 decimals (`0.912345`) — cannot change validity or the argmax; makes files diffable. Minor |
| D10 | **Organizer items the plugin cannot fix** (§3) | — | the metric notebook's allowed set `(0, 1)` → `range(6)` and its messages; the two-class `kaggle_content/*.md` pages; `Rules.md` 20/day vs 100/day. All must land before the sandbox is created, or every submission of a class ≥ 2 is rejected |
| D11 | Unreadable / missing test image at predict time | n/a (table rows); the kit predicts a grey placeholder and fills missing ids with `0, 0.5` | (a) fail the job (the kit was verified per file at import; corruption later is worth a loud stop + "Re-run Import"); (b) mirror the kit. **Recommend (a)** |
| D12 | Reading Kaggle's `status` / `error_description` / `private_score` back | never read; a metric rejection is invisible | read `status` + `error_description` once after submit (or on Status refresh) and surface "Kaggle scored it: 0.7233" / "Kaggle rejected it: …" on the submit banner or the Status row. New behaviour; small |

Everything else in §1 is verbatim or falls under an allowed difference. The kit-side checkpoint defect
(§4, last row) is a note for the kit's owner, not a plugin decision.

## 10. Implementation (2026-10-01, `7c3904b` + `71568a4`)

What shipped, module by module, against §1/§8/§9:

- **`predictor.py`** — `list_runs` / `assess_run` (the picker's `usable` + `reason`: still training ·
  failed · interrupted · no best checkpoint saved · best checkpoint missing on disk · provenance check
  failed · best checkpoint changed on disk (sha256 mismatch) · the Run's record disagrees with the train
  record); `resolve_checkpoint` (THE gate: `train_job_id` wins, a path in the body is ignored, a bare
  path is refused, the file must exist, recorded = on disk = on the Run, `provenance_ok` must be true;
  cancelled-with-best allowed); `test_inputs` (every `sample_submission.csv` id's image under
  `<kit_dir>/data/test` matched to `files.json` by size + sha256, ≈ 4 s for 1,800 files, cached per
  path/size/mtime); the val check over the LOCKED val revision (`VAL_TOLERANCE_PP = 0.25`; a miss fails
  the job); the test pass (val transform, `build_model`, `weights_only` strict, eval, batch 32, workers
  `default_workers()`); the CSV under `plugin_home()/predictions/<Run folder>/submission_<stamp>.csv`
  (LF, UTF-8, no BOM, six decimals, the written file is what is checked and hashed; `.INVALID.csv` on a
  failed check); the seven format checks (§3) plus three more rows the panel shows under Checkpoint /
  Test inputs; the distribution card (D5); the durable `predict_state` / `submit_state` records with the
  Train pid rule; `predict_submit_state` (`empty / running / predicted / submitted / stale / failed`,
  the CSV re-verified by sha256); `csv_path_for` (record, then the ledger); `run_kaggle_submit`.
- **`kaggle_client.py`** — presence check (`KAGGLE_API_TOKEN`, `~/.kaggle/access_token[.txt]`,
  `<KAGGLE_CONFIG_DIR|~/.kaggle>/kaggle.json`, `KAGGLE_USERNAME`+`KEY`), `authenticated_api` (catches
  `SystemExit`), `connection`, `submit`, `classify_error`, `read_back`. **Found at G3:** on the
  unlaunched event competition `ListSubmissions` answers **403** (ExDark's LAUNCH-VERIFY note, confirmed),
  while `GetSubmission(ref)` and `GetSubmissionLimits` both answer — so the D12 read-back goes by ref and
  the used-today counter comes from the limits call (`num_today`, `num_allowed_now`); the list is the
  fallback for both (`71568a4`).
- **`ledger.py`** — `append` / `read` / `find` over `ledger.jsonl` (append-only; per-entry `ts`).
- **Routes** `GET /runs`, `GET /predict/preflight`, `GET /submit/state`, `GET /kaggle/connection`,
  `GET /submissions/{job_id}/download`; `_meta.predict_state` on `/config` (the stepper's "submit
  done"). Job kinds `predict`, `kaggle_submit` in `run_job`.
- **The fragment** — the §1 table as written under D1–D12: the Run picker (`run_folder · N epochs · best
  val accuracy 57.58 % · Sep 30`, unusable rows disabled with the reason, the note with the trained
  checkpoint's sha), the locked **Test images** row + gate (idle / checking / verified / missing with
  Go to Import), Device (mirrors the session device with Train's field), the progress block
  ("Val check: n / 1,200 images" then "Inference: n / 1,800 images", one bar), the grouped checks
  (Checkpoint · Test inputs · Submission format), the **Predicted-class distribution** card, the hero
  ("66.67 % · Val accuracy · Your locked validation split, not the leaderboard."), the CSV row, the
  basis line, the connection card (manifest limit; "N of 100 submissions left today" when the limits
  call answers), the locked **Competition** row, the message field, the confirm ("1 of your 100 daily
  submissions"), the success banner + the D12 line ("Kaggle scored it: public score 0.65666" /
  "Kaggle rejected it. <error_description>" / "Kaggle's verdict could not be read back"), the soft
  callouts, the failure banners with Copy diagnostics, revisit + "New prediction", tab-open
  resolution through `PluginJobs.list` + `/submit/state`. Fixtures: `submit-state1`, `submit-gated`,
  `submit-gate-missing`, `submit-provenance-blocked`, `submit-checkpoint-changed`, `submit-inference`,
  `submit-results` (= `submit-results-val`), `submit-results-low`, `submit-checks-fail`,
  `submit-nokaggle`, `submit-notjoined`, `submit-limit`, `submit-success`, `submit-rejected`,
  `submit-fail`, `submit-kaggle-error`, `submit-revisit` — all 17 rendered in the harness without a
  console error (2026-10-01).
- **Tests** — `tests/test_predictor.py` (46): the gate (id wins / bare path / tampered / Run disagrees /
  provenance blocked / cancelled allowed), the seven checks on synthetic rows, the CSV bytes, the
  distribution card, the test-images gate on a synthetic kit (ok / tampered / missing / idle), the
  revisit resolution, the ledger, the client (credentials absent never raise `SystemExit`, the sources,
  `classify_error`, the read-back by ref and by list, the submit outcomes, the connection states), the
  submit job's records + ledger + write-back, and the heavy end-to-end predict on the synthetic kit
  after a real one-epoch run (10/10 checks, the val check reproduces the recorded accuracy exactly on
  CPU, a tampered `best.pt` and a flipped `provenance_ok` are refused, an unreadable image fails D11).

Deviations from the mirror table, all small: the CSV directory is the **Run folder's** name (unique on
disk; equals the run name unless tlc suffixed it); the checks panel carries **ten** rows (the three-way
sha, the test inputs, the val check + the seven format checks) and says "checks passed", not "format
checks"; Kaggle's `private_score` is kept in the **ledger** only (it is what the API returned to the
host account) and never rendered.

## 11. Gates (2026-10-01, laptop, `3lc-hub-11`, catalog install pinned to `7c3904b` then `71568a4`)

Driven by `../3lc-hub-11/gates_session4.py` through the in-process host (the service stopped, the
redirected home, the regenerated `cdn/` served locally on :8765 as the manifest remote so the new slug
is the one in force — the dev bucket still carried the TBC slug). The run used: `recheck_gpu`
(`8c77fefd`, cuda, recorded best val accuracy 66.67 %, sha `b0d7efc6e7ab…`). RTX 3070 Ti Laptop GPU.

| Gate | Result |
|---|---|
| G0 | manifest `3lc-scene-classification-challenge`, `daily_limit` 100; 15 runs listed, 13 usable, `g5c_restart` disabled "interrupted"; connection card **ready as `rishikeshjadhav3lc`** (Kaggle's limit 100 = the manifest's), the test-images gate 1,800 / 1,800 verified |
| G1 | **PASS** — 38 s end to end on cuda (auto): val check 1,200 images, **66.67 % recorded · 66.67 % now** (delta 0.00), test pass 1,800 images; 10/10 checks; CSV 1,800 rows in `sample_submission.csv` order, header exact, six decimals, LF, no BOM, sha256 on disk = the record's; distribution buildings 257 · forest 310 · glacier 354 · mountain 420 · sea 141 · street 318, mean confidence 0.613, 642 below 0.5, no skew warning (every class within 5–50 %) |
| G2 | **PASS** — a tampered `best.pt` (one byte appended): the picker says "best checkpoint changed on disk (sha256 mismatch)", the job fails with the SHA_MISMATCH sentence; a foreign file of the same size at the checkpoint path: the same refusal; `provenance_ok` flipped to false on the record: "provenance check failed" / "failed a provenance check at training time…"; a bare `weights_path`: "Direct weights files are not accepted…"; one test image corrupted (same size): the gate reports 1 mismatch and the job refuses "could not be verified … Re-run Import"; the G1 record stayed the revisit basis through all five refusals; no ledger entry for any refused job |
| G3 | **PASS** — submission **56756858** accepted ("Successfully submitted to 3LC Scene Classification Challenge"), read back by ref 10 s later: **COMPLETE, public score 0.65666** (private 0.62666 in the ledger), `error_description` empty — so **D10 is settled**: the real metric accepts classes 0–5. Local val accuracy 66.67 % vs public 65.67 %. The first run's submission **56756714** scored the same (read back out of band after the 403 finding). Used today after G3: 2 of 100 (`GetSubmissionLimits`) |
| G4 | **PASS** — `GET /submissions/<job>/download` 200 `text/csv`, `Content-Disposition: attachment; filename="submission_….csv"`, 50,431 bytes, sha256 = the record's; an unknown job 404. The daily-limit and API-error paths are the mocked cases in `tests/test_predictor.py` (limit_reached with "(100/day)", not_joined from the pre-probe without spending an attempt, a 500 → failed job, no credentials → skipped) |
| G5 | **PASS** — `ledger.jsonl` under the plugin home: one `predict` entry (run URL, checkpoint path + the three equal sha256s, the contract from the Run, `test_inputs {count 1800, files_json_sha256, sample_submission_sha256}`, device, `csv {path, sha256, rows}`, the ten checks, the distribution, the val score, the manifest provenance `remote ec0c60cf45f0…`) and one `submit` entry per submission (`csv_sha256`, slug, message, `kaggle {status submitted, ref, response, submitted_at, read_back_* (COMPLETE, 0.65666, …)}`) |

Not reproducible live, covered by the unit test instead: D11 proper (an image that passes the sha gate
but fails to decode) — the gate catches any byte change first, so only a decoder fault reaches the
loader; `test_an_unreadable_test_image_fails_the_job_after_the_gate` pins the loud failure.

**For the demo (2026-10-02):** the compute service resolves the manifest remote → cache → bundled, so
`kaggle/intel-scene/manifest.json` (2,967 B, `ec0c60cf…bdfcb`) must be uploaded to the dev bucket
before the Hub is opened against `https://competitions.dev.3lc.ai`, or the bucket's old document (the
TBC slug) wins over the bundled one. The credentials the worker reads are the compute host's:
`3lc-hub-11/home/.kaggle/access_token` on this laptop (`USERPROFILE` is redirected; the plugin only
checks the file exists).

## 12. Decided in session 5 (review later) — 2026-10-05

- **The val-edit warning on Predict** (brief B2): the same amber callout Train shows under its locked
  Val row renders under Predict's locked **Test images** row whenever `import_state.val_edited` is
  true ("Your edits to val in the Dashboard are ignored; every run is scored on `<dataset>/<locked
  revision>`."). The val check (D6) keeps scoring the LOCKED revision; nothing else changes.
  Decision record: `docs/TRAIN_MIRROR.md` §14 S5-B2.
- **Continue to Status** now lands on the ungated Status tab (`docs/STATUS_MIRROR.md`); the pending
  line "The Status tab shows the result." on the submit banner is therefore true.

## 13. Decided in session 8 (2026-10-08) — the guided connect flow (1.0.0rc10)

| | Item | ExDark | Ours (rc10) |
|---|---|---|---|
| D13 | **The not-connected state of the connection card** — the one deliberate divergence from the mirror rule on this tab. The rc9 hand test: the card's help paragraph (ExDark's `_token_setup_commands`, verbatim) was far too detailed and told the participant to write the token under `$env:USERPROFILE` / `~` — the BROWSING user's profile — while the compute service may run with a redirected home (the tester environments do), so the token landed where the service never looks | err callout "**Kaggle account not connected.**" + the server help: a paragraph with the PowerShell and POSIX commands, `KAGGLE_API_TOKEN`, legacy `kaggle.json`, the manual-upload sentence | warn callout "**Kaggle isn't connected yet.**" + one line ("Get a token at kaggle.com › Settings › API › Create New Token, then paste it here.") + a masked field (`type=password`, autocomplete off) + **Connect**; a collapsed **Other ways to connect** (the service's OS-specific one-line command with the resolved path filled in, the env var, the legacy file — one line each, Copy on each). The card is rendered OUTSIDE `#ps-step2-body`'s muted block so Kaggle can be connected before a prediction exists; the competition row, the message and Submit stay muted |
| D13a | The write | none (the participant writes the file) | `POST /kaggle/connect {token}` → `kaggle_client.connect`: trim (whitespace, newlines, BOM, surrounding quotes) · `KGAT_[A-Za-z0-9_\-]{16,}` and ASCII · refuse while `KAGGLE_API_TOKEN` is set on the service (it would shadow the file) · write to `token_path()` = `os.path.expanduser("~/.kaggle/access_token")` — the exact expression kagglesdk evaluates, so a redirected `USERPROFILE` / `HOME` is honoured by construction — plain ASCII, no BOM, no trailing newline, `0700` dir / `0600` file on POSIX · verify through the client's own `authenticate()` (one introspect call) · answer `{ok, username, path, connection}` with the fresh card. Failures: one plain sentence each (`REASONS`), the field kept: not a token · Kaggle rejected it (the file is removed again) · no network (the file is kept) · `KAGGLE_API_TOKEN` override · could not save · a scrubbed error |
| D13b | The token's exposure | never read by the plugin | written once, never read back, never logged, never in a response / reason / record / the diagnostics; masked in the field, cleared from the browser after the connect; the Doctor and the card carry `token_path` (the path, never a value). `tests/test_kaggle_connect.py` connects with a planted token and scans every served surface; the bundle's `SECRET_PATTERNS` still refuse a `KGAT_` string |
| D13c | The CSV row | file name · middle-truncated path · Copy · Download | file name · Copy · Download · the FULL path on its own line (monospace, never truncated) · the note "Saved at that path on the compute service's machine. You can upload it by hand on Kaggle's Submit page …" — always, so a participant who never connects from here still knows where the file is |
| D13d | Connection status before predicting | only the card inside Step 2 | a one-line status at the top of the tab body (`#ps-kstatus`: "Kaggle: not connected · Connect in Step 2" / "connected as <user> · N of M left today" / "connected as <user> · competition not joined"), fed by the same `GET /kaggle/connection`; the Doctor's Kaggle row reads "not connected · connect on the Predict + Submit tab (Step 2)" and copies the token path; the Status tab's Kaggle live callout and a skipped submit point at Step 2 |
| D13e | Where this is documented | — | CONTEXT.md decisions 2026-10-08, CHANGELOG 1.0.0rc10, TESTING.md §7.5b (the optional Connect step: a wrong paste, then a real token; still no submit) |

## 14. Decided in session 9 (2026-10-09) — rc11, the UI pass (back-port to ExDark after the event)

| # | Decision | ExDark | rc11 |
|---|---|---|---|
| D14 | **Submit confirmation** | native `window.confirm("This spends 1 of your N daily submissions. Submit "<run>" now?")` | an in-app callout in `#ps-banner` naming the run, its val accuracy and the submissions left today, **Submit** / **Cancel** (`psRenderSubmitConfirm` → `psSubmitNow`) |
| D15 | **Other ways to connect** | collapsed; the command wraps | collapsed by default (unchanged); each line on one row with horizontal scroll and Copy |
| D16 | **"Run inference first"** | shown whenever no basis is set | never shown while a prediction record exists (`kgSubmitState.state` predicted / submitted); the basis follows from the revisit |
| D17 | **The Doctor after Connect** | refreshed on the next Status visit | `stLoadDoctor(true)` right after a successful Connect |
