# TRAIN_MIRROR.md — the Train tab mirrors 3lc-compute-plugin-kaggle v1.2.15

Session 3, Phase 0 (2026-09-28): study only, no code. The governing rule is
`docs/EXDARK_MIRROR.md`: the Train tab mirrors ExDark's Train tab **control for control, in
functionality and looks**. Allowed differences: (1) competition identity, (2) dataset,
(3) competition rules, (4) task-inherent (timm resnet18 classification, no Ultralytics).
Anything else that would differ is listed under **D (needs decision)** with ExDark's version and
the reason it cannot be copied. The decided requirements from the session-3 brief are applied
where they map; where they go beyond ExDark they are marked **+brief**.

References (read-only): `../3lc-compute-plugin-kaggle/src/tlc_plugin_kaggle/ui.html` (Train markup
771–1058, Train JS 4340–5250, pickers 1805–1870, session helpers 1903–2100, fixtures 3832–4030),
`trainer.py`, `routes.py`, `jobs.py`, `importer.py` (563–704), `docs/ui-notes.md` §9, §13, §18 and the
Train fixture map; `../reference/3lc-compute-plugin-timm/src/tlc_plugin_timm/{trainer.py,__init__.py}`
(`b77f252`, `src/` identical to v0.2.6); tlc 3.3.2 `tlc/integration/torch/samplers/_sampler.py`
and `tlc/_core/objects/mutable_objects/run.py` in this repo's `.venv`; SDK 0.3.3
`tlc_plugin_sdk/{worker.py,job_context.py,shared/model_storage.py,shared/job_tracker.py}`; compute 1.1.0
`tlc_compute/plugins/job_manager.py` in `../3lc-hub-11/.venv`;
`../hackathon_starter/intel-kit/{train.py,config.yaml,predict.py,register_tables.py,README.md}` (the Intel
kit's original scripts, read-only, on the laptop since 2026-09-29); `../hackathon_starter/Starter_Kit/`
(the Chihuahua vs Muffin predecessor the Phase 0 study used as a stand-in).

Legend: **V** ported verbatim · **A(n)** adapted under allowed difference n (how) ·
**X** dropped (why) · **D** needs decision · **+brief** required by the session-3 brief, no ExDark
equivalent.

## 1. Train tab, in render order

| # | ExDark element (ui.html line) | Ours | Mark |
|---|---|---|---|
| 1 | `.card-header`: `plugin-section-number` 2 · "Train"; subtitle "Trains the competition baseline on your imported tables and records verified provenance (the pinned official init) on the 3LC Run." (774) | same; subtitle "…records verified provenance (the locked from-scratch contract) on the 3LC Run." | A(3) |
| 2 | Locked-contract banner `.format-selected-banner` (lock icon; `.fmt-name` "YOLOv11n from the official COCO-pretrained checkpoint · 640px" with `.kg-contract-imgsz`; `.fmt-desc` "Enforced server-side; identical starting weights for every participant."; `title` tooltip) (780) | same geometry; name "resnet18 from random init (timm) · 150px" from `model.*`; desc "Enforced server-side; identical architecture, init rule and resolution for every participant."; tooltip reworded ("The leaderboard measures your data work, not your compute." kept) | A(3) |
| 3 | `.kg-locked-rows`: Model "YOLOv11n" · Init "yolo11n.pt · sha256 <12>…" (`#tr-lock-init`, full hash in `title`) · Image size 640; each a `.kg-locked-row` with lock glyph and a `title` tooltip; no ids collected into the POST (792) | five rows, same component: **Model** "resnet18 · timm 1.0.29" · **Init** "random (pretrained = false) · seed 42" · **Image size** "150" · **Val table** the import record's locked val URL (middle-truncated, full URL in `title`, Copy) · **Inference** "single forward pass". The last two are the brief's locked fields; tooltips written for the classification contract | A(3) +brief |
| 4 | `#tr-conn-banner` (807) | verbatim | V |
| 5 | `#tr-banner` success/failure slot (810) | verbatim | V |
| 6 | **Tables** section: `.kg-sec-head` "Tables"; two columns, each `.form-group.kg-pick-wrap` with a required label, a `.kg-url-row` (text input + `kgi-btn` picker button `title="Pick a table revision"` `aria-haspopup="listbox"`) and a `.kg-pop` popover listing the split's dataset and its lineage-ordered revisions with a LATEST badge and row counts (`kgBindTablePicker` 1805, `GET /tables/list`) (812) | **Train table URL** verbatim (picker scoped to `manifest.dataset_name("train")`, `GET /tables/list` ported from ExDark's `list_project_tables`); **Val table URL** is NOT a field: it renders as the locked row in #3 (the brief locks val to the import record's URL). The right column keeps its place with the locked val row so the two-column geometry holds | A(3) |
| 7 | Field handlers on both URL inputs: 500 ms debounce, blur, Enter, paste → `kgSetUrlOverride` (session `overrides.train_table_url`, `classify_override` keep/suppress/drop) + `trVerifyTables` (5010) | verbatim for the train field; `session.py` already carries `URL_SEG_PATTERNS` and `classify_override`; the fragment gains `kgUrlSeg` (the JS/Python regex parity test `test_url_regex_parity.py` goes live) | V |
| 8 | `#tr-tables-gate` (`trVerifyTables` 4877 → `trEvaluateGate` 4765 → `trRenderGate` 4825): states idle ("Run Import (tab 1) to create the competition tables, or paste table URLs from the Dashboard."), checking, green `.kg-preflight-ok` "Tables verified: exdark_train/initial · exdark_val/initial · trains on the latest revision of each / these exact revisions", amber `.kg-callout.warn` with per-problem rows: split identity (DP-11, `kgUrlSeg` dataset segment), same-table, not on disk (+ Go to Import), row budget (`_meta.contract.max_rows`, evaluated against `latest_row_count` when Use latest is on; unknown counts never refuse) via `GET /import/revisions` | same states and renderer, fed by a new read-only `GET /train/preflight?train_url=&use_latest=` that returns what ExDark's `/import/revisions` returns (exists, row_count, latest_url, latest_row_count, revisions) PLUS: `descends_from_seed` (lineage walk from the resolved revision back to `import_state.lineage_root.train_url`; a revision outside that lineage is an amber problem, +brief), the usable-row summary (#9) and class coverage (#10). Split identity kept (a pasted `intel-scene_val` URL is refused). Row budget: ceiling = `manifest.expected_rows("train")` = 6,600 (labeling a pool row never changes the count, so the rule transfers unchanged). The Intel kit's own cap on weight-1 rows (3,000) is a different rule — D14 | V / +brief |
| 9 | — (no equivalent: ExDark's gate shows only table names and the revision choice) | **Usable-row summary** under the green line, in the gate block, re-evaluated with the Use-latest checkbox like the row budget: "603 labeled rows in use · 5,997 excluded as undefined · 0 excluded at weight 0". Amber `.kg-callout.warn` when undefined rows carry weight > 0: "N unlabeled rows have weight > 0 and will be skipped until you label them." Hard stop (amber, Start disabled) when 0 usable rows: "No usable rows: every row is undefined or at weight 0. Label images in the Dashboard first." Placement is my proposal (the gate is the one surface that already re-evaluates per revision) | +brief · D1 (placement) |
| 10 | — | **Class coverage warning** (amber, does not block): "No usable rows for: glacier, sea. Those classes cannot be learned in this run." | +brief |
| 11 | "Use latest revision" checkbox (checked) + help "Trains on each table's newest revision (`.latest()`), picking up your label edits from the Dashboard." (834); picking a non-latest revision unchecks it; toggling re-evaluates the gate (5057) | verbatim, train table only ("Trains on the train table's newest revision…"); default = latest (the brief) | V |
| 12 | **Training parameters** `.kg-sec-head` + two-column grid; every bounded field: label, `input[type=number]` with `value/min/max/step`, `.form-help` ending in "Range a to b.", a `.kg-field-slot` for the inline error (841) | same grid; the field list is §6; defaults and bounds come from `manifest.training` and are served in `_meta` (the fragment carries no literal, so `value/min/max` are filled on config load like `.kg-contract-imgsz`) | A(3) |
| 13 | Epochs (20, 1–300) with the pretrained-calibrated help + `#tr-duration` ETA hint slot (843) | Epochs (10, 1–50) with help calibrated after G1 ("A from-scratch run reaches about X % val accuracy by epoch N…" — the number comes from the gate run; until then the help states the range only); `#tr-duration` slot verbatim (§5) | A(3) |
| 14 | Batch (16, 1–128) "Images per training step. Raise it until GPU memory runs out; lower it on CUDA out-of-memory. 16 fits an 8 GB card at 640px." (853) | Batch size (16, 8–128), same help with "16 fits any card at 150px" | A(3) |
| 15 | lr0 (0.01, 0.0001–0.1) "Initial learning rate…" (860) | Learning rate (0.0001, 0.00001–0.01) "Initial learning rate for Adam. Higher learns faster but can diverge…" | A(3) |
| 16 | lrf (0.01, 0.01–1.0) "Final learning rate as a fraction of lr0. The schedule decays toward lr0 × lrf…" (867) | no direct equivalent (Ultralytics' linear/cosine decay). The baseline kit uses `StepLR(step_size=5, gamma=0.1)` with no exposed knob. Options: (a) fixed StepLR from the kit, no field (the schedule is part of the baseline recipe); (b) expose the kit's two numbers as fields with bounds added to schema v1; (c) mirror lrf's meaning with a cosine decay to `lr × lrf` | D2 |
| 17 | Optimizer `select` Auto / SGD / Adam / AdamW (874) | Optimizer `select` with the manifest default `adam` selected. ExDark offers the four Ultralytics names; ours can offer what `torch.optim` gives (`adam`, `adamw`, `sgd`). The choice list is a competition rule → belongs in schema v1 (`training.optimizers`) or is fixed to `adam` only (the kit) | D3 |
| 18 | Patience (100, 0–100) early stopping (888) | no equivalent: neither the kit nor the timm plugin stops early. (a) drop (task-inherent: the baseline has no early stopping); (b) add early stopping on val accuracy as a new field with bounds in schema v1 | D4 |
| 19 | — | **Weight decay** (0.0, 0–0.1): in the manifest's defaults and bounds already, not in ExDark's form. Render as a fourth core field, or keep it in Advanced | +manifest · D5 |
| 20 | — | **Seed** (42): in the manifest's defaults, no bounds. The contract needs it recorded; expose as an Advanced field (bounds `[0, 2147483647]` added to schema v1) or lock it (a locked row) | D6 |
| 21 | **Advanced** disclosure (`kgl-acc-summary` "Advanced", opens on load when a saved value is non-default) (897) | verbatim mechanics | V |
| 22 | Device (text, blank = auto; help "Blank = auto-select: CUDA GPU if available, else `mps`, else CPU. Or set a GPU index…") (905) | verbatim field and help (`resolve_device` already ported); +brief: when the accelerator fails at model/first-batch time the job retries on CPU and says so (§7) | V +brief |
| 23 | Workers (0, 0–16) "Keep 0 on Windows (the safe default); raise on Linux…" (912) | same field; the DEFAULT is device-aware and served from `_meta` (0 on Windows, `min(4, cpu_count)` elsewhere — my proposal for the non-Windows value), bounds 0–16 as a plugin constant (not a competition fact) | V +brief |
| 24 | Extra args (free text, locked-key guard) (920) | dropped: Ultralytics kwargs; our trainer has no free-form kwargs | X A(4) |
| 25 | **3LC settings** section: Project as a locked row "Session · <project>" with help "The run lands in the same project as the imported tables." (929) | verbatim | V |
| 26 | Run name (placeholder `kaggle_run_<timestamp>`, "Blank generates a timestamped name.") (941) | placeholder `intel-scene_run_<timestamp>` (`<competition id>_run_<YYYYMMDD_HHMMSS>`, CONTEXT.md naming) | A(1) |
| 27 | Conf threshold (0.1) · Max detections (300) (945) | dropped: detection collection settings | X A(4) |
| 28 | **Metrics collection (advanced)** disclosure: Image embeddings off/2D/3D · reducer pacmap/umap · Instance embeddings · Collection start / Interval · checkboxes Collect loss / GT instance embeddings / Val split only / Exclude zero-weight rows (collection) / Disable collection / Use sampling weights / Exclude zero-weight rows (training) (960) | every one of these is fixed by the brief (3D UMAP with PCA fallback, loss always, end of training only, both splits, all rows collected, sampling weights on, weight 0 excluded, undefined excluded). (a) drop the disclosure; (b) keep it collapsed as read-only locked rows titled "Metrics collection (locked)" so the looks match | D7 |
| 29 | Action row: **Start Training** `btn-primary btn-lg` (play icon; "Re-run Training" + refresh icon after a failure) · **Cancel** `btn-secondary` disabled until a job runs · spinner · `#tr-state` live text ("Starting…", "Training… (safe to navigate away)", "Cancelling… (stops at the next checkpoint)", "Fix the highlighted fields first.") (1033) | verbatim; Start disabled while `trRunning`, the gate is not green, any field is invalid, or `kgDevMode` (`trCheckReady` 5089) | V |
| 30 | Inline bounds (`TR_BOUNDS` 5073, `trValidateField` 5094): red `.kg-field-err` under the field on blur "epochs must be between 1 and 300 (got 999).", clears on input, blocks Start; `applyConfig` resets a persisted out-of-range value to the field default | verbatim; the bounds object is filled from `_meta.training.bounds` (message shape unchanged); the server re-checks on the merged kwargs (`build_train_kwargs`) | V |
| 31 | Start click (5129): `trValidateAll`, gate safety net, `saveTabConfig('train')`, body of every field, `kgStartJob('train', body)` (`POST /validate/train` then host `/run`), fresh `#tr-progress` structure, 2 s poll | same, through our `kgStartJob` (no `/validate`: the preflight is the gate and `run_training` re-validates, EXDARK_MIRROR #36); the body carries `train_table_url`, `use_latest`, the fields of §6, `project_name`, `run_name`, `device`, `workers`; the locked facts are never posted (val URL, arch, image size come from the record and the manifest server-side) | A(plumbing) |
| 32 | Cancel click (5199): `window.confirm("Stop this training run? It stops at the next checkpoint; epochs completed so far are kept.")` → `POST /api/plugins/jobs/<id>/cancel` | verbatim via `PluginJobs.cancel` | V |
| 33 | In-run view `trRenderRunView` (4464): `.kg-run-head` (run name · status badge · "Epoch n/N" · "elapsed …" · "≈ … remaining" / "finishing up…") · `.plugin-progress-wrap` bar (indeterminate only during epoch 0 before the first batch; then `(epoch + batch_i/batch_n)/total`) · `.kg-metric-strip` of `.kg-metric` chips with `kgSparkline` (last 30 epochs) · `#tr-run-note` stage note; structure built once, values swapped with `kgSwapText` | verbatim structure; chips = **Train loss · Val loss · Val accuracy** (the brief's per-epoch metrics; three chips, not four); the header meta gains "· cuda (auto)" (the brief's device display, D8 for placement); `#tr-run-note` shows the final-pass note "Collecting per-sample metrics and embeddings on 7,800 rows…" after the last epoch (ExDark's note is the checkpoint fetch at epoch 0) | A(4) +brief |
| 34 | Poll `pollTrain` (4690) every 2 s against the job store; log accordion fills from `job.log`; `kgSetTitleProgress`; connection loss → "Connection lost. Training continues on the host." and the guard resumes | the job channel (`PluginJobs.track` + `stage_progress` / `log_line` / `fact` / `checks` events into the ExDark record shape, as Import does); title progress and the guard verbatim | A(plumbing) |
| 35 | "Show log" accordion `#tr-log-details` (1046) | verbatim, filled from `log_line`; no log on revisit (the host keeps none, Import #33) | V / A(plumbing) |
| 36 | Provenance panel `trRenderProvenance` (4540): verdict "Verified provenance recorded" / "Provenance check failed", group head "Provenance verified", four checks read back from the Run's parameters (model, imgsz, pretrained, checkpoint sha256), cascade on live arrival | same renderer; checks read back from `run.parameters`: `arch == resnet18`, `image_size == 150`, `pretrained == False`, `timm_version == 1.0.29`, `seed` recorded, `train_table_url` (the revision trained on) recorded, `val_table_url == the locked val`, `best_checkpoint_sha256` recorded and equal to the file on disk — eight checks | A(3) |
| 37 | Success banner `trRenderSuccessBanner` (4584): `alert-success` "Training complete: best.pt saved" · weights path (middle-truncated, `title`, Copy "Copy weights path (a path on the compute service's machine)") · **Continue to Submit** · **Open Run in Dashboard** (`dashRunLink`, `?run=`) · **Open Run in Projects** (`/projects/<project>#runs` from the RUN's recorded project) · (revisit) **Start new run** | "Training complete: best checkpoint saved" · checkpoint path + Copy · **Continue to Submit** (D9: verbatim, or "Continue to Predict" since step 1 of tab 3 is Predict) · Open Run in Dashboard verbatim · Open Run in Projects only on `hub.3lc.ai` / `hub-beta.3lc.ai` (our Import rule) · Start new run | V / A(1) |
| 38 | `#tr-result` info callout "Review per-sample losses and embeddings in the Run to find label issues; fix them in the Dashboard and retrain with "Use latest revision" on. That loop is the competition." (4661) | verbatim | V |
| 39 | Cancelled banner (4634): `.kg-callout.info` "Training cancelled after N epochs. The best checkpoint so far was kept. Start Training begins a new run."; the Run is `set_status_cancelled`; a cancelled run with weights is USABLE in Predict (`/runs` 218) | the brief: a clean, clearly-marked cancelled Run and no half-written checkpoint marked best. ExDark keeps the best-so-far (complete) checkpoint. (a) mirror: keep best-so-far, Predict may use it; (b) a cancelled run never offers a checkpoint | D10 |
| 40 | Failure banner (4624): `alert-error` "Training failed" / "Training was interrupted" (status `stale`) + message + Copy diagnostics (`trBuildDiagnostics`: job id, `[train params]`, provenance checks, log tail); CTA → "Re-run Training" | verbatim; `stale` is our restart detection (§7) | V |
| 41 | Terminal orchestrator `trRenderTerminal` (4649): completed on revisit hides the form behind Start new run, live completed keeps the form; failed/stale/cancelled keep the form; focus + scroll to the banner on a live finish | verbatim | V |
| 42 | Tab-open resolution `trInitTrainTab` (5221): `GET /jobs?kind=train` → running record reconnects into the in-run view; else the newest terminal record renders static (full record fetched for checks + log); else the form. Tab enter `trOnTabEnter` (4998): re-derive URLs (`kgApplyDerivedUrls`, `GET /tables/defaults`), re-verify, reload duration stats | same precedence from a durable **train record** (`train_state` in the session store, §7) plus `PluginJobs.list` for the running case; `/tables/defaults` ported for the derivation | A(plumbing) |
| 43 | Session projection (2048–2100): `CFG_FIELDS.train` snapshot saved on Start (`POST /config {train: {...}}`), restored by `applyConfig`; disclosures auto-open for non-default saved values; `kgApplySession` sets `tr-device` from `session.device` | verbatim with our field ids; `session.py` `_ALLOWED_TABS` already has `train` | V |
| 44 | `renderPipeline` (2255): train step done = any train job with `facts.weights` | done = a train record with a best checkpoint on disk | V |
| 45 | `GET /runs` (routes 218): the Predict selector's list (job id, run name, weights, run_url, status, created_at, epochs_completed, best metric, provenance_ok, contract, usable + reason) | session 4 consumes it; the train record written now carries every field it needs (§7), including `best_val_accuracy` and the checkpoint sha256 | plumbing |
| 46 | `?kgdev` Train fixtures `trDevForce` (3832): `train-state1`, `train-state2`, `train-state2-missing`, `train-state2-rows`, `train-state2-invalid`, `train-state3`, `train-state4`, `train-state4-noproject`, `train-state5`, `train-state6`; deterministic curve, `JOB()` composition, numbers imitating backend strings allowed | ported with our names and numbers derived from the served manifest, plus the states the brief adds: `train-state2-undefined-weight` (the > 0 warning), `train-state2-zero-rows` (hard stop), `train-state2-class-empty` (class warning), `train-state2-lineage` (revision outside the seed lineage), `train-state3-collecting` (final pass note), `train-state3-cpu-retry` (accelerator failed, CPU), `train-state5-stale` (interrupted), `train-state6-cancelled` | A + brief |

Components the fragment still lacks and ports verbatim from ExDark for this tab: `kgBindTablePicker`,
`kgTableRef`, `kgUrlSeg` / `kgSplitDataset`, `kgApplyDerivedUrls` / `kgSetUrlOverride` /
`kgRenderProjectFact` / `kgRenderLoopFixLabels`, `kgSwapText`, `kgSparkline`, `kgAccOpenInstant`,
`saveTabConfig` / `applyConfig` (train subset), `dashRunLink`, and the CSS blocks `.kg-locked-row*`
(378–385), `.kg-select-wrap` (389), `.kg-check-cols` / `.kg-field-err` (405–411), `.kg-run-head` /
`.kg-metric-strip` / `.kg-metric` / `.kg-spark` (415–432), `.kg-url-row` / `.kg-pick-wrap` / `.kg-pop*`
(481–505).

## 2. The baseline: training defaults

### The kit (re-checked against the Intel kit's original scripts, 2026-09-29)

- `../hackathon_starter/intel-kit/{train.py, config.yaml, predict.py, register_tables.py, README.md,
  sample_submission.csv}` — the Intel Scene kit as shipped (read-only reference, never modified). The
  table below is read from these files.
- `../hackathon_starter/Starter_Kit/` — the Hack[AI]thon 2.0 kit (Chihuahua vs Muffin, tlc 2.22 API)
  the Phase 0 study used as a stand-in. A `diff` of the two `train.py` files shows the Intel kit is the
  same script with three changes: six classes (`NUM_CLASSES = 6`, `undefined` = index 6 and "MUST remain
  last"), image size **150** in both transform stacks (Chihuahua: 128), and a **labeling budget** block
  (`MAX_WEIGHT1_ROWS = 3000`) that counts the `weight > 0` rows of the loaded train revision and exits
  before any Run is created when the count exceeds 3,000. `config.yaml` differs the same way
  (`constraints.max_weight1_rows: 3000`, `training.image_size: 150`, six class names, split counts) and
  says of itself "informational only — scripts use their own constants". Every other training default is
  byte-identical, so the Phase 0 table needed one value corrected (image size, already the manifest's)
  and four rows added (budget, undefined-with-weight behaviour, val revision, per-epoch logging).
- The bundled manifest (`manifests/intel-scene-v1.yaml`) `training.defaults` = epochs 10, batch_size 16,
  lr 0.0001, weight_decay 0.0, optimizer adam, seed 42; `model.image_size` 150 — all confirmed against
  the kit. The manifest carries **no** labeling budget, and PLAN §A locks "Labeling: No cap" (D14).
- Native image sizes (measured over `../data-source/3-lc-hack-nova-scene-classification-challenge/data`,
  2026-09-29): 150 × 150 for 6,582 of 6,600 train, 1,193 of 1,200 val and 1,797 of 1,800 test images;
  the other 28 are 150 wide and 72–149 high. `Resize(150)` resizes the shorter side, so the
  resize-then-crop pair is an identity on every 150 × 150 image.

### Intel kit defaults (`train.py` constants; `config.yaml` "reference" block agrees)

| Setting | Kit value | Where | Ours (proposed) | Forced change? |
|---|---|---|---|---|
| Epochs | 10 | `EPOCHS` | 10 (manifest) | no |
| Batch size | 16 for the train and val loaders and for collection (`predict.py` uses 32) | `BATCH_SIZE` | 16 (manifest) | no |
| Optimizer | `torch.optim.Adam(lr=1e-4)`: no weight decay (Adam's default 0), default betas; `optimizer.zero_grad()` per batch, no grad clipping, no AMP | `train()` | adam, lr 1e-4, wd 0.0 (manifest) | no |
| Schedule | `StepLR(step_size=5, gamma=0.1)`, `scheduler.step()` once per epoch after validation: lr 1e-4 for epochs 1–5, 1e-5 for 6–10 | `train()` | not in the manifest — D2 | no (a schema-v1 addition if exposed) |
| Seed / determinism | 42: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`; `cudnn.deterministic=True`, `cudnn.benchmark=False`; `PYTHONHASHSEED` | `set_seed` | 42 (manifest), same procedure | no |
| Image size | **150**: `Resize(150)` → `RandomCrop(150)` (train) / `CenterCrop(150)` (val, collection and `predict.py`) | transforms | 150 (manifest, locked), same recipe; on this data the pair only rescales-and-crops the 28 shorter images | no (corrected: Phase 0 carried the Chihuahua 128) |
| Augmentation | `RandomHorizontalFlip()`, `RandomAffine(0, shear=10, scale=(0.8, 1.2))`, `ToTensor`; val = resize + center crop + `ToTensor` | transforms | the kit's, fixed (not fields). Effective train augmentation here is flip + affine (the crop is an identity on 150 × 150). The timm plugin's `create_transform` (RandomResizedCrop + flip + colour jitter) would be a different baseline | D11 (kit vs timm pipeline) |
| Normalization | ImageNet mean/std spelled out, `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`, on every path; images converted to RGB first | `train_fn`/`val_fn` | the same constants (spelled out, or from `model.pretrained_cfg` — identical values for timm resnet18) | no |
| Workers | 0 in every loader (`config.yaml` `num_workers: 0`) | `DataLoader` | device-aware (brief) | brief |
| `drop_last` / `pin_memory` | not set (False / False) | `DataLoader` | False / False (the timm plugin uses True / True; `drop_last=True` would drop 8 of 600 rows per epoch) | no |
| Sampler | `train_table.create_sampler(exclude_zero_weights=True)` (2.x Table method; `weighted=True` by default → `WeightedRandomSampler`, epoch = non-zero-weight rows); val loader `shuffle=False` | `train()` | `tlc.integration.torch.samplers` semantics, effective weights in memory (§3) | **yes** (tlc 3.x: the Table method is gone) |
| Weights at registration | labels 0–5 → weight 1.0; `undefined` (label 6) → weight 0.0; `SampleWeightSchema` column | `register_tables.py` | the importer writes the same | no |
| Undefined rows in training | excluded ONLY through weight 0; there is no label filter. An `undefined` row given weight > 0 in the Dashboard IS sampled, and `CrossEntropyLoss` with target 6 against 6 logits raises (`IndexError` on CPU, a device-side assert on CUDA): the kit crashes mid-epoch. Only the post-training metrics pass masks label 6 (`labels < num_logits`) | `train()` / `metrics_fn` | filtered out regardless of weight (PLAN §A, brief); #9 warns "N unlabeled rows have weight > 0 and will be skipped" instead of failing | brief (a deliberate softening of a kit defect) |
| Labeling budget | `MAX_WEIGHT1_ROWS = 3000`: `sum(1 for row in train_table.table_rows if row["weight"] > 0)` over the loaded revision (every `weight > 0` row counts, undefined included, though the message says "weight = 1"); over the cap → printed remedy (set weights back to 0, or pin an earlier revision by URL), `sys.exit(1)`, no Run created. `config.yaml` `constraints.max_weight1_rows: 3000`; README: "your final train table may have at most 3,000 rows with weight = 1 (the 600 seed rows count toward this)" | `train()` before `tlc.init` | **not in PLAN §A ("Labeling: No cap") and not in the manifest** — D14 | competition-design decision |
| Table revisions | train AND val loaded by name with `.latest()` (OPTION 1, default); OPTION 2 (commented out) pins both by URL; the budget check runs on either path | `train()` | train: latest by default with the picker (#6, #11); val: LOCKED to the import record's URL (brief; PLAN §A "val 1,200 … locked") — the kit would follow val edits, ours does not | brief |
| Model | torchvision `resnet18(weights=None)`, `fc = Identity`, custom head 512→256→ReLU→Dropout(0.3)→128→ReLU→Dropout(0.3)→6 | `ResNet18Classifier` | `timm.create_model("resnet18", pretrained=False, num_classes=6)` — the plain timm head (PLAN §A, locked). The kit's MLP head is NOT mirrored | **yes** (the locked contract names timm) |
| Loss | `CrossEntropyLoss()` (mean reduction) | | same | no |
| Best checkpoint | by val accuracy (strict `>`, so epoch 1 always becomes the first best), state dict kept in memory, saved once at the end as `best_model.pth` beside the script (overwritten each run) | | best by val accuracy + last, saved under the run (§4, D12) | brief |
| Per-epoch logging | `tlc.log({"epoch", "val_accuracy"})` only; train loss is never aggregated and val loss never computed | | train loss · val loss · val accuracy per epoch (#33, brief) | brief |
| Per-sample metrics | after training, on the best model, TRAIN only: loss (masked to `labels < num_logits`, masked rows set to 1.0), predicted, accuracy, confidence via `FunctionalMetricsCollector`; embeddings from the `fc` (Identity) layer via `EmbeddingsMetricsCollector` + `tlc.Predictor`, batch 16, workers 0; `run.reduce_embeddings_by_foreign_table_url(train_url, method="umap", n_neighbors=15, n_components=3)` inside a try/except that only warns | `train()` | train AND val, hand-rolled per the timm plugin, loss masked (absent, not 1.0), per-class probabilities added, UMAP fit on train / transform val, PCA fallback (§4) | **yes** (tlc 3.x removed `table.map()`; `Predictor`/collectors are not used by the reference trainer, STUDY G-1) |
| tlc API | `tlc.init(project_name, description)`, `tlc.log({epoch, val_accuracy})`, `run.set_status_completed()`, `register_project_url_alias` | | `tlc.init(project_name, run_name, description)`, `tlc.log` per epoch, `run.set_parameters(contract)`, `set_status_collecting/completed/cancelled`; no alias | no |
| Device | `cuda` if available else `cpu`, chosen at import time; no `mps`, no retry | module level | ExDark's `resolve_device` + the CPU retry (#22) | brief |

**tlc 3.3 forces**: `table.map(fn)` → `table.with_transform(fn)` (`TableView`, not a `Table`);
`Table.create_sampler` → the `tlc.integration.torch.samplers` factories; metrics via
`run.add_metrics(metrics, schema=, foreign_table_url=, constants=)` (signature confirmed in 3.3.2).
`run.reduce_embeddings_by_foreign_table_url` still exists in 3.3 but swallows failures
(returns `{}` on exception) and offers no PCA fallback, so the reducer is hand-rolled as in the timm plugin.
**timm forces** nothing beyond the model call; `model.pretrained_cfg` still resolves for a from-scratch
resnet18 (test_timm_model.py), so ImageNet mean/std can come from it or be spelled out as the kit does.

## 3. Sample weights: tlc 3.3's sampler

Source: `tlc/integration/torch/samplers/_sampler.py` (3.3.2; the 3.3.0 host venv exposes the same
API). `tlc.Table` has NO `create_sampler` method in 3.x (grep of `_core/objects/table.py`); the kit's
`train_table.create_sampler(...)` becomes `tlc.integration.torch.samplers.create_sampler(train_table, ...)`.

`create_sampler(table, exclude_zero_weights=True, weighted=True, shuffle=True, repeat_by_weight=False)`
dispatches:

| Flags | Returns | Epoch length | Weight semantics |
|---|---|---|---|
| `weighted=True` (default), `exclude_zero_weights=True` (default) | `create_weighted_sampler` → `torch.utils.data.WeightedRandomSampler(weights=<ALL rows' weights, zeros included>, num_samples=<count of rows with weight != 0>, replacement=True)` | number of non-zero-weight rows | draw probability ∝ weight; weight 0 → probability 0 (never drawn); with replacement, so a row can be drawn 0 or several times per epoch; expected draws per epoch = `num_samples × w_i / Σw` |
| `weighted=True`, `exclude_zero_weights=False` | same sampler with `num_samples=len(table)` | full table length (zero rows still never drawn) | as above |
| `weighted=False`, `shuffle=True` | `SubsetRandomSampler(non_zero_indices)` / `RandomSampler` | non-zero rows / all | uniform; weights ignored beyond the zero cut |
| `weighted=False`, `shuffle=False` | `SubsetSequentialSampler` / `RangeSampler` | as above | sequential |
| `repeat_by_weight=True` (needs both flags True) | `RepeatByWeightSampler`: row `i` appears `int(w_i)` times plus once more with probability `frac(w_i)`, shuffled; weights must be floats | ≈ Σ weights | weight = exact repeat count per epoch |
| `weighted=True, shuffle=False` | `ValueError("Cannot create a weighted sampler without shuffling.")` | | |

Preconditions (`_get_weights_and_non_zero_indices`): the table must have a column with the
sample-weight number role (`table.weights_column_name`, else `ValueError`); with
`exclude_zero_weights=True` and no non-zero row → `ValueError("Cannot create a sampler with
exclude_zero_weights=True because the table has no rows with non-zero weights.")`. Weights are read
row by row from `table.table_rows` (the Table's own weight column) — the factory cannot take
weights from anywhere else.

**Which matches "weight = sampling frequency, 0 = excluded"?** Both the default weighted sampler and
`repeat_by_weight`. The default (`weighted=True, exclude_zero_weights=True` →
`WeightedRandomSampler`) is: what `create_sampler` returns with no flags, what the kit called
(`create_sampler(exclude_zero_weights=True)` with 2.x's `weighted=True` default), what the timm plugin
uses when "Use sampling weights" is ticked, and what G4's statistical test describes ("weight 2
sampled about twice as often"). Epoch length = usable rows, which is also what the ETA scales by.
Recommendation: the default weighted sampler. `repeat_by_weight` would make weight 2 exactly two
draws and change the epoch length to Σw; listed for completeness, not proposed.

**Effective weights (undefined → 0, in memory, never written back).** Because the factory reads
the Table's column, the plugin builds the same torch sampler itself from an in-memory copy:
`eff = weights with eff[label == undefined] = 0`;
`WeightedRandomSampler(weights=eff, num_samples=int(count(eff != 0)), replacement=True)` — identical
to `create_weighted_sampler` except for the weight source; the table is never modified. The
pre-training summary (#9) derives from the same arrays: labeled in use = `count(eff != 0)`,
excluded as undefined = `count(label == undefined)`, excluded at weight 0 = `count(weight == 0 and
label != undefined)`, warning count = `count(label == undefined and weight > 0)`; zero usable rows
is the hard stop before the sampler is built (the factory's `ValueError` never reaches the participant).
Determinism: `WeightedRandomSampler` draws from torch's global generator, so `torch.manual_seed(42)`
fixes the draw sequence; G4 samples many epochs and compares per-row draw counts to `w_i / mean(w)`.

**How the timm plugin uses it** (`trainer.py` 240–258): only when `sampling_weights` or
`exclude_zero_weight_training` is set, `create_sampler(train_table, exclude_zero_weights=excl,
weighted=sw)` and `shuffle=False` on the `DataLoader` (`drop_last=True`, `pin_memory=True`); on ANY
exception it logs a warning and falls back to plain shuffling over every row — which we must not do
(hard stop instead). Both flags default to off there, so weight-0 rows train in the timm plugin
unless ticked; ours are always on.

## 4. The timm plugin's trainer: what we reuse

`../reference/3lc-compute-plugin-timm/src/tlc_plugin_timm/` (`src/` identical between `b77f252`
and the published v0.2.6). Apache-2.0, no relicensing header needed; STUDY G-5 lists only ExDark
adaptations.

| Part | File:lines | Reuse |
|---|---|---|
| `_SampleTransform` (top-level, picklable; dict samples by column name; `_as_rgb_image` through `tlc_plugin_sdk.shared.images.load_image`) | `trainer.py` 40–88 | reuse as is (workers > 0 off Windows needs the picklable class) |
| Table load, `num_classes` from `get_value_map` with fallbacks | 149–178 | not reused: `num_classes` is the manifest's; the table's value map is VERIFIED against `import_state.label_map` (a mismatch is a refusal) |
| Model: `timm.create_model(name, pretrained=…, num_classes)`, optional checkpoint load | 188–218 | the `pretrained=False` branch only; no checkpoint path |
| Transforms: `resolve_data_config(model.pretrained_cfg)` + `create_transform(is_training=…)` | 220–235 | D11 — the kit's torchvision recipe is proposed instead |
| `TableView`s via `with_transform`; loaders (`shuffle=False` with a sampler, `pin_memory`, `drop_last=True`) | 237–270 | shape reused; our sampler; `drop_last=False` (§2) |
| `_create_optimizer` (sgd momentum 0.9 / adam / adamw / lamb) | 1038–1055 | reuse for the chosen optimizer list (D3) |
| `_create_scheduler` (cosine / step T/3 / plateau=None, linear warmup) | 1058–1103 | only if D2 picks a timm-style schedule; the kit's `StepLR(5, 0.1)` is one line |
| Training loop: per-batch cancel check, batch progress every 10 %, epoch train/val loss + val accuracy, `tlc.log`, best state dict on CPU by val accuracy, `best_epoch` | 366–470 | reuse; batch progress becomes ExDark's throttled `batch_i/batch_n` flush (≤ 1/s) and the per-epoch `history` for the chips |
| Cancel path: break out, `run.set_status_cancelled()`, skip collection, no checkpoint saved | 471–490 | reuse the status handling; D10 for the checkpoint |
| Restore best, final collection pass on the best model with the val transform | 491–520 | reuse |
| `_collect_metrics`: `forward_features` → `forward_head(pre_logits=True)` embeddings + `forward_head` logits, softmax → predicted / confidence, `CrossEntropyLoss(reduction="none")`, fit reducer on train + `transform` val, `run.add_metrics(..., foreign_table_url, schema={embeddings: Float32Schema(shape=(dim,)), predicted: deepcopy(label schema)}, constants={"epoch": epoch})` | 744–968 | reuse with four changes: per-class probabilities as `prob_<class>` columns; loss MASKED (their loop computes CE for every row — with label 6 ≥ 6 logits that raises, so the mask is required, and the masked rows carry no value); the val pass writes to the LOCKED val URL; a single final pass (no periodic collection) |
| `_fit_reducer` (UMAP `n_neighbors=min(15, n-1)`, `min_dist=0.1`, `random_state=42`; pacmap) + `_transform_embeddings` | 970–1008 | reuse the UMAP branch; drop pacmap; add the PCA fallback (`sklearn.decomposition.PCA(n_components)`) and record `reducer` on the run |
| Checkpoints: `save_model_to_run(run_url, state_dict, "best_model.pt")` → `<run>/model/best_model.pt`; `store_model_info_in_run` → `run.set_parameters({model_name, model_path})` | 522–540; SDK `shared/model_storage.py` | proposed (D12): `model/best.pt` + `model/last.pt` under the run folder via the SDK helper, sha256 of each in `run.parameters` and in the train record. The alternative is ExDark's plugin-home layout (`<plugin home>/runs/<run id>/weights/best.pt`) |
| `run_job` wiring: `on_epoch` / `on_status` / `is_cancelled` callbacks; `_build_timing` (`elapsed_s`, `avg_epoch_s` from measured epoch durations, `eta_s`); `epoch_progress` → `ctx.progress(percent, label, timing)`; `ctx.result(run_url)`; `ctx.emit` custom events; final `ctx.progress(100, "Done")` | `__init__.py` 150–290 | reuse the callback shape and the timing dict; our `_JobCtxAdapter.set_progress` already emits `stage_progress` (the ExDark progress payload rides in it) |
| GPU reclaim | SDK `worker.py` 168 `release_gpu_memory()` after every job | nothing to add (ExDark's `reclaim_gpu_memory` predates this) |
| Not reused | config store, alias overrides, mixup/cutmix/auto-augment/reprob/colour-jitter params, periodic collection, `collect` mode, task detection, `generate_name` | our run name is `<competition id>_run_<timestamp>` |

## 5. Estimated time to completion — ExDark's actual behaviour, and the brief's extension

ExDark v1.2.15 (`ui.html` 4966–4996, 4464–4530; `trainer.py` `on_fit_epoch_end`):

1. **Pre-run hint** (`trLoadDurationStats`, on tab enter): `GET /jobs?kind=train` → the five most recent
   records with status `completed` or `cancelled` that carry a numeric `progress.avg_epoch_s` → their
   **median** → `trEpochSecs`. Rendered under Epochs (`#tr-duration`), recomputed on every Epochs
   keystroke: "Recent runs averaged ~96 s/epoch on this machine. 20 epochs ≈ 32m 0s." (`~N min/epoch`
   when ≥ 60 s). Rendered only when history exists — no placeholder on a first run.
2. **Backend measurement**: `avg_epoch_s = (now − train_start) / epoch` at every epoch boundary
   (`train_start` is taken before `model.train()`, so setup time is amortised in), `eta_s = avg ×
   (total − epoch)`, both flushed with the epoch's progress; `batch_i/batch_n` flushed at most once per
   second.
3. **In-run remaining**: on every 2 s poll `perEpoch = progress.avg_epoch_s || trEpochSecs`,
   `remaining = max(0, perEpoch × total − (now − created_at))`; "≈ 12m 4s remaining", "finishing up…"
   below 1 s, nothing when no pace is known (first run before epoch 1). The bar is
   `(epoch + batch_i/batch_n) / total`.
4. **Not in ExDark**: no scaling by row count (its splits never change size), no device
   classification, no bundled benchmark, no collection-pass term.

The brief asks for the same behaviour **scaled by epochs and usable row count, with a bundled
per-device benchmark for the first run**. That is an extension; the shape I propose, for your go:

- Per finished run the train record stores `device_class` (`cuda` / `mps` / `cpu`), `usable_rows`,
  `avg_epoch_s`, `epoch_s_per_row = avg_epoch_s / usable_rows`, and `collect_s` (the final pass,
  which is a fixed cost scaled by total rows, 7,800 here).
- Hint = median `epoch_s_per_row` over the last five runs on the SAME device class × usable rows
  (from the gate's summary, so it moves with the revision) × epochs + median `collect_s`; copy:
  "Recent runs averaged ~X s/epoch on this machine (603 usable rows). 10 epochs ≈ D."
- First run on a device class: a bundled `benchmark.py` constant `{device_class: {epoch_s_per_row,
  collect_s_per_row}}` seeded from G1 (a plugin fact, not a competition fact, so a code constant is
  allowed); copy: "Estimated from a reference GPU run: 10 epochs ≈ D." The in-run seed and the
  recompute rule stay ExDark's.
- The stats live in the train record (`train_state.history`), because compute 1.1.0 keeps job records
  in memory only (`job_manager.py` `_MAX_RECORDS = 50`, nothing survives a restart) and its generic
  list carries no `kind`, `params` or `progress.avg_epoch_s`.

## 6. Hyperparameter fields (from the mapping; final list after D2–D7)

| Field | Default | Bounds | Source | Units / notes |
|---|---|---|---|---|
| Epochs | 10 | 1–50 | manifest | passes over the usable rows |
| Batch size | 16 | 8–128 | manifest | images per step |
| Learning rate | 0.0001 | 0.00001–0.01 | manifest | Adam initial lr |
| Weight decay | 0.0 | 0.0–0.1 | manifest | L2 (D5: core or Advanced) |
| Optimizer | adam | choice list | D3 | |
| Schedule | StepLR(5, 0.1) | — | D2 | |
| Seed | 42 | [0, 2147483647] to add to schema v1 | manifest default; D6 (field or locked) | |
| Device | blank (auto) | free | session | Advanced |
| Workers | device-aware (0 on Windows) | 0–16 (plugin constant) | brief | Advanced |
| Run name | blank → `intel-scene_run_<ts>` | free | | |
| Presets `quick / standard / long` (manifest) | | | ExDark has no presets; PLAN §C names them | D13: render (how? ExDark has no control for it) or ignore for the mirror |

Every bounded field: description, default, bounds and units in the `.form-help` ("Range a to b."),
inline `.kg-field-err` on blur, Start disabled while invalid, server re-check on the merged kwargs
with the same message.

## 7. Plumbing (invisible, ours) the brief requires

- **Durable train record** `train_state` (add to `session.py` `_ALLOWED_TABS`): written at start
  (`status: running`, job id, params, resolved train revision URL, locked val URL, device, usable-row
  summary, worker pid, `started_at`, heartbeat), updated at every epoch boundary (history,
  `avg_epoch_s`) and on every terminal state; `history[]` keeps the last N finished runs for the ETA and
  for session 4's run list. This replaces ExDark's on-disk job store (`jobs.py`), since the host's
  records are in-memory and generic.
- **Restart / sleep detection** (ExDark: `_mark_if_orphaned` by worker pid at read time). Ours, at
  `GET /train/state`: a `running` record whose job id is absent from the host's job list (`PluginJobs.list`
  — the host forgets everything on restart; the supervisor kills workers on shutdown and reaps idle
  ones after 300 s) or whose heartbeat is older than a threshold while no host record is active →
  `stale`, banner "Training was interrupted" (ExDark's wording), never `completed`. While the host is
  up and the worker dies, the host itself marks the job `failed` with the transport error. A sleep
  that resumes shows as a heartbeat gap logged into the run ("No progress for 23 min — the machine
  slept?") and continues; it is never reported as complete.
- **Double-click Start**: the fragment disables the button synchronously (`trRunning`, ExDark); the
  worker refuses a second train while the record says `running` with a live heartbeat
  (`ctx.fail("A training run is already in progress (…)")`). The host's `GpuQueue` would serialise
  (queue) a second GPU job, not run it concurrently, but it must not start at all.
- **Cancel**: cooperative at batch and epoch boundaries (`ctx.cancelled`); the Run is
  `set_status_cancelled`; checkpoints are written atomically (`.tmp` + `os.replace`) and `best` is only
  ever pointed at a fully written file, so a cancel mid-write leaves no half-written best (D10 decides
  whether a cancelled run keeps its best at all).
- **Accelerator failure → CPU retry** (brief): the device is resolved in the worker; if model
  creation or the first training batch raises a device error (`RuntimeError` naming CUDA/MPS, out of
  memory included), the run logs it, re-resolves to `cpu`, restarts from epoch 1 and the in-run view
  shows the note; the record stores `device_requested`, `device`, `device_fallback_reason`.
- **Tables are never modified**: reads only (`Table.from_url`, `.latest()`, `table_rows`), metrics
  tables are written under the Run by `add_metrics`, effective weights stay in memory. A test
  snapshots the table folders' mtimes across a training run.
- **New routes**: `GET /train/preflight` (#8–#10), `GET /train/state`, `GET /tables/list`,
  `GET /tables/defaults` (ExDark's, ported). No `/validate/train` (EXDARK_MIRROR #36).

## 8. Decisions (taken by Rishikesh on 2026-09-29; the recommendations below were all accepted)

Re-checked against the Intel kit's original scripts on 2026-09-29 (§2): the kit confirms the values
behind D2 (StepLR 5 / 0.1), D3 (Adam only), D4 (no early stopping), D6 (seed 42) and D11 (the
torchvision recipe at 150 px). No recommendation changed. One decision was added: **D14**, the kit's
3,000-row labeling budget, which PLAN §A rules out ("Labeling: No cap") and the manifest does not carry.

**Taken:** D1 gate block · D2 (a) fixed StepLR as a locked row "LR schedule: ×0.1 every 5 epochs"
with a tooltip · D3 (a) Adam as a locked row · D4 (a) dropped · D5 core grid · D6 Advanced field,
bounds `[0, 2147483647]` in the bundled manifest (plugin fallback when a manifest lacks them),
default 42, the kit's determinism · D7 (a) dropped · D8 header meta · D9 verbatim · D10 (a) ·
D11 the kit's recipe verbatim · D12 SDK layout, sha256 of best and last on the Run and in the record,
PLAN.md "Predict uses best" · D13 not rendered · D14 (a) No cap. Plus: the model is timm resnet18
with its standard head (the kit's MLP head is not replicated; PLAN §A notes that baseline accuracy
may differ from past HackNova runs). Implemented in `aaa58a9` (Phase 1); §9 lists what still
differs from ExDark beyond the four allowed differences.

| # | Question | ExDark | Recommendation |
|---|---|---|---|
| D1 | Where the usable-row summary and its warnings render | no equivalent | inside the gate block under the green line (re-evaluates with the revision) |
| D2 | Learning-rate schedule | `lrf` field (Ultralytics decay) | fixed `StepLR(5, 0.1)` from the kit, no field (kit-confirmed: stepped once per epoch, lr 1e-4 → 1e-5 at epoch 6) |
| D3 | Optimizer choices | Auto / SGD / Adam / AdamW select | `adam` only (the kit; no other optimizer appears in it) as a locked-looking select, or `training.optimizers` in schema v1 |
| D4 | Patience / early stopping | field, 0–100 | drop (kit-confirmed: no early stopping; best-by-val-accuracy only) |
| D5 | Weight decay placement | — | core grid, fourth field (the kit has no knob and uses 0.0; the manifest already bounds it 0–0.1) |
| D6 | Seed | — | Advanced field, bounds added to schema v1 (kit-confirmed default 42 and the full determinism procedure, §2) |
| D7 | Metrics-collection disclosure | 12 controls | drop; the facts are stated in the contract tooltip and the result callout |
| D8 | Where the resolved device shows during a run | log line only | in-run header meta "· cuda (auto)" |
| D9 | Success CTA label | "Continue to Submit" | "Continue to Submit" verbatim (the tab is Predict + Submit in both plugins) |
| D10 | Cancelled run keeps its best-so-far checkpoint and is usable in Predict | yes | yes (mirror); "no half-written best" is guaranteed by atomic writes |
| D11 | Augmentation recipe | n/a | the kit's torchvision recipe, fixed: `Resize(150)` → `RandomCrop(150)` → flip → `RandomAffine(0, shear=10, scale=(0.8, 1.2))` → ImageNet normalize; val/collection `Resize(150)` → `CenterCrop(150)` → normalize. Kept verbatim although the crop pair is an identity on the 99.7 % of images that are natively 150 × 150 (§2): it is the kit's recipe and it handles the 28 shorter images the same way the kit does |
| D12 | Checkpoint location | `<plugin home>/runs/<name>/weights/best.pt` (Ultralytics layout) | `<run>/model/best.pt` + `last.pt` via the SDK helper, sha256 recorded on the run and in the record; PLAN.md gets "Predict uses best" |
| D13 | Manifest presets (`quick / standard / long`) | none | not rendered (no ExDark control); presets stay a manifest fact for later |
| D14 | **Labeling budget.** The Intel kit refuses to train (exit before any Run) on a revision with more than 3,000 `weight > 0` rows (`MAX_WEIGHT1_ROWS`, `config.yaml` `constraints.max_weight1_rows`, README "at most 3,000 rows with weight = 1, the 600 seed rows count"). PLAN §A locks "Labeling: No cap"; the manifest has no such field. Options: (a) keep No cap (PLAN §A stands; the gate's usable-row summary is informational only); (b) adopt the cap as a competition rule: a schema-v1 manifest field (`splits.train.max_weight1_rows`, absent = no cap — a competition constant never lives in code), a hard stop in `GET /train/preflight` with ExDark's row-budget geometry (amber problem row, Start disabled: "This revision has N labeled rows in use; the competition allows at most 3,000. Set weights back to 0 in the Dashboard or pick an earlier revision."), counted as labeled rows at weight > 0 (#9's "labeled rows in use", which is what the kit's count means once undefined rows are excluded), recorded in `run.parameters` and the ledger, and PLAN §A's Labeling row rewritten | ExDark's only budget is the table row count (`max_rows`) | **(a)** unless the new competition's published rules keep the 3,000 cap. This is a competition-design decision (CLAUDE.md A5), not mine: PLAN §A was locked with the kit in view and says No cap. If the rules keep it, (b) exactly as written, and the count must be the labeled-at-weight->0 number, not the kit's literal `weight > 0` (which would count undefined rows the plugin never trains on) |

## 9. What the port differs in, beyond the four allowed differences (for Rishikesh, not invented)

Everything below is a place where the shipped Train tab is NOT ExDark's, and why. Each is either a
consequence of a decision above, of the brief, or of the plumbing (the host job channel instead of a
polled job store). Anything you want copied back verbatim is a one-line change.

| # | Ours | ExDark | Why |
|---|---|---|---|
| 1 | **Val table** is a locked row in the Tables grid (revision name · row count · Copy), no URL field, no picker | a second URL field + picker | the brief locks val to the import record (#3, #6) |
| 2 | **Six locked rows** rendered from `/config`: Model (arch · timm version), Init, Image size, Optimizer, LR schedule, Inference | three static rows: Model, Init (checkpoint sha), Image size | D2, D3 and the brief's "single forward pass at inference"; the values are served, so the rows render rather than sit in markup |
| 3 | The gate's green block carries the **usable-row line** and the two amber warnings (undefined at weight > 0, empty classes); the amber problems add **not in the seed lineage** and **class map not the imported one** | table names + revision choice + not-on-disk / same-table / row budget | D1; the brief's lineage rule and label-map verification |
| 4 | Training parameters = **Epochs · Batch size · Learning rate · Weight decay**; Advanced = **Device · Workers · Seed** | Epochs · Batch · lr0 · lrf · Optimizer · Patience; Advanced = Device · Workers · Extra args | D2–D6; no free-form kwargs exist in our trainer |
| 5 | 3LC settings = Project + Run name only; **no Metrics-collection disclosure** | + Conf threshold · Max detections; a 12-control disclosure | task-inherent (no detections); D7 |
| 6 | In-run chips: **Train loss · Val loss · Val accuracy (%)**; the header meta shows **"· cuda (auto)"** (or "· cpu after a GPU failure"); the stage note also shows during the final per-sample pass | four detection chips; device only in the log; the note only during the checkpoint fetch at epoch 0 | task-inherent; D8; the brief's collection pass |
| 7 | The **ETA hint** scales the history median per usable row by the gate's row count and adds the final pass; a first run uses the bundled per-device benchmark ("Estimated from a reference GPU run") | median `avg_epoch_s` of the last five runs, no scaling, no first-run hint | the brief (§5) |
| 8 | The in-run "remaining" adds the collection pass to the epoch estimate | epochs only | our final pass is a fixed cost ExDark does not have |
| 9 | **Provenance = eight checks** (arch, image size, pretrained, timm version, seed, train revision, locked val, best checkpoint sha256 == file on disk) | four (model, imgsz, pretrained, checkpoint sha) | the from-scratch contract records different facts |
| 10 | Success banner: "Training complete: **best checkpoint saved**", Copy "checkpoint path"; **Open Run in Projects** renders only on `hub.3lc.ai` / `hub-beta.3lc.ai` | "best.pt saved", "weights path"; the Projects link always renders | naming; the Import tab's origin rule |
| 11 | Terminal state comes from the **durable train record** (`GET /train/state`): checks, result, facts and the last 300 log lines; on revisit the log shows from that record | the job store's full record and log | the host keeps records in memory only (PLAN §A3) |
| 12 | A running record the host no longer lists renders as **"Training was interrupted"** (`stale`), never as running or complete; the server marks it stale by worker pid and by a 15-minute heartbeat gap | `_mark_if_orphaned` by service pid at read time | the brief's restart / sleep rule; same shape, our record |
| 13 | **Cancel** disables its button until the cancel request returns; the state text switches to "Cancelling…" from the response | the button stays enabled; the text switches immediately | a double-cancel guard; cosmetic |
| 14 | Start posts a **client token**; the worker refuses a second start carrying a consumed token or while a run is in progress | the synchronous button disable only | the brief's double-click rule through the host's serial GPU queue |
| 15 | No `/validate/train` (no fail-fast 400): the gate is the preflight and the job's refusal (`ctx.fail`) carries the message verbatim | `POST /validate/train` before `/run` | EXDARK_MIRROR #36 |
| 16 | The Loop's **fix labels** link follows the import record's latest train revision (session 2.5 decision), not the Train field's value | `kgRenderLoopFixLabels` follows the train field | already decided in EXDARK_MIRROR #4 |
| 17 | `kgApplyDerivedUrls` derives only the train field; `KG_URL_FIELDS` has one entry | three fields (train, val, test) | #1 and no test table |
| 18 | The **Epochs help** reads the reference accuracy from the served benchmark ("A from-scratch run reached about X % by epoch N on the reference GPU") once gate G1 seeds it; until then the range only | a hand-written pretrained calibration sentence | the fragment carries no literal |

## 10. Gates (2026-09-29, laptop, `3lc-hub-11`, catalog install pinned to `aaa58a9`)

Driven by `../3lc-hub-11/gates_session3.py` through the in-process host (live auth exempt, the
service stopped, the redirected home), after `reinstall_pinned.py` (30 s, plugin state kept). The
tables are the real ones under `C:/Users/rishi/AppData/Local/3LC/3LC/projects/intel-scene`; the
`manual-test` EditedTable is `latest()` of `intel-scene_train/initial`. RTX 3070 Ti Laptop GPU (8 GB),
CPython 3.12.13, torch 2.14 cu126, timm 1.0.29, tlc 3.3.2, batch 16, workers 0.

| Gate | Result |
|---|---|
| G1 default run (auto → `cuda`, 10 epochs, use latest) | **PASS** — trained on `manual-test`, 609 usable rows; 4.1 s/epoch (0.00673 s/row); best val accuracy **57.58 %** at epoch 9; final per-sample pass 75.2 s over 7,800 rows (includes UMAP's numba compile; a later run's pass took 22 s); **148 s end to end**; 8/8 provenance checks; best.pt and last.pt under the run |
| G1b ETA accuracy (2 epochs) | history hint ≈ 70 s vs actual 32 s (train 7.2 s + pass 22.3 s): the hint carried the first run's cold 75 s collection pass; the median self-corrects after two runs. In-run "remaining" uses the measured `avg_epoch_s` from epoch 1 |
| G1-cpu (forced `cpu`, 3 epochs) | **PASS** — 25.8 s/epoch (0.04231 s/row), best val accuracy 47.08 % at epoch 3, pass 81.8 s, 162 s end to end; device recorded `cpu`, requested `cpu` |
| G2 Run + per-sample metrics | **PASS** — the Run under `intel-scene` carries two metrics tables: train 6,600 rows and val 1,200 rows with `predicted`, `confidence`, `prob_<6 classes>`, `loss`, `embeddings` (3-D), `epoch`; the 5,991 undefined rows have predictions, confidence and embeddings and **no loss** (NaN); every val row has a loss. Dashboard colouring by confidence / predicted / loss is Rishikesh's visual check |
| G3 labeling loop | **PASS** — `use_latest` resolved `initial` → `manual-test`, usable rows **609** (the laptop's revision has 9 pool images labeled; the brief's 603 was the office count) |
| G4 weight semantics | **PASS** — `tests/test_trainer.py::test_weight_semantics_gate_g4` (weight 0 never drawn, undefined at weight > 0 never drawn and warned, weight 2 drawn 2.0× ± 10 %, epoch length = usable rows, table bytes unchanged); the preflight on the real seed: 600 in use / 6,000 undefined; on `manual-test`: 609 / 5,991, `buildings` 109 |
| G5a cancel mid-run | **PASS** — cancelled during epoch 2 of 5, record `cancelled`, best.pt from epoch 1 kept on disk, the Run's status `cancelled`, provenance checks recorded, no collection pass |
| G5b double-click Start | **PASS** — two `POST /run` with the same client token: the first completed, the second (queued behind it on the GPU queue) failed with "This Start request was already used by the previous run (a double click?). Press Start again." and left no run |
| G5c compute restart mid-run | **PASS** — the host shut down at epoch 1 (workers killed); a fresh host's `GET /train/state` answered `stale` with "Interrupted: the compute service restarted while this run was training.", the host listed no running job, best.pt from epoch 1 on disk |

Found and fixed along the way: none in the plugin. The gate script itself needed UTF-8 stdout
(the console is cp1252) and a worker-kill filter that does not match its own PowerShell process.
