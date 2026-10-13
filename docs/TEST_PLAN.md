# TEST_PLAN.md: the hardening campaign before the hackathon (session 15, 2026-10-12)

The plan for 1.0.0rc14. Every item has a method, an expected result and a status filled in by the
run; the run's evidence is TESTING_PROOF.md run 14. Findings are fixed on `develop` as they appear
and ranked by participant impact in the rc14 report. Environment: `3lc-hub-fresh` (the participant
layout, fresh redirected home, project root under `home\projects`), upgraded from the hosted catalog to
the build under test. The AppData projects are never touched. Kaggle submissions are made only on
Rishikesh's go-ahead; predict runs freely.

Methods: **unit** = a test in `tests/` (the suite, heavy venv); **host** = a script against the
in-process compute host of the fresh env's venv (the proof pattern: live auth exempt, services
stopped) or against the running services through the admin routes; **hand** = the Hub in the browser;
**sim** = the condition simulated (monkeypatch, env var, firewall-free network cut by a closed port);
**n/a** = cannot run on this machine, listed for the testers (section G).

Status values: PASS, FAIL (finding number), FIXED (finding number), OPEN, N/A.

## Pre-items

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| P1 | TESTING.md step 2 drops the host `3lc==3.3.0` pin; the plugin extra pins `3lc` to the exact version the matrix passes on | unit (`test_packaging`) + doc | participants get the 3lc `uv pip install 3lc-compute` resolves; the plugin venv is pinned | |
| P2 | NVIDIA driver requirement in TESTING.md prerequisites (R580+ for cu130) | doc | one row, one sentence | |
| P3 | The Doctor names the installed driver version when it is too old for the build | unit + host | `device.gpu.driver` and a plain sentence; red row | |
| P4 | Windows login under the redirected home: the start script logs in when no key is in the tester home (3lc 3.4.1 reads the redirected `LOCALAPPDATA`) | host + doc | no "API key not found" restart loop for a Windows tester who followed TESTING.md | |

## A. Environment matrix (install from the hosted catalog each time)

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| A1 | Windows username with a space and a non-ASCII character in the path | n/a (needs a second Windows account; the tester folder itself is placed under such a path instead: `C:\Users\Owner\Desktop\3LC Hackathons\tést år\3lc-hub` as a stand-in) | install, import, train, predict work; every path is quoted | |
| A2 | Desktop redirected to OneDrive | n/a (no OneDrive here); the stand-in is a tester folder under a path with spaces and a `.` component | | |
| A3 | Path length near 260 characters | host: a tester folder whose plugin state path reaches 240+ characters | the managed venv and the kit extract succeed, or fail with a message naming the path length | |
| A4 | Python 3.11 and 3.12 | host: a venv on 3.11 (3lc has 3.11 wheels?) and on 3.12 | both provision; the plugin floors at 3.11 | |
| A5 | CPU-only (CUDA hidden) incl. ETA accuracy and a full train / predict cycle | host, sim (`CUDA_VISIBLE_DEVICES=""` for the service and its workers) | auto = cpu, the hint and the in-run ETA within 2× of actual, train + predict complete | |
| A6 | A GPU outside the wheel's arch list (hide sm_120) on Train and Predict | unit (the sm_120-less arch list) + host sim (`gpu_kernel_support` monkeypatch) | the Doctor says it plainly; Train and Predict run on CPU with `cpu (fallback: GPU found …)` | |
| A7 | An NVIDIA driver older than R580 | sim (driver string 560.xx) | the Doctor names the installed driver and the needed one; auto = cpu | |
| A8 | Host 3lc at the oldest and newest versions compatible with 3lc-compute 1.1.0 | host: `uv pip install 3lc-compute==1.1.0 3lc==<oldest>` and `3lc==<newest>` in scratch venvs; the plugin's routes against each | the plugin's SDK pin resolves; `/config`, `/status/doctor` answer | |
| A9 | Plugin upgrade rc13 -> rc14 with state | host (the proof pattern) | records, ledger, kit, tables carried; the Doctor shows rc14 | |
| A10 | Uninstall and reinstall | host (`DELETE /api/admin/plugins/<id>`, then install) | the plugin state survives under `plugin-state`; the kit record is intact | |
| A11 | Service restart with the plugin installed | host (stop / start) | the plugin loads without re-provisioning; a running train is marked stale | |

## B. Kit download and import

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| B1 | Network cut mid-shard then resume | unit (`test_completed_shard_skipped_and_partial_resumed`, `test_cancel_between_shards_stays_resumable`) + host sim (the base URL moved to a closed port after shard 1) | the next download resumes from the `.part` | |
| B2 | Corrupted shard | unit (`test_corrupt_shard_on_disk_is_redownloaded`, `test_persistent_sha_mismatch_fails_with_participant_message`) | re-downloaded once, then a participant message | |
| B3 | A kit file modified after verify | unit (`test_verify_now_passes_then_names_the_tamper`) + host (edit one JPEG, Re-verify, then Import) | the Re-verify names the file; Import's kit check fails before any table | |
| B4 | Kit folder deleted after import | host | the Import tab shows the stale kit; Train refuses with the images-missing message; the Doctor's Table images row is red | |
| B5 | Manual kit folder with the CDN blocked | host sim (base URL on a closed port) + hand | the Kaggle Data route, Verify files, Import | |
| B6 | Wrong folder, nested folder, trailing slash | unit + host (`/import/preflight?kit_dir=`) | each refused with its own sentence; the trailing slash is accepted | |
| B7 | Re-import fresh three times | host | `initial-2`, `initial-3`, `initial-4` | |
| B8 | Project renamed | host (rename the project folder on disk) | the stale view names both tables; a re-import into the new name works | |
| B9 | Two imports started at once | host (two `POST` within a second) | one runs, the other is refused or queued; no partial tables | |

## C. Train

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| C1 | Batch size that OOMs on this GPU | host (batch 4096 on 16 GB) | `OutOfMemoryError` -> `cpu (fallback: …)`, or a clear refusal if the bound stops it first | |
| C2 | Cancel at epoch 1 | host | record cancelled, best.pt from epoch 1 kept | |
| C3 | Compute service killed mid-train | host (kill the service process) | on restart: the record is stale, "Interrupted: the compute service restarted…", no orphan worker | |
| C4 | Browser closed mid-train and reopened | hand + unit (`test_ui_train_reconcile`) | the in-run view reconnects, or the terminal view if it finished | |
| C5 | Two trains queued | host (two `POST /run`) | the second waits on the GPU queue and runs after, or is refused with the double-click sentence when the token repeats | |
| C6 | Label edits in the Dashboard then "Use latest revision" | host (an EditedTable revision written with tlc) | the run trains on the revision; lineage check passes; foreign rows refused | |
| C7 | Every bound on every hyperparameter field | unit (`test_build_train_kwargs_refuses_out_of_bounds_and_locked_overrides`) + host (each field at min-1, min, max, max+1) | the bounds message names the field and the range | |
| C8 | Epochs 1 and 50 | host (1 epoch full; 50 epochs started and cancelled at 3, the ETA read) | both accepted | |
| C9 | A run name with spaces and unicode | host | the run folder is created and listed; the bundle exports it | |

## D. Predict and Submit

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| D1 | Predict on a cancelled, failed and running run | unit (`test_the_gate_blocks_a_failed_provenance_check_and_allows_cancelled_with_best`) + host | cancelled-with-best allowed, failed refused, running refused | |
| D2 | Checkpoint modified on disk | unit (`test_the_three_way_sha_gate_refuses_a_tampered_checkpoint`) + host | refused by the three-way sha gate | |
| D3 | Submission CSV edited by hand | unit (`test_missing_or_changed_csv_is_empty_despite_the_record`) + host | the Submit step refuses (sha mismatch); the Status row shows no CSV | |
| D4 | Invalid, revoked and expired Kaggle tokens | unit (`test_connect_removes_a_token_kaggle_rejects`) + host (a bad-format token, a random KGAT_ string) | each refused with its sentence; nothing written for a bad format | |
| D5 | `KAGGLE_API_TOKEN` env var vs the file | unit (`test_connect_refuses_when_the_env_var_would_shadow_the_file`) | the Connect route names the shadowing | |
| D6 | Kaggle unreachable mid-submit | unit (`test_submit_outcomes_through_a_fake_client`) + host sim (api host unresolvable via `KAGGLE_API_ENDPOINT`?) | the ledger records the outcome; the CSV stays | |
| D7 | A user without competition access | unit (`test_kaggle_live_reports_unlaunched_when_the_list_403s…`) + host | the card says not joined; submit refused before the upload | |
| D8 | The 100/day cap reached | sim (`submissions_used_today` = limit) | the Submit button and the route refuse with the count | |
| D9 | Submission message with unicode and 500+ characters | unit + host (fake client) | trimmed or refused per the manifest's limit; never a crash | |

## E. Status and bundle

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| E1 | Bundle with 0, 1 and 2 submitted runs | unit (`test_bundle_checkpoint_modes…`, `test_both_rule_picks…`) + host | the rule's checkpoints; the README names the count | |
| E2 | Bundle under a redirected home | host (the fresh env is one) | the zip lands under the plugin home; the paths inside are relative | |
| E3 | Doctor on every failure state above | host | each state is one row, never a crash | |
| E4 | No `KGAT_` in any bundle | unit (`test_a_planted_secret_refuses_the_whole_export`) + host (grep every bundle produced in this run) | zero hits | |

## F. Shared-resource load

| # | Item | Method | Expected | Status |
|---|---|---|---|---|
| F1 | 50 concurrent kit downloads from the dev CDN | host script (50 threads, each the full shard set, Range requests on; throughput, error count, CloudFront status codes) | no 5xx; throughput reported | |
| F2 | 20 concurrent manifest and index fetches | host script | all 200, latency reported | |
| F3 | Catalog fetch under load | host script (20 concurrent raw.githubusercontent fetches) | all 200 | |
| F4 | Back-to-back submissions (Kaggle rate limiting) | hand, each on Rishikesh's go-ahead | the outcome text per submission | OPEN (needs go-ahead) |
| F5 | A 3-hour soak of train -> predict cycles with memory tracked | host script (cycles of 1-epoch train + predict, RSS of the worker sampled per cycle) | no growth beyond the first cycle's plateau | |
| F5b | **The release gate** (Rishikesh, 2026-10-12): a 30-cycle verification soak on the build with the between-jobs memory release (`41fba65`) | host script, same as F5 | the worker's working-set slope is at most **2 MB per cycle** (least-squares over cycles 2 to 30; cycle 1 carries the warm-up: the CUDA context, the UMAP compile); above that, no tag: report instead. **Amended 2026-10-13 (Rishikesh):** the private-bytes slope over cycles 15 to 30 is reported too; when the 2-to-30 slope is above 2 MB only because of warm-up and the 15-to-30 slope is under 0.5 MB per cycle, the gate passes, with both slopes and the reason recorded in the proof | |

## G. Platform coverage for the testers (not runnable here)

A checklist for TESTING.md: macOS (mps) and CPU-only Linux / Windows testers.

## H. Anti-cheat review

Every way a participant could submit predictions not produced by a plugin-trained run on the locked
val split, and whether each is blocked or detectable in the bundle. Written up in the report and in
docs/PREDICT_MIRROR.md.
