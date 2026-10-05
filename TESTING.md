# TESTING.md — try the Kaggle Classification plugin (release candidate 1.0.0rc5)

You are testing a 3LC Hub plugin that runs an image-classification Kaggle hackathon end to end:
download a starter kit, import it as 3LC tables, label a few images in the Dashboard, train the
fixed baseline for two epochs, predict on the test images, and look at the Status tab. The
competition itself is **not launched yet**, so there is nothing to submit to Kaggle and no
leaderboard; the plugin says so where it matters. Plan on 30–45 minutes the first time, most of it
waiting for downloads.

Everything you install lands in one folder you create. Delete that folder afterwards and nothing is
left behind except your 3LC login.

## 1. Prerequisites

| | Needed | Where |
|---|---|---|
| A 3LC account and its API key | yes | https://account.3lc.ai (Settings → API key). The services refuse to start without a login. |
| `uv` | yes | https://docs.astral.sh/uv/getting-started/installation/ — the compute service builds the plugin's environment with it. |
| Python 3.12 | yes, via uv | `uv python install 3.12` (any OS). The 3LC packages have no wheel for Python 3.14. |
| Chrome or Edge | yes | the Hub is a web page at https://hub.3lc.ai; it talks to the two services on your machine |
| A GPU | optional | an NVIDIA GPU makes the 2-epoch training about 1 minute instead of about 5. CPU is fine. |
| Disk | about 6 GB | torch (≈ 2.5 GB download on the first plugin install), the two Python environments, the 113 MB starter kit |
| OS | Windows 10/11, macOS, Linux | the start script exists for all three |

A Kaggle account is **not** needed: the services run with their home folder redirected to your
tester folder, so even a Kaggle token elsewhere on the machine is not seen, and nothing can be
submitted.

## 2. Install the two services into a venv

Create an empty folder and work inside it (examples: `C:\3lc-tester` on Windows, `~/3lc-tester`
on macOS / Linux).

**Windows (PowerShell):**

```powershell
mkdir C:\3lc-tester; cd C:\3lc-tester
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe --index-url https://pypi.org/simple "3lc-compute==1.1.0" "3lc==3.3.0"
.\.venv\Scripts\3lc.exe login
```

`3lc login` asks for the API key (type or paste it; it is never shown). On Windows the key is stored
under your real profile, so this login is also valid for any other 3LC use on the machine.

**macOS / Linux (bash or zsh):**

```bash
mkdir -p ~/3lc-tester && cd ~/3lc-tester
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple "3lc-compute==1.1.0" "3lc==3.3.0"
```

(No login yet: on macOS / Linux the start script logs you in on its first run, inside the tester
folder, because it redirects `HOME` there.)

## 3. Get the start script

Download the one for your OS into the tester folder (next to `.venv`):

- Windows: https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc5/tester/start_tester.ps1
- macOS / Linux: https://raw.githubusercontent.com/3lc-ai/3lc-compute-plugin-kaggle-classification/release/1.0.0rc5/tester/start_tester.sh

What the script does: it starts the 3LC object service and the 3LC compute service with their home
folder redirected to `.\home` under the tester folder, points the compute service at the test tier of
the competition manifest (`KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL=https://competitions.dev.3lc.ai`),
and lists the release candidate's catalog next to the default plugin catalog
(`TLC_COMPUTE_PLUGIN_CATALOG_URLS`). Logs go to `.\logs`. If a service exits it is restarted after
5 s and the exit is logged.

Defaults: object service on port **5015**, compute service on port **5020** (what the Getting Started
page expects). If those ports are taken on your machine, pass others (`-ObjectPort 5018 -ComputePort
5024` on Windows; `OBJECT_PORT=5018 COMPUTE_PORT=5024 ./start_tester.sh` elsewhere) and use them in
step 5.

Optional: `-ProjectRoot C:\3lc-tester\projects` (Windows) or `PROJECT_ROOT=~/3lc-tester/projects`
(macOS / Linux) keeps the plugin's tables and runs in their own 3LC project root instead of your
default one; useful when you already use 3LC on this machine. Optional: an existing `UV_CACHE_DIR` in
your environment is reused (saves the torch download if you have it cached).

## 4. Start the services

**Windows:**

```powershell
powershell -ExecutionPolicy Bypass -File .\start_tester.ps1
```

**macOS / Linux:**

```bash
chmod +x start_tester.sh && ./start_tester.sh
```

**What you should see.** On Windows two console windows open, "3LC tester: object :5015" and "3LC
tester: compute :5020"; leave them open (minimising is fine). On macOS / Linux the services run in
the background and the script asks for your 3LC API key once. Within about 15 s the script prints:

```
Ready: open https://hub.3lc.ai/gettingstarted/ and connect the object service on 127.0.0.1:5015 and the compute service on 127.0.0.1:5020
```

**If it does not match:** the script prints the reason (missing venv, missing `uv`, a port in use,
no 3LC login). Send `start_tester.last.log` and the newest file in `logs\`.

## 5. Connect the Hub

1. Open https://hub.3lc.ai/gettingstarted/ in Chrome or Edge and sign in with your 3LC account.
2. The page shows an Object Service and a Compute Service indicator with their URLs. Enter
   `http://127.0.0.1:5015` for the object service and `http://127.0.0.1:5020` for the compute service
   (or the ports you chose) and press **Check connection**.
3. The first time, Chrome asks whether `hub.3lc.ai` may access devices on your local network: choose
   **Allow**. (If you dismissed it: click the site-settings icon left of the address bar, set "Local
   network access" to Allow, reload.)

**What you should see:** both indicators green. **If not:** a screenshot of the page and the newest
`logs\compute-*.log`.

## 6. Install the plugin

1. In the Hub open **Plugins → Available**. The card **Kaggle Classification** shows version
   **1.0.0rc5**. On the very first start the compute service first installs its eight stock plugins
   (about a minute with a fast connection, longer on a slow one); the card appears once that is done,
   so reload the page if the list is still empty after a minute.
2. Click **Install**. The compute service builds the plugin's own Python environment (torch is the
   big download, 2–15 minutes depending on your connection; a fresh machine downloads about 2.5 GB).
   The card shows the progress; the service window / log shows `uv` at work.
3. When the card says installed, open **AI Tools → Kaggle Classification** in the Hub's sidebar.

**What you should see:** a page headed **3LC Scene Classification Challenge** with three constraint
chips (resnet18 · from scratch · 150px / 6 classes · 6,000 unlabeled / Scored by accuracy), "The
Loop" row, four tabs — **1 Import · 2 Train · 3 Predict + Submit · 4 Status** — and the Import tab
open with a **Download starter kit** button. The footer reads `3LC Kaggle Classification plugin
v1.0.0rc5`. **If not:** a screenshot, and the newest `logs\compute-*.log`. If the page says
"Setting up the plugin environment. The first run takes a few minutes." just wait: that is the
first-use provisioning, it continues by itself.

## 7. The test checklist

Do the steps in order. For every step there is what you should see and what to send if it differs.
"Copy diagnostics" is a button on the plugin page that copies a text block to your clipboard — paste
it into your feedback. "The Doctor" is the collapsed panel at the bottom of the Status tab.

### 7.1 Download the starter kit (Import tab)

Click **Download starter kit**.

- You should see: progress rows (Manifest → Disk space → Download → Verify → Extract), a progress
  line with the shard and bytes ("Downloading shard 2/5 · 74 of 113 MB"), then a green line "Starter
  kit downloaded and verified. 9,601 files match the published manifest. The starter kit folder below
  is filled in and ready to import." About 70 s on a fast connection, a few minutes on a slow one.
- If not: Copy diagnostics (the failure banner has the button), a screenshot.

### 7.2 Import

Click **Import & Validate** (keep Project name and Table name as they are).

- You should see: three progress rows (Import train / Import val / Validate), then "Imported &
  validated: intel-scene_train · intel-scene_val", **18/18 checks passed** (collapsed; click to
  expand), two rows with **CREATED** badges: train **6,600** rows, val **1,200** rows, each with
  **Open in Dashboard**. A few seconds. The tab bar now marks **1 Import** done.
- If not: Copy diagnostics, a screenshot of the checks expanded.

### 7.3 Label a batch in the Dashboard and commit

In the Loop row click **fix labels** (it opens the 3LC Dashboard on the train table).

1. In the Dashboard, filter the `label` column to **undefined** (the 6,000 unlabeled pool).
2. Select 10–20 images that clearly belong to one class (for example sea), set their `label` to that
   class AND their `weight` to **1** (two edits — a pool image enters training only with a real label
   and a weight above 0), and **Commit**. The Dashboard writes a new revision of the train table;
   nothing in the original is changed.
3. Back on the plugin page open **2 Train**.

- You should see on the Train tab: "Tables verified: intel-scene_train/initial · val locked · trains
  on the latest revision (intel-scene_train/<your revision name>)" and the line "**6NN labeled rows in
  use** · 5,9NN excluded as undefined · 0 excluded at weight 0" where 6NN = 600 plus the number you
  labeled. The revision picker (the layers button beside the Train table URL) lists `initial` with
  your revision underneath, marked LATEST.
- If not: a screenshot of the Train tab's Tables section and of the Dashboard after Commit. If you
  could not find the Edit or Commit controls in the Dashboard, say so — that is useful feedback too.

### 7.4 Train 2 epochs

On the Train tab set **Epochs** to **2** (leave everything else), then click **Start Training**.

- You should see: the in-run header "<run name> · running · Epoch 1/2 · cuda (auto)" (or "cpu
  (auto)" without a GPU), a progress bar filling per batch, three chips (Train loss · Val loss · Val
  accuracy) with sparklines, "Show log" filling live; after epoch 2 the note "Collecting per-sample
  metrics and embeddings on 7,800 rows…"; then the green banner "**Training complete: NN.N% val
  accuracy at epoch N** (the checkpoint Predict uses)", a **Verified provenance recorded** panel with
  9 checks, buttons **Continue to Submit**, **Open Run in Dashboard**, **Open Run in Projects**. On a
  GPU about 1–2 minutes end to end; on a CPU about 5–8 minutes. Val accuracy after 2 epochs is low
  (30–55 %), that is expected.
- If not: Copy diagnostics (on the failure banner), the Show log text, a screenshot.

### 7.5 Predict

Click **Continue to Submit** (or open **3 Predict + Submit**).

- You should see: your run preselected in the **Run** picker with the note "Verified provenance on
  this run's record · best checkpoint sha256 …", the locked row "Kit · …\data\test · 1,800 images"
  and the green line "Test images verified: 1,800 files match the kit's files.json". Click **Run
  inference**: a progress block "Val check: n / 1,200 images" then "Inference: n / 1,800 images",
  then **10/10 checks passed**, the **Predicted-class distribution** card (six class tags with
  counts; after only two epochs an amber "The predicted-class distribution is skewed …" note under
  the tags is normal), the hero stat "NN.NN % · Val accuracy · Your locked validation split, not the
  leaderboard.", and a CSV row `submission_<timestamp>.csv` with **Copy CSV path** and **Download
  CSV**. Under **Step 2 · Submit to Kaggle** the connection card reads **Kaggle account not
  connected** (expected: the services have no Kaggle token) and the Submit button stays disabled.
- If not: Copy diagnostics, a screenshot.

### 7.6 Download CSV

Click **Download CSV**.

- You should see: the browser saves `submission_<timestamp>.csv`. Open it: a header line
  `image_id,prediction,confidence` and 1,800 data lines, predictions 0–5, confidences between 0 and 1
  with six decimals.
- If not: the browser's download error text, Copy diagnostics.

**Do not try to submit to Kaggle.** The competition is unlaunched; the button is disabled by design.

### 7.7 Status tab

Open **4 Status**.

- You should see: a hero strip (Best public score "no scores yet", Latest activity with your
  prediction, no Kaggle budget block), **Runs** with one row — your run, the revision it trained on,
  the labeled rows, "NN.N% at epoch N", the device, elapsed, COMPLETED, Dashboard / Projects links —
  **History** with one row — your prediction, its val accuracy, "–" under Public score and Δ, Outcome
  "CSV generated (not submitted)", Copy CSV path and Download CSV icons — and **Kaggle live** with the
  callout "Connect your Kaggle account…" (expected). "Updated just now" with a refresh button; click
  it and the line resets.
- Expand **Doctor** at the bottom: one row per fact — Plugin (v1.0.0rc5 and a commit), Compute
  service (v1.1.0), SDK · 3lc (0.3.3 · 3.3 or newer), torch · torchvision (**2.14.0** · **0.29.0**, with
  `+cu126` on Windows / Linux), CUDA in the worker, Manifest (remote or cache · a sha256 · intel-scene
  kit v1 — "cache" means the last fetched copy, the same document),
  Kit (v1 · 9,601 files verified at download), Kaggle (not connected), Plugin home, Free disk space,
  Python, Records (1 run · 1 prediction · 0 submissions). Click **Copy diagnostics** and paste the
  block into your feedback: it is the single most useful thing you can send.
- If not: a screenshot of the Status tab and the Doctor, Copy diagnostics.

### 7.8 Export verification bundle

In the **Verification** section click **Export verification bundle**.

- You should see: the browser saves `verification-bundle_intel-scene_<timestamp>.zip`; the note
  beside the button reads "Exported …". Open the zip: `README.txt`, `plugin.json`,
  `manifest_provenance.json`, `import_record.json`, `train_revisions.json`, `runs/<id>.json`,
  `predictions/<id>.json`, `ledger.jsonl` — and nothing else (no images, no CSV, no checkpoint).
- If not: the error shown beside the button, Copy diagnostics.

### 7.9 Stop

Windows: `powershell -ExecutionPolicy Bypass -File .\start_tester.ps1 -Stop`. macOS / Linux:
`./start_tester.sh --stop`. Both report "Stopped: ports … are free." Delete the tester folder when
you are done; your 3LC login (Windows) is the only thing outside it.

## 8. Feedback template

Paste this into your message, filled in (one per problem is better than one for everything):

```
Plugin test — 1.0.0rc5
OS / GPU:            (e.g. Windows 11, RTX 3060 · macOS 15, M2 · Ubuntu 24.04, no GPU)
Step that differed:  (7.1 … 7.9, or "install" / "connect")
What I expected:     (from the "You should see" line)
What I saw:          (one or two sentences; attach a screenshot)
Copy diagnostics:    (paste the ```3lc-kaggle-diagnostics block from the failing step, or from the Doctor)
Doctor rows:         (paste or screenshot the Doctor panel if you reached the Status tab)
Log lines:           (the last 30 lines of logs\compute-*.log if a service misbehaved)
Timing:              (kit download / plugin install / training, roughly)
```

Things we specifically want to hear about: any step where the words on screen did not match this
file, anything that took much longer than the times above, anything that needed a page reload, and
anything in the Dashboard labeling step that was not obvious.

## 8b. Upgrading from an earlier candidate

If you tested an earlier candidate: download the new start script (the catalog URL changed), start
the services as before, open **Plugins** in the Hub and press **Update** on Kaggle Classification
(or Install from Available). Open the plugin: the Import tab must still show your imported tables,
the Train tab your previous runs, the Status tab your history, and the Doctor's "Plugin home" row
ends with "carried forward from the previous version". Nothing is re-downloaded. If any of that is
missing, send the Doctor rows and the newest `logs\compute-*.log`.

## 9. Known limits of this candidate

- The competition is unlaunched: no Kaggle submit, no leaderboard, no submission list; the Status tab
  says "available after the competition launches".
- The Hub's **Open in Projects** links render only when the page runs on hub.3lc.ai (they do).
- The first training run on a machine pays a one-time compile of the embedding reducer in its final
  pass (the plugin pre-warms it after Import, so it is usually already done).
- Windows first: the macOS / Linux start script was written against the same services but was proven
  on Windows (see `TESTING_PROOF.md`).
