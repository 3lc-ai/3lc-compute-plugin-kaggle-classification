# Laptop to PC migration (2026-10-09)

Two scripts. `export.ps1` stages and zips everything the PC needs from the laptop; `import.ps1` unpacks it on the
PC into the same layout, verifies every file, rebuilds the venvs and prints the manual steps. The repo itself is
not in the zip: clone it on the PC at the same path and check out the commit `MANIFEST.json` names.

What the zip holds: `3lc-hub-11` and `3lc-hub-rc9-clean` (no `.venv`, no `__pycache__`, no `.kaggle`, no `*.log`
older than the export day), the staged kit `datasets\intel-scene-kit-v1`, the 3LC project root (`intel-scene`,
`intel-scene-demo`, `test1`), Claude Code's memory for this repo and its `settings.json`, and `MANIFEST.json`
(sha256 and size per file, the laptop's Python / 3lc / 3lc-compute / plugin versions, the repo commit). The
redirected homes' package caches (`home\AppData\Local\uv`, gigabytes of wheels and git checkouts) are never staged.
The export refuses to zip if any file name or text matches the release audit's secret patterns (tokens,
`kaggle.json`, `3lc_api_key`, `mapping.csv`, `solution*.csv`); the only `KGAT_` strings it lets through are the
repo's documented fake fixtures (`KGAT_proofclean…`, `KGAT_test0123…`, `KGAT_other0123…`, `KGAT_abcdefghijklmnop0123`).

## On the laptop

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\rishi\Desktop\3LC Hackathons\3lc-compute-plugin-kaggle-classification\tools\migrate\export.ps1"
```

Output: `C:\Users\rishi\Desktop\3LC Hackathons\migration-2026-10-09\migration-2026-10-09.zip` (plus `staging\` and
`export.log` beside it). Copy the zip to the PC. Stop the services first if they run
(`start_tester_rc12.ps1 -Stop` in the clean env, `start_demo.ps1 -Stop` in hub-11).

## On the PC

```powershell
git clone https://github.com/3lc-ai/3lc-compute-plugin-kaggle-classification.git "C:\Users\rishi\Desktop\3LC Hackathons\3lc-compute-plugin-kaggle-classification"
```

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\rishi\Desktop\3LC Hackathons\3lc-compute-plugin-kaggle-classification\tools\migrate\import.ps1" -Zip "<where you copied it>\migration-2026-10-09.zip"
```

`import.ps1` unpacks to `C:\Users\rishi\Desktop\3LC Hackathons`, renames any folder it would overwrite to
`<name>-old-<date>`, places the projects under `%LOCALAPPDATA%\3LC\3LC\projects` and the memory under
`%USERPROFILE%\.claude\projects\...\memory`, rebuilds both `.venv` folders with `uv` (3lc-compute and 3lc pinned to
the laptop's versions), runs `3lc-hub-11\check_intel_scene_untouched.ps1`, and prints the manual steps:

1. `git -C "<repo>" reset --hard <commit from MANIFEST.json>`, then `uv sync --python 3.12 --extra kaggle-classification --group dev` and `uv run pytest`.
2. Claude Code login, open the repo folder.
3. `3lc login` in `3lc-hub-11\.venv\Scripts\3lc.exe` (the key store is per machine).
4. Kaggle: paste a fresh token into Connect on the Predict + Submit tab (the token was never exported).
5. Getting Started: object `http://127.0.0.1:5015`, compute `http://127.0.0.1:5020`, then hard refresh the plugin page.

Then the migration check: start the clean env, Import revisit, a 1-epoch train, and cut rc13 (CONTEXT.md, the
session handoff of 2026-10-09).
