# CLAUDE.md — kaggle-classification plugin (build phase: session 1 of 6)

Vocabulary lives in [CONTEXT.md](CONTEXT.md). This file is the operating
protocol; depth lives in the docs linked in §E — link, don't duplicate. The
workspace root has its own `../CLAUDE.md` (folder canonicality, the Hub
environments, credential locations); it points here at the repo boundary.

## A. Hard rules

1. **No silent assumptions.** If a task is ambiguous or conflicts with
   `docs/PLAN.md`, state the conflict and wait. Never pick an interpretation
   silently.
2. **No over-engineering.** Smallest change that resolves the finding. If a fix
   wants to exceed ~2x its apparent size, pause and report (the circuit-breaker
   rule).
3. **No orthogonal changes.** Touch only what the finding requires. Refactors
   are their own proposed task, never a rider.
4. **Plan first on non-trivial work.** Anything beyond a one-file fix: plan +
   conflicts + fragment impact, then wait for go-ahead.
5. **The locked decisions in `docs/PLAN.md` §A are not code tasks.** Model,
   splits, kit format, licence, SDK window, per-file verification: changes there
   are competition-design decisions.

## B. Repo operating rules

- **Branches.** Work on `develop`; push at the end of each phase. Tags `vX.Y.Z`
  only when RELEASING.md says so. `core.autocrlf` is `false` in this repo
  (the machine's system gitconfig says `true`; files are LF).
- **No competition constants in code.** Every class name, split size, arch,
  slug, kit URL and UI copy comes from the manifest (`manifest.py`). The UI
  renders what `GET /config` serves and defines nothing. `test_manifest.py::
  test_no_competition_literal_outside_the_manifest` and
  `test_routes.py::test_plugin_compute_and_fragment` enforce it.
- **License lineage.** Apache-2.0 on every module (`SPDX-License-Identifier`
  header). Code adapted from `3lc-compute-plugin-kaggle` carries the
  relicensing header verbatim and is listed in `docs/STUDY.md` G-5; the word
  AGPL never appears in code. Adapt only modules with no Ultralytics import or
  derived logic. `test_packaging.py` enforces all three.
- **Reference repos are read-only.** `../reference/*`, `../3lc-compute-plugin-kaggle`,
  the Intel data dir and `../hackathon_starter` are never edited; the Intel
  `solution.csv` / `submission*.csv` and the HackNova `solution.csv` are never
  read by code or copied.
- **Import stays light.** `import kaggle_classification` and the manifest /
  session / kit modules import nothing heavy at module level (torch, timm,
  tlc, yaml, litestar, PIL, numpy live inside functions).
  `test_packaging.py::test_package_import_is_light` enforces it.
- **Never derive a path from HOME first.** Everything the plugin writes goes under
  `storage.plugin_home()` (the redirected-home lesson). `Path.home()` appears only as
  the last rule in `storage.py`.
- **`workers=0` anywhere data loads.** Windows host. Device-aware in session 3.
- **Fragment rule.** Any change to `ui/ui.html` or `plugin.toml` means the
  running install is stale. End the task by stating which applies: worker
  **reload** (code/fragment edits on a folder source), venv **re-provision**
  (dependency changes), or **fresh install from a new tag** (catalog installs
  never see working-copy changes).
- **Tests.** `uv run pytest` before any push. The suite is a divergence guard
  first: version strings, description parity, SDK-window overlap with the
  latest 3lc-compute, wheel contents, import weight, licence headers, the
  manifest-literal census. `test_timm_model.py` needs the heavy extra; it
  skips without it, so a green run in a light venv is not a green run.
- **Windows-first.** Paths contain spaces: quote everything. Files are LF,
  BOM-less. Docs speak PowerShell.
- **Published docs describe current behaviour, not history.** Rationale for a
  settled decision goes in `docs/PLAN.md` or `docs/STUDY.md`, not in README or
  RELEASING.

## C. The dev loop

```powershell
uv sync --extra kaggle-classification --group dev
uv run pytest
```

Folder-source registration on the 1.x Hub (`../3lc-hub-ga/`, compute :5022):
start the service with `--plugin-dir "<repo>\src"` (or
`TLC_COMPUTE_EXTERNAL_PLUGIN_DIRS`), then after edits:

```powershell
curl -X POST http://localhost:5022/api/admin/plugins/dirs/reload -H "Content-Type: application/json" -d "{\"directory\": \"<repo>\\src\"}"
```

Manifest against a local mock: set `KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL`
to a `python -m http.server` root serving `index.json` + `intel-scene/manifest.json`.

## D. Self-maintenance of these files

At the end of any session that ships a tag, changes a contract/process rule, or
adds vocabulary: check whether CLAUDE.md / CONTEXT.md / PLAN.md need a line,
and update them in the same commit series. Keep them terse.

## E. Pointers

| Doc | Answers |
|---|---|
| docs/PLAN.md | Locked decisions, the labeling-loop contract, component contracts, session map, open TBCs |
| docs/STUDY.md | What the reference plugins do and what we reuse / adapt / write new; the Gate 0 findings; G-5 relicensing list |
| RELEASING.md | Tag → catalog flow, the version census, kit staging rules |
| CHANGELOG.md | What each version shipped |
| ../3lc-compute-plugin-kaggle/docs/ui-notes.md | The UI playbook the tabs follow from session 2 (six-state machines, motion, icons, copy) |
