# CONTEXT.md — shared language (kaggle-classification plugin)

One line per term. Depth: docs/PLAN.md (decisions), docs/STUDY.md (references).

## Where we are (2026-09-22)

- **Session 1 of 6** — scaffold, manifest schema and loader, session store, kit download stage, kit builder. No release yet; `develop` only.
- **Next:** session 2 = Import tab (download UI, table registration, revisit view).

## Competition & contract

- **the manifest** — the remote competition document (schema v1) every competition fact comes from; resolved remote → cache → bundled; `manifest.py`. "Bundled" is the copy inside the wheel, "cache" the last good remote copy on disk.
- **competition id** — the stable CDN id (`intel-scene`), never the Kaggle slug; names the manifest path, the default project, the dataset prefix and the kit directory.
- **the slug** — the Kaggle URL slug (`competition.slug`); used only by Submit/Status.
- **the contract (locked)** — `arch` from the manifest (`resnet18`), `pretrained=false`, `image_size` from the manifest, `timm==1.0.29`; identical init for every participant; `pretrained: true` is rejected at manifest load.
- **undefined** — the unlabeled pool: label id `num_classes` (the LAST map entry), weight 0, filtered out of training regardless of weight, given predictions/confidence/embeddings but no loss.
- **the labeling loop / loop contract** — the per-sample metrics a run writes back (PLAN §B): predicted, confidence, per-class probabilities, loss (masked for undefined), 3D embeddings with UMAP fit on train (labeled + undefined), val transformed into the same space.
- **starter kit** — `starter_kit/data/{train/<class>,train/undefined,val/<class>,test}` + `sample_submission.csv` + `files.json`; images renamed to salted opaque ids and re-encoded; the judge's `mapping.csv` never ships.
- **files.json** — the per-file index inside the kit (relpath, sha256, bytes, kit_version); the download stage verifies every file against it.
- **kit{} block** — the manifest's `kit{base_url, version, shards[{name, sha256, bytes}]}`; emitted by `tools/build_kit.py`, pasted into the manifest; an empty `shards` list means "not published" and the download refuses.
- **download_kit** — the job kind: shards (sha256, Range resume, .part files) → extract → per-file verify → split counts vs manifest → record + `session.kit_dir`; states `empty / success / superseded / stale`.
- **the session object** — `{project_name, table_name, kit_dir, device, overrides}` in `~/.3lc-kaggle-classification/ui_config.json`; tabs render projections; defaults derive from the manifest; retired keys 400.
- **REUSED vs CREATED** — per-split import outcome (session 2): identical existing table reused, else created.
- **plugin-run-only** — participants predict only from runs this plugin trained (session 4).
- **the ledger / verification bundle** — the append-only record of every step and the per-run zip an organizer verifies a leaderboard entry against (session 5).

## Ops & environments

- **the plugin home** — `~/.3lc-kaggle-classification/`: `ui_config.json`, `kit/<id>.json` (the download record), `data/<id>/<kit version>/` (the kit).
- **hosts** — `../3lc-hub-ga/` (compute 1.0.1 + 3lc 3.3.0 + SDK 0.3.2, :5022/:5016) is where click-through happens today; `min_service_version` is 1.1.0, so the card greys out there until that environment moves to 1.1.x. Compute 1.1.0 is on PyPI and declares SDK `>=0.3.3,<0.4.0`.
- **the SDK window** — `>=0.3.1,<0.4.0`; `test_packaging.py` checks it overlaps the latest 3lc-compute release's declared range (snapshot offline, PyPI live).
- **folder source vs tag install** — a dev Hub registers `src/` and picks up edits on worker reload; a tester's catalog install pins a tag and never sees the checkout.
- **the catalog** — `catalog.json`: one entry per released version, newest first, manifest pasted verbatim, `source` pinned to the tag; `HEAD/catalog.json` on the default branch is the URL hubs consume.
- **the four tabs** — Import · Train · Predict + Submit · Status; the tab bar is the stepper.
- **?kgdev fixtures, six-state machine, motion tokens, glance card, verdict line** — the UI playbook vocabulary, inherited from the ExDark plugin's docs/ui-notes.md from session 2 onward.

## Naming

- **project** — default `intel-scene` (the competition id); **tables** — `initial` by default; **datasets** — `<competition id>_<split>` (`intel-scene_train`, `intel-scene_val`).
- **run names** — `<competition id>_run_<YYYYMMDD_HHMMSS>` when blank (session 3).
- **tags** — `vX.Y.Z`; the version string is identical in pyproject, plugin.toml and the catalog entry (`scripts/release_version.py`, `test_packaging.py`). Kit data uses the manifest's `kit.version` (`v1`, `v2`, …), never a git tag.
- **plugin id** — `kaggle-classification`; the entry-point key equals it.
