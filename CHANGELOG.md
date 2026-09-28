# Changelog

All notable changes to `3lc-compute-plugin-kaggle-classification` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased] — 0.1.0 (sessions 1 and 2)

### Changed (session 2.5, the ExDark mirror)
- The plugin now mirrors 3lc-compute-plugin-kaggle v1.2.15 (`docs/EXDARK_MIRROR.md`): the shell and
  the Import tab are a port of its fragment (innerHTML + `esc()`, icons, motion, connection guard,
  diagnostics, preflight, progress rows, result and failure banners, revisit, download section,
  `?kgdev` fixtures) under the four allowed differences; existing tables are REUSED and re-validated;
  Re-import fresh writes fresh `initial-N` tables beside the old ones; the hero title is the
  manifest's display name; a stale result clears when the preflight turns amber or red.
  `GET /download/verify`; the kit record carries the manifest provenance of the download.

### Changed (session 2.5, superseded passes)
- Competition display name in the bundled manifest: "3LC Scene Classification Challenge"; `cdn/` regenerated
  (manifest sha256 `d9f34aed…db83c`, index `e0e9c4bc…f262d`), not yet uploaded.
- The fragment at the ExDark presentation standard (`docs/UI_PARITY.md`): hero with constraint chips
  and the Loop row, the tab bar as stepper with state glyphs and keyboard support, Hub card and
  form classes, one callout geometry, stepper rows with status badges and elapsed, checks as a
  collapsed verdict that auto-expands on failure with remedies, Technical details and Show log
  disclosures, truncated paths with Copy, "Open in Dashboard" and the Hub project link, the
  connection guard, Copy diagnostics, "Next: train your first model", gated later tabs, reduced
  motion honoured everywhere. `import_state` now carries `latest` (the newest revision per split).
- Progressive disclosure after the State 1 review (`docs/UI_PARITY.md` §3b): one status line and one
  primary action per tab by default; step rows under "Show steps" (auto-open + scroll on failure),
  one progress line with shard, bytes and ETA while running; the import form hidden until the kit
  is on disk, Project / Table name under "Advanced", the re-import toggle only on a collision; no
  `undefined` class tag (the pool is explained on the classes chip); the manifest source moved
  into Technical details; `submission.daily_limit` in the bundled manifest is 100 (was the
  session-1 placeholder 3), so the chip reads "100 submissions per day"; `cdn/` regenerated
  (manifest sha256 `6a10e44f…efed2`), not uploaded.

### Changed (session 2)
- Manifest schema v1: `kit.base_url` replaced by a relative `kit.path`; shard URLs resolve against the
  URL the manifest was fetched from; the index's `manifest_url` is relative to the index. Layout
  `kaggle/classification-index.json`, `kaggle/<id>/manifest.json`, `kaggle/<id>/starter-kit/<v>/`.
- Host allowlist: prod only by default; the dev CDN and loopback only under the base-URL override.

### Added (session 2)
- Importer (`importer.py`, job kind `import`): kit validated against the manifest (structure,
  counts, `sample_submission.csv` ids, every image decodes), `train`/`val` tables on tlc 3.3 with a
  distinct `undefined` label value and per-row weights, collision refusal with explicit re-import
  to fresh `<table>-N` tables, no partial tables on failure or cancel, the import record with
  lineage root, locked val, checks, timings and manifest provenance. Routes `GET /import/preflight`
  and `GET /import/state`.
- Import tab: one stepper for download + import, preflight gate with the collision callout,
  per-check pass/fail, table links into the Hub, revisit from the import record, first-run
  provisioning rendered as an expected state.
- Line endings normalized to LF (`.gitattributes`).
- `docs/PROMOTION.md` §6: the dev tier verified (served headers, sha256s of all seven objects) and
  the console upload steps as performed; `tests/test_deletion_safety.py` (the deletion audit).
- `tools/make_cdn_tree.py` (the bucket mirror + `upload-plan.json`), `tools/verify_cdn.py`
  (served headers and sha256 verification), `docs/PROMOTION.md`.


### Added
- Plugin scaffold from the 3LC template: `plugin.toml` (id `kaggle-classification`,
  `min_service_version` 1.1.0), Apache-2.0, SDK window `>=0.3.1,<0.4.0`, torch from the
  cu126 index as the timm plugin declares it, `timm==1.0.29`.
- Competition manifest schema v1 (`manifest.py`) with field-naming validation, unknown-field
  warnings, the bundled `manifests/intel-scene-v1.yaml`, and the derived facts
  (`num_classes`, `undefined_label_id`, `dataset_name`, `expected_rows`).
- Session store (`session.py`): one canonical session derived from the manifest, retired-key
  rejection, atomic writes, URL parsing by position in the layout tail.
- Kit download stage (`kit.py`): sha256 + Range-resumable shards, zip-slip guard, per-file
  verification against `files.json`, split counts checked against the manifest, revisit states.
- `tools/build_kit.py`: the kit build pipeline (salted opaque ids from `--salt-file`, RGB JPEG q92 re-encode with EXIF/ICC stripped, `files.json`, `sample_submission.csv`, deterministic `intel-scene-v1-NN.zip` shards, `kit-manifest-block.yaml`, judge-only `mapping.csv`) and a participant-side verification pass; the bundled manifest carries the v1 build's `kit{}` block.
- Manifest resolution: remote index + manifest (5 s budget, one retry, server-side only) →
  cache with `{fetched_at, source_url, sha256}` sidecar → bundled; invalid remote falls back
  with a visible warning; competition picker when several are active; job-start provenance
  (`resolve_manifest_for_job`); kit host allowlist, https-only help links, markup-free display
  strings; background refresh so the fragment never waits on the network.
- `storage.py`: the plugin home resolved env → SDK helper → worker state root →
  `<cwd>/.plugin-state/<id>` → `~`, reported on `GET /config`.
- Four-tab fragment shell, `GET/POST /config`, the `download_kit` job kind.
- Test suite: manifest, session, kit stage, packaging + SDK-window overlap + import weight +
  license lineage, the ctx adapter, the timm offline model check, the release-version script.
