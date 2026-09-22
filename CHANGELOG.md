# Changelog

All notable changes to `3lc-compute-plugin-kaggle-classification` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.

## [Unreleased] — 0.1.0 (session 1)

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
