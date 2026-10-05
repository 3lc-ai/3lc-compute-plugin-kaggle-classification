# 3lc-compute-plugin-kaggle-classification

A [3LC Hub](https://docs.3lc.ai) compute-service plugin that runs an image-classification
Kaggle hackathon end to end: download and verify the starter kit, import it as 3LC tables,
train the fixed baseline, inspect and label in the Dashboard, predict, submit. Everything
competition-specific comes from a remote competition manifest, so one plugin build serves
any classification hackathon.

It is the classification sibling of
[3lc-compute-plugin-kaggle](https://github.com/3lc-ai/3lc-compute-plugin-kaggle) (ExDark,
detection). This one is **Apache-2.0** and links no Ultralytics code.

## Status

Release candidate **1.0.0rc5** (2026-10-05): all four tabs — Import, Train, Predict + Submit,
Status — mirror the ExDark plugin control for control (`docs/EXDARK_MIRROR.md`,
`docs/TRAIN_MIRROR.md`, `docs/PREDICT_MIRROR.md`, `docs/STATUS_MIRROR.md`). Testers start
with `TESTING.md`. See `docs/PLAN.md` for the locked decisions, `CLAUDE.md` for the
operating protocol, `CONTEXT.md` for vocabulary, `CHANGELOG.md` for what shipped.

## Run it locally

```powershell
uv sync --extra kaggle-classification --group dev
uv run pytest
```

Point a compute service (>= 1.1.0) at `src/` with `--plugin-dir` or
`TLC_COMPUTE_EXTERNAL_PLUGIN_DIRS`, then reload after edits:

```powershell
curl -X POST http://localhost:5022/api/admin/plugins/dirs/reload -H "Content-Type: application/json" -d "{\"directory\": \"<repo>\\src\"}"
```

## License

Apache-2.0. See `LICENSE`. Modules adapted from 3lc-compute-plugin-kaggle carry a header
recording that 3LC, the copyright holder, relicensed them for this project.
