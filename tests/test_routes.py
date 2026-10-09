# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GET /config`` serves a populated session and a ``_meta`` block the fragment renders and never
defines: version, repository, the manifest with its provenance, the kit destination and state."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


import kaggle_classification
from kaggle_classification import routes, session


def test_config_payload_is_populated_and_served_not_restated(home, manifest):
    out = routes.config_payload()
    assert out["session"] == session.default_session(manifest)
    meta = out["_meta"]
    assert meta["version"] == kaggle_classification.__version__
    assert meta["repository_url"] == kaggle_classification.REPOSITORY_URL
    assert meta["manifest_source"] == "bundled"
    assert meta["manifest"]["model"] == {
        "backbone": "torchvision_resnet18", "head": "kit_mlp_512_256_128_d03", "arch": "resnet18",
        "pretrained": False, "image_size": 150,
    }
    assert [c["name"] for c in meta["manifest"]["classes"]] == manifest.class_names
    assert meta["kit_dest"] == str(home / "data" / manifest.competition.id)
    assert meta["kit_state"] == {"state": "empty"}
    assert meta["plugin_home"] == {"path": str(home), "resolved_by": "env"}
    assert meta["manifest_provenance"]["manifest_source"] == "bundled"
    assert meta["manifest_candidates"] == []
    # The config path never blocks on the network: it kicks the refresh and reports its state.
    assert meta["manifest_refresh"]["state"] in ("running", "done", "failed")


def test_config_path_never_fetches(home, monkeypatch):
    from kaggle_classification import manifest as manifest_mod

    def boom(*a, **k):
        raise AssertionError("GET /config fetched the network synchronously")

    monkeypatch.setattr(manifest_mod, "_http_get", boom)
    monkeypatch.setattr(manifest_mod, "refresh_in_background", lambda **kw: {"state": "skipped"})
    out = routes.config_payload()
    assert out["_meta"]["manifest_refresh"] == {"state": "skipped"}


def test_config_payload_reflects_a_saved_session(home, manifest):
    session.save({"session": {**session.default_session(manifest), "project_name": "probe-x"}})
    out = routes.config_payload()
    assert out["session"]["project_name"] == "probe-x"
    assert out["session"]["table_name"] == "initial"


def test_route_handlers_build_when_litestar_is_present():
    import pytest

    pytest.importorskip("litestar")
    handlers = routes.get_route_handlers()
    paths = {p for h in handlers for p in h.paths}
    assert paths == {
        "/config", "/manifest", "/manifest/select", "/import/preflight", "/import/state", "/download/verify",
        "/train/preflight", "/train/state", "/tables/list", "/tables/defaults",
        "/runs", "/predict/preflight", "/submit/state", "/kaggle/connection", "/kaggle/connect",
        "/submissions/{job_id:str}/download",
        "/status/history", "/status/kaggle", "/status/doctor", "/status/bundle", "/status/bundle/preview",
    }


def test_route_handler_annotations_resolve_like_litestar_does():
    """The compute 1.1.0 worker died at startup because a handler's string return annotation
    (``from __future__ import annotations``) named a ``Response`` that was not in the routes
    module's globals. Litestar resolves hints with ``typing.get_type_hints``; so do we."""
    import typing

    import pytest

    pytest.importorskip("litestar")
    for handler in routes.get_route_handlers():
        hints = typing.get_type_hints(handler.fn, globalns=vars(routes))
        assert "return" in hints, handler.fn.__name__


def test_plugin_compute_and_fragment():
    plugin = kaggle_classification.KaggleClassificationPlugin()
    info = plugin.compute({})
    assert info["plugin"] == "kaggle-classification"
    assert info["implemented"] == ["download_kit", "import", "train", "predict", "kaggle_submit"]
    html = plugin.get_ui_fragment()
    assert 'class="kgc"' in html and "kaggle-classification" in html
    for needle in ("buildings", "resnet18", "6000", "1800", "HackNova"):
        assert needle not in html, f"competition literal {needle!r} in the fragment"


# ── The ExDark mirror (docs/EXDARK_MIRROR.md): the port's shape, and its safety pattern ──────

FRAGMENT = Path(kaggle_classification.__file__).parent / "ui" / "ui.html"


def _script() -> str:
    html = FRAGMENT.read_text(encoding="utf-8")
    return html[html.index("<script>") + len("<script>") : html.rindex("</script>")]


def test_fragment_is_the_exdark_import_tab():
    """The shell and Import tab elements ExDark renders, in ExDark's ids and copy (EXDARK_MIRROR §1–2)."""
    html = FRAGMENT.read_text(encoding="utf-8")
    for needle in (
        'class="kg-id-row"', "Competition constraints", 'id="kg-loop"', 'id="kg-loop-inspect"', 'id="kg-loop-fixlabels"',
        'id="kg-tabs"', 'id="kg-state-import"', "Starter kit → 3LC tables", "Predict → CSV → Kaggle", "History &amp; leaderboard",
        'id="kg-conn-banner"', 'id="kg-dl-section"', 'id="kg-dl-offer"', 'id="kg-dl-dest"', 'id="kg-dl-btn"', 'id="kg-dl-progress"',
        'id="kg-import-banner"', 'id="kg-import-form"', 'for="kg-kit"', 'id="kg-project"', 'id="kg-table"',
        'id="kg-glance"', 'id="kg-splits-body"', "Splits to import", 'id="kg-import-btn"', "Import &amp; Validate",
        'id="kg-import-progress"', 'id="kg-checks"', 'id="kg-result"',
        'id="kg-log-toggle"', "Show log", "data-kg-footer", "Continue to Train", "<span>Explore</span>",
        "Tables not found.", "Go to Import", "Compute service unreachable, retrying", "Reconnected.", "Copy diagnostics",
        "Importing… (safe to navigate away)", "Dataset at a glance", "Revision name for the imported tables.",
        "Path on the machine running the compute service.", "function kgSplitsPlaceholder", "function kgRenderSplitsLocked",
        "Detected: train / val · ", "Matches the competition manifest.", "Explore ' + split", "kg-badge-reused", "kg-badge-created",
        "Re-import fresh", "function kgReimportFresh", "kg-reimport-slot",
        "var KG_REMEDIES", "function kgRenderProgress", "function renderResult", "function renderFailBanner",
        "function dlRenderQuiet", "function dlRenderSuperseded", "function dlRenderProgress", "prefers-reduced-motion: no-preference",
        # rc11 (CONTEXT.md decisions 2026-10-09, back-port to ExDark after the event): the shell pass and the Import
        # tab's behaviour, kept by rc12 under the rc10 layout.
        'id="kg-reimport-confirm"', 'id="kg-manifest-note"', 'id="kg-kaggle-fallback"',
        'id="kg-kit-manual"', 'id="kg-hero-toggle"', "function kgRenderImported", "function kgRenderStale", "function kgReflectRecord",
        "function kgRenderImportFromState", "function kgConfirmReimport", "function kgReloadConfig", "function kgRenderManifestNote",
        "function kgShowKaggleFallback", "function kgRenderKitManual", "function kgHeroInit", "function kgDisplayPath",
        "Found existing tables", "Re-import…", "Your existing tables and label edits are kept.",
        "Starter kit ready", "images + manifest", "Re-verify", "/competitions/' + slug + '/data", "Open the Data page on Kaggle",
        "is missing on disk", "checks passed", "if (kgImporting) { return; }",
    ):
        assert needle in html, needle
    # Dropped on purpose (allowed difference 4): the Ultralytics band and the YOLO format banner on
    # the Import tab (the Train tab reuses the banner geometry for the locked contract, TRAIN_MIRROR #2).
    import_panel = html[html.index('id="kg-panel-import"') : html.index('id="kg-panel-train"')]
    assert "format-selected-banner" not in import_panel
    for gone in ("kg-license-note", "Ultralytics", "YOLO", "exdark_", "dataset.yaml", "Explore test"):
        assert gone not in html, gone
    # Plumbing that stays ours: the host job channel, the provisioning state, the remote manifest.
    for needle in ("PluginJobs.start(", "PluginJobs.track(", "PluginJobs.on(NS, 'checks'", "PluginJobs.on(NS, 'stage_progress'",
                   "PluginJobs.on(NS, 'log_line'", "PluginJobs.cancel(", "PluginJobs.list(", "/import/preflight", "/import/state",
                   "/download/verify", "'download_kit'", "Setting up the plugin environment.", "status === 'provisioning'",
                   "?table=", "object_service=", "/manifest'", "refresh.state === 'done'"):
        assert needle in html, needle
    # Decisions of 2026-09-28 (EXDARK_MIRROR §3): Re-import fresh in ExDark's slot, writing fresh
    # tables (mode=reimport, never overwrite); ExDark's exact stepper dot; the hero title is the
    # manifest's display name; the ?kgdev fixtures ported.
    assert "kgStartImport({ reimport: mode === 'reimport'" in html
    assert "mode: opts.reimport ? 'reimport' : 'import'" in html
    # Gone on purpose: Start over and rc10's thin revisit view (rc11), the native confirm() on Submit (rc11), the
    # Advanced disclosure and the rc11 facts line (rc12: the rc10 form is always visible).
    for gone in ("Start over</button>", "window.confirm('This spends", "function kgRenderRevisit", 'id="kg-adv-toggle"',
                 'id="kg-imported"', "kg-imported-facts", "kgOpenAdvanced", "under Advanced"):
        assert gone not in html, gone
    # rc12: the Imported view keeps the form on screen (never display: none on the import record's view).
    imported_fn = html[html.index("function kgRenderImported(") : html.index("function kgReflectRecord(")]
    assert "el('kg-import-form').style.display = ''" in imported_fn and "display = 'none'" not in imported_fn
    assert "var currentSeen = false;" in html
    assert "querySelector('.kg-id-title').textContent = comp.display_name" in html
    assert "function kgDevForce(mode)" in html and "function kgDevDisableActions()" in html
    for state in ("state1", "state1-bundled", "state2", "state2-existing", "state2-mismatch", "state2-error", "state3",
                  "state4", "state5", "state6", "state6-found", "state6-stale", "state6-superseded", "dl-empty", "dl-running",
                  "dl-verify", "dl-verify-fail", "dl-success", "dl-fail", "dl-fail-checks", "dl-fail-kaggle", "dl-cancelled",
                  "dl-revisit", "dl-superseded"):
        assert f"mode === '{state}'" in html, state
    # Fix c: a stale result never sits next to a new amber/red preflight.
    assert "function kgClearStaleResult()" in html and "if (transitioned && (kind === 'mismatch' || kind === 'error')) { kgClearStaleResult(); }" in html


def test_fragment_is_the_exdark_train_tab():
    """The Train tab elements ExDark renders, in ExDark's ids and copy, under the session-3 decisions
    (docs/TRAIN_MIRROR.md §1, D1–D14 of 2026-09-29)."""
    html = FRAGMENT.read_text(encoding="utf-8")
    train_panel = html[html.index('id="kg-panel-train"') : html.index('id="kg-panel-submit"')]
    for needle in (
        "format-selected-banner", 'id="tr-contract-name"', 'id="tr-locked-rows"', 'id="tr-conn-banner"', 'id="tr-banner"',
        'id="tr-form"', 'for="tr-train-url"', 'id="tr-train-pick"', 'id="tr-train-pop"', 'id="tr-val-locked"',
        'id="tr-tables-gate"', 'id="tr-latest"', "Use latest revision", 'for="tr-epochs"', 'for="tr-batch"', 'for="tr-lr"',
        'for="tr-wd"', 'id="tr-duration"', 'id="tr-adv-toggle"', 'for="tr-device"', 'for="tr-workers"', 'for="tr-seed"',
        'id="tr-project-fact"', 'for="tr-runname"', 'id="tr-start-btn"', 'id="tr-cancel-btn"', 'id="tr-spinner"',
        'id="tr-state"', 'id="tr-progress"', 'id="tr-checks"', 'id="tr-result"', 'id="tr-log-toggle"',
        'data-range="epochs"', 'data-range="seed"', 'data-range="weight_decay"',
    ):
        assert needle in train_panel, needle
    # Dropped per the decisions: no val URL field (locked), no lrf / optimizer select / patience (D2–D4),
    # no extra args, no conf / max-det, no metrics-collection disclosure (D7), no presets (D13).
    for gone in ('for="tr-val-url"', 'id="tr-lrf"', 'id="tr-patience"', 'id="tr-extra"',
                 'id="tr-conf"', 'id="tr-maxdet"', 'id="tr-mc-toggle"', 'id="tr-embdim"', "preset"):
        assert gone not in train_panel, gone
    for needle in (
        "function kgBindTablePicker", "function kgUrlSeg", "function kgOverrideDisposition", "function kgApplyDerivedUrls",
        "function kgSetUrlOverride", "function kgSparkline", "function trRenderRunView", "function trRenderProvenance",
        "function trRenderSuccessBanner", "function trRenderFailBanner", "function trRenderCancelledBanner",
        "function trRenderTerminal", "function trEvaluateGate", "function trRenderGate", "function trVerifyTables",
        "function trRenderDurationHint", "function trLoadDurationStats", "function trInitTrainTab", "function trDevForce",
        "function trApplyContract", "function trApplyFields", "function trUsableLine", "var CFG_FIELDS", "var TR_BOUNDS",
        "Verified provenance recorded", "Training complete", "the checkpoint Predict uses", "Continue to Submit",
        "Open Run in Dashboard", "Open Run in Projects", "Start new run", "Training was interrupted",
        "Watch run in Projects", 'id="tr-run-watch"',
        "Training cancelled after", "Stop this training run?", "Training… (safe to navigate away)",
        "Cancelling… (stops at the next checkpoint)", "Fix the highlighted fields first.", "Tables verified: ",
        "excluded as undefined", "will be skipped until you label", "cannot be learned in this run",
        "No usable rows: every row is undefined or at weight 0.", "is not derived from the imported train table",
        "more rows than the competition split", "runs averaged ", "Estimated from a reference ",
        "/train/preflight?train_url=", "/train/state", "/tables/list?project=", "/tables/defaults?project=",
        "client_token: trClickToken", "trRunning = true;", "window.PluginJobs.cancel(trainJobId)",
        "'Train loss'", "'Val loss'", "'Val accuracy'", "function trDeviceLabel",
        # Item 1 of the re-check: the header names the reason and Use these settings never copies the device.
        "if (facts.device_label) { return '· ' + facts.device_label; }", "(forced in Advanced)",
        "The device is never copied",
        # Item 2: the estimate's inputs are keyed by the resolved device class; the in-run terms follow the run's.
        "function trStatsFor", "r.device_class === cls;",
        # Session 4 closeout: Previous runs lists the session project's runs only (project_runs).
        "var runs = (kgTrainState.project_runs || kgTrainState.runs || []).filter(",
        # ... and the Train banner, the Predict revisit and the stepper follow the same project.
        "function kgRefreshProjectScope", "kgTrainState.current_in_project === false ? null : kgTrainState.current",
        "kgRefreshProjectScope();   // a new project's Train / Predict start from their forms", "function trAwaitDeviceProbe", "function trRunEtaTerms",
        "Recent ' + clsText + ' runs averaged",
        # Item 3: the config load retries while the worker starts and never renders an empty form.
        "function kgStartupError", "name === 'AbortError'", "function kgCacheConfig", "function kgCachedConfig",
        "function kgApplyLoadedConfig", "function kgConfigFallback", "Loading saved settings… the plugin worker is starting",
        "from the last successful load", "so the form stays hidden rather than empty", "class=\"btn btn-ghost btn-sm kg-cfg-retry\"",
        "if (!r.ok) { var err = new Error('HTTP ' + r.status); err.status = r.status; throw err; }",
        # Item 4: Use latest revision is on for every new run (not persisted; Start new run resets it);
        # a pinned revision with newer ones warns.
        "el('tr-latest').checked = true;   // item 4", "'Pinned to '", "newer revision", "Turn on Use latest revision",
        "'train-state2-pinned'", "followed: useLatest && info.has_revisions, pinned: pinned",
        # Item 5: the runs dropdown shows the Run folder's unique name and says "interrupted", never "stale".
        "function trStatusLabel", "return status === 'stale' ? 'interrupted' : (status || '?');", "function trRunFolder",
        "return trRunFolder(r) + ' · ' + trStatusLabel(r.status)", "esc(trStatusLabel(r.status))", "esc(trStatusLabel(status))",
        # Item 6: older runs show backfilled settings; unrecoverable ones disable Use these settings with a reason.
        "r.params_missing ? 'not recorded'", "(from the Run\\'s record)", "if (!r.params_missing) { el('tr-run-use')",
        # Item 7: the tree connector is inline before the name and names never wrap; a three-level fixture.
        '<span class="kg-pop-name">', ".kg-pop-name { flex: 1 1 auto; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }",
        ".kg-pop-tree { display: inline;", "'train-state2-tree'", "depth: 2, parent: aUrl", "var kgDevTables = null;",
        # Closeout: the fixtures re-derive the manifest numbers on every apply (usable rows never 0 after the config);
        # the pinned fixture pins the seed with its rows while two newer revisions exist.
        "function derive() {", "          derive();\n          deriveText();", "base: pinnedBase, latest: newest, latest_url: rev2Url",
    ):
        assert needle in html, needle
    assert "Defaults are shown; reload the page to try again." not in html
    assert "'tr-latest': 'use_latest'" not in html
    for needle in ():
        assert needle in html, needle
    assert "el('tr-device').value = String(p.device)" not in html
    for needle in ():
        assert needle in html, needle
    for state in ("train-state1", "train-state2", "train-state2-missing", "train-state2-rows", "train-state2-invalid",
                  "train-state2-undefined-weight", "train-state2-zero-rows", "train-state2-class-empty",
                  "train-state2-lineage", "train-state2-pinned", "train-state2-tree", "train-state3", "train-state3-collecting", "train-state3-cpu-retry",
                  "train-state4", "train-state4-noproject", "train-state5", "train-state5-stale", "train-state6",
                  "train-state6-cancelled"):
        assert f"'{state}'" in html, state
    # The fragment defines no training literal: defaults, bounds, the optimizer and the schedule are served.
    for gone in ('value="10"', 'value="16"', 'value="0.0001"', "StepLR(5", "step_size: 5", "'adam'"):
        assert gone not in train_panel, gone
    # Part B: every training field group is tagged; the two choice fields are selects that render only
    # when the manifest lists them under training.editable (locked rows otherwise); the Start body
    # carries editable fields only.
    for needle in ('data-field="epochs"', 'data-field="seed"', 'data-field="optimizer" hidden', 'data-field="schedule" hidden',
                   'id="tr-optimizer"', 'id="tr-schedule"'):
        assert needle in train_panel, needle
    for needle in ("function trIsEditable", "function trLockedFieldRows", "group.hidden = !trIsEditable(key)",
                   "if (key === 'workers' || trIsEditable(key)) { body[key] = el(id).value; }"):
        assert needle in html, needle


ESC = "function esc(s) {"

# Values that may enter a markup template unescaped: calls that emit markup they built themselves
# or text they escaped/formatted, our own accumulators and constants, and the numbers, enum values
# and boolean flags the fragment itself computes (used in ternaries over literals).
SAFE_CALLS = ("esc(", "kgIcon(", "fmtCount(", "fmtDur(", "dlMB(", "kgCheckIcon(", "kgDiagBtn(", "kgClassTint(",
              "dashTableLink(", "kgWithObjectService(", "encodeURIComponent(", "kgFmtAgo(", "link(", "trUsableLine(",
              "psKaggleLine(", "kgWhenSpan(", "stHeroBlock(", "kgValEditedWarning(",
              # rc10: the "Other ways to connect" row (esc + kgIcon inside)
              "psOtherWayRow(",
              # rc13: the shared checks accordion and its rows (markup built from literals, esc + kgIcon inside)
              "kgChecksAccordion(", "kgChecksSummary(", "kgCheckRow(")
SAFE_IDENTS = {
    # markup accumulators / constants the fragment builds from literals and the calls above
    "html", "banner", "mhtml", "chips", "lines", "badge", "elapsed", "fade", "entering", "text", "head", "tail", "counts",
    "KG_HINT_HTML", "KG_BTN_IMPORT", "KG_BTN_RERUN", "KG_BTN_DL", "KG_BTN_DL_RESUME", "KG_BTN_TOPUP", "KG_ICONS", "KG_STAGE_LABELS",
    "orig", "glyph", "label", "when",
    # rc13: the shared checks accordion (markup it built, its id prefix, the summary and body it is handed)
    "prefix", "summary", "body",
    # numbers, enum values and loop variables the fragment defines
    "passed", "total", "i", "s", "split", "status", "pct", "quartile", "r", "g", "b", "cls", "name", "g.title", "kind",
    # boolean flags used only to pick between literal branches
    "animate", "allOk", "c.ok", "t.reused", "dashboardUrl", "kgRevisitActive", "resume", "dlMode", "isCur", "isDone",
    "model.pretrained", "pool", "files", "have", "current", "first", "res.updated", "icon", "prov", "prov.state", "repo",
    "p.detail", "c.detail", "d.detail", "remedy", "matched", "fileCount", "v.error", "true", "false", "null", "undefined",
    # Train tab (session 3): sparkline geometry numbers, the flags and enum values its templates branch on
    "w", "h", "cx", "cy", "coords", "open", "revisit", "problems", "data.help", "data.noProjectHint", "parts",
    "summary.labeled_in_use", "summary.undefined_with_weight", "summary.excluded_undefined",
    "summary.excluded_zero_weight", "empty.length", "epochs", "weights", "link", "projectsHref", "showNote",
    "p.stage_note", "c.ok", "allOk", "checks", "html",
    # ExDark's gate and picker templates: pre-escaped problem rows and help lines, the LATEST flag, the
    # inline-error flag, the usable-row parts (all built from fmtCount / esc above)
    "t.latest", "parts.join", "problems.length", "problems.map", "data.help.map", "function", "return", "join",
    "p", "h", "bad",
    # the revision tree (part D): the out-of-lineage flag and the indentation depth
    "off", "depth",
    # the previous-runs card (part E): rows are [label, pre-escaped value] pairs built above
    "row",
    # the duration hint's device class word (item 2 of the re-check): 'CPU' or 'GPU', picked over literals
    "clsText",
    # the config-load banners (item 3): the pre-escaped reason and the Retry button markup built above
    "reason", "retryBtn",
    # the bundle's checkpoint checklist (session 6, STATUS_MIRROR D16): the checkbox attribute flags and
    # the pre-escaped text / helper-built markup pieces assembled above the template
    "attrs", "metaText", "ruleTag", "note",
    # the pinned-revision warning (item 4): built from esc / fmtCount above
    "data.pinned",
    # the Predict + Submit tab (session 4): the button constants, pre-escaped text and flags its
    # templates branch on (ExDark's names), the grouped-checks loop, the download button markup
    "KG_BTN_PREDICT", "KG_BTN_PREDICT_RERUN", "KG_BTN_SUBMIT", "okText", "dirText", "budgetTxt", "exhausted", "isInfo",
    "dl", "enter", "phase", "groups", "score", "score.value", "sub.ref", "ss.ref", "k.status", "psBasis.persisted",
    "psBasis.when", "ps.finished_at", "ss.finished_at", "ss.status", "ss.reason", "s.username", "s.probe_error",
    "sanity.warning", "names.length", "n", "pc", "b.left", "b.limit", "data.ok", "data.checking", "data.count",
    "run", "run.usable",
    # rc10, the guided connect flow: the caret constant, the ?kgdev flag (disabled attributes over literals), the
    # status line's connect-link flag
    "KG_CARET_SVG", "kgDevMode", "withConnect",
    # the Status tab (session 5): ExDark's hero-block arguments (pre-escaped markup and the big/val flag),
    # the pre-built table cells (runCell, actions, pub, deltaCell, o.html — all from esc / kgIcon above)
    # and the launched flag the Kaggle callout branches on
    "big", "mainHtml", "noteHtml", "runCell", "actions", "pub", "deltaCell", "o.html", "s.launched",
    # rc11 (the Import tab and shell pass): the pre-escaped pieces and the flags the new templates branch on
    "reusedAll", "validated", "revText", "rowsText", "whenText", "pathShown", "kitShown", "whenLbl", "found", "v", "kit",
    "facts", "missing.length", "imgs", "state.source", "state.verified_at", "state.imported_at", "left", "fb", "mode",
    "t.exists", "t.latest_revision", "t.revision", "res.image_count", "res.file_count", "failed.length", "hasCsv", "url",
    "known", "kitDir", "testN", "KG_BTN_REIMPORT",
}
HEAD = re.compile(
    r"(?:\.innerHTML\s*\+?=|\bvar (?:html|banner|mhtml|chips|lines|badge|elapsed|text|head|tail|KG_[A-Z_]+)\s*=|"
    r"\b(?:html|banner|mhtml|chips)\s*\+=|\breturn\s+(?='<))"
)


def _statement_from(script: str, pos: int) -> str:
    """The statement starting at ``pos`` up to its terminating ';' outside strings and brackets."""
    depth = 0
    i = pos
    n = len(script)
    while i < n:
        ch = script[i]
        if ch == "'" or ch == '"' or ch == "`":
            q = ch
            i += 1
            while i < n and script[i] != q:
                i += 2 if script[i] == "\\" else 1
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif ch == ";" and depth == 0:
            break
        i += 1
    return script[pos:i]


def _template_statements(script: str):
    """Statements that BUILD MARKUP: an innerHTML assignment, an accumulator, or a returned
    template — recognised by carrying an HTML literal ('<…) or an icon call."""
    for m in HEAD.finditer(script):
        stmt = _statement_from(script, m.start())
        if "'<" in stmt or "kgIcon(" in stmt or "kgCheckIcon(" in stmt or "kgDiagBtn(" in stmt:
            yield script.count("\n", 0, m.start()) + 1, stmt


def _operands(stmt: str):
    """The non-literal top-level operands of the statement's concatenation."""
    body = re.sub(r"^.*?(?:\+=|=|\breturn)\s*", "", stmt, count=1, flags=re.S)
    depth = 0
    cur = ""
    ops = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in "'\"`":
            q = ch
            j = i + 1
            while j < len(body) and body[j] != q:
                j += 2 if body[j] == "\\" else 1
            cur += body[i : j + 1]
            i = j + 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "+" and depth == 0:
            ops.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    ops.append(cur)
    out = []
    for o in ops:
        o = o.strip()
        if not o or re.fullmatch(r"'(?:[^'\\]|\\.)*'", o):
            continue  # a string literal
        out.append(o)
    return out


def test_fragment_is_the_exdark_status_tab():
    """The Status tab elements ExDark renders, in ExDark's ids and copy, under the session-5 decisions
    (docs/STATUS_MIRROR.md §1, D1–D10), plus the +brief sections (Runs, Verification, Doctor)."""
    html = FRAGMENT.read_text(encoding="utf-8")
    status_panel = html[html.index('id="kg-panel-status"') : html.index("<script>")]
    for needle in (
        "Your best score, latest activity, and submission history.", 'id="st-conn-banner"', 'id="st-gate"', 'id="st-body"',
        'id="st-hero"', 'id="st-updated"', 'id="st-refresh-btn"', 'id="st-runs"', 'id="st-history"', 'id="st-kaggle"',
        'id="st-verify"', 'id="st-export-btn"', 'id="st-export-note"', "Export verification bundle",
        'id="st-doctor-toggle"', 'id="st-doctor-panel"', 'id="st-doctor"', "Never images, prediction CSVs, tokens or answer keys.",
        'id="st-bundle-runs"', "the best checkpoint of up to two submitted runs",
        "History", "Kaggle live", "Verification", "Doctor",
    ):
        assert needle in status_panel, needle
    assert "arrives in a later session" not in html
    for needle in (
        "function stHeroBlock", "function stOutcome", "function stRenderHero", "function stRenderRuns", "function renderHistoryFrom",
        "function renderKaggleFrom", "function renderKaggleLive", "function stDoctorRows", "function renderDoctorFrom",
        "function stBuildDiagnostics", "function stLoadDoctor", "function stExportBundle", "function stFetchAll",
        "function stLoadBundlePreview", "function stRenderBundlePreview", "function stBundleQuery",
        "Checkpoints to include", "checkpoints=selected&runs=", "/status/bundle/preview?",
        "function refreshStatus", "function stStartAuto", "function stStopAuto", "function stOnTabEnter", "function stDevForce",
        "function kgWhenSpan", "function stLiveKind",
        # ExDark's outcome vocabulary and copy
        "Validation failed", "Interrupted", "Running…", "CSV generated (daily limit reached)", "CSV generated (not joined on Kaggle)",
        "Submission rejected", "CSV generated (not submitted)", "rejected by Kaggle",
        "No predictions yet. Train a model and predict on the test set to see", "Go to Train",
        "Best public score so far: ", "Leaderboard: rank ", "No Kaggle submissions yet.", "View leaderboard on Kaggle", "Show details",
        "become available after the competition launches", "a known Kaggle API limitation",
        "Per-day counter unavailable on this competition.", "Resets 00:00 UTC.", "'Live now'", "Open tab", "'Latest activity'",
        "'Best public score'", "no scores yet", "no activity yet",
        # the routes and the refresh loop
        "/status/history?live=", "/status/kaggle", "/status/doctor", "/status/bundle", "computeUrl + '/health'",
        "}, 15000);", "setInterval(stRenderUpdated, 5000)", "document.addEventListener('visibilitychange'",
        "if (name === 'status') { stOnTabEnter(); } else { stStopAuto(); }",
        # the gate: Status ungated behind the import record, its body hidden otherwise
        "['st-gate', 'Status', 'st-body']", "el(pair[2]).hidden = !imported;",
        # Continue to Status lands on it
        "id=\"ps-continue-status\"", "el('ps-continue-status').addEventListener('click', function () { showTab('status'); });",
        # the Doctor rows (D6) and Copy diagnostics
        "'Compute service'", "'CUDA in the worker'", "'Free disk space'", "'Plugin home'", "kgBindDiag(box, stBuildDiagnostics)",
        "'[doctor]'",
        # rc3: the Doctor reloads on every job completion and on the poll (without Kaggle), one budget wording,
        # the manifest row's refresh state, the Loop's inspect link bound to the latest run or disabled.
        "function stOnJobDone", "if (terminal) { stOnJobDone(); }", "'/status/doctor?kaggle='",
        "stLoadDoctor(!!manual || !!live || !stDoctorLoaded);", "' left today'", "m.refresh_state === 'failed'",
        "carried forward from the previous version", "function kgSetLoopInspect", "kg-loop-off",
        "No run in this project yet. Train first;",
    ):
        assert needle in html, needle
    assert "submissions left today" not in html and "' today'" not in html
    # The fixtures (D7): ExDark's five names; fixture pages disable the export button too.
    for state in ("status-empty", "status-history", "status-live-running", "status-kaggle-live", "status-kaggle-403"):
        assert f"mode === '{state}'" in html, state
    assert "if (mode && mode.indexOf('status-') === 0) { stDevForce(mode); return; }" in html
    assert "'st-export-btn'].forEach" in html


def test_every_interpolated_value_in_an_innerhtml_template_is_escaped():
    """ExDark's safety pattern, as a census over the fragment's markup-building statements: a value
    that reaches innerHTML is a string literal, one of our own accumulators/constants, a number or
    flag the fragment computed, or wrapped in esc() (or a helper that only emits markup it built)."""
    script = _script()
    assert ESC in script
    offenders = []
    for line, stmt in _template_statements(script):
        for op in _operands(stmt):
            if any(call in op for call in SAFE_CALLS):
                continue
            idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", re.sub(r"'(?:[^'\\]|\\.)*'", "", op)))
            if not idents or idents <= SAFE_IDENTS:
                continue
            offenders.append(f"line {line}: {op[:100]}")
    assert offenders == [], "unescaped interpolation(s) in a markup template:\n" + "\n".join(offenders)
    # The census saw the templates it exists for.
    assert sum(1 for _ in _template_statements(script)) >= 25


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to execute the fragment's esc()")
def test_markup_in_a_manifest_string_renders_as_literal_text():
    """A class name or display name of `<img src=x onerror=alert(1)>` must render as text. Every
    manifest string enters the DOM through esc() (census above); this runs the fragment's own esc()
    on the payload and checks the escaped form, which browsers render literally."""
    script = _script()
    start = script.index(ESC)
    end = script.index("}", script.index("replace(/\"/g, '&quot;');", start)) + 1
    esc_src = script[start:end]
    payload = "<img src=x onerror=alert(1)>"
    out = subprocess.run(
        ["node", "-e", esc_src + "\nprocess.stdout.write(esc(" + json.dumps(payload) + "));"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout == "&lt;img src=x onerror=alert(1)&gt;"
    # The two manifest strings a participant sees first go through it: the lock chip and the class tags.
    assert "esc(model.arch)" in script and "esc(n)" in script


def test_constraint_chips_read_the_manifest(home, manifest):
    """ExDark's constraint chips are the model, the class count and the metric (the daily limit belongs
    to Predict + Submit). Whatever renders reads the manifest; the fragment holds no count of its own."""
    html = FRAGMENT.read_text(encoding="utf-8")
    assert not re.search(r"\d+ submissions per day", html)
    assert "esc(sub.metric)" in html and "fmtCount(classes.length)" in html and "esc(model.image_size)" in html
    served = routes.config_payload()["_meta"]["manifest"]
    assert served["submission"]["daily_limit"] == manifest.submission.daily_limit == 100
    assert served["competition"]["display_name"] == "3LC Scene Classification Challenge"
