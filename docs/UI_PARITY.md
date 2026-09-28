# UI_PARITY.md — bringing the fragment to the ExDark plugin's standard (session 2.5, Phase 1)

Read-only study of `../3lc-compute-plugin-kaggle/src/tlc_plugin_kaggle/ui.html` (6,550 lines:
CSS 3–568, markup 569–1215, JS 1216–6549) and `docs/ui-notes.md` (the UI playbook, §1–§18 and
the appendix), against our `src/kaggle_classification/ui/ui.html` (≈560 lines) as shipped at
`f6dad81`. Nothing here has been changed yet; Phase 2 applies the "Planned" column to the shell
and the Import tab as reusable components for Train, Predict + Submit and Status.

**Ground rules carried over unchanged:** every value comes from `GET /config` or the job channel
(no competition literals, `test_routes.py` census); manifest strings only via `textContent`
(ExDark builds markup with `innerHTML` + `esc()`; we keep `textContent`/`createElement` because
our census test forbids `innerHTML` — the components below are written that way); jobs through
`window.PluginJobs` (host dispatch), never a plugin socket; Windows-first, paths quoted.

## 1. The parity table

| # | Pattern (ExDark, ui-notes §) | ExDark does | Ours today | Planned change (Phase 2) |
|---|---|---|---|---|
| 1 | **Page frame** (§8, appendix) | `.plugin-page-narrow` (Hub class), `padding: 24px; max-width: 900px; margin: 0 auto`, Hub `.card / .card-header / .card-title / .card-subtitle / .card-body`, `.plugin-section-number` badges, no `font-family` anywhere | own `.kgc` wrapper, own `.kgc-card` with private CSS, `max-width: 980px`, no Hub card classes | Adopt `.plugin-page-narrow` + Hub `.card*` classes and `.plugin-section-number`; keep a private `kgc-` namespace only for what the Hub has no class for. Drop `.kgc-card` styling in favour of the Hub's |
| 2 | **Header identity** (§3, §16 "static furniture") | `.plugin-hero` > `.kg-id-row`: 20px monochrome icon in `--accent`, `h1.kg-id-title` (16px/700), `p.kg-id-sub` one-liner in `--text-secondary`; no entrance motion | `h1` 20px + `p.kgc-sub` that is the manifest's `loop_banner_text` (so the tagline changes when the banner text does) | Port `.kg-id-row` verbatim (our plugin.toml `icon_svg` as the 20px identity icon). Tagline = a fixed plugin one-liner ("Import the starter kit, train the fixed baseline, label in the Dashboard, predict and submit.") — keeps "label in the Dashboard". The manifest's `loop_banner_text` moves to the Loop row (#4) |
| 3 | **Constraints chips** (§13, §18 "the contract is served") | `.kg-micro-label` "Competition constraints" + `.kg-chips`: lock chip "YOLOv11n · COCO-pretrained · 640px", "12 classes", "Scored by mAP@0.5"; every chip has a `title` tooltip; values fill from `_meta.contract`, slots ship EMPTY | the Competition card's four `kgc-kv` cells (Name / Model / Kit / Manifest) + class pills; no tooltips | Replace the kv grid with chips filled from `_meta.manifest`: lock chip "resnet18 · from scratch · 150 px" (arch/pretrained/image_size), "6 classes + undefined", "Scored by accuracy" (`submission.metric`), "Kit v1", "3 submissions/day" (`daily_limit`). `title` tooltips on each, copy in the inform-not-instruct tone. Class names stay as tinted tags (#20) under the chips |
| 4 | **The Loop banner** (§1, header) | `.kg-loop` row: micro-label "The Loop" + `import › train › inspect ↗ › fix labels ↗ › retrain › submit`; `data-goto` links switch tabs, external ones deep-link the Dashboard with `?object_service=`; the active tab's step reads stronger | none (the tagline paraphrases the loop) | Port `.kg-loop` verbatim: `import › train › inspect ↗ (Dashboard, ?table=<train url>) › label ↗ (Dashboard) › retrain › predict › submit`; the manifest's `loop_banner_text` renders as the row's `title`/lead sentence. Steps for tabs that do not exist yet (train, predict, submit) still switch to the tab, which shows its gate (#17) |
| 5 | **Tab bar as stepper** (§1, `renderPipeline`) | `.kg-tabs` with `flex:1` centred tabs "1 Import / 2 Train / 3 Predict + Submit / 4 Status", each with a `.kg-tab-sub` line and a `.kg-tab-state` glyph (check-circle done · dot current · circle pending) driven by `GET /pipeline`; keyboard (`tabindex=0`, arrow keys); last tab remembered in `localStorage` | `.kgc-tabs` plain buttons, no sub-line, no state glyph, no keyboard, no memory | Port the tab bar. Sub-lines: "Starter kit → 3LC tables" / "resnet18, from scratch" (from the manifest) / "Predict → CSV → Kaggle" / "History & leaderboard". State glyphs from `_meta.import_state` now (Import done when a verified record exists) and from later tabs' records later. Remember the tab in `localStorage` (per-viewer convenience, try/catch) |
| 6 | **Section stepper rows** (§1 State 3, `KG_STAGE_LABELS`) | stock queue-panel anatomy: `.kg-job-row` label + `badge badge-status-*` (QUEUED / RUNNING / COMPLETED / FAILED) + elapsed; badges fade in only on status change | own `.kgc-steps` list with a coloured dot, label and detail text; no badges, no elapsed | Keep one list (Manifest · Disk space · Download · Verify · Extract · Register · Validate) but render each row with the Hub `badge badge-status-*` vocabulary + elapsed; badge swaps only on transitions. Detail text stays (the shard/bytes line is what makes a slow link not read as stuck, §15) |
| 7 | **Status chips / badges** (§6, `.kg-badge-created`/`reused`) | pill badges (10px/700) for per-table outcomes, `badge badge-status-*` for job states | none | Add `.kg-badge-*` pills: CREATED on import rows, FRESH on a re-import row; job-state badges from #6 |
| 8 | **Hover help and tooltips** (§4, 34 `title=` + 39 `.form-help`) | every editable field has a `.form-label` + `.form-help` line under it; every locked fact carries a `title`; chips carry `title`; help text informs, never instructs; no em dashes | `label` + `input` only; zero `title`, zero help lines | Every field gets `.form-label`/`.form-help` (Project: "`intel-scene` is the competition's project; tables land under it." Table name: "`initial` is the convention for a first import; a re-import gets `-2`, `-3`." Kit directory: "Filled by the download; read-only."). Every locked fact (chips, kit dir, dest) gets a `title`. Copy audited for the no-em-dash rule |
| 9 | **Empty states** (§1 State 1) | form + placeholder panels + `guide-pulse` on the first required control + primary CTA disabled with a reason | "No kit on this machine yet…" line and a disabled Import CTA; no pulse, no reason next to the CTA | State 1: the Download CTA gets the `guide-next` pulse (stock keyframes, reduced-motion gated); the Import card shows a quiet placeholder ("Import unlocks after the kit is on disk.") instead of the amber preflight error |
| 10 | **Error and warning presentation** (§5, `.kg-callout`) | one callout geometry (icon column + text, `padding 9px 12px`, `radius-lg`) in four tints `ok / info / warn / err`; failure banners = `alert alert-error` + **Copy diagnostics** (`kgBuildDiagnostics`: version/OS/time header + preflight JSON + checks + log tail in a fenced block); remediation hints under failed checks (`KG_REMEDIES`) | own `.kgc-callout` (left border only), first-line error in the callout, traceback in the log accordion; no Copy diagnostics, no remedies | Port `.kg-callout` geometry and tints (icon column). Add `kgDiagBtn` + a `buildDiagnostics()` (version, host, time, manifest provenance, params, checks, log tail) on every failure surface. Add a small remedy map for our check labels (kit defect → "report to the organizers"; collision → "Re-import or another table name"; sha mismatch → "run again, it resumes") |
| 11 | **Light/dark theming** (appendix) | only Hub tokens: `--text/--text-secondary/--text-muted`, `--border/--border-light`, `--accent/--accent-light`, `--bg-card/--bg-secondary`, `--success/--warning/--danger`, `--radius-lg`; fallback literals = the stock plugin's (`#d97706`, `#ef4444`, `rgba(5,150,105,…)`); tints via rgba so they survive both themes | uses `--text/--text-muted/--border/--accent/--bg/--bg-card`, but also invents `--accent-contrast`, `--error`, hardcoded `#fff`, and solid `--success` backgrounds on dots | Use exactly ExDark's token set and fallbacks (`--danger` not `--error`; no `--accent-contrast`; tints as rgba). Nothing solid-filled except the primary button (Hub `.btn-primary`) |
| 12 | **Long-path handling** (§8, `midTrunc`, `kgBindCopy`) | paths never render raw in participant view: `.path` cells `overflow: hidden; white-space: nowrap` + `midTrunc(s, 64)` + a ghost **Copy** button (`data-copy`, "Copied" swap, `prompt()` fallback); full paths only in the log / diagnostics; `code { word-break: break-all }` as the backstop | raw plugin-state paths and full table URLs in the kit line, the destination line, the footer and the success list | Component `pathCell(url)`: middle-truncated text with the full value in `title` + Copy button. Used for kit dir, destination, table URLs (with "Open in Dashboard" and the Hub project link beside), and the footer's state path moves into Technical details (#13) |
| 13 | **Collapsed details** (§5, `kgBindAccordion`) | button + region accordion (`aria-expanded`, `hidden`, animated `grid-template-rows`), caret rotates; "Show log" collapsed in every state; advanced options behind the same disclosure | native `<details>` for "Details" and "Show log"; the record block prints job id, lineage URLs, timing JSON | Port the accordion (button + region, `aria-controls`). Two disclosures on the Import card: **Technical details** (plugin home, kit dir, dest, job id, timings, manifest sha256 + document URL, lineage root, locked val, each with Copy) and **Show log**. Default collapsed; the participant view above them carries no raw path, id or JSON |
| 14 | **Checks presentation** (§5) | verdict line first (check-circle "9/9 checks passed" / x-circle "7/9 — 2 failed"), two-column grouped grid with uppercase group heads, compress-on-pass, expand-on-fail with italic remedy, thousands separators | verdict line + a flat two-column list; every check always visible; details echo the assertion | Verdict line becomes the accordion toggle: **collapsed** "18/18 checks passed" that expands on click; **any failure auto-expands** with only failed checks open by default. Groups: Kit · Tables. Pass rows compressed ("train labeled per class · 100"), fail rows expanded with remedy |
| 15 | **Busy and disabled states** (§6, `spinner` ×31, `aria-live` ×14) | CTA disabled while a job runs, label swaps to "Importing…" with a `spinner` span, `aria-live="polite"` status spans, Cancel visible; hover lift on the primary next-step button; `.kg-step-muted` (opacity .55 + pointer-events none, plus real `disabled`) for a step that is not yet available | CTA disabled + Cancel shown; no spinner, no live region, no label swap | Add the Hub `spinner` to the running CTA ("Importing…"/"Downloading…"), `aria-live="polite"` on the pipeline subtitle and the kit state line, `.kg-step-muted` on the Import card while the kit is missing |
| 16 | **Revisit views** (§1 State 6, §15) | straight into the success view; form hidden; **Start over** ghost button in the banner; a solved problem earns one quiet line (`kg-preflight-ok`) not a panel; kit section hides on Import revisit except for a running download or a superseded kit | success callout + table list + all 18 checks + Details + Start over; kit card always visible with a "Download again" CTA | Success view = `alert alert-success` banner ("Imported · train 6,600 rows · val 1,200 rows") with the next-step primary (#18) and Start over ghost; checks collapsed (#14); the kit section collapses to one quiet line with Verify/Download again as ghost actions (visible only for running/superseded, per §15) |
| 17 | **Tab gating** (Train gate, `trRenderGate`) | Train/Predict verify their tables on tab entry; without them an amber `.kg-callout.warn`: "**Tables not found.** … Looking in project `x` — no import has produced these tables there yet …" + a **Go to Import** secondary button with a trailing arrow | Train / Predict + Submit / Status panels say "arrives in session N" | One `gateCard(tab)` component: while `_meta.import_state.state !== "success"` each of the three tabs renders the info callout "**Import the kit first.** The Train / Predict + Submit / Status steps use the tables Import creates." + **Go to Import**. When a record exists, the panel shows its own (still-stub) content. Tab-state glyphs (#5) stay pending |
| 18 | **Next-step hints** (`kg-continue-train`) | success banner carries the banner-primary "Continue to Train →" which actually switches the tab; the Loop row highlights the current step | none ("Kit imported. Continue on the Train tab." as text) | Banner primary **"Next: train your first model →"** switches to the Train tab (which shows its gate content or, from session 3, the form) |
| 19 | **Copy tone** (§4) | facts, sentence case, terminal periods on sentences, no em dashes (rendered copy), "1 to 300" not "1–300" | mostly fine; a few instructions ("Download the starter kit first."), one em dash in the kit line, `—` in summaries | Audit every string against §4; keep imperative only in remedies |
| 20 | **Class tags / palette** (§8 class-tag palette) | `kgClassTint(i)` from the Hub `--chart-1…12` categorical variables as quiet tints | plain pills "0 buildings" | Tint each class pill with `--chart-N` (index = manifest id); `undefined` stays untinted muted |
| 21 | **Footer** (§ "Version footer") | one muted line per tab: version, "Source" link, state path removed from view | version + raw state path + Source | Version + Source only; the state path moves to Technical details |
| 22 | **Connection guard** (§7 `kgConn`) | network-level fetch failures show a retrying banner (backoff 2/5/10/15 s) in every tab, resume polls/gates on reconnect, never restart work | none: a failed `/config` load writes "Could not load the plugin config" into the tagline | Port a minimal `connGuard`: retrying banner slot at the top of each panel, resumes `load()`/gate on reconnect; job starts are never retried automatically |
| 23 | **Manifest source label** (our #kgc-source) | n/a (ExDark has no manifest) | shows the first resolution ("cache (fetched …)") and only re-renders when the poll settles inside 8 s; the label then stays "cache" although the refresh finished `remote` | Render the source from the **refresh result** when `manifest_refresh.state === "done"` (`refresh.source`), and re-read `/config` on tab activation so the label tells the truth without a reload |
| 24 | **Provisioning state** (ours, session 2) | ExDark predates first-use provisioning | info callout "Setting up the plugin environment…" + poll | Keep; restyle as `.kg-callout.info` with the Hub `spinner` |
| 25 | **`?kgdev` fixtures** (§1 dev affordance, §12) | every state renderable from static fixtures, actions disabled under `?kgdev` | none | Not in Phase 2 scope (playbook item, not a participant-facing gap); listed so it is not forgotten for the Train session |

## 2. ExDark components that port directly

Ported as-is (CSS verbatim, JS re-expressed with `createElement`/`textContent` to satisfy our census test):

- **CSS blocks:** motion tokens + the whole `@media (prefers-reduced-motion: no-preference)` gate,
  `.kgi*` icon sizing, `.kg-id-row/-icon/-title/-sub`, `.kg-micro-label`, `.kg-chips/.kg-chip`,
  `.kg-tabs/.kg-tab*`, `.kg-loop*`, `.kg-job-row`, `.kg-checks-grid/.kg-check*/.kg-verdict`,
  `.kg-table-row*`, `.kg-badge-created/-reused`, `.kg-log-details` + `.kgl-acc*` accordion,
  `.kg-callout` (four tints), `.kg-preflight-ok`, `.kg-sec-head`, `.kg-locked-rows/-row`,
  `.kg-footer`, `.kg-visually-hidden`, overflow containment rules, the `@media (max-width: 720px)`
  block, `.kg-btn-danger-hover`, `.kg-step-muted`, `.form-control.guide-next` + `guide-pulse`.
- **Icon set** (`KG_ICONS` + `kgIcon`): lock, check, x, check-circle, x-circle, alert-triangle,
  info, copy, arrow-right, upload, refresh-cw, chevron-right, external-link, circle, dot.
- **Helpers:** `midTrunc`, `fmtCount`, `fmtDur`, `kgBindCopy` (data-copy + Copied swap + prompt
  fallback), `kgBindAccordion` / `kgAccOpenInstant`, `kgSwapText`/`kgSwapHtml` (as text swaps),
  `kgMotionOK`, `kgScrollTo`, `kgBuildDiagnosticsCore` + `kgChecksSection` + `kgLogTailSection` +
  `kgDiagBtn`, `kgConn` (connection guard), `kgClassTint`, the tab keyboard handler, the
  `localStorage` tab memory, `renderPipeline`'s glyph logic (fed from `_meta` instead of a
  `/pipeline` route).
- **Markup:** the hero (`plugin-hero` + id row + constraints + loop), the tab bar, the card
  header anatomy with `plugin-section-number`, the success banner with banner-primary + ghost
  Start over, the "Show log" accordion.

Adapted, not copied: the Train gate callout becomes the generic `gateCard` (#17); the ExDark
download section's two blurbs become one blurb reading the manifest (`kit.version`, shard count,
bytes from `_meta`); the checks grid gains the collapsed verdict toggle (#14).

Not ported: the Ultralytics license band (§16, AGPL-specific), the locked-format banner (YOLO
format), the Status hero strip and history table (session 6), the segmented source toggle and
sanity card (sessions 4–5), the sparkline. **Deferred, not dropped:** the revision picker popover
(`kgBindTablePicker`, ui-notes §7) ports in session 3 with the Train tab's table-URL fields.

## 2b. Adjustments from the Phase 2 go (2026-09-28)

- Loop steps for classification: **import › train › label & weight in Dashboard ↗ › retrain ›
  predict › submit**; the Dashboard step deep-links the **latest train revision**
  (`import_state.latest.train`, tlc's own `latest()`), not the seed table.
- Connection guard: ExDark's **full** `kgConn` (retrying banner with 2/5/10/15 s backoff in every
  tab's slot, resume callbacks on reconnect, never restarts or duplicates work).
- Class chip wording: **"6 classes · 6,000 unlabeled to label"**, both numbers from the manifest.
- Remedy map, at minimum: disk full, network or CDN failure, sha256 mismatch, file locked
  (antivirus or sync client), tables already exist, plugin environment still provisioning.
- Reduced motion: the guide pulse and every other animation live inside the
  `prefers-reduced-motion: no-preference` gate; JS waits are gated by `kgMotionOK()`.
- Hub project link: built from `window.location.origin` only when the origin is `hub.3lc.ai` or
  `hub-beta.3lc.ai`; omitted elsewhere.
- Element builders + `textContent` everywhere; the `innerHTML` ban stays.

## 3. Decisions already taken for Phase 2 (from the brief)

- Competition display name → **"3LC Scene Classification Challenge"** in the bundled manifest;
  regenerate `cdn/kaggle/intel-scene/manifest.json` + `upload-plan.json`, do NOT upload;
  PROMOTION.md gets the new sha256; new tables' descriptions pick up the name, existing keep theirs.
- Link text **"Open in Dashboard"** (the Hub's own Open button goes to dashboard.3lc.ai, so the
  target is right). Add a **Hub project link** if derivable: the Hub is `https://hub.3lc.ai` and
  the observed project page form is `hub.3lc.ai/projects/<project>`; the fragment has no Hub URL
  from `PLUGIN_API` (only `dashboard_url`, `object_service_url`, `compute_service_url`), so the
  host is derived from `window.location.origin` when the fragment runs inside the Hub, and the
  link is omitted (not guessed) when it runs elsewhere. To be confirmed in the visual review.
- Tagline keeps "label in the Dashboard".
- Manifest source label reflects the latest resolution (#23).
- Participant view: no raw plugin-state paths, job ids or timing JSON; they live behind
  **Technical details** with Copy buttons; long paths truncated with Copy (#12, #13).
- The 18 checks collapse to "18/18 checks passed"; failures always expand (#14).
- Success next step: **"Next: train your first model"** → Train tab (#18).
- Tab gating: Train, Predict + Submit and Status show "Import the kit first" until an import
  record exists (#17).

## 4. Constraints the port must respect

- `tests/test_routes.py`: no `innerHTML` / `insertAdjacentHTML` / `outerHTML` / `document.write` /
  `eval(`; no competition literals (`buildings`, `resnet18`, `6000`, …) in the fragment; the
  PluginJobs needles stay. ExDark's `esc()`+`innerHTML` idiom is therefore re-expressed with
  element builders; the icon set is inlined as SVG element builders, not strings.
- `docs/PLAN.md` §A2 hardening: manifest strings only through `textContent`; help-link URLs https.
- Fragment rule (CLAUDE.md §B): any `ui.html` change means a **worker reload** on a folder source and
  a **fresh install from a new ref** for the catalog-installed plugin on `3lc-hub-11`
  (`catalog-test.json` re-pinned, `DELETE` + install as in `phase_d_import.py`).
