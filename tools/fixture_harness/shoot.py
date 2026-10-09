# SPDX-License-Identifier: Apache-2.0
"""Screenshot the fragment's ?kgdev fixtures without a Hub login.

The compute service's plugin routes answer 403 without the Hub JWT, so a real service cannot be screenshotted
from a script. This harness serves index.html + a copy of src/.../ui.html + config.json (make_config.py) + the
Hub's public CSS (fetched once) on a loopback port and drives it with Playwright.

    python tools/fixture_harness/shoot.py <out dir> <state,state,...>

Needs an interpreter with ``playwright`` and a Chromium channel (``channel`` below; msedge needs no download).
Writes import-<state>.png (full page, 1100 px wide) and shoot_index.json (what each page rendered and any
console error) into <out dir>. The copies it makes beside itself (ui.html, main.css, plugin-common.css,
config.json, shoot_index.json) are gitignored."""
from __future__ import annotations

import json
import shutil
import sys
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FRAGMENT = REPO / "src" / "kaggle_classification" / "ui" / "ui.html"
CSS = {"main.css": "https://hub.3lc.ai/static/css/main.css", "plugin-common.css": "https://hub.3lc.ai/static/css/plugin-common.css"}
PORT = 8777
CHANNEL = "msedge"


def prepare() -> None:
    shutil.copyfile(FRAGMENT, HERE / "ui.html")
    for name, url in CSS.items():
        if not (HERE / name).exists():
            print("fetching", url)
            with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (a fixed https URL)
                (HERE / name).write_bytes(r.read())
    if not (HERE / "config.json").exists():
        raise SystemExit("config.json missing: run make_config.py first")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    states = [s for s in sys.argv[2].split(",") if s]
    prepare()
    from playwright.sync_api import sync_playwright

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):  # noqa: D102 (no request log in the shot output)
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Quiet, directory=str(HERE)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    index = []
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel=CHANNEL, headless=True)
            ctx = b.new_context(viewport={"width": 1100, "height": 900}, device_scale_factor=1, reduced_motion="reduce")
            for st in states:
                pg = ctx.new_page()
                errors: list[str] = []
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append("console." + m.type + ": " + m.text) if m.type in ("error", "warning") else None)
                pg.goto(f"http://127.0.0.1:{PORT}/index.html?kgdev={st}")
                pg.wait_for_function("window.HARNESS && window.HARNESS.mounted && window.HARNESS.configCalls >= 1")
                pg.wait_for_function("document.querySelector('#kg-chips') && document.querySelector('#kg-chips').children.length > 0")
                pg.wait_for_timeout(1500)
                path = out / f"import-{st}.png"
                pg.screenshot(path=str(path), full_page=True)
                info = pg.evaluate("""() => ({
                    activeTab: (document.querySelector('#kg-tabs .kg-tab.active') || {}).dataset ? document.querySelector('#kg-tabs .kg-tab.active').dataset.tab : null,
                    stepper: ['import','train','submit'].map(s => s + ':' + (document.getElementById('kg-state-' + s) || {}).className),
                    dlHidden: (document.getElementById('kg-dl-section') || {}).hidden,
                    formHidden: (document.getElementById('kg-import-form') || {}).style.display,
                    banner: (document.getElementById('kg-import-banner') || {}).innerText || '',
                    checksToggle: (document.getElementById('kg-checks-toggle') || {}).innerText || '',
                    checksOpen: (document.getElementById('kg-checks-toggle') || {}).getAttribute ? document.getElementById('kg-checks-toggle').getAttribute('aria-expanded') : null,
                    rows: Array.from(document.querySelectorAll('#kg-result .kg-table-row')).map(r => r.innerText.replace(/\\s+/g, ' ').trim()),
                    preflight: (document.getElementById('kg-preflight') || {}).innerText || '',
                    splits: (document.getElementById('kg-splits-body') || {}).innerText || '',
                    emDash: document.body.innerText.indexOf('\\u2014') >= 0,
                    otherCalls: window.HARNESS.otherCalls, configCalls: window.HARNESS.configCalls,
                    height: document.documentElement.scrollHeight })""")
                info.update({"state": st, "file": str(path), "errors": [e for e in errors if "404" not in e][:5]})
                index.append(info)
                print(json.dumps({k: info[k] for k in ("state", "formHidden", "checksToggle", "checksOpen", "emDash", "errors", "height")}), flush=True)
                pg.close()
            b.close()
    finally:
        srv.shutdown()
    (out / "shoot_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    bad = [i["state"] for i in index if i["errors"] or i["emDash"]]
    print("done:", len(index), "shots;", ("problems in " + ", ".join(bad)) if bad else "no page errors, no em dash")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
