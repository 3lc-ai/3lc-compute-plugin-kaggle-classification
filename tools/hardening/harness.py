# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The hardening campaign's harness (docs/TEST_PLAN.md): drives the plugin through the in-process compute
host of a tester environment exactly the way the Hub does (the catalog install route, the plugin's /run
route, the host's job list), under the tester layout's redirected home. Imported by the per-section
scripts in this folder; run from the tester env's venv python with the services STOPPED.

Environment (the start script's, set by `env_for(root)`): USERPROFILE / LOCALAPPDATA / APPDATA / HOME
-> <root>/home, TLC_PROJECT_ROOT_URL -> <root>/home/projects, TLC_COMPUTE_PORT, the manifest tier, and
TLC_COMPUTE_PLUGIN_CATALOG_URLS = the default catalog + a local catalog listing the build under test.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PLUGIN_ID = "kaggle-classification"
API = f"/api/plugins/{PLUGIN_ID}"
DEFAULT_CATALOG = "https://3lc-public-examples-2-2.s3.amazonaws.com/hub/catalog.json"
REPO_GIT = "https://github.com/3lc-ai/3lc-compute-plugin-kaggle-classification.git"
MANIFEST_BASE = "https://competitions.dev.3lc.ai"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Environment ───────────────────────────────────────────────────────────────


def env_for(root: Path, *, port: int = 5020, catalog: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    home = root / "home"
    env = {
        "USERPROFILE": str(home), "LOCALAPPDATA": str(home / "AppData" / "Local"), "APPDATA": str(home / "AppData" / "Roaming"),
        "HOME": str(home), "TLC_PROJECT_ROOT_URL": str(home / "projects"), "TLC_COMPUTE_PORT": str(port),
        "KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL": MANIFEST_BASE,
        "TLC_COMPUTE_PLUGIN_CATALOG_URLS": DEFAULT_CATALOG + ("," + catalog if catalog else ""),
    }
    env.update(extra or {})
    return env


def apply_env(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v


def write_catalog(root: Path, version: str, ref: str, manifest: dict[str, Any]) -> Path:
    """A one-entry catalog for the build under test (a tag or a commit), the shape of catalog-test.json."""
    source = f"3lc-compute-plugin-kaggle-classification[kaggle-classification] @ git+{REPO_GIT}@{ref}"
    cat = {
        "schema_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_note": f"hardening campaign: {version} at {ref}",
        "plugins": [{"id": PLUGIN_ID, "versions": [{"version": version, "source": source, "wheel_url": "", "manifest": manifest}]}],
    }
    path = root / f"catalog-campaign-{version}.json"
    path.write_text(json.dumps(cat, indent=1), encoding="utf-8")
    return path


def plugin_manifest(repo: Path) -> dict[str, Any]:
    import tomllib

    return tomllib.loads((repo / "src" / "kaggle_classification" / "plugin.toml").read_text(encoding="utf-8"))


# ── The host ──────────────────────────────────────────────────────────────────


def host_client(timeout: float = 1800):
    # A plugin worker outlives the in-process host that spawned it (the TestClient's shutdown does not
    # reach it), and on Windows a live worker holds the managed venv's python.exe, so the next host's
    # update or reinstall fails with "Access is denied" (seen 2026-10-12, section A). Every host session
    # starts clean; the scenarios that kill a worker on purpose do so explicitly.
    stale = kill_workers()
    if stale:
        log(f"stopped {stale} plugin worker(s) left by a previous host")
    import tlc_compute.plugins.live_auth as live_auth

    live_auth.is_live_auth_exempt = lambda scope: True  # type: ignore[assignment]
    from litestar.testing import TestClient

    from tlc_compute.app import create_app, load_config

    asgi = create_app(load_config())
    app = getattr(asgi, "other_asgi_app", asgi)
    return TestClient(app=app, timeout=timeout)


def card_of(client) -> dict[str, Any] | None:
    cards = client.get("/api/plugins").json()
    return next((c for c in cards if c.get("id") == PLUGIN_ID), None)


def wait_ready(client, timeout: float = 1800) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/admin/plugins/install/{PLUGIN_ID}/status").json()
        card = card_of(client)
        ready = bool(card) and (card.get("provisioning") or {}).get("state") == "ready" and not card.get("placeholder")
        kick = client.get(f"{API}/config")
        if st.get("state") != "installing" and ready and kick.status_code == 200 and kick.json().get("_meta"):
            return round(time.time() - t0, 1)
        time.sleep(3)
    msg = "the plugin never became ready"
    raise TimeoutError(msg)


def install(client, version: str, timeout: float = 1800) -> dict[str, Any]:
    """Install (or update to) `version` from the catalogs the host was given; returns the install facts."""
    t0 = time.time()
    source = None
    while time.time() - t0 < 600 and not source:
        r = client.get("/api/admin/plugins/catalog")
        if r.status_code == 200:
            cards = r.json()
            cards = cards.get("plugins") if isinstance(cards, dict) else cards
            card = next((c for c in (cards or []) if c.get("id") == PLUGIN_ID), None)
            if card:
                source = next((v.get("source") for v in (card.get("versions") or []) if v.get("version") == version), None)
                # The 1.1.0 host's card carries the newest entry's source at the top level and an empty
                # `versions` list for a one-entry catalog (seen 2026-10-12).
                if not source and card.get("version") == version and card.get("source"):
                    source = card["source"]
        if not source:
            time.sleep(5)
    if not source:
        msg = f"the catalogs never listed {PLUGIN_ID} {version}"
        raise RuntimeError(msg)
    r = client.post("/api/admin/plugins/install", json={"source": source, "id": PLUGIN_ID})
    log(f"POST install {version} -> {r.status_code} {r.text[:160]}")
    st: dict[str, Any] = {}
    while time.time() - t0 < timeout:
        st = client.get(f"/api/admin/plugins/install/{PLUGIN_ID}/status").json()
        if st.get("state") in ("succeeded", "failed", "error"):
            break
        time.sleep(5)
    facts = {"state": st.get("state"), "error": st.get("error"), "install_s": round(time.time() - t0, 1), "source": source}
    if st.get("state") != "succeeded":
        return facts
    facts["ready_s"] = wait_ready(client, timeout)
    facts["version"] = (client.get(f"{API}/config").json().get("_meta") or {}).get("version")
    return facts


def doctor(client, *, kaggle: bool = False, timeout: float = 120) -> dict[str, Any]:
    """The Doctor once its device probe has finished (a fresh worker starts the probe on the first call)."""
    t0 = time.time()
    doc = client.get(f"{API}/status/doctor?kaggle={1 if kaggle else 0}").json()
    while (doc.get("device") or {}).get("probe") != "done" and time.time() - t0 < timeout:
        time.sleep(2)
        doc = client.get(f"{API}/status/doctor?kaggle={1 if kaggle else 0}").json()
    return doc


def uninstall(client) -> int:
    """The host's uninstall: ``DELETE /api/admin/plugins/install/{id}`` (the install path, reused)."""
    return client.delete(f"/api/admin/plugins/install/{PLUGIN_ID}").status_code


# ── Jobs ──────────────────────────────────────────────────────────────────────


def jobs(client) -> list[dict[str, Any]]:
    return client.get("/api/plugins/jobs").json()


def job_of(client, job_id: str) -> dict[str, Any] | None:
    return next((j for j in jobs(client) if j.get("id") == job_id), None)


def start(client, kind: str, params: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    r = client.post(f"{API}/run", json={"kind": kind, **(params or {})})
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text[:300]}
    log(f"POST /run {kind} {json.dumps(params or {})[:100]} -> {r.status_code} {json.dumps(body)[:160]}")
    return r.status_code, body


def cancel(client, job_id: str) -> int:
    return client.post(f"/api/plugins/jobs/{job_id}/cancel").status_code


def wait_terminal(client, job_id: str, *, timeout: float = 3600, until=None, quiet: bool = False) -> dict[str, Any]:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        job = job_of(client, job_id)
        line = ((job or {}).get("status"), ((job or {}).get("progress") or {}).get("label"))
        if line != last and not quiet:
            log(f"  {time.time() - t0:7.1f}s {line[0]} · {line[1]}")
            last = line
        if until and job and until(job):
            job["elapsed_s"] = round(time.time() - t0, 1)
            return job
        if job and job.get("status") in ("completed", "failed", "cancelled"):
            job["elapsed_s"] = round(time.time() - t0, 1)
            return job
        time.sleep(2)
    return {"status": "timeout", "elapsed_s": round(time.time() - t0, 1)}


def run_and_wait(client, kind: str, params: dict[str, Any] | None = None, **kw) -> dict[str, Any]:
    code, body = start(client, kind, params)
    if not (200 <= code < 300) or not (body.get("job_id") or body.get("id")):   # the plugin's /run answers 201
        return {"status": "refused", "http": code, "body": body}
    jid = body.get("job_id") or body.get("id")
    job = wait_terminal(client, jid, **kw)
    job["id"] = jid
    return job


def kill_workers(scope: str | None = None) -> int:
    """Stop the plugin worker processes of ONE tester environment (the compute-service-killed scenarios, the
    orphan sweep): a worker's command line names its state root under that environment's home, so a
    worker of another environment (the fresh env's running services) is left alone. scope defaults to
    the home this process was pointed at (USERPROFILE, set by apply_env)."""
    scope = (scope or os.environ.get("USERPROFILE") or "").replace("'", "''")
    if not scope:
        return 0
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*tlc_plugin_sdk.worker*' -and "
          f"$_.CommandLine.IndexOf('{scope}', [StringComparison]::OrdinalIgnoreCase) -ge 0 }} | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, capture_output=True, text=True)
    return len([line for line in out.stdout.splitlines() if line.strip().isdigit()])


# ── Records ───────────────────────────────────────────────────────────────────


class Results:
    """The campaign's findings file: one JSON per section, appended item by item."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": {}}

    def put(self, item: str, status: str, **facts: Any) -> None:
        self.data["items"][item] = {"status": status, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **facts}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        log(f"{item}: {status} {json.dumps(facts, default=str)[:300]}")


def plugin_state_dir(root: Path) -> Path:
    return root / "home" / ".3lc-compute" / "plugin-state" / PLUGIN_ID


def setup_stdio() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
