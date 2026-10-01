# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""The Kaggle API surface the Predict + Submit tab uses (docs/PREDICT_MIRROR.md §7).

Credentials are read by the ``kaggle`` client from its own sources only — ``KAGGLE_API_TOKEN``,
``~/.kaggle/access_token``, the legacy ``~/.kaggle/kaggle.json`` (``KAGGLE_CONFIG_DIR`` relocates
it), or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``. The plugin never reads a token's value, never stores
it and never forwards it; ``~`` is the WORKER's home, which on a redirected-home host is the
service's, not the browsing participant's (the connection card's help says so).

kaggle 2.x authenticates inside ``authenticate()`` and, when nothing is found, prints help and
calls ``exit(1)`` — a ``SystemExit`` inside the plugin worker. ``credentials_present`` runs first and
``authenticated_api`` also catches ``SystemExit``, so the worker never dies on a missing token.

Calls: ``get_competition`` (entered flag, Kaggle's daily limit, title), ``competition_submissions``
(the used-today counter and the D12 read-back), ``competition_submit``. The slug always comes from
the manifest (D8); the daily limit the UI shows is the manifest's (D7), Kaggle's refusal is
authoritative. Import-light: the client is imported inside functions.
"""

from __future__ import annotations

import os
import time
from datetime import UTC
from pathlib import Path
from typing import Any

# D12: how long the one status read-back waits for Kaggle to score the submission.
READ_BACK_SCHEDULE_S = (3.0, 4.0, 5.0, 6.0, 7.0)   # ≈ 25 s in total, stops at the first non-pending answer


def competition_url(slug: str) -> str:
    return f"https://www.kaggle.com/competitions/{str(slug).strip()}"


def config_dir() -> Path:
    """Where the client looks for the legacy ``kaggle.json`` (``KAGGLE_CONFIG_DIR`` or ``~/.kaggle``)."""
    override = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    return Path(override) if override else Path(os.path.expanduser("~")) / ".kaggle"


def credential_sources() -> dict[str, str]:
    """The paths and variables the client consults, for the help text (values are never read)."""
    home = Path(os.path.expanduser("~"))
    return {
        "access_token": str(home / ".kaggle" / "access_token"),
        "kaggle_json": str(config_dir() / "kaggle.json"),
        "env_token": "KAGGLE_API_TOKEN",
        "env_pair": "KAGGLE_USERNAME + KAGGLE_KEY",
    }


def credentials_present() -> bool:
    """Any of the auth sources the kaggle client reads (presence only, never the contents)."""
    home = Path(os.path.expanduser("~"))
    return (
        bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (home / ".kaggle" / "access_token").is_file()
        or (home / ".kaggle" / "access_token.txt").is_file()
        or (config_dir() / "kaggle.json").is_file()
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )


def _token_setup_commands() -> str:
    """Dual-platform token-save commands, compute-host platform first (ExDark's, verbatim). The
    file lives on the COMPUTE host; byte-exactness matters: plain text, no BOM, no trailing newline."""
    import sys

    win = (
        'Windows (PowerShell): mkdir "$env:USERPROFILE\\.kaggle" -Force; '
        'Set-Content -Path "$env:USERPROFILE\\.kaggle\\access_token" '
        '-Value "KGAT_<your token>" -NoNewline -Encoding ascii.'
    )
    posix = (
        "macOS / Linux: mkdir -p ~/.kaggle && "
        "printf '%s' \"KGAT_<your token>\" > ~/.kaggle/access_token "
        "(printf, not echo: echo would append a newline)."
    )
    first, second = (win, posix) if sys.platform == "win32" else (posix, win)
    return (
        "Create an API token on kaggle.com (Settings -> API -> Create New "
        "Token; new tokens look like KGAT_...) and save it to "
        "~/.kaggle/access_token on the machine running the compute service, "
        f"byte-exact: plain text, no BOM, no trailing newline. {first} "
        f"{second} (Or set the KAGGLE_API_TOKEN environment variable; legacy "
        "~/.kaggle/kaggle.json also works.)"
    )


def credentials_help() -> str:
    """Participant-facing no-credentials message (the connection card)."""
    return (
        "Kaggle credentials not found. " + _token_setup_commands() +
        " The generated submission.csv is saved locally, so you can always "
        "upload it manually on the competition's Submit page."
    )


def authenticated_api() -> tuple[Any, str]:
    """``(api, "")`` on success, ``(None, reason)`` otherwise. The client is imported here, and the
    ``exit(1)`` the client calls without credentials is caught (``SystemExit``) as a plain failure."""
    if not credentials_present():
        return None, credentials_help()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api, ""
    except SystemExit:
        return None, credentials_help()
    except Exception as exc:
        return None, f"Kaggle authentication failed: {exc}"


def api_username(api: Any) -> str:
    try:
        return str(api.config_values.get("username", "") or "")
    except Exception:
        return ""


def competition_info(api: Any, slug: str) -> dict[str, Any]:
    """One GetCompetition call: joined state, Kaggle's daily limit, the title."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

    with api.build_kaggle_client() as client:
        request = ApiGetCompetitionRequest()
        request.competition_name = str(slug).strip()
        comp = client.competitions.competition_api_client.get_competition(request)
    return {
        "user_has_entered": bool(comp.user_has_entered),
        "max_daily_submissions": int(comp.max_daily_submissions or 0),
        "title": str(comp.title or ""),
    }


def _submission_dict(s: Any) -> dict[str, Any]:
    status = getattr(s, "status", None)
    status_name = getattr(status, "name", None) or (str(status) if status is not None else "")
    date = getattr(s, "date", None)
    return {
        "ref": str(getattr(s, "ref", "") or ""),
        "date": date.isoformat() if hasattr(date, "isoformat") else (str(date) if date else ""),
        "description": str(getattr(s, "description", "") or ""),
        "status": str(status_name or "").upper(),
        "public_score": _score(getattr(s, "public_score", None)),
        "private_score": _score(getattr(s, "private_score", None)),
        "error_description": str(getattr(s, "error_description", "") or ""),
        "file_name": str(getattr(s, "file_name", "") or ""),
    }


def _score(raw: Any) -> float | None:
    try:
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def list_submissions(api: Any, slug: str, page_size: int = 20) -> list[dict[str, Any]]:
    subs = api.competition_submissions(str(slug).strip(), page_size=page_size) or []
    return [_submission_dict(s) for s in subs if s is not None]


def submissions_used_today(api: Any, slug: str) -> int | None:
    """The proactive counter. Returns None on any error (the list call is known to 403 on some
    private competitions); the card then shows "N submissions/day"."""
    try:
        from datetime import datetime

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return sum(1 for s in list_submissions(api, slug, page_size=100) if s["date"].startswith(today))
    except Exception:
        return None


def connection(slug: str, daily_limit: int, log: Any = None) -> dict[str, Any]:
    """The three-state connection card: ``no_credentials`` → ``not_joined`` → ``ready``. The daily
    limit shown is the manifest's (D7); Kaggle's own value is reported beside it when it answers."""
    out: dict[str, Any] = {
        "slug": slug, "competition_url": competition_url(slug), "daily_limit": int(daily_limit),
        "daily_limit_source": "manifest", "credential_sources": credential_sources(),
    }
    api, reason = authenticated_api()
    if api is None:
        return {**out, "state": "no_credentials", "help": reason}
    out["username"] = api_username(api)
    try:
        info = competition_info(api, slug)
    except Exception as exc:
        return {**out, "state": "ready", "probe_error": str(exc)}
    out["competition_title"] = info["title"]
    if info["max_daily_submissions"]:
        out["kaggle_daily_limit"] = info["max_daily_submissions"]
        if info["max_daily_submissions"] != int(daily_limit) and log:
            log(f"Kaggle reports {info['max_daily_submissions']} submissions/day; the manifest says {daily_limit}.")
    if not info["user_has_entered"]:
        return {**out, "state": "not_joined"}
    used = submissions_used_today(api, slug)
    if used is not None:
        out["submissions_used_today"] = used
    return {**out, "state": "ready"}


def classify_error(exc: BaseException) -> str:
    """``daily_limit`` | ``not_joined`` | ``error`` from a submit-call exception (substring matching on
    Kaggle's own message, which kagglesdk surfaces verbatim)."""
    low = str(exc).lower()
    if ("daily" in low and ("limit" in low or "submission" in low)) or "submission limit" in low or (
        "maximum" in low and ("per day" in low or "today" in low)
    ):
        return "daily_limit"
    if ("rules" in low and "accept" in low) or "must accept" in low or "not accepted" in low:
        return "not_joined"
    return "error"


def read_back(api: Any, slug: str, ref: str, *, schedule: tuple[float, ...] = READ_BACK_SCHEDULE_S,
              sleep: Any = time.sleep) -> dict[str, Any]:
    """D12: Kaggle's own verdict on the submission — ``status`` (PENDING / COMPLETE / ERROR), the
    public score and the error description — read from the submissions list, waiting a bounded time
    for the scoring to finish. Never raises: an unreachable list answers ``{status: "unknown"}``."""
    last: dict[str, Any] = {"status": "unknown", "ref": str(ref)}
    for wait in schedule:
        sleep(wait)
        try:
            match = next((s for s in list_submissions(api, slug) if s["ref"] == str(ref)), None)
        except Exception as exc:
            last = {"status": "unknown", "ref": str(ref), "error": f"{type(exc).__name__}: {exc}"}
            continue
        if match is None:
            last = {"status": "unknown", "ref": str(ref), "error": "not listed yet"}
            continue
        last = match
        if match["status"] != "PENDING":
            break
    return last


def submit(csv_path: str, message: str, slug: str, daily_limit: int, log: Any,
           *, read_back_after: bool = True) -> dict[str, Any]:
    """Upload through the kaggle client. Non-fatal outcomes are friendly states, never failures:
    ``skipped`` (no credentials), ``not_joined``, ``limit_reached`` leave the validated CSV on disk."""
    api, reason = authenticated_api()
    if api is None:
        return {"status": "skipped", "reason": reason}
    # Cheap pre-probe: a submit against a not-joined competition cannot succeed. Probe failure is
    # not a verdict — fall through and let Kaggle answer.
    kaggle_limit: int | None = None
    try:
        info = competition_info(api, slug)
        kaggle_limit = info["max_daily_submissions"] or None
        if not info["user_has_entered"]:
            return {
                "status": "not_joined",
                "reason": (
                    "Join the competition on Kaggle first (accept the rules on the "
                    f"competition page), then submit again: {competition_url(slug)}"
                ),
            }
    except Exception as exc:
        log(f"Competition probe failed ({type(exc).__name__}: {exc}); submitting anyway.")
    try:
        response = api.competition_submit(file_name=csv_path, message=message, competition=slug)
        ref = getattr(response, "ref", None)
        text = str(getattr(response, "message", "") or "") or str(response)
        if not ref:
            # The client reports an upload failure as a response without a ref (see
            # COMPETITION_SUBMIT_UPLOAD_FAILED_MESSAGE); treat it as Kaggle's rejection.
            return {"status": "failed", "reason": f"Kaggle rejected the submission: {text}", "response": text}
        log(f"Kaggle accepted the submission: {ref}")
        out: dict[str, Any] = {"status": "submitted", "response": text, "ref": str(ref)}
        if read_back_after:
            log("Reading the submission status back from Kaggle…")
            out["kaggle"] = read_back(api, slug, str(ref))
            k = out["kaggle"]
            if k.get("status") == "COMPLETE":
                log(f"Kaggle scored it: public score {k.get('public_score')}")
            elif k.get("status") == "ERROR":
                log(f"Kaggle rejected it: {k.get('error_description') or 'no description'}")
            else:
                log(f"Kaggle status: {k.get('status')} ({k.get('error') or 'still scoring'})")
        return out
    except Exception as exc:
        kind = classify_error(exc)
        limit = kaggle_limit or int(daily_limit)
        if kind == "daily_limit":
            return {
                "status": "limit_reached",
                "reason": (
                    f"Daily submission limit reached ({limit}/day), resets midnight UTC. "
                    "Your CSV is saved and validated. Submit it tomorrow from here, "
                    "or upload it manually on the competition's Submit page."
                ),
                "detail": str(exc),
            }
        if kind == "not_joined":
            return {
                "status": "not_joined",
                "reason": (
                    "Kaggle rejected the submission because the competition rules are "
                    f"not accepted yet. Join here, then submit again: {competition_url(slug)}"
                ),
                "detail": str(exc),
            }
        return {"status": "failed", "reason": f"Kaggle rejected the submission: {exc}", "detail": str(exc)}
