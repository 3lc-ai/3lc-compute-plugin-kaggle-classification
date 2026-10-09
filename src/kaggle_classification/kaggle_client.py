# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""The Kaggle API surface the Predict + Submit tab uses (docs/PREDICT_MIRROR.md §7).

Credentials are read by the ``kaggle`` client from its own sources only, ``KAGGLE_API_TOKEN``,
``~/.kaggle/access_token``, the legacy ``~/.kaggle/kaggle.json`` (``KAGGLE_CONFIG_DIR`` relocates
it), or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``. ``~`` is the WORKER's home, which on a redirected-home
host is the service's, not the browsing participant's.

The guided connect flow (rc10, a deliberate divergence from ExDark, CONTEXT.md decisions 2026-10-08):
``connect(raw)`` takes the token the participant pasted into the Predict + Submit tab, trims it,
checks the ``KGAT_`` shape, writes it to ``token_path()``: the exact file kagglesdk reads
(``os.path.expanduser("~/.kaggle/access_token")``, so a redirected ``USERPROFILE`` / ``HOME`` is
honoured by construction), as plain ASCII, no BOM, no trailing newline, user-only permissions on
POSIX, then verifies it through the client's own ``authenticate()`` (one introspect call) and
reports the username. The token's value is never logged, never echoed in a response or a reason,
never stored anywhere else; a token Kaggle rejects is removed again. ``connect_help()`` is the
"Other ways to connect" block: the OS-specific one-line command with the resolved path filled in,
the environment variable and the legacy file.

kaggle 2.x authenticates inside ``authenticate()`` and, when nothing is found, prints help and
calls ``exit(1)``: a ``SystemExit`` inside the plugin worker. ``credentials_present`` runs first and
``authenticated_api`` also catches ``SystemExit``, so the worker never dies on a missing token.

Calls: ``get_competition`` (entered flag, Kaggle's daily limit, title), ``get_submission_limits``
(the used-today counter; ``competition_submissions`` is the fallback, the list call answers 403 on
an unlaunched competition, verified 2026-10-01), ``competition_submit``, ``get_submission`` (the D12
read-back by ref; the list is its fallback). The slug always comes from the manifest (D8); the daily
limit the UI shows is the manifest's (D7), Kaggle's refusal is authoritative. Import-light: the
client is imported inside functions.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
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


def token_path() -> Path:
    """The ONE file the connect flow writes: what kagglesdk reads as ``~/.kaggle/access_token``
    (``os.path.expanduser``: ``USERPROFILE`` on Windows, ``HOME`` elsewhere, so a redirected home on the
    compute service resolves to the redirected folder, never to the browsing participant's profile)."""
    return Path(os.path.expanduser("~/.kaggle/access_token"))


def credential_sources() -> dict[str, str]:
    """The paths and variables the client consults, for the help text (values are never read)."""
    return {
        "access_token": str(token_path()),
        "kaggle_json": str(config_dir() / "kaggle.json"),
        "env_token": "KAGGLE_API_TOKEN",
        "env_pair": "KAGGLE_USERNAME + KAGGLE_KEY",
    }


def credentials_present() -> bool:
    """Any of the auth sources the kaggle client reads (presence only, never the contents)."""
    path = token_path()
    return (
        bool(os.environ.get("KAGGLE_API_TOKEN"))
        or path.is_file()
        or path.with_name("access_token.txt").is_file()
        or (config_dir() / "kaggle.json").is_file()
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )


TOKEN_RE = re.compile(r"^KGAT_[A-Za-z0-9_\-]{16,}$")   # Kaggle's access tokens (Settings -> API -> Create New Token)
TOKEN_SOURCE = "kaggle.com > Settings > API > Create New Token"

# One plain sentence per failure (the card keeps the field). Never a token value.
REASONS = {
    "empty": "Paste the token first.",
    "format": f"That doesn't look like a Kaggle access token: it should start with KGAT_ ({TOKEN_SOURCE}).",
    "env_override": (
        "The compute service has KAGGLE_API_TOKEN set, which overrides a saved token; unset it on the service "
        "and connect again."
    ),
    "rejected": "Kaggle rejected the token (revoked, expired or mistyped); create a new one and try again.",
    "unreachable": (
        "Kaggle could not be reached from the compute service (no network); the token is saved and will be "
        "checked again."
    ),
    "write_failed": "The token could not be saved on the compute service: {detail}",
    "error": "Kaggle answered with an error: {detail}",
}


class TokenRejected(Exception):
    """The client found no usable credential after the write (kaggle 2.x prints help and exits)."""


class KaggleUnreachable(Exception):
    """A network-level failure while talking to Kaggle."""


def _scrub(text: Any) -> str:
    """Any token-shaped substring in an error message is masked before it can reach a response or a log."""
    return re.sub(r"KGAT_[A-Za-z0-9_\-]+", "KGAT_…", str(text))


def normalize_token(raw: Any) -> str:
    """Trim what a paste carries (whitespace, newlines, a BOM, surrounding quotes); the exact bytes written."""
    text = str(raw or "").replace("\ufeff", "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def validate_token(raw: Any) -> tuple[str, str]:
    """``(token, "")`` when the pasted text has the access-token shape, else ``("", reason)``."""
    token = normalize_token(raw)
    if not token:
        return "", REASONS["empty"]
    if not token.isascii() or not TOKEN_RE.fullmatch(token):
        return "", REASONS["format"]
    return token, ""


def save_token(token: str) -> Path:
    """Write the token byte-exact to ``token_path()``: plain ASCII, no BOM, no trailing newline; the
    directory and the file are user-only on POSIX (0700 / 0600; Windows keeps the profile's ACL)."""
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        with contextlib.suppress(OSError):
            os.chmod(path.parent, 0o700)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("ascii"))
    finally:
        os.close(fd)
    if os.name != "nt":
        os.chmod(path, 0o600)
    return path


def _authenticate_raw() -> Any:
    """The client's own authentication (the one introspect call): the api, or ``SystemExit`` when no
    source yields a usable credential, or the network error the client raised. Split out so tests can
    stand in for it."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _is_network_error(exc: BaseException) -> bool:
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "SSLError", "ProxyError", "gaierror"}:
        return True
    low = str(exc).lower()
    return any(k in low for k in ("max retries", "name resolution", "getaddrinfo", "network is unreachable",
                                  "connection refused", "timed out", "temporary failure"))


def verify_connection() -> str:
    """Authenticate through the client's own sources and return the username; ``TokenRejected`` when the
    client finds nothing usable, ``KaggleUnreachable`` on a network-level failure."""
    try:
        api = _authenticate_raw()
    except SystemExit as exc:
        raise TokenRejected() from exc
    except Exception as exc:
        if _is_network_error(exc):
            raise KaggleUnreachable(_scrub(exc)) from exc
        raise
    return api_username(api)


def connect(raw: Any) -> dict[str, Any]:
    """The guided connect flow. ``{ok: True, username, path}`` or ``{ok: False, kind, reason}``; the
    response never carries the token, and nothing here logs it."""
    token, reason = validate_token(raw)
    if not token:
        return {"ok": False, "kind": "empty" if reason == REASONS["empty"] else "format", "reason": reason}
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"ok": False, "kind": "env_override", "reason": REASONS["env_override"]}
    had_file = token_path().is_file()
    try:
        path = save_token(token)
    except OSError as exc:
        return {"ok": False, "kind": "write_failed", "reason": REASONS["write_failed"].format(detail=_scrub(exc))}
    try:
        username = verify_connection()
    except TokenRejected:
        # A rejected token must not linger as a "credential": the card would stay not connected with
        # a dead file in the way. (A previous token file, if any, was already replaced by the write.)
        with contextlib.suppress(OSError):
            path.unlink()
        return {"ok": False, "kind": "rejected", "reason": REASONS["rejected"], "replaced_previous": had_file}
    except KaggleUnreachable:
        return {"ok": False, "kind": "unreachable", "reason": REASONS["unreachable"], "path": str(path)}
    except Exception as exc:
        detail = _scrub(f"{type(exc).__name__}: {exc}")
        return {"ok": False, "kind": "error", "reason": REASONS["error"].format(detail=detail)}
    return {"ok": True, "username": username, "path": str(path)}


def connect_help(platform: str | None = None) -> dict[str, str]:
    """"Other ways to connect", with the resolved path filled in: the one-line terminal command for the
    COMPUTE SERVICE's OS (the file lives on that machine), the environment variable, the legacy file."""
    plat = platform or sys.platform
    path = token_path()
    if plat == "win32":
        command = (
            f'New-Item -ItemType Directory -Force "{path.parent}" | Out-Null; '
            f'Set-Content -LiteralPath "{path}" -Value "KGAT_<your token>" -NoNewline -Encoding ascii'
        )
        shell = "PowerShell"
    else:
        command = f"mkdir -p \"{path.parent}\" && printf '%s' 'KGAT_<your token>' > \"{path}\" && chmod 600 \"{path}\""
        shell = "Terminal"
    return {
        "platform": plat,
        "shell": shell,
        "path": str(path),
        "command": command,
        "env_var": "KAGGLE_API_TOKEN=KGAT_<your token> in the compute service's environment, then restart it",
        "legacy": f'{config_dir() / "kaggle.json"} with {{"username": "<you>", "key": "<legacy API key>"}}',
        "token_source": TOKEN_SOURCE,
    }


def credentials_help() -> str:
    """Participant-facing no-credentials sentence (the connection card renders the connect form)."""
    return f"Kaggle isn't connected yet. Paste an access token ({TOKEN_SOURCE}) and press Connect."


def authenticated_api() -> tuple[Any, str]:
    """``(api, "")`` on success, ``(None, reason)`` otherwise. The client is imported here, and the
    ``exit(1)`` the client calls without credentials is caught (``SystemExit``) as a plain failure."""
    if not credentials_present():
        return None, credentials_help()
    try:
        return _authenticate_raw(), ""
    except SystemExit:
        return None, credentials_help()
    except Exception as exc:
        return None, f"Kaggle authentication failed: {_scrub(exc)}"


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


def submission_limits(api: Any, slug: str) -> dict[str, Any]:
    """GetSubmissionLimits: ``num_today`` / ``num_allowed_now`` / ``num_total``: answers on an
    unlaunched competition where ListSubmissions does not (verified 2026-10-01)."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionLimitsRequest

    with api.build_kaggle_client() as client:
        req = ApiGetSubmissionLimitsRequest()
        req.competition_name = str(slug).strip()
        lim = client.competitions.competition_api_client.get_submission_limits(req)
    return {
        "num_today": int(getattr(lim, "num_today", 0) or 0),
        "num_allowed_now": int(getattr(lim, "num_allowed_now", 0) or 0),
        "num_total": int(getattr(lim, "num_total", 0) or 0),
        "limited_by_total": bool(getattr(lim, "limited_by_total", False)),
    }


def get_submission(api: Any, ref: str) -> dict[str, Any]:
    """GetSubmission by ref: the status, scores and error description of ONE submission."""
    from kagglesdk.competitions.types.competition_api_service import ApiGetSubmissionRequest

    with api.build_kaggle_client() as client:
        req = ApiGetSubmissionRequest()
        req.ref = int(ref)
        return _submission_dict(client.competitions.competition_api_client.get_submission(req))


def submissions_used_today(api: Any, slug: str) -> int | None:
    """The proactive counter: the limits call first, the list (UTC-dated) as the fallback. Returns
    None when neither answers; the card then shows "N submissions/day"."""
    try:
        return submission_limits(api, slug)["num_today"]
    except Exception:
        pass
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
        "token_path": str(token_path()),
    }
    api, reason = authenticated_api()
    if api is None:
        return {**out, "state": "no_credentials", "help": reason, "connect": connect_help()}
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
    """D12: Kaggle's own verdict on the submission, ``status`` (PENDING / COMPLETE / ERROR), the
    public score and the error description, read by ref (``get_submission``; the submissions list is
    the fallback), waiting a bounded time for the scoring to finish. Never raises: when neither call
    answers the result is ``{status: "unknown"}``."""
    last: dict[str, Any] = {"status": "unknown", "ref": str(ref)}
    for wait in schedule:
        sleep(wait)
        try:
            match: dict[str, Any] | None = get_submission(api, ref)
        except Exception as exc:
            try:
                match = next((s for s in list_submissions(api, slug) if s["ref"] == str(ref)), None)
            except Exception as exc2:
                last = {"status": "unknown", "ref": str(ref),
                        "error": f"{type(exc).__name__}: {exc}; list: {type(exc2).__name__}: {exc2}"}
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
    # not a verdict, fall through and let Kaggle answer.
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
