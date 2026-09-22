# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes, as relative Litestar handlers under ``/api/plugins/kaggle-classification/``.

* ``GET /config`` — the populated session plus the ``_meta`` block the fragment renders and
  never defines: version, repository, the manifest as resolved WITHOUT the network (cache or
  bundled, so the page renders at once), its provenance and warnings, the picker candidates,
  the plugin home, the kit destination and revisit state, and the background-refresh status.
  The first call in a while kicks off a remote refresh on a thread; the fragment polls
  ``GET /manifest`` until it settles and re-renders. The browser never fetches the CDN.
* ``GET /manifest`` — the same resolution payload alone, for that poll.
* ``POST /manifest/select`` — pick one of several active competitions.
* ``POST /config`` — merge per-tab snapshots; retired keys answer 400.

Handlers are ``def`` with ``sync_to_thread=True`` (Litestar runs them in a threadpool)
because they touch the disk store. Built fresh per call, for per-app registration.

Litestar is imported at MODULE level on purpose, like the timm and ExDark plugins' routes
modules: with ``from __future__ import annotations`` the handlers' return annotations are
strings that Litestar resolves against this module's globals when the worker mounts them,
and a ``Response`` imported inside ``get_route_handlers`` is not in those globals — the
worker then dies at startup with ``NameError: name 'Response' is not defined`` (found live on
compute 1.1.0). This module is imported lazily by ``get_route_handlers`` in ``__init__``, so
the package import stays light.
"""

from __future__ import annotations

from typing import Any

from litestar import Response, get, post
from litestar.status_codes import HTTP_400_BAD_REQUEST


def manifest_payload(*, kick_refresh: bool = True) -> dict[str, Any]:
    """The manifest as the fragment should render it right now (no network on this path)."""
    from kaggle_classification import manifest

    resolution = manifest.resolve(network=False)
    refresh = manifest.refresh_in_background() if kick_refresh else manifest.refresh_status()
    return {**resolution.to_dict(), "refresh": refresh}


def config_payload() -> dict[str, Any]:
    """What ``GET /config`` returns. Pure function so tests call it without Litestar."""
    import kaggle_classification
    from kaggle_classification import kit, manifest, session, storage

    payload = manifest_payload()
    current = manifest.resolve(network=False).manifest
    out = session.load()
    out["session"] = session.populated_session(current)
    out["_meta"] = {
        "version": kaggle_classification.__version__,
        "repository_url": kaggle_classification.REPOSITORY_URL,
        # The competition contract, SERVED rather than restated: the fragment renders these and
        # carries no literal of its own.
        "manifest": payload["manifest"],
        "manifest_source": current.source,
        "manifest_provenance": payload["provenance"],
        "manifest_warnings": payload["warnings"],
        "manifest_candidates": payload["candidates"],
        "manifest_refresh": payload["refresh"],
        "plugin_home": storage.describe(),
        "kit_dest": str(kit.default_dest(current)),
        "kit_state": kit.download_state(current),
    }
    return out


def get_route_handlers() -> list[Any]:
    @get("/config", sync_to_thread=True)
    def get_config() -> dict[str, Any]:
        return config_payload()

    @post("/config", status_code=200, sync_to_thread=True)
    def save_config(data: dict[str, Any]) -> Response[dict[str, Any]]:
        from kaggle_classification import session

        if not isinstance(data, dict):
            return Response({}, status_code=200)
        try:
            return Response(session.save(data), status_code=200)
        except ValueError as exc:
            return Response({"error": str(exc)}, status_code=HTTP_400_BAD_REQUEST)

    @get("/manifest", sync_to_thread=True)
    def get_manifest() -> dict[str, Any]:
        return manifest_payload(kick_refresh=False)

    @post("/manifest/select", status_code=200, sync_to_thread=True)
    def select_manifest(data: dict[str, Any]) -> Response[dict[str, Any]]:
        from kaggle_classification import manifest

        try:
            manifest.select_competition(str((data or {}).get("id") or ""))
        except ValueError as exc:
            return Response({"error": str(exc)}, status_code=HTTP_400_BAD_REQUEST)
        manifest.refresh_in_background(force=True)
        return Response(manifest_payload(kick_refresh=False), status_code=200)

    return [get_config, save_config, get_manifest, select_manifest]
