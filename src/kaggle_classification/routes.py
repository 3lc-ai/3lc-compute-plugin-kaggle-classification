# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Custom routes, as relative Litestar handlers under ``/api/plugins/kaggle-classification/``.

Session 1 serves the config surface only: ``GET /config`` (the populated session plus the
``_meta`` block the fragment renders and never defines — version, repository, the manifest
and its provenance, the kit destination and revisit state) and ``POST /config``. Job
submission stays host-managed via ``/run``.

Handlers are ``def`` with ``sync_to_thread=True`` (Litestar runs them in a threadpool)
because they touch the disk store. Built fresh per call, for per-app registration.
"""

from __future__ import annotations

from typing import Any


def config_payload() -> dict[str, Any]:
    """What ``GET /config`` returns. Pure function so tests call it without Litestar."""
    import kaggle_classification
    from kaggle_classification import kit, manifest, session

    current = manifest.load_manifest()
    out = session.load()
    out["session"] = session.populated_session(current)
    out["_meta"] = {
        "version": kaggle_classification.__version__,
        "repository_url": kaggle_classification.REPOSITORY_URL,
        # The competition contract, SERVED rather than restated: the fragment renders these and
        # carries no literal of its own.
        "manifest": current.to_dict(),
        "manifest_source": current.source,
        "manifest_warnings": list(current.warnings),
        "kit_dest": str(kit.default_dest(current)),
        "kit_state": kit.download_state(current),
    }
    return out


def get_route_handlers() -> list[Any]:
    from litestar import Response, get, post
    from litestar.status_codes import HTTP_400_BAD_REQUEST

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

    return [get_config, save_config]
