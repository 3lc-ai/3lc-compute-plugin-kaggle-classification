# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ledger and verification bundle (session 5) — STUB.

Contract (docs/PLAN.md, "Ledger"):

* An append-only JSON-lines ledger under the plugin home records every import, train, predict
  and submit with its inputs (table URLs and revisions, params, run URL, checkpoint sha256,
  submission file sha256, Kaggle submission ref) and the manifest version in force.
* ``build_verification_bundle(run_url)`` zips the ledger slice for one run plus the run's
  recorded contract and the submission CSV, so an organizer can verify a leaderboard entry
  was produced through the plugin. Teams are per machine; no table sharing.
"""

from __future__ import annotations

from typing import Any


def append(entry: dict[str, Any]) -> None:
    """Not implemented in session 1."""
    msg = "The ledger is scheduled for session 5 (docs/PLAN.md)."
    raise NotImplementedError(msg)
