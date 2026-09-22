# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Predict + Submit stage (sessions 4 and 5) — STUB.

Contract (docs/PLAN.md, "Predict + Submit"):

* Predictions run ONLY from runs this plugin created: the run must carry the recorded contract
  (``trainer``) and its checkpoint must sit under the run folder; a hand-typed weights path is
  refused. Test images are read from ``<kit_dir>/data/test/`` — the test split is never a table.
* Output: ``submission.csv`` with exactly ``manifest.submission.columns`` in the id order of
  ``manifest.splits.test.ids_from``; ``prediction`` is the class id, ``confidence`` the max
  softmax. Row count must equal ``manifest.splits.test.count``.
* Submit uses the Kaggle API client against ``manifest.competition.slug``; the daily budget is
  ``manifest.submission.daily_limit``. Every prediction and submission lands in the ledger.
* Device resolution is ``trainer.resolve_device``.
"""

from __future__ import annotations

from typing import Any

from kaggle_classification.manifest import Manifest


def run_predict(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """Not implemented in session 1."""
    msg = "Predict is scheduled for session 4 (docs/PLAN.md)."
    raise NotImplementedError(msg)


def run_submit(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """Not implemented in session 1."""
    msg = "Submit is scheduled for session 5 (docs/PLAN.md)."
    raise NotImplementedError(msg)
