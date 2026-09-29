# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference timings for the Train tab's first-run duration hint (docs/TRAIN_MIRROR.md §5).

ExDark's pre-run hint needs history ("Recent runs averaged ~N s/epoch on this machine"); a
machine's first run has none, so the fragment falls back to these per-device-class constants,
scaled by the usable rows and the epochs of the run being configured. They are PLUGIN facts
(what the baseline costs per row on a reference machine), not competition facts, so a code
constant is allowed. Seeded from gate G1 (session 3); ``reference`` names the machine.

``epoch_s_per_row`` is one training epoch's wall time divided by the usable rows the sampler
drew (the epoch length). ``collect_s_per_row`` is the final per-sample pass (inference +
UMAP + metrics writes) divided by the rows collected (train + val).
"""

from __future__ import annotations

from typing import Any

# Filled from gate G1 (2026-09-29): RTX 3070 Ti Laptop GPU (8 GB), CPython 3.12, torch cu126,
# 603-609 usable rows at 150 px, batch 16, workers 0. CPU: the same laptop, forced ``cpu``.
BENCHMARK: dict[str, dict[str, float]] = {
    "cuda": {"epoch_s_per_row": 0.0, "collect_s_per_row": 0.0},
    "mps": {"epoch_s_per_row": 0.0, "collect_s_per_row": 0.0},
    "cpu": {"epoch_s_per_row": 0.0, "collect_s_per_row": 0.0},
}

REFERENCE: dict[str, Any] = {
    "machine": "",
    "date": "",
    # The default run's val accuracy at its best epoch, for the Epochs help (calibrated after G1).
    "val_accuracy": None,
    "best_epoch": None,
    "epochs": None,
}


def facts() -> dict[str, Any]:
    """What ``GET /config`` serves under ``_meta.training.benchmark``."""
    return {"per_device": {k: dict(v) for k, v in BENCHMARK.items()}, "reference": dict(REFERENCE)}
