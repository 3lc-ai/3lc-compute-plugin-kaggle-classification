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

# Gate G1 (2026-09-29): RTX 3070 Ti Laptop GPU (8 GB), CPython 3.12.13, torch 2.14 cu126, timm 1.0.29,
# 609 usable rows at 150 px, batch 16, workers 0: 4.1 s/epoch (0.00673 s/row), the final per-sample
# pass 75.2 s over 7,800 rows (0.00964 s/row), 148 s end to end for 10 epochs. The collection figure is
# a FIRST-run figure: it includes UMAP's numba compile (a second run's pass took 22 s); the history
# median replaces it once a run has finished. CPU (gate G1-cpu, the same laptop, forced ``cpu``):
# 25.8 s/epoch (0.04231 s/row), the pass 81.8 s (0.01049 s/row). No Apple silicon was measured; ``mps``
# carries the CPU figures until it is.
BENCHMARK: dict[str, dict[str, float]] = {
    "cuda": {"epoch_s_per_row": 0.00673, "collect_s_per_row": 0.00964},
    "mps": {"epoch_s_per_row": 0.04231, "collect_s_per_row": 0.01049},
    "cpu": {"epoch_s_per_row": 0.04231, "collect_s_per_row": 0.01049},
}

REFERENCE: dict[str, Any] = {
    "machine": "NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB)",
    "date": "2026-09-29",
    # The default run's val accuracy at its best epoch, for the Epochs help (gate G1: 10 epochs,
    # 609 usable rows of the manual-test revision).
    "val_accuracy": 57.58,
    "best_epoch": 9,
    "epochs": 10,
}


def facts() -> dict[str, Any]:
    """What ``GET /config`` serves under ``_meta.training.benchmark``."""
    return {"per_device": {k: dict(v) for k, v in BENCHMARK.items()}, "reference": dict(REFERENCE)}
