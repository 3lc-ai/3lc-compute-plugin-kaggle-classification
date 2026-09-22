# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
# Adapted from 3lc-compute-plugin-kaggle (Copyright 3LC AI), relicensed by the copyright holder
# under Apache-2.0 for this project.
"""Train stage (session 3) — STUB, plus ``resolve_device`` which the predictor shares.

Contract (docs/PLAN.md, "Train" and "The labeling-loop contract"):

* Model: ``timm.create_model(manifest.model.arch, pretrained=False, num_classes=manifest.num_classes)``
  with ``arch`` allowlisted to the single manifest value — any other value is rejected
  server-side with a participant-facing message; ``pretrained`` is never a parameter.
* Data: ``tlc.Table.from_url`` (optionally ``.latest()``), ``table.with_transform`` views,
  ``DataLoader(num_workers=0)``. Undefined rows (``label == manifest.undefined_label_id``) are
  filtered out of training REGARDLESS of weight; no labeling cap.
* Params come from ``manifest.training.defaults`` merged with the participant's form, bounded
  by ``manifest.training.bounds`` on the merged kwargs; presets are named partial overrides.
* Per-sample metrics after training (on the best epoch, val transform, both splits):
  ``predicted`` (categorical, mapped to class names), ``confidence`` (max softmax), per-class
  probabilities, ``loss`` (masked out for undefined rows, never fabricated), and
  ``embeddings`` (``n_components``-D). UMAP is fit on the TRAIN embeddings (labeled + undefined)
  and val is transformed into that space; PCA is the fallback. Written with ``run.add_metrics``.
* The run records the contract (arch, image_size, pretrained=false, timm version, seed) so a
  prediction can prove it came from a plugin-trained run.
"""

from __future__ import annotations

from typing import Any

from kaggle_classification.manifest import Manifest


def resolve_device(raw: Any) -> str:
    """Resolve the participant's Device field to a ``torch.device`` string.

    Blank = auto: CUDA -> MPS -> CPU. A bare index (``"0"``) means that CUDA
    device; any other non-blank string passes through (``"cpu"``, ``"cuda:1"``).
    Must NOT be called from the host request path (validation stays torch-free).
    """
    s = str(raw).strip() if raw is not None else ""
    if s.isdigit():
        return f"cuda:{int(s)}"
    if s:
        return s
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_training(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """Not implemented in session 1."""
    msg = "Training is scheduled for session 3 (docs/PLAN.md)."
    raise NotImplementedError(msg)
