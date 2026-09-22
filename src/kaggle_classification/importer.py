# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import stage (session 2) — STUB. Registers the kit's train and val splits as 3LC tables.

Contract (docs/PLAN.md, "Import"):

* Input: ``session.kit_dir`` (a verified kit, see ``kit.py``) and the manifest.
* ``train`` table: ``labeled_per_class × N`` rows with ``label`` in ``0..N-1`` and
  ``weight = 1.0``, plus the ``undefined`` pool rows with ``label = manifest.undefined_label_id``
  (the LAST map entry) and ``weight = 0.0``. ``val``: labeled rows only. ``test`` is NEVER
  registered — the predictor reads ``data/test/`` directly.
* Table URLs are derived under the configured project root
  (``tlc.config.project_root_url``, never a tlcconfig internal) as
  ``<root>/<project>/datasets/<manifest.dataset_name(split)>/tables/<table_name>``; an identical
  existing table is REUSED, otherwise CREATED (the per-split outcome the Import tab renders).
* Row counts are checked against ``manifest.expected_rows(split)`` and the class map against
  ``manifest.class_names`` before success is reported.
* tlc 3.x API only: ``tlc.Table.from_image_folder`` or ``tlc.TableWriter`` with
  ``tlc.schemas.CategoricalLabelSchema`` / ``SampleWeightSchema`` (docs/STUDY.md G-1).
* DataLoader/scan ``num_workers=0`` everywhere data loads.
"""

from __future__ import annotations

from typing import Any

from kaggle_classification.manifest import Manifest


def run_import(params: dict[str, Any], ctx: Any, manifest: Manifest) -> dict[str, Any]:
    """Not implemented in session 1."""
    msg = "Import is scheduled for session 2 (docs/PLAN.md)."
    raise NotImplementedError(msg)
