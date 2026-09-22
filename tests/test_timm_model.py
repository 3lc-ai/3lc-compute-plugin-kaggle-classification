# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The locked model: ``timm.create_model(arch, pretrained=False, num_classes=N)`` builds the
manifest's architecture with NO network access and accepts the manifest's image size.

Requires the heavy extra (torch + timm) in the dev venv; skipped where they are absent so the
light CI still passes, but the Gate 1 record must show it ran."""

from __future__ import annotations

import socket
import urllib.request

import pytest

timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")


def _no_network(monkeypatch):
    def boom(*args, **kwargs):
        msg = "network access attempted while creating the model"
        raise AssertionError(msg)

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_timm_is_the_pinned_version():
    import tomllib
    from pathlib import Path

    py = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    (pin,) = [d for d in py["project"]["optional-dependencies"]["kaggle-classification"] if d.startswith("timm==")]
    assert timm.__version__ == pin.split("==", 1)[1]


def test_manifest_arch_builds_offline_and_accepts_the_manifest_image_size(manifest, monkeypatch):
    _no_network(monkeypatch)
    model = timm.create_model(
        manifest.model.arch, pretrained=manifest.model.pretrained, num_classes=manifest.num_classes
    )
    model.eval()
    size = manifest.model.image_size
    x = torch.zeros(2, 3, size, size)
    with torch.no_grad():
        logits = model(x)
        features = model.forward_features(x)
        embeddings = model.forward_head(features, pre_logits=True)
    assert logits.shape == (2, manifest.num_classes)
    assert embeddings.ndim == 2 and embeddings.shape[0] == 2 and embeddings.shape[1] > manifest.num_classes
    assert model.pretrained_cfg.get("input_size") is not None  # resolve_data_config still works from scratch
