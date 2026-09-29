# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The locked model is the Intel kit's, exactly: torchvision ``resnet18(weights=None)`` with ``fc``
replaced by Identity and the kit's MLP head (512 → 256 → ReLU → Dropout 0.3 → 128 → ReLU →
Dropout 0.3 → N). Built with NO network access; the seeded init is deterministic; the manifest's
allowlist is enforced.

Requires the heavy extra (torch + torchvision); skipped where absent, so a green run in a light venv
is not a green run (CLAUDE.md §B)."""

from __future__ import annotations

import socket
import urllib.request

import pytest

from kaggle_classification import manifest as manifest_mod
from kaggle_classification import trainer

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _no_network(monkeypatch):
    def boom(*args, **kwargs):
        msg = "network access attempted while creating the model"
        raise AssertionError(msg)

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def test_the_manifest_names_the_kits_model(manifest):
    assert manifest.model.backbone == "torchvision_resnet18" and manifest.model.head == "kit_mlp_512_256_128_d03"
    assert manifest.model.arch == "resnet18" and manifest.model.pretrained is False
    assert manifest_mod.BACKBONES == ("torchvision_resnet18",) and "linear" in manifest_mod.HEADS


def test_the_kit_head_is_replicated_exactly(manifest, monkeypatch):
    from torch import nn

    _no_network(monkeypatch)
    model = trainer.build_model(manifest.model.backbone, manifest.model.head, manifest.num_classes)
    assert isinstance(model.resnet.fc, nn.Identity)
    layers = list(model.classifier)
    kinds = [type(x).__name__ for x in layers]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"]
    assert (layers[0].in_features, layers[0].out_features) == (512, 256)
    assert (layers[3].in_features, layers[3].out_features) == (256, 128)
    assert (layers[6].in_features, layers[6].out_features) == (128, manifest.num_classes)
    assert layers[2].p == 0.3 and layers[5].p == 0.3
    model.eval()
    size = manifest.model.image_size
    x = torch.zeros(2, 3, size, size)
    with torch.no_grad():
        assert model(x).shape == (2, manifest.num_classes)
        assert model.features(x).shape == (2, 512)


def test_the_seeded_init_is_deterministic(manifest):
    trainer.set_seed(42)
    a = trainer.build_model(manifest.model.backbone, manifest.model.head, manifest.num_classes)
    trainer.set_seed(42)
    b = trainer.build_model(manifest.model.backbone, manifest.model.head, manifest.num_classes)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb), ka
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False


def test_the_linear_head_and_the_allowlist(manifest):
    model = trainer.build_model("torchvision_resnet18", "linear", manifest.num_classes)
    model.eval()
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        assert model(x).shape == (1, manifest.num_classes) and model.features(x).shape == (1, 512)
    with pytest.raises(trainer.TrainRefused):
        trainer.build_model("timm_resnet18", "linear", 6)
    with pytest.raises(trainer.TrainRefused):
        trainer.build_model("torchvision_resnet18", "mlp_1024", 6)


def test_the_manifest_rejects_unknown_backbones_and_heads(manifest):
    data = manifest_mod.load_yaml_text(manifest_mod.bundled_path().read_text(encoding="utf-8"))
    data["model"]["backbone"] = "torchvision_resnet50"
    with pytest.raises(manifest_mod.ManifestError, match="model.backbone"):
        manifest_mod.parse_manifest(data)
    data["model"]["backbone"] = "torchvision_resnet18"
    data["model"]["head"] = "kit_mlp_1024"
    with pytest.raises(manifest_mod.ManifestError, match="model.head"):
        manifest_mod.parse_manifest(data)
    # The pre-session-3 spelling still loads as the kit's model, with a warning.
    del data["model"]["backbone"], data["model"]["head"]
    data["model"]["arch"] = "resnet18"
    m = manifest_mod.parse_manifest(data)
    assert m.model.backbone == "torchvision_resnet18" and m.model.head == "kit_mlp_512_256_128_d03"
    assert any("deprecated" in w for w in m.warnings)
