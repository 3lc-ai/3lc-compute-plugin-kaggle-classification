# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The leak hunt's isolation test for the predict half: one stage of the predict job repeated alone, the
process's private bytes after each round. Stages: ``table`` (open the locked val table with tlc and read
every row), ``decode`` (PIL decode + the val transform over the val and test images), ``forward`` (the
competition model on the GPU over random batches of the same count), ``loader`` (the real DataLoader over
the val table view, CPU, no model). Reads a tester env's import record; writes nothing.

  uv run python tools/hardening/predict_leak_repro.py --root <tester> --stage table --rounds 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", choices=["table", "decode", "forward", "loader"], required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--drop", action="store_true", help="drop tlc's object cache after each round, as the worker does")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    from kaggle_classification import leak_probe
    from kaggle_classification import trainer as trainer_mod

    import harness as h  # this folder (the campaign's own plugin state, harness.plugin_state_dir)

    state = h.plugin_state_dir(Path(args.root))
    record = json.loads((state / "ui_config.json").read_text(encoding="utf-8"))["import_state"]
    val_url = record["val_locked"]["url"]
    kit_dir = Path(record["kit_dir"])
    series = []
    import torch

    model = None
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        if args.stage == "table":
            import tlc

            t = tlc.Table.from_url(tlc.Url(val_url))
            n = sum(1 for _ in t.table_rows)
            del t
            if args.drop:
                trainer_mod.drop_tlc_cache()
        elif args.stage == "decode":
            from PIL import Image

            from kaggle_classification import trainer

            _, val_tf = trainer.build_transforms(150)
            paths = sorted((kit_dir / "data" / "val").rglob("*.jpg")) + sorted((kit_dir / "data" / "test").rglob("*.jpg"))
            n = 0
            for p in paths:
                with Image.open(p) as im:
                    val_tf(im.convert("RGB"))
                n += 1
        elif args.stage == "forward":
            from kaggle_classification import trainer

            if model is None or True:   # a fresh model per round, as each predict job loads one
                model = trainer.build_model("torchvision_resnet18", "kit_mlp_512_256_128_d03", 6).to("cuda").eval()
            n = 0
            with torch.no_grad():
                for _ in range(3000 // 64 + 1):
                    model(torch.randn(64, 3, 150, 150, device="cuda"))
                    n += 64
            del model
            model = None
            torch.cuda.empty_cache()
        else:
            import tlc
            from torch.utils.data import DataLoader

            from kaggle_classification import trainer

            _, val_tf = trainer.build_transforms(150)
            t = tlc.Table.from_url(tlc.Url(val_url))
            view = t.with_transform(trainer._SampleTransform(val_tf))
            n = 0
            for images, _labels in DataLoader(view, batch_size=64, shuffle=False, num_workers=0):
                n += len(images)
            del t, view
        gc.collect()
        mem = leak_probe._process_memory()
        series.append(mem["private_mb"])
        print(f"round {r} ({args.stage}, {n} items): private {mem['private_mb']} MB, working set {mem['working_set_mb']} MB, {time.time() - t0:.1f} s", flush=True)
    print(f"SERIES {args.stage} {series} private growth per round after the first: {(series[-1] - series[1]) / max(len(series) - 2, 1):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
