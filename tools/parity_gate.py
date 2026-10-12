# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The kit-parity gate's plugin side (docs/TRAIN_MIRROR.md §11), re-runnable on any build: one
``trainer.run_training`` per seed on a COPY of the imported ``initial`` seed tables in a scratch project
root, the kit's hyperparameters (10 epochs, batch 16, lr 1e-4, weight decay 0, workers 0, Use latest
off), auto device. Then, with ``--predict``, one ``predictor.run_predict`` on the first seed's checkpoint
(auto device) so GPU inference is exercised on the same build.

The kit side is fixed (the kit's ``train.py`` numbers recorded in §11: seeds 42 / 43 / 44 = 68.75 /
71.83 / 71.00 %, mean 70.53 ± 1.59); the verdict is the gate's rule: PASS when the difference of means
is within the larger std + 2 points.

Read-only towards the source project: the two ``initial`` tables are copied; the images they point at
(the kit folder recorded in the import record) are read in place; every run, record and CSV lands under
``--scratch`` (default ``<repo>/.parity-gate``, gitignored by being untracked).

  uv run python tools/parity_gate.py --source "<home>/projects/intel-scene" --seeds 42 43 44 --predict

The kit folder comes from the copied tables' image paths; the import record the trainer reads is
rebuilt here in the scratch plugin home.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

KIT_SIDE = {42: 68.75, 43: 71.83, 44: 71.00}   # TRAIN_MIRROR.md §11, the kit's train.py, 2026-09-29


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Ctx:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.checks: list = []
        self.fields: dict = {}

    def log(self, m: str) -> None:
        log("  " + m)

    def set_checks(self, c: list) -> None:
        self.checks = c

    def set_progress(self, p: dict) -> None:
        pass

    def set_field(self, k: str, v: object) -> None:
        self.fields[k] = v

    def is_cancelled(self) -> bool:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="the imported project dir holding datasets/intel-scene_train and _val")
    ap.add_argument("--scratch", default=str(Path(__file__).resolve().parents[1] / ".parity-gate"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--project", default="intel-scene-parity")
    ap.add_argument("--manifest-base", default="https://competitions.dev.3lc.ai")
    ap.add_argument("--predict", action="store_true", help="run predictor.run_predict on the first seed's checkpoint")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    scratch = Path(args.scratch)
    root = scratch / "projects"
    plugin_home = scratch / "plugin-home"
    os.environ["KAGGLE_CLASSIFICATION_HOME"] = str(plugin_home)
    os.environ["TLC_PROJECT_ROOT_URL"] = root.as_posix()
    os.environ.setdefault("KAGGLE_CLASSIFICATION_MANIFEST_BASE_URL", args.manifest_base)
    if scratch.exists():
        shutil.rmtree(scratch)
    source = Path(args.source)
    for split in ("train", "val"):
        src = source / "datasets" / f"intel-scene_{split}" / "tables" / "initial"
        dst = root / args.project / "datasets" / f"intel-scene_{split}" / "tables" / "initial"
        shutil.copytree(src, dst)
        log(f"copied {src} -> {dst}")
    plugin_home.mkdir(parents=True)

    import torch
    import torchvision

    log(f"torch {torch.__version__} torchvision {torchvision.__version__} cuda {torch.cuda.is_available()} python {sys.version.split()[0]}")
    import tlc

    from kaggle_classification import manifest as manifest_mod
    from kaggle_classification import predictor, session, trainer

    log(f"gpu: {trainer.gpu_kernel_support()}")
    current = manifest_mod.resolve(network=False).manifest
    train_url = (root / args.project / "datasets" / "intel-scene_train" / "tables" / "initial").as_posix()
    val_url = (root / args.project / "datasets" / "intel-scene_val" / "tables" / "initial").as_posix()
    t = tlc.Table.from_url(tlc.Url(train_url))
    first_image = str(next(iter(t.table_rows))["image"])
    kit_dir = str(Path(first_image).parents[3])   # <kit>/data/train/<class>/<file> -> <kit>
    log(f"train rows {t.row_count}; kit dir from the image paths: {kit_dir}")
    label_map = {str(k): v for k, v in sorted({int(c.id): c.name for c in current.classes}.items())}
    label_map[str(current.undefined_label_id)] = "undefined"
    session.save({"import_state": {
        "project_name": args.project, "table_name": "initial", "kit_dir": kit_dir, "kit_version": current.kit.version,
        "tables": {"train": {"url": train_url, "rows": t.row_count}, "val": {"url": val_url, "rows": 1200}},
        "lineage_root": {"train_url": train_url, "val_url": val_url},
        "val_locked": {"url": val_url, "editable": True}, "label_map": label_map, "checks": [], "timings": {},
        "manifest_provenance": current.provenance, "job_id": "parity-import", "completed_at": time.time(),
    }})
    session.publish_kit_dir(current, kit_dir)

    results = []
    for seed in args.seeds:
        params = {"train_table_url": train_url, "use_latest": False, "epochs": str(args.epochs), "batch_size": "16",
                  "lr": "0.0001", "weight_decay": "0.0", "seed": str(seed), "workers": "0", "device": "",
                  "run_name": f"parity_seed{seed}", "project_name": args.project}
        ctx = Ctx(f"parity-seed{seed}")
        t0 = time.time()
        result = trainer.run_training(params, ctx, current)
        elapsed = round(time.time() - t0, 1)
        row = {"seed": seed, "best_val_accuracy": float(result["best_val_accuracy"]), "best_epoch": result.get("best_epoch"),
               "device": result.get("device"), "device_label": ctx.fields.get("device_label"), "elapsed_s": elapsed,
               "kit_side": KIT_SIDE.get(seed), "train_job_id": ctx.job_id}
        results.append(row)
        log(json.dumps(row))

    plugin_mean = statistics.mean(r["best_val_accuracy"] for r in results)
    plugin_std = statistics.stdev(r["best_val_accuracy"] for r in results) if len(results) > 1 else 0.0
    kit = [KIT_SIDE[r["seed"]] for r in results if r["seed"] in KIT_SIDE]
    kit_mean = statistics.mean(kit) if kit else None
    kit_std = statistics.stdev(kit) if len(kit) > 1 else 0.0
    threshold = max(plugin_std, kit_std) + 2.0
    diff = (plugin_mean - kit_mean) if kit_mean is not None else None
    verdict = ("PASS" if abs(diff) <= threshold else "FAIL") if diff is not None else "NO KIT SIDE"
    out = {
        "torch": torch.__version__, "torchvision": torchvision.__version__, "gpu": trainer.gpu_kernel_support(),
        "runs": results, "plugin_mean": round(plugin_mean, 2), "plugin_std": round(plugin_std, 2),
        "kit_mean": round(kit_mean, 2) if kit_mean is not None else None, "kit_std": round(kit_std, 2),
        "difference_of_means": round(diff, 2) if diff is not None else None, "threshold": round(threshold, 2), "verdict": verdict,
    }
    if args.predict:
        ctx = Ctx("parity-predict")
        t0 = time.time()
        pred = predictor.run_predict({"train_job_id": results[0]["train_job_id"], "device": ""}, ctx, current)
        out["predict"] = {"device": pred.get("device"), "device_label": ctx.fields.get("device_label"), "rows": pred.get("rows"),
                          "local_score": pred.get("local_score"), "checks_ok": sum(1 for c in (pred.get("checks") or []) if c.get("ok")),
                          "checks": len(pred.get("checks") or []), "elapsed_s": round(time.time() - t0, 1)}
        log(json.dumps(out["predict"]))
    (scratch / "parity_gate.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log(f"VERDICT: {verdict} (plugin {out['plugin_mean']} ± {out['plugin_std']} vs kit {out['kit_mean']} ± {out['kit_std']}; "
        f"difference {out['difference_of_means']}, threshold {out['threshold']})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
