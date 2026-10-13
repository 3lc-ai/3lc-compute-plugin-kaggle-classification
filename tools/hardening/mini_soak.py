# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""An in-process mini soak on the synthetic kit (the test helpers): N cycles of a 1-epoch CPU train plus a
predict, this process's working set after each cycle. Used to verify the between-jobs memory release
(trainer.release_memory) that the campaign's F5 soak asked for. Run from the repo's dev venv:

  uv run python tools/hardening/mini_soak.py --cycles 8 [--no-release]
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as w
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tools"))


def working_set_mb() -> int:
    import subprocess

    out = subprocess.run(["powershell", "-NoProfile", "-Command", f"[math]::Round((Get-Process -Id {os.getpid()}).WorkingSet64 / 1MB)"],
                         capture_output=True, text=True, check=False)
    return int(out.stdout.strip() or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--no-release", action="store_true", help="skip the between-jobs release (the before picture)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    tmp = Path(tempfile.mkdtemp(prefix="kg-mini-soak-"))
    os.environ["KAGGLE_CLASSIFICATION_HOME"] = str(tmp / "home")
    os.environ["TLC_PROJECT_ROOT_URL"] = (tmp / "root").as_posix()
    (tmp / "home").mkdir()
    (tmp / "root").mkdir()
    from helpers import FakeCtx, make_kit_tree, small_manifest_data

    import build_kit
    from kaggle_classification import importer, predictor, session, trainer
    from kaggle_classification import manifest as manifest_mod

    importer.project_root_url = lambda: (tmp / "root").as_posix()  # type: ignore[assignment]
    if args.no_release:
        trainer.drop_tlc_cache = lambda: None  # type: ignore[assignment]
    data = small_manifest_data("v1")
    kit_root = make_kit_tree(tmp / "tree", data, real_images=True)
    build_kit.write_files_index(kit_root, competition_id=data["competition"]["id"], kit_version="v1")
    manifest = manifest_mod.parse_manifest(data, source="test", source_detail="synthetic")
    session.publish_kit_dir(manifest, kit_root)
    importer.run_import({}, FakeCtx(), manifest)
    print(f"after import: {working_set_mb()} MB", flush=True)
    series = []
    for n in range(1, args.cycles + 1):
        ctx = FakeCtx()
        ctx.job_id = f"soak-{n}"
        t0 = time.time()
        trainer.run_training({"epochs": "1", "batch_size": "8", "device": "cpu", "workers": "0", "run_name": f"soak{n}"}, ctx, manifest)
        pctx = FakeCtx()
        pctx.job_id = f"pred-{n}"
        predictor.run_predict({"train_job_id": ctx.job_id, "device": "cpu"}, pctx, manifest)
        mb = working_set_mb()
        series.append(mb)
        print(f"cycle {n}: {mb} MB ({time.time() - t0:.1f} s)", flush=True)
    print(f"SERIES {series} growth per cycle {(series[-1] - series[0]) / max(len(series) - 1, 1):.1f} MB (release {'off' if args.no_release else 'on'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
