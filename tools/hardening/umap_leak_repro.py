# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The leak hunt's isolation test: does UMAP (fit on N train rows, transform M val rows, the plugin's settings)
grow the process per call? Prints private bytes and the number of live llvmlite modules after each round.

  uv run python tools/hardening/umap_leak_repro.py --rounds 6 [--stage fit|transform|both]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--rows", type=int, default=6600)
    ap.add_argument("--val", type=int, default=1200)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--stage", choices=["fit", "transform", "both", "plugin"], default="both")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    import numpy as np
    import umap

    from kaggle_classification import leak_probe, trainer

    trainer.numba_cache_env()

    def modules() -> int:
        import llvmlite.binding.module as m

        n = 0
        for o in gc.get_objects():
            try:
                n += isinstance(o, m.ModuleRef)
            except ReferenceError:   # a dead weakref proxy
                continue
        return n

    rng = np.random.default_rng(0)
    train = rng.normal(size=(args.rows, args.dim)).astype(np.float32)
    val = rng.normal(size=(args.val, args.dim)).astype(np.float32)
    fitted = None
    if args.stage == "transform":
        fitted = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42).fit(train)
    series = []
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        if args.stage in ("fit", "both"):
            fitted = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42).fit(train)
        if args.stage in ("transform", "both"):
            fitted.transform(val)
        if args.stage == "plugin":   # the plugin's own path: a fresh fit, then trainer._transform
            fitted = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42).fit(train)
            trainer._transform(fitted, val, 3)
        if args.stage == "fit":
            fitted = None
        gc.collect()
        mem = leak_probe._process_memory()
        series.append(mem["private_mb"])
        print(f"round {r} ({args.stage}): private {mem['private_mb']} MB, working set {mem['working_set_mb']} MB, "
              f"llvmlite modules {modules()}, {time.time() - t0:.1f} s", flush=True)
    print(f"SERIES {args.stage} {series} private growth per round after the first: "
          f"{(series[-1] - series[1]) / max(len(series) - 2, 1):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
