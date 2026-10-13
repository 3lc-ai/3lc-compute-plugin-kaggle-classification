# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""A diagnostic for the worker's memory (session 15, the leak hunt): what this worker process holds, read
from inside it between jobs. Served by ``GET /debug/leak_probe`` only when the compute service runs with
``KAGGLE_CLASSIFICATION_LEAK_PROBE=1``; otherwise the route answers 404 and nothing here is imported.

Each call returns the facts and the change since the previous call: process memory (working set, private
bytes; Windows), tracemalloc's Python allocations (top 20 by size and by count, grouped by line), gc's
object counts by type, the logging handlers on the root and every named logger, torch's global hooks and
live modules and CUDA's allocator, and every module-level container in the tlc / tlcurl / tlcsaas /
tlc_plugin_sdk / kaggle_classification packages with more than a few entries (tlc's object registry, its
URL adapters, caches), so a cache that is not the object registry shows up by name.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import Counter
from typing import Any

ENV = "KAGGLE_CLASSIFICATION_LEAK_PROBE"
_PACKAGES = ("tlc", "tlcurl", "tlcsaas", "tlc_plugin_sdk", "tlc_compute", "kaggle_classification", "umap", "numba", "sklearn")
_state: dict[str, Any] = {"snapshot": None, "types": None, "containers": None, "calls": 0, "at": None}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == "1"


def _process_memory() -> dict[str, int | None]:
    out: dict[str, int | None] = {"working_set_mb": None, "private_mb": None}
    if sys.platform != "win32":
        try:
            import resource

            out["max_rss_mb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        except Exception:
            pass
        return out
    import ctypes
    import ctypes.wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = w.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD]
    psapi.GetProcessMemoryInfo.restype = w.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        out["working_set_mb"] = int(pmc.WorkingSetSize / 1048576)
        out["private_mb"] = int(pmc.PrivateUsage / 1048576)
    return out


def _logging_handlers() -> dict[str, Any]:
    import logging

    per: dict[str, int] = {}
    root_n = len(logging.getLogger().handlers)
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and lg.handlers:
            per[name] = len(lg.handlers)
    return {"root": root_n, "named_total": sum(per.values()), "named": dict(sorted(per.items(), key=lambda kv: -kv[1])[:10]),
            "loggers": len(logging.root.manager.loggerDict)}


def _torch_facts() -> dict[str, Any]:
    if "torch" not in sys.modules:
        return {"imported": False}
    import torch

    mod = torch.nn.modules.module
    facts: dict[str, Any] = {"imported": True}
    for attr in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks", "_global_backward_pre_hooks",
                 "_global_module_registration_hooks", "_global_parameter_registration_hooks", "_global_buffer_registration_hooks"):
        v = getattr(mod, attr, None)
        if v is not None:
            facts[attr] = len(v)
    mods = tens = 0
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.nn.Module):
                mods += 1
            elif isinstance(o, torch.Tensor):
                tens += 1
        except Exception:
            continue
    facts["live_modules"], facts["live_tensors"] = mods, tens
    if torch.cuda.is_available():
        facts["cuda_allocated_mb"] = int(torch.cuda.memory_allocated() / 1048576)
        facts["cuda_reserved_mb"] = int(torch.cuda.memory_reserved() / 1048576)
    return facts


def _containers() -> dict[str, int]:
    """Every module-level dict / list / set / deque (and a class's such attributes) in the watched packages."""
    out: dict[str, int] = {}
    for mname, mod in list(sys.modules.items()):
        if not mod or mname.split(".")[0] not in _PACKAGES:
            continue
        try:
            items = list(vars(mod).items())
        except Exception:
            continue
        for aname, val in items:
            n = _size(val)
            if n is not None and n >= 5:
                out[f"{mname}.{aname}"] = n
            if isinstance(val, type) and val.__module__ == mname:
                for cname, cval in list(vars(val).items()):
                    n = _size(cval)
                    if n is not None and n >= 5:
                        out[f"{mname}.{aname}.{cname}"] = n
    return out


def _size(val: Any) -> int | None:
    import collections

    if isinstance(val, (dict, list, set, frozenset, collections.deque, collections.OrderedDict)):
        try:
            return len(val)
        except Exception:
            return None
    cache_info = getattr(val, "cache_info", None)   # functools.lru_cache
    if callable(cache_info):
        try:
            return int(cache_info().currsize)
        except Exception:
            return None
    return None


def _safe(fn: Any) -> Any:
    """One section's facts, or its error: a probe that raises would hide every other section."""
    try:
        return fn()
    except Exception as exc:
        import traceback

        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1200:]}


def probe(reset: bool = False) -> dict[str, Any]:
    import tracemalloc

    gc.collect()
    if not tracemalloc.is_tracing():
        tracemalloc.start(1)
    snap = tracemalloc.take_snapshot().filter_traces((tracemalloc.Filter(False, tracemalloc.__file__),))
    types: Counter = Counter()
    for o in gc.get_objects():
        try:
            types[type(o).__module__ + "." + type(o).__qualname__] += 1
        except Exception:
            continue
    containers = _safe(_containers) or {}
    if "error" in containers:
        containers = {}
    out: dict[str, Any] = {
        "call": _state["calls"] + 1, "at": time.time(), "since_s": round(time.time() - _state["at"], 1) if _state["at"] else None,
        "process": _process_memory(), "tracemalloc_mb": round(sum(s.size for s in snap.statistics("filename")) / 1048576, 1),
        "logging": _safe(_logging_handlers), "torch": _safe(_torch_facts), "gc_objects": sum(types.values()),
    }
    prev = _state["snapshot"]
    if prev is not None and not reset:
        diff = snap.compare_to(prev, "lineno")
        out["top_by_size"] = [{"where": str(d.traceback[0]), "size_kb": round(d.size_diff / 1024, 1), "count": d.count_diff}
                              for d in sorted(diff, key=lambda d: -d.size_diff)[:20]]
        out["top_by_count"] = [{"where": str(d.traceback[0]), "count": d.count_diff, "size_kb": round(d.size_diff / 1024, 1)}
                               for d in sorted(diff, key=lambda d: -d.count_diff)[:20]]
        ptypes = _state["types"] or Counter()
        out["gc_types_grown"] = [{"type": t, "now": types[t], "delta": types[t] - ptypes.get(t, 0)}
                                 for t in sorted(types, key=lambda t: -(types[t] - ptypes.get(t, 0)))[:25] if types[t] - ptypes.get(t, 0) > 0]
        pcont = _state["containers"] or {}
        out["containers_grown"] = [{"name": k, "now": v, "delta": v - pcont.get(k, 0)}
                                   for k, v in sorted(containers.items(), key=lambda kv: -(kv[1] - pcont.get(kv[0], 0)))[:25] if v - pcont.get(k, 0) > 0]
    out["containers_largest"] = dict(sorted(containers.items(), key=lambda kv: -kv[1])[:15])
    _state.update({"snapshot": snap, "types": types, "containers": containers, "calls": out["call"], "at": out["at"]})
    return out
