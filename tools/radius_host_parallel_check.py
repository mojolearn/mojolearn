#!/usr/bin/env python3
"""Exactness and wall-clock gate for CPU RadiusNeighbors row parallelism.

Build the binding first with ``sh bindings/build_core_host.sh``.  An
alternate directory containing ``_mojolearn_core_host.so`` may be passed as
the sole argument, which makes this useful for an isolated candidate build.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
from pathlib import Path
import statistics
import sys
import time


def _fixture(n: int, seed: int):
    out = (ctypes.c_float * n)()
    state = seed
    for i in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        out[i] = ((state >> 8) / float(1 << 24) - 0.5) * 6.0
    return out


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: radius_host_parallel_check.py [host-binding-dir]")
    if len(sys.argv) == 2:
        sys.path.insert(0, sys.argv[1])
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "mojolearn" / "host"))
    binding = importlib.import_module("_mojolearn_core_host")

    ni, nq, nf = 5000, 192, 12
    index = _fixture(ni * nf, 937)
    queries = _fixture(nq * nf, 1823)

    def run(threads: str | None, metric: int = 5, metric_arg: float = 2.0):
        if threads is None:
            os.environ.pop("MOJOLEARN_CPU_THREADS", None)
        else:
            os.environ["MOJOLEARN_CPU_THREADS"] = threads
        indptr = (ctypes.c_int32 * (nq + 1))()
        nnz = binding.radius_neighbors_count(
            ctypes.addressof(index), ctypes.addressof(queries),
            ctypes.addressof(indptr), [ni, nq, nf, 3.0, metric, metric_arg],
        )
        cols = (ctypes.c_int32 * nnz)()
        dists = (ctypes.c_float * nnz)()
        got = binding.radius_neighbors_fill(
            ctypes.addressof(index), ctypes.addressof(queries),
            ctypes.addressof(indptr), ctypes.addressof(cols),
            ctypes.addressof(dists),
            [ni, nq, nf, 3.0, nnz, 1, metric, metric_arg],
        )
        if got != nnz:
            raise AssertionError((got, nnz))
        digest = hashlib.sha256(bytes(indptr) + bytes(cols) + bytes(dists)).hexdigest()
        return nnz, digest

    serial = run("1")
    for metric, metric_arg in ((5, 2.0), (3, 2.0), (7, 2.0), (9, 3.0)):
        one = run("1", metric, metric_arg)
        many = run(None, metric, metric_arg)
        if one != many:
            raise AssertionError(
                f"metric {metric}/{metric_arg}: serial {one} != parallel {many}"
            )

    samples = []
    for _ in range(5):
        start = time.perf_counter()
        if run(None) != serial:
            raise AssertionError("repeat changed output bytes")
        samples.append(time.perf_counter() - start)
    print(
        "radius_host_parallel_check OK:", serial[0], "edges, sha256",
        serial[1], "parallel median", f"{statistics.median(samples):.6f}s",
    )


if __name__ == "__main__":
    main()
