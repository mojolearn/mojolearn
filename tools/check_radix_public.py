#!/usr/bin/env python3
"""Bounded native radix regression through NearestNeighbors; no build steps.

Run against the consolidated IDENTICAL GPU bindings. Integer coordinates make
squared distances exact in Float32, so NumPy supplies an independent distance
and (distance, row index) order oracle without floating-point tolerances.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

ROWS = 2053
KS = (1024, 1025, 2000)
QUERY_TILES = (1, 3)


def fixtures():
    i = np.arange(ROWS)
    queries = np.array([[0, 0], [1, -1], [-3, 2]], dtype=np.float32)
    yield "mixed-ties", np.column_stack(((i * 17) % 31 - 15,
                                         (i * 7) % 13 - 6)).astype(np.float32), queries
    yield "all-ties", np.zeros((ROWS, 2), dtype=np.float32), queries


def oracle(index, queries, k):
    delta = queries[:, None, :].astype(np.int64) - index[None, :, :].astype(np.int64)
    squared = np.sum(delta * delta, axis=2)
    # Integer coordinates are intentionally bounded: both the pinned expanded
    # L2 formula and this direct distance are exact, including cancellation.
    if np.max(squared) >= 2 ** 24:
        raise ValueError("fixture distances are outside exact Float32 integers")
    order = np.argsort(squared, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(squared, order, axis=1).astype(np.float32), order


def assert_answer(got, expected, context):
    distances, indices = map(np.asarray, got)
    want_d, want_i = expected
    if indices.shape != want_i.shape or not np.array_equal(indices, want_i):
        raise AssertionError(f"{context}: index order differs from independent (distance,index) oracle")
    if distances.dtype != np.float32 or distances.shape != want_d.shape:
        raise AssertionError(f"{context}: expected Float32 distances with shape {want_d.shape}")
    if not np.array_equal(distances.view(np.uint32), want_d.view(np.uint32)):
        raise AssertionError(f"{context}: squared-distance bits differ from exact integer oracle")


def run_checks(neighbors_class):
    results = []
    for name, index, queries in fixtures():
        for k in KS:
            expected = oracle(index, queries, k)
            for tile in QUERY_TILES:
                model = neighbors_class(n_neighbors=k, query_tile=tile,
                                        metric="sqeuclidean", algorithm="brute").fit(index)
                started = time.monotonic()
                for repeat in range(2):
                    got = model.kneighbors(queries)
                    assert_answer(got, expected, f"{name} k={k} tile={tile} repeat={repeat}")
                results.append(dict(fixture=name, k=k, requested_query_tile=tile,
                                    used_query_tile=getattr(model, "used_query_tile_", None),
                                    repeats=2, rows=ROWS, queries=len(queries),
                                    seconds=round(time.monotonic() - started, 4)))
                print(f"RADIX CASE PASS {name} k={k} tile={tile}", flush=True)
                del model
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-out", required=True)
    args = p.parse_args()
    import mojolearn as ml
    from mojolearn import _backend, _verify
    if _backend.numeric_mode() != "identical" or _backend.vendor() not in ("metal", "cuda", "hip"):
        raise RuntimeError("radix regression requires IDENTICAL GPU bindings; CPU replay does not exercise this kernel")
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    started = time.monotonic()
    results = run_checks(ml.NearestNeighbors)
    report = dict(format="mojolearn.radix-public-regression.v1", passed=True,
                  source_commit=commit, vendor=_backend.vendor(), cases=results,
                  scope="public GPU kNN radix selection and wide rounds, exact tied-distance/index ordering, query tiling, repeated resident inference",
                  seconds=round(time.monotonic() - started, 4),
                  bindings=_verify.binding_artifacts(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"RADIX PUBLIC PASS {len(results)} cases {report['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
