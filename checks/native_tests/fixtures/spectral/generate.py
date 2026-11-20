# SPDX-License-Identifier: Apache-2.0
"""Original Spectral inputs and independent dense preparation oracles.

Independent fixtures run in processes across os.cpu_count(). Random draws
within a fixture stay in their original order. No Mojolearn/product import,
native operation, fitted-output oracle or identity measurement is executed.
"""
from concurrent.futures import ProcessPoolExecutor
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np


DIRECTORY = Path(__file__).resolve().parent


def write(name, values, kind, role):
    values = np.asarray(values, dtype=kind, order="C")
    data = values.tobytes(order="C")
    path = DIRECTORY / name
    path.write_bytes(data)
    return {"file": name, "shape": list(values.shape), "dtype": kind,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "role": role}


def matrix(n, seed, plant):
    rnd = random.Random(seed)
    pool = [0.0, 0.0, -0.0, 1.0, 1.0, 2.5, 1e-45, 1.1754943508222875e-38, 0.25, 0.25]
    values = [rnd.choice(pool) if plant else rnd.random() for _ in range(n * n)]
    return np.asarray(values, dtype="<f4").reshape(n, n)


def affinity(query, training):
    p, q = training.astype(np.float64), query.astype(np.float64)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    cross = ((q[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
    threshold = float(np.median(d2))
    return np.ascontiguousarray(np.where(cross < threshold, 1.0 / (1.0 + cross), 0.0).astype(np.float32))


def blobs(count, seed):
    rng = np.random.default_rng(seed)
    centers = np.asarray([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    rows = [centers[i % 3] + rng.normal(0.0, 0.7, 2) for i in range(count)]
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))


def fixture(job):
    family, seed = job
    artifacts = []
    if family == "coo":
        n = 17 + seed
        values = matrix(n, seed, True)
        values.flat[3] = float("nan")
        values.flat[5] = float("inf")
        # Original independent Python row-major scan, including Float32
        # values widened to Python float for the nonzero comparison.
        triples = [(i, j, float(values[i, j])) for i in range(n) for j in range(n)
                   if float(values[i, j]) != 0.0]
        artifacts += [write(f"coo-{seed}-input.f32", values, "<f4", "original input"),
                      write(f"coo-{seed}-rows.i32", [v[0] for v in triples], "<i4", "independent oracle"),
                      write(f"coo-{seed}-cols.i32", [v[1] for v in triples], "<i4", "independent oracle"),
                      write(f"coo-{seed}-values.f32", [v[2] for v in triples], "<f4", "independent oracle")]
        return {"name": f"coo-{seed}", "rows": n, "nonzero": len(triples), "artifacts": artifacts}
    if family == "knn":
        n, k = 23 + seed, 3 + seed
        values = matrix(n, 100 + seed, seed % 2 == 0)
        directed = np.zeros((n, n), dtype=np.float32)
        for i, row in enumerate(values.tolist()):
            candidates = [(value, column) for column, value in enumerate(row)]
            for _, column in sorted(candidates)[:k]:
                directed[i, column] = 1.0
        expected = (0.5 * (directed + directed.T)).astype(np.float32)
        artifacts += [write(f"knn-{seed}-input.f32", values, "<f4", "original input"),
                      write(f"knn-{seed}-affinity.f32", expected, "<f4", "independent sorted-candidate oracle")]
        return {"name": f"knn-{seed}", "rows": n, "neighbors": k, "artifacts": artifacts}
    if family == "embedding":
        values = blobs(150, 20260920)
        dense = affinity(values, values)
        hollow = dense.copy()
        np.fill_diagonal(hollow, 0.0)
        artifacts += [write("embedding150-input.f32", values, "<f4", "original input"),
                      write("embedding150-affinity.f32", dense, "<f4", "original precomputed input"),
                      write("embedding150-hollow.f32", hollow, "<f4", "original hollow precomputed input"),
                      write("embedding80-affinity.f32", affinity(values[:80], values[:80]), "<f4", "original tolerance input")]
    elif family == "predict":
        values = blobs(120, 20260915)
        far = np.asarray([[1e4, -1e4], [-3e4, 2.5e4]], dtype=np.float32)
        pre_queries = np.asarray([[1e4, 1e4], [4.0, 4.0]], dtype=np.float32)
        artifacts += [write("predict120-input.f32", values, "<f4", "original input"),
                      write("predict240-duplicated.f32", np.concatenate([values, values]), "<f4", "original duplicate input"),
                      write("predict120-far.f32", far, "<f4", "original nearest-neighbor query"),
                      write("predict120-affinity.f32", affinity(values, values), "<f4", "original precomputed input"),
                      write("predict120-query-affinity.f32", affinity(pre_queries, values), "<f4", "original zero-degree and central queries")]
    elif family in ("resident", "assignment"):
        rng = np.random.default_rng(11 if family == "resident" else 7)
        values = np.ascontiguousarray(np.concatenate([rng.normal(c, 0.4, size=(60, 3)) for c in (0.0, 4.0, 8.0)]), dtype=np.float32)
        artifacts += [write(f"{family}180-input.f32", values, "<f4", "original input")]
        if family == "resident":
            artifacts += [write("resident180-query.f32", values[::5] + np.float32(0.05), "<f4", "original resident query")]
    else:
        rng = np.random.default_rng(2)
        values = np.concatenate([rng.standard_normal((60, 3)) + 6.0, rng.standard_normal((60, 3)) - 6.0]).astype(np.float32)
        artifacts += [write("host120-input.f32", values, "<f4", "original separated-class input")]
    return {"name": family, "artifacts": artifacts, "status": "input preparation only; fitted numerical oracles pending typed API"}


def main():
    jobs = [(family, seed) for family in ("coo", "knn") for seed in range(6)]
    jobs += [(family, 0) for family in ("embedding", "predict", "resident", "assignment", "host")]
    workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fixtures = list(pool.map(fixture, jobs))
    root = DIRECTORY.parents[3]
    sources = {}
    for name in ("test_native_dense_coo.py", "test_spectral_embedding.py", "test_spectral_predict.py",
                 "test_cluster_resident_native.py", "test_x_cluster_twice.py", "test_cpu_training_e2.py"):
        path = root / "python/mojolearn/tests" / name
        source = path.read_text()
        sources[str(path.relative_to(root))] = {"sha256": hashlib.sha256(source.encode()).hexdigest(),
            "functions": {node.name: {"line": node.lineno, "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()}
                          for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}}
    report = {"status": "independent reference/input preparation only", "workers": workers,
              "independent_fixtures": len(fixtures), "numpy": np.__version__, "python": sys.version,
              "platform": platform.platform(), "sources": sources, "fixtures": fixtures,
              "limits": ["No Mojolearn imports or native execution.", "No fitted-output or old-release hash is an oracle.",
                         "Original full Spectral input bytes retained; fit/eigen/query expectations await typed API."]}
    (DIRECTORY / "references.json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts = [item for case in fixtures for item in case["artifacts"]]
    print(json.dumps({"fixtures": len(fixtures), "artifacts": len(artifacts), "bytes": sum(x["bytes"] for x in artifacts),
                      "workers": workers, "product_execution": False}))


if __name__ == "__main__":
    main()
