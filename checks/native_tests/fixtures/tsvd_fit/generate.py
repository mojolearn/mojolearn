# SPDX-License-Identifier: Apache-2.0
"""Existing TSVD inputs plus independent NumPy Float64 SVD/variance oracles.

Variant preparation and individual file writes use the host-sized pool.
This script never imports or executes Mojolearn.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def prepare(name):
    if name == "host-512x5":
        x = np.random.default_rng(7).standard_normal((512, 5)).astype("f4")
    else:
        x = np.array([[1, 2, 0], [3, 0, 1], [0, 1, 4], [2, 3, 1],
                      [1, 0, 2], [4, 1, 0]], dtype="f4")
    xd = x.astype("f8")
    _, singular, vectors = np.linalg.svd(xd, full_matrices=False)
    scores = xd @ vectors[:2].T
    variance = scores.var(axis=0)
    full = xd.var(axis=0).sum()
    selected = xd[:, [0, 2]].var(axis=0)
    outputs = {name + ".f32": x,
               name + "-singular.f64": singular,
               name + "-vectors.f64": vectors,
               name + "-variance.f64": variance,
               name + "-ratio.f64": variance / full,
               name + "-selected-variance.f64": selected,
               name + "-selected-ratio.f64": selected / full}
    return list(outputs.items())


def write(item):
    filename, value = item
    data = value.astype("<f4" if filename.endswith(".f32") else "<f8", copy=False).tobytes()
    (ROOT / filename).write_bytes(data)
    return {"file": filename, "shape": list(value.shape), "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(prepare, ("host-512x5", "boundary-6x3")))
        records = list(pool.map(write, (item for group in groups for item in group)))
    metadata = {"source_commit": "14ad3ca7ab", "numpy": np.__version__,
                "source_files": ["test_cpu_training_e.py", "test_tsvd_native_boundary.py"],
                "seed": 7, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": records,
                "limits": "Independent reference generation only; no Mojolearn compile, execution, timing or identity evidence."}
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"files": len(records), "numpy": np.__version__, "metadata": str(ROOT / "metadata.json")}))
