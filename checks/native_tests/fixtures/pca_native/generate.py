# SPDX-License-Identifier: Apache-2.0
"""Original PCA inputs and independent Float64 NumPy spectra; no product run.

Each independent variant runs in the host-sized pool. NumPy performs the
small SVD; no PCA numerical implementation is copied into this generator.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def source(rows, cols):
    rng = np.random.default_rng(9021)
    return (rng.normal(size=(rows, cols)) * np.arange(1, cols + 1)
            + np.arange(cols) / 3).astype("f4")


def record(name):
    if name == "full-8x3":
        x = np.array([[0, 1, 2], [2, 0, 1], [1, 3, -1], [-2, 1, 0],
                      [4, -1, 2], [0, -3, 1], [3, 2, 4], [-1, 0, -2]], dtype=np.float32)
    elif name == "rank-20x5":
        base = source(20, 2)
        x = np.column_stack((base, base[:, 0] * 2, base[:, 1] * 4, base[:, 0] * -2)).astype("f4")
    else:
        x = source(*( (21, 4) if name == "exact-21x4" else (4, 7) ))
    outputs = {name + ".f32": x.astype("<f4", copy=False)}
    if name != "rank-20x5":
        xd = x.astype("f8")
        mean = xd.mean(axis=0)
        _, singular, vectors = np.linalg.svd(xd - mean, full_matrices=False)
        variance = singular ** 2 / (len(x) - 1)
        outputs.update({name + "-mean.f64": mean,
                        name + "-singular.f64": singular,
                        name + "-variance.f64": variance,
                        name + "-ratio.f64": variance / variance.sum(),
                        name + "-vectors.f64": vectors})
    return list(outputs.items())


def write(item):
    filename, value = item
    data = value.astype("<f4" if filename.endswith(".f32") else "<f8", copy=False).tobytes()
    (ROOT / filename).write_bytes(data)
    return {"file": filename, "shape": list(value.shape), "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(record, ("exact-21x4", "full-8x3", "wide-4x7", "rank-20x5")))
        records = list(pool.map(write, (item for group in groups for item in group)))
    metadata = {"source_commit": "e53128fe7f9206abe748dbe32052a81ed864ef3c",
                "source_files": ["test_pca_exact_native_owner.py", "test_pca_full_surface.py"],
                "numpy": np.__version__, "seed": 9021,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": records,
                "limits": "Independent fixture generation only; no Mojolearn execution or correctness/timing/identity claim."}
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"files": len(records), "numpy": np.__version__, "metadata": str(ROOT / "metadata.json")}))
