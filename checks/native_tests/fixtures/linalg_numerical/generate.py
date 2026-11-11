# SPDX-License-Identifier: Apache-2.0
"""Record the original linalg inputs and an independent NumPy SVD oracle.

No Mojolearn import or execution. Independent fixture variants share a pool
whose worker count comes from the host. NumPy owns the small SVD operation;
its numerical implementation is neither copied nor replaced here.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def matrix(rows, columns):
    return np.ascontiguousarray(
        np.random.default_rng(5).standard_normal((rows, columns)).astype(np.float32)
    )


def record(kind):
    if kind == "constant-64x10":
        base = matrix(64, 4) * np.asarray([1e4, 1.0, 1e4, 1.0], np.float32)
        constant = np.broadcast_to(
            np.asarray([1.0, 3.0, 0.0, 250.0, 0.0, 7.5], np.float32), (64, 6)
        )
        value = np.ascontiguousarray(np.hstack([base, constant]).astype(np.float32))
    else:
        rows, columns = {"qr-48x5": (48, 5), "qr-16x4": (16, 4), "rank-48x3": (48, 3)}[kind]
        value = matrix(rows, columns)
    data = value.astype("<f4", copy=False).tobytes(order="C")
    path = ROOT / (kind + ".f32")
    path.write_bytes(data)
    records = [{"file": path.name, "shape": list(value.shape), "sha256": hashlib.sha256(data).hexdigest()}]
    if kind == "constant-64x10":
        reference = np.linalg.svd(value.astype(np.float64), compute_uv=False)
        data = reference.astype("<f8", copy=False).tobytes()
        path = ROOT / (kind + "-singular.f64")
        path.write_bytes(data)
        records.append({"file": path.name, "shape": [10], "sha256": hashlib.sha256(data).hexdigest()})
    return records


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(record, ("qr-48x5", "qr-16x4", "rank-48x3", "constant-64x10")))
    records = [record for group in groups for record in group]
    metadata = {
        "source": "test_linalg_decompositions.py at e53128fe7f9206abe748dbe32052a81ed864ef3c",
        "numpy": np.__version__,
        "seed": 5,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": records,
        "limits": "Independent input/reference generation only; no Mojolearn execution, timing or identity evidence.",
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"files": len(records), "numpy": np.__version__, "metadata": str(ROOT / "metadata.json")}))
