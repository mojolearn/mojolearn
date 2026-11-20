# SPDX-License-Identifier: Apache-2.0
"""Record the blob inputs of test_gmm_sample.py's REPEAT, MOMENTS and WHITEN
arms: `_blobs(seed, n)`, two NumPy-normal groups (seed 1 n 3000, seed 2
n 6000, seed 3 n 3000), stacked and cast to float32 in the original order.

No Mojolearn import or execution. The three independent draws run in a
thread pool sized from the host's cores.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def blobs(item):
    seed, n = item
    rng = np.random.default_rng(seed)
    a = rng.normal([0.0, 0.0, 0.0], [1.0, 0.5, 2.0], size=(n // 3, 3))
    b = rng.normal([8.0, -6.0, 3.0], [0.7, 1.5, 0.4], size=(n - n // 3, 3))
    value = np.ascontiguousarray(np.vstack([a, b]).astype(np.float32))
    name = "blobs-seed%d-%dx3.f32" % (seed, n)
    payload = value.astype("<f4", copy=False).tobytes()
    (ROOT / name).write_bytes(payload)
    return {"file": name, "shape": list(value.shape), "sha256": hashlib.sha256(payload).hexdigest()}


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        files = list(pool.map(blobs, ((1, 3000), (2, 6000), (3, 3000))))
    metadata = {
        "source": "python/mojolearn/tests/test_gmm_sample.py::_blobs",
        "numpy": np.__version__,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": files,
        "limits": "Input generation only; no Mojolearn execution, timing or identity evidence.",
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
