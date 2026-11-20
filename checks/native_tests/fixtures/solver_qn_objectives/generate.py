# SPDX-License-Identifier: Apache-2.0
"""Record the inputs of test_qn_objectives_surface.py's `_data()` (NumPy
seed 0): X = standard_normal((n, 5)) as float32, yr = X @ (arange(1, 6) / 5)
+ 0.5 + 0.1 * standard_normal(n) as float32, for n = 400 and n = 64 (the
two draws share X's prefix but not y's), and the three-class codes of
test_linear_svc_one_vs_rest_and_labels, np.digitize(yr, quantile(yr, [1/3, 2/3]))
on the n = 400 draw. yc = (yr > 0.5) is formed by the check.

No Mojolearn import or execution. Two small draws, so it runs in this
process (nothing to split across cores).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def data(n, d=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w = np.arange(1, d + 1, dtype=np.float32) / d
    yr = (x @ w + 0.5 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return x, yr


if __name__ == "__main__":
    x, yr = data(400)
    _, yr64 = data(64)
    y3 = np.digitize(yr, np.quantile(yr, [1 / 3, 2 / 3])).astype(np.int32)
    files = []
    for name, value, kind in (("x-400x5.f32", x, "<f4"), ("yr-400.f32", yr, "<f4"),
                              ("yr-64.f32", yr64, "<f4"), ("y3-400.i32", y3, "<i4")):
        payload = np.ascontiguousarray(value).astype(kind, copy=False).tobytes()
        (ROOT / name).write_bytes(payload)
        files.append({"file": name, "shape": list(value.shape), "sha256": hashlib.sha256(payload).hexdigest()})
    metadata = {
        "source": "python/mojolearn/tests/test_qn_objectives_surface.py::_data",
        "numpy": np.__version__,
        "seed": 0,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": files,
        "limits": "Input generation only; no Mojolearn execution, timing or identity evidence.",
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
