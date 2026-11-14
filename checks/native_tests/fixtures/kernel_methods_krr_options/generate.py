# SPDX-License-Identifier: Apache-2.0
"""Record the inputs of test_krr_options.py's exact properties: `_data()`
(NumPy seed 3): X = integers in [-4, 4] as float32 (96 x 5) and
y = (X @ normal(5) + normal(96)) as float32, in the original draw order.
The kernel matrices are integer Gram matrices, exact under any summation
order, and are formed in Mojo by the check.

No Mojolearn import or execution. One small draw, so it runs in this
process (nothing to split across cores).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    a = rng.integers(-4, 5, size=(96, 5)).astype(np.float32)
    y = (a @ rng.normal(size=5) + rng.normal(size=96)).astype(np.float32)
    files = []
    for name, value in (("x-96x5.f32", a), ("y-96.f32", y)):
        data = np.ascontiguousarray(value).astype("<f4", copy=False).tobytes()
        (ROOT / name).write_bytes(data)
        files.append({"file": name, "shape": list(value.shape), "sha256": hashlib.sha256(data).hexdigest()})
    metadata = {
        "source": "python/mojolearn/tests/test_krr_options.py::_data",
        "numpy": np.__version__,
        "seed": 3,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": files,
        "limits": "Input generation only; no Mojolearn execution, timing or identity evidence.",
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
