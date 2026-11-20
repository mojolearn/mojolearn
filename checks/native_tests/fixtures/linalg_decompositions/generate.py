# SPDX-License-Identifier: Apache-2.0
"""Record the symmetric 5x5 input of test_linalg_decompositions.py's eigh
device/host comparison: `_matrix(64, 5)` (NumPy seed 5), `sym = a.T @ a`
in float32, then `((sym + sym.T) * 0.5)` in float32, exactly the original
expression. The QR and svdvals inputs are the linalg_numerical fixtures.

No Mojolearn import or execution. One record: the work is one 64x5 product,
so it runs in this process (nothing to split across cores).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    a = np.ascontiguousarray(np.random.default_rng(5).standard_normal((64, 5)).astype(np.float32))
    sym = a.T @ a
    sym = np.ascontiguousarray(((sym + sym.T) * np.float32(0.5)).astype(np.float32))
    data = sym.astype("<f4", copy=False).tobytes()
    (ROOT / "eigh-64x5-sym.f32").write_bytes(data)
    metadata = {
        "source": "python/mojolearn/tests/test_linalg_decompositions.py::test_eigh_device_equals_host_byte_for_byte",
        "numpy": np.__version__,
        "seed": 5,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": [{"file": "eigh-64x5-sym.f32", "shape": [5, 5], "sha256": hashlib.sha256(data).hexdigest()}],
        "limits": "Input generation only; NumPy's float32 product is part of the recorded bytes.",
    }
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
