# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Launch every accumulator specialization through the installed Python binding.

Run in separate processes with MOJOLEARN_NUMERIC_MODE=fast and identical,
after building each mode with bindings/build_trees.sh.
"""
import os

import numpy as np
from mojolearn.extratrees import ExtraTreesClassifier, ExtraTreesRegressor

rng = np.random.default_rng(2021)
x = rng.standard_normal((1537, 13), dtype=np.float32)
for classes in (1, 2, 4, 5, 8, 9, 16, 17, 32):
    y = np.arange(len(x), dtype=np.int64) % classes
    kwargs = dict(n_estimators=3, max_depth=6, random_state=2021)
    gpu = ExtraTreesClassifier(device="gpu", **kwargs).fit(x, y)
    cpu = ExtraTreesClassifier(device="cpu", **kwargs).fit(x, y)
    np.testing.assert_array_equal(gpu.predict(x), cpu.predict(x))
    np.testing.assert_array_equal(gpu.predict_proba(x), cpu.predict_proba(x))
    probabilities = gpu.predict_proba(x)
    assert probabilities.shape == (len(x), classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=2e-7, rtol=0)
    print(f"PASS Python accumulator dispatch classes={classes}")

y = x[:, 0]
gpu = ExtraTreesRegressor(device="gpu", **kwargs).fit(x, y)
cpu = ExtraTreesRegressor(device="cpu", **kwargs).fit(x, y)
np.testing.assert_allclose(gpu.predict(x), cpu.predict(x), atol=1e-4, rtol=0)
print(f"PASS Python regression smoke requested_mode={os.environ['MOJOLEARN_NUMERIC_MODE']}")
native = gpu._bind("_mojolearn_trees")
print("loaded_trees_extension", native.__file__)
compiled_mode = native.trees_numeric_mode()
expected_mode = {"fast": 0, "identical": 1, "deterministic": 2}[os.environ["MOJOLEARN_NUMERIC_MODE"]]
assert compiled_mode == expected_mode, (compiled_mode, expected_mode)
expected_mask = int(os.environ.get("MOJOLEARN_ET_EXPECT_SHARED_MASK", "14" if native.trees_vendor() == "metal" else "0"))
assert native.trees_shared_counts_mask() == expected_mask
print("compiled_numeric_mode", compiled_mode, "shared_counts_mask", native.trees_shared_counts_mask())
