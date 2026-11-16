# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the gbdt-depthwise and
gbdt-lossguide lanes: a runtime check that runs only where the gbdt host
binding is built and the package took the CPU-only path (skipped, and SAID
to be skipped, otherwise). A small Depthwise fit and a small Lossguide fit
survive a save and load. Their two-fit repeat, refusals and `ntree` records
are the native check `gbdt_public_fits`. The bit claim against the GPU
columns is the release verification's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt_trees
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def test_non_symmetric_save_load_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built():
        print("SKIP: the gbdt host binding is not built")
        return
    import os
    import tempfile
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gbdt_host")
    assert not bool(module.gbdt_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    y = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    for kwargs in (dict(max_depth=3, grow_policy="Depthwise"),
                   dict(max_leaves=8, grow_policy="Lossguide")):
        # The two-fit repeat and the model text moved to the native check
        # gbdt_public_fits_cpu (lane tmig-trees); the archive round trip stays.
        m = mojolearn.GradientBoosting(n_estimators=3, loss="Logloss", **kwargs).fit(x, y)
        expected = np.asarray(m.predict(x)).tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            m.save(path)
            back = type(m).load(path)
            assert np.asarray(back.predict(x)).tobytes() == expected, kwargs


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
