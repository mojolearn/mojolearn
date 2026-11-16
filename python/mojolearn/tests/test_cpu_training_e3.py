# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E batch 3, CPU training for the rf-clf and rf-reg lanes, run
only where the rf host binding is built and the package took the CPU-only
path (skipped, and SAID to be skipped, otherwise): both estimators fit twice
through the host binding on a small draw and return the same bytes,
predictions in range, and the binding reads back sabotage False. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def test_random_forests_fit_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_rf_host" not in _backend.host_families_built():
        print("SKIP: the rf host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_rf_host")
    assert not bool(module.rf_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    yc = (x[:, 3] + 0.5 * x[:, 4] > 0).astype(np.int32)
    yr = (x[:, 0] - 2.0 * x[:, 1]).astype(np.float32)
    outs = []
    for _ in range(2):
        c = mojolearn.RandomForestClassifier(n_estimators=4, max_depth=5, random_state=7).fit(x, yc)
        r = mojolearn.RandomForestRegressor(n_estimators=4, max_depth=5, random_state=7).fit(x, yr)
        outs.append((np.asarray(c.predict_proba(x)).tobytes(), np.asarray(r.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    proba = np.asarray(mojolearn.RandomForestClassifier(
        n_estimators=4, max_depth=5, random_state=7).fit(x, yc).predict_proba(x))
    assert proba.shape == (600, 2) and float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
