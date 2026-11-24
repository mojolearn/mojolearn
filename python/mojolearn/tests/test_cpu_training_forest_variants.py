# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training of the parallel_groves forest engine, through the public
estimators where the rf and trees host bindings are built and the package
took the CPU-only path (skipped, and SAID to be skipped, otherwise): the
parallel_groves engine's predictions are not the sequential engine's bytes
(the grove reduction is reached, not a sequential fallback). The two-fit
repeats of the forest variants are the verify lanes rf-clf-entropy-log2-noboot,
rf-clf-balanced-parallel, rf-reg-poisson, rf-reg-gamma-ig,
et-clf-entropy-bestfirst and et-reg-bootstrap-parallel (tools/identity_break.py),
run on the CPU side and the GPU side.

    cd python && python3 -m mojolearn.tests.test_cpu_training_forest_variants
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def test_parallel_groves_is_reached_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    built = _backend.host_families_built()
    if "_mojolearn_rf_host" not in built or "_mojolearn_trees_host" not in built:
        print("SKIP: the rf or trees host binding is not built")
        return
    import numpy as np
    for name in ("_mojolearn_rf_host", "_mojolearn_trees_host"):
        module = _backend.load_host_module(name)
        prefix = "rf" if name == "_mojolearn_rf_host" else "trees"
        assert not bool(getattr(module, prefix + "_host_sabotage")()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(1)
    x = rng.standard_normal((500, 7)).astype(np.float32)
    yc = (x[:, 2] - 0.4 * x[:, 5] > 0.3).astype(np.int32)
    yr = (x[:, 0] - 2.0 * x[:, 1]).astype(np.float32)
    kw = dict(n_estimators=4, max_depth=5, random_state=7)

    def fits(engine):
        return (
            mojolearn.RandomForestClassifier(class_weight="balanced", inference_engine=engine,
                                             **kw).fit(x, yc).predict_proba(x),
            mojolearn.ExtraTreesRegressor(bootstrap=True, max_samples=0.5, inference_engine=engine,
                                          **kw).fit(x, yr).predict(x),
        )

    groves = [np.asarray(a).tobytes() for a in fits("parallel_groves")]
    sequential = [np.asarray(a).tobytes() for a in fits("sequential")]
    assert groves[0] != sequential[0], "RF parallel_groves returned the sequential engine's bytes"
    assert groves[1] != sequential[1], "ET parallel_groves returned the sequential engine's bytes"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
