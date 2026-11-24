# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the gbdt-rmse lane: a runtime check that runs only where
the gbdt host binding is built and the package took the CPU-only path.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small RMSE fit runs twice through the host
binding and returns the same model text and predictions, the text carries a
`bias` record, and `leaf_estimation_iterations=2` trains twice to the same bytes. It is a
plumbing check. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e3_gbdt_rmse
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


# test_binding_dispatches_rmse_and_lifts_only_its_refusals deleted: a source-text check of
# bindings/_mojolearn_gbdt_host.mojo. 285ac89b42 moved the host binding's RUN (the RMSE
# dispatch and its refusals) into gbdt/mojolearn_gbdt_host_run.mojo. RMSE/MAE/Quantile/MAPE
# host fits and their refusals run natively: checks/native_tests/gbdt_public_fits.mojo.


def test_gradient_boosting_rmse_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gbdt_host" not in _backend.host_families_built():
        print("SKIP: the gbdt host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gbdt_host")
    assert not bool(module.gbdt_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 6)).astype(np.float32)
    y = (x[:, 3] + 0.5 * x[:, 4] + 2.0).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.GradientBoosting(n_estimators=3, max_depth=3, loss="RMSE").fit(x, y)
        outs.append((m.model_, np.asarray(m.predict(x)).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"
    assert "\nbias " in str(m.model_), "an unset boost_from_average under RMSE must seed a bias"
    assert m.bias_ != 0.0
    # b3333df510 (lane gbdt-cpu-rest-4-sym) trains any RMSE leaf iteration count on SymmetricTree on the CPU,
    # and 76661ed4d3 (gbdt-cpu-rest-4-simple) deleted the host's by-name refusal: two leaf iterations now
    # train on the host binding, twice to the same bytes, where this check expected the deleted refusal.
    twice = []
    for _ in range(2):
        m2 = mojolearn.GradientBoosting(n_estimators=2, max_depth=3, loss="RMSE",
                                        leaf_estimation_iterations=2).fit(x, y)
        twice.append((m2.model_, np.asarray(m2.predict(x)).tobytes()))
    assert twice[0] == twice[1], "two host fits at two leaf iterations returned different bytes"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
