# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the arima lanes: the loaded arima host binding is not a
sabotage build (runs only where that binding is built and the package took the
CPU-only path).

The runtime arithmetic check (the three lane orders fit twice through the host
fit body and return the same bytes, `forecast(h)` equals `predict(n_obs,
n_obs + h)`, an in-sample prediction agrees with one straddling the end of
the series, and a second AR coefficient refuses by name) is native:
checks/native_tests/tsa_training_arima.mojo.

    cd python && python3 -m mojolearn.tests.test_cpu_training_arima
"""

# Gate-runner scope: the binding probe requires the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

from mojolearn import _backend


def test_host_binding_is_not_a_sabotage_build():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_arima_host" not in _backend.host_families_built():
        print("SKIP: the arima host binding is not built")
        return
    module = _backend.load_host_module("_mojolearn_arima_host")
    assert not bool(module.arima_host_sabotage()), "a sabotage build loaded outside the gate"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
