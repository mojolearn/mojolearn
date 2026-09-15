# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the par-* lanes whose driver shards in Python
(lane/cpu-training-par-classical, 2026-09-15): par-scaler, par-arima and
par-holtwinters; since lane/cpu-training-par-wave2 (the same day) also the
query-sharded and reference-sharded neighbor drivers.

Source checks (run on a box with nothing built): the manifest declares each
lane on the family whose host binding serves its shard fits; the pool's
CPU_OPERATIONS are exactly the worker operations those drivers send, each
sent from a NON-cooperative pool (the driver splits and merges in Python);
no cooperative driver's operation is admitted.

Runtime checks (skipped, and SAID to be skipped, when a binding is absent or
a GPU set loaded): inside reference_training a column-sharded scaler fit
sends one worker request per shard and equals the plain fit byte for byte; a
cooperative driver and a non-cooperative operation outside the route refuse
by name before any worker starts.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_par_classical
"""
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, _parallel_pool, host_surface

ROOT = Path(__file__).resolve().parents[3]

LANES = {"par-scaler": "preprocessing", "par-arima": "arima", "par-holtwinters": "tsa",
         # wave 2 (lane/cpu-training-par-wave2, 2026-09-15): the neighbor drivers
         "par-queries-knn": "core", "par-queries-radius": "core", "par-queries-kde": "estimators",
         "par-reference-knn": "core", "par-reference-knn-reg": "core",
         "par-forest": "rf", "par-forest-et": "trees"}
DRIVERS = {
    "python/mojolearn/parallel_preprocessing.py": ("scaler_fit", "scaler_transform"),
    "python/mojolearn/parallel_classical.py": ("arima_fit", "holtwinters_fit"),
    "python/mojolearn/parallel_neighbors.py": ("neighbor_query",),
    "python/mojolearn/parallel_neighbors_reference.py": ("neighbor_reference", "neighbor_vote"),
    "python/mojolearn/parallel_ensemble.py": ("forest_fit",),
}


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_par_lanes():
    covered = host_surface.covered_lanes()
    for lane, family in LANES.items():
        assert lane in covered, f"{lane} is not a covered training lane"
        assert lane in host_surface.family(family)["training_lanes"], f"{lane} is not a {family} training lane"
        assert host_surface.TRAINING_LANE_NAMES[lane] in host_surface.training_sentence()


def test_cpu_operations_are_the_python_sharded_drivers():
    wanted = set()
    for rel, operations in DRIVERS.items():
        text = _read(rel)
        for op in operations:
            wanted.add(op)
            # Each admitted operation is sent from a function whose pool is
            # NOT cooperative: find the function body that names the op.
            bodies = [b for b in re.split(r"^def ", text, flags=re.M) if f"'{op}'" in b]
            assert bodies, f"{op} is not sent from {rel}"
            for body in bodies:
                assert "DevicePool(devices)" in body and "cooperative=True" not in body, (
                    f"{op} in {rel} is not sent from a non-cooperative pool")
    assert set(_parallel_pool.CPU_OPERATIONS) == wanted, sorted(_parallel_pool.CPU_OPERATIONS)
    cooperative = set()
    for rel in ("python/mojolearn/parallel_classical.py", "python/mojolearn/parallel_preprocessing.py"):
        for body in re.split(r"^def ", _read(rel), flags=re.M):
            if "cooperative=True" in body:
                cooperative |= set(re.findall(r"\('([a-z_]+)', ", body))
    assert cooperative and not (cooperative & set(_parallel_pool.CPU_OPERATIONS)), sorted(cooperative)


def _cpu_only_with(*basenames):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    for basename in basenames:
        if not Path(_backend.host_module_path(basename)).exists():
            print(f"SKIP: {basename} is not built")
            return False
    return True


def test_refusals_come_before_any_worker():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    for cooperative, op, words in ((True, "glm_fit", "cooperative multi-GPU driver glm_fit"),
                                   (False, "mlp_gradient", "parallel worker operation mlp_gradient")):
        pool = _parallel_pool.DevicePool((0,), cooperative=cooperative)
        try:
            pool.map([(op, None, None)])
        except NotImplementedError as exc:
            assert "no CPU implementation of the " + words in str(exc), str(exc)
        else:
            raise AssertionError(f"{op} was admitted on a CPU-only install")
        assert pool._workers == [], "a worker started before the refusal"


@reference_training()
def test_sharded_scaler_equals_the_plain_fit_when_built():
    if not _cpu_only_with("_mojolearn_preprocessing_host"):
        return
    import numpy as np
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    X = np.random.default_rng(5).standard_normal((40, 10)).astype(np.float32)
    sent = []
    call = _parallel_pool.DevicePool._call

    def spy(worker, request):
        sent.append(request[2][0] if request[0] == "cpu_reference" else request[0])
        return call(worker, request)

    _parallel_pool.DevicePool._call = staticmethod(spy)
    try:
        par = fit_scaler(mojolearn.StandardScaler(), X, columns_per_shard=4)
        t = transform_scaler(par, X, columns_per_shard=4)
    finally:
        _parallel_pool.DevicePool._call = staticmethod(call)
    assert sent == ["scaler_fit"] * 3 + ["scaler_transform"] * 3, sent
    plain = mojolearn.StandardScaler().fit(X)
    for name in ("mean_", "var_", "scale_"):
        assert np.asarray(getattr(par, name)).tobytes() == np.asarray(getattr(plain, name)).tobytes(), name
    assert np.asarray(t).tobytes() == np.asarray(plain.transform(X)).tobytes()


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
