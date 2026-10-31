# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the par-* lanes whose driver shards in Python
(lane/cpu-training-par-classical, 2026-09-15): par-scaler, par-arima and
par-holtwinters; since lane/cpu-training-par-wave2 (the same day) also the
query-sharded and reference-sharded neighbor drivers, and since
lane/cpu-verifier-par-samba (wave 3, 2026-09-16) the Samba stack's gradient
shards (par-samba and par-samba-clip), the same ParallelNeuralTrainer route
par-mlp takes.

Source checks (run on a box with nothing built): every
Python-sharded driver is one native call whose owner runs its shards on the
native local pool (bindings/parallel_shard_common.mojo `shard_run`, one
worker process per device index on the CPU route); every cooperative driver
binds a device plan and refuses the CPU route by name
(bindings/parallel_plan_door.mojo `plan_devices`). Since 2026-10-24 there is
no Python pool (docs/plans/NATIVE_PARALLEL_DRIVER_20261024.md).

Runtime checks (skipped, and SAID to be skipped, when a binding is absent or
a GPU set loaded): a column-sharded scaler fit runs one worker request per
shard and equals the plain fit byte for byte; a cooperative driver refuses
by name; the driver negative control is off by default, needs both
variables and is inert at one device.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_par_classical
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend

ROOT = Path(__file__).resolve().parents[3]

#: The Python-sharded drivers: {module: (its native owner entry, the operations it calls)}.
DRIVERS = {
    "python/mojolearn/parallel_preprocessing.py": ("_native_parallel_shard", ("fit_scaler", "transform_scaler")),
    "python/mojolearn/parallel_classical.py": ("_native_parallel_shard", ("transform_rbf_sampler",)),
    "python/mojolearn/parallel_neighbors.py": ("_native_parallel_shard", ("parallel_queries",)),
    "python/mojolearn/parallel_neighbors_reference.py": ("_native_parallel_shard", ("reference_query",)),
    "python/mojolearn/parallel_ensemble.py": ("_native_parallel_shard", ("fit_forest",)),
    "python/mojolearn/parallel_gaussian_process.py": ("_native_parallel_shard",
                                                      ("fit_gaussian_process_classifier",
                                                       "predict_gaussian_process_classifier")),
    "python/mojolearn/parallel_ivf.py": ("_native_parallel_shard", ("ivf_from_index", "ivf_search")),
}


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_python_sharded_drivers_are_native_pools():
    for rel, (entry, operations) in DRIVERS.items():
        text = _read(rel)
        assert "DevicePool" not in text and "cooperative=True" not in text, rel
        for op in operations:
            assert f"{entry}('{op}'" in text, f"{op} is not one native call in {rel}"
    owner = _read("bindings/parallel_shard_native.mojo")
    common = _read("bindings/parallel_shard_common.mojo")
    forecast = _read("bindings/parallel_forecast_native.mojo")
    # one worker process per device index, never a cooperative pool, every
    # worker admitted by its own identity before it receives work
    assert "var pool = process_pool_create(devices, False, child)" in common
    assert "native_require_worker_inventory(identities, p[].vendor, pids, " in common
    assert "var pool = shard_open(devices, child)" in common
    assert owner.count("process_pool_create(") == 0 and forecast.count("shard_run(requests, devices, _CHILD)") == 1
    for op in ("transform_rbf_sampler", "fit_scaler", "transform_scaler", "fit_gaussian_process_classifier",
               "predict_gaussian_process_classifier", "fit_forest", "reference_query", "parallel_queries",
               "ivf_from_index", "ivf_search"):
        assert f'if operation == "{op}":' in owner, op
    for op in ("fit_arima", "fit_exponential_smoothing", "predict_arima", "predict_exponential_smoothing"):
        assert f'if operation == "{op}":' in forecast, op
    # a cooperative driver binds a device plan and refuses the CPU route
    door = _read("bindings/parallel_plan_door.mojo")
    assert '"NotImplementedError: no CPU implementation of the cooperative multi-GPU driver "' in door
    family = _read("bindings/family_native.mojo")
    assert "var devices = plan_devices(args[1], driver, True, context)" in family
    # ParallelNeural's public API now enters one native owner. Its native
    # owner retains the same plain gradient / cooperative update distinction;
    # complete-operation and child audits live in test_parallel_neural_native.
    trainer = _read("python/mojolearn/parallel_training.py")
    assert 'return _native_parallel_neural("init", (self, model, devices, logical_shards))' in trainer
    assert 'return _native_parallel_neural("train", (self, shards))' in trainer
    coordinator = _read("bindings/parallel_neural_native.mojo")
    assert 'operation = "mlp_gradient"' in coordinator
    assert 'operation = "samba_gradient"' in coordinator
    assert 'process_pool_create(devices, False, child_module)' in coordinator
    assert 'process_pool_create(pool["devices"], True, child_module)' in coordinator
    assert '"samba_update" if samba else "mlp_update"' in coordinator
    assert 'var grads = neural_ordered_sum(args)' in coordinator
    forecasting = _read("python/mojolearn/parallel_forecasting.py")
    assert "_binding()._module.parallel_forecast_call" in forecasting
    assert "DevicePool" not in forecasting and "MOJOLEARN_" not in forecasting
    causal = _read("python/mojolearn/models/parallel_causal_lm.py")
    assert "DevicePool" not in causal and "MOJOLEARN_" not in causal
    assert 'return _native_parallel_causal("route", (layer_devices,))' in causal
    cv = _read("python/mojolearn/parallel_model_selection.py")
    assert "DevicePool" not in cv and "cooperative=True" not in cv and "MOJOLEARN_" not in cv
    assert "_selection._NATIVE(None, 'family_msel_parallel_score'," in cv
    cv_owner = _read("bindings/parallel_cv_native.mojo")
    assert 'var pool = process_pool_create(devices, False, "mojolearn._native_parallel_cv_worker")' in cv_owner
    assert "native_require_worker_inventory(identities, " in cv_owner
    assert not (ROOT / "python/mojolearn/_parallel_worker.py").exists()
    assert "class DevicePool" not in _read("python/mojolearn/_parallel_pool.py")


def _cpu_only_with(*basenames):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    for basename in basenames:
        if not Path(_backend.host_module_path(basename)).exists():
            print(f"SKIP: {basename} is not built")
            return False
    return True


def test_cooperative_drivers_refuse_the_cpu_route_by_name():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    import numpy as np
    from mojolearn.parallel_classical import fit_logistic
    X = np.zeros((4, 2), np.float32)
    try:
        fit_logistic(mojolearn.LogisticRegression(), X, np.array([0, 1, 0, 1]), devices=(0,))
    except NotImplementedError as exc:
        assert "no CPU implementation of the cooperative multi-GPU driver glm_fit" in str(exc), str(exc)
    else:
        raise AssertionError("a cooperative driver was admitted on a CPU-only install")


def test_the_gpc_class_fit_worker_follows_the_public_fit_guard():
    """THE GPC SHARD CARRIES THE PUBLIC FIT'S GUARD (lane/unlaned-public-
    algorithms, 2026-09-20). The other admitted fit operations run their
    estimator's PUBLIC `fit` and inherit `_mode._guard_cpu_training`; the GPC
    shard calls `_fit_binary`, below that decorator, so the worker calls
    `_cpu_reference.require_training` itself. The two routes must answer
    the same way on a CPU-only install. Since dac6e4d15 (CPU training is
    public) that answer is "train": the plain fit and the driver both fit,
    outside `reference_training()`, and agree byte for byte."""
    if not _cpu_only_with("_mojolearn_gp_host"):
        return
    import numpy as np
    from mojolearn.parallel_gaussian_process import fit_gaussian_process_classifier
    X = np.ascontiguousarray(np.random.default_rng(11).standard_normal((48, 3)).astype(np.float32))
    y = (X[:, 0] > 0).astype(np.int64)
    kernel = mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0)
    plain = mojolearn.GaussianProcessClassifier(kernel=kernel).fit(X, y)
    par = fit_gaussian_process_classifier(mojolearn.GaussianProcessClassifier(kernel=kernel), X, y,
                                          devices=(0,))
    for name in ("L_", "pi_", "W_sr_"):
        assert (np.asarray(getattr(par.estimators_[0], name)).tobytes()
                == np.asarray(getattr(plain.estimators_[0], name)).tobytes()), name


@reference_training()
def test_sharded_gpc_equals_the_plain_fit_when_built():
    """The class shards are real and the merge is byte exact: one
    `gpc_class_fit` per one-vs-rest column (ONE for two classes, THREE for
    three), one `gpc_class_predict` per fitted class, and the published state
    and the merged posterior equal the plain estimator's bytes. This is the
    in-cell oracle the `par-gpc-fit` and `par-gpc-predict` lanes carry, held
    here as well so a CPU box with no record can still see it fail."""
    if not _cpu_only_with("_mojolearn_gp_host"):
        return
    import numpy as np
    from mojolearn.parallel_gaussian_process import (fit_gaussian_process_classifier,
                                                     predict_gaussian_process_classifier)
    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.standard_normal((64, 3)).astype(np.float32))
    q = np.ascontiguousarray(rng.standard_normal((16, 3)).astype(np.float32))
    kernel = mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0)
    from mojolearn._native_parallel_shard import _native_parallel_shard

    def sent():
        # every request the native pools served since the last read
        return [call["operation"] for entry in _native_parallel_shard("trace_take", ())
                for call in entry["calls"]]

    for labels, n_fits in (((X[:, 0] > 0).astype(np.int64), 1),
                           ((X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0).astype(np.int64), 3)):
        sent()
        par = fit_gaussian_process_classifier(
            mojolearn.GaussianProcessClassifier(kernel=kernel), X, labels, devices=(0,))
        proba = predict_gaussian_process_classifier(par, q, devices=(0,), method="predict_proba")
        served = sent()
        assert served == ["gpc_class_fit"] * n_fits + ["gpc_class_predict"] * n_fits, served
        plain = mojolearn.GaussianProcessClassifier(kernel=kernel).fit(X, labels)
        assert len(par.estimators_) == n_fits, len(par.estimators_)
        for i in range(n_fits):
            for name in ("L_", "pi_", "W_sr_"):
                assert (np.asarray(getattr(par.estimators_[i], name)).tobytes()
                        == np.asarray(getattr(plain.estimators_[i], name)).tobytes()), (i, name)
        assert np.asarray(proba).tobytes() == np.asarray(plain.predict_proba(q)).tobytes()


@reference_training()
def test_sharded_scaler_equals_the_plain_fit_when_built():
    if not _cpu_only_with("_mojolearn_preprocessing_host"):
        return
    import numpy as np
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    X = np.random.default_rng(5).standard_normal((40, 10)).astype(np.float32)
    from mojolearn._native_parallel_shard import _native_parallel_shard
    _native_parallel_shard("trace_take", ())
    par = fit_scaler(mojolearn.StandardScaler(), X, columns_per_shard=4)
    t = transform_scaler(par, X, columns_per_shard=4)
    sent = [call["operation"] for entry in _native_parallel_shard("trace_take", ()) for call in entry["calls"]]
    assert sent == ["scaler_fit"] * 3 + ["scaler_transform"] * 3, sent
    plain = mojolearn.StandardScaler().fit(X)
    for name in ("mean_", "var_", "scale_"):
        assert np.asarray(getattr(par, name)).tobytes() == np.asarray(getattr(plain, name)).tobytes(), name
    assert np.asarray(t).tobytes() == np.asarray(plain.transform(X)).tobytes()


@reference_training()
def test_sharded_samba_publishes_the_ordered_sum_of_its_shard_gradients_when_built():
    """par-samba's shards are real: the gradient a two-shard step publishes
    is the ordered sum of the two shard gradients, byte for byte (wave 3,
    lane/cpu-verifier-par-samba).

    Since 2e09ef628 ParallelNeuralTrainer is a native coordinator over the
    compiled process pool (bindings/parallel_neural_native.mojo), so the old
    spy on the Python `DevicePool._send/_receive` sees no request. The shard
    gradients are now taken independently: the same starting model trained
    on each shard alone (logical_shards=1 publishes that shard's gradient).

    What this does NOT check, and cannot: the fold's ORDER. Both lanes use
    two logical shards, and IEEE addition is commutative, so `g0 + g1` and
    `g1 + g0` are the same bytes; only associativity fails, and that needs
    three or more parts. Reversing the parts of `ordered_sum_gradients`
    therefore moves nothing here and is not a control (measured 2026-09-16:
    every cell stayed byte-identical under the reversal). What holds this
    test to something real is the shard's CONTRIBUTION: dropping the second
    shard from the fold makes this assertion fail and every cell of both
    lanes DIVERGENT.
    """
    if not _cpu_only_with("_mojolearn_training_host", "_mojolearn_mamba_host",
                          "_mojolearn_transformer_host"):
        return
    import numpy as np
    from mojolearn.parallel_training import ParallelNeuralTrainer

    def fresh():
        config = mojolearn.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"),
                                       n_heads=2, intermediate=64)
        return mojolearn.SambaStack(config, generator=mojolearn.training.Generator(1), lr=1e-3)

    ids = np.arange(4 * 17, dtype=np.int32).reshape(4, 17) % 256
    shards = [(ids[2 * k: 2 * k + 2, :-1], ids[2 * k: 2 * k + 2, 1:]) for k in range(2)]
    with ParallelNeuralTrainer(fresh(), devices=(0,), logical_shards=2) as trainer:
        trainer.train_step(shards)
        published = trainer.export_gradients()
    shard_gradients = []
    for shard in shards:
        with ParallelNeuralTrainer(fresh(), devices=(0,), logical_shards=1) as one:
            one.train_step([shard])
            shard_gradients.append(one.export_gradients())
    first, second = shard_gradients
    # The oracle is NOT ordered_sum_gradients: checking the published
    # gradient against the very function that produced it passes even when
    # that function is broken (measured 2026-09-16: with the fold patched to
    # drop every shard after the first, such an assertion still passed while
    # every identity cell went DIVERGENT). Two shards fold with ONE pairwise
    # elementwise add, which is plain IEEE single-precision addition, so
    # numpy computes the same bytes without calling the code under test.
    assert len(published) == len(first) == len(second), (len(published), len(first), len(second))
    contributed = False
    for i, (got, a, b) in enumerate(zip(published, first, second)):
        got, a, b = np.asarray(got), np.asarray(a), np.asarray(b)
        assert got.tobytes() == (a + b).tobytes(), f"gradient {i} is not shard 0 + shard 1"
        if got.tobytes() != a.tobytes():
            contributed = True
    # and the second shard is really in there: a fold that ignored it would
    # publish shard 0 unchanged.
    assert contributed, "the published gradient is shard 0 alone; the second shard never reached the fold"


def test_par_driver_sabotage_switch_is_off_by_default_and_inert_at_one_device():
    """The non-cooperative drivers' negative control, now native
    (bindings/parallel_shard_common.mojo `shard_read_shift`).

    Three properties, and the third is the one that was measured WRONG first.
    A switch that quietly returns wrong answers on ONE env var is a footgun
    (`model_selection._sabotage_fold_order`, 2026-09-17). A switch never seen
    to fire is not a switch, so the ON case is here too. And an arm that is not
    INERT AT ONE DEVICE cannot attribute a moved cell to itself rather than to
    the second device: gated on the shard INDEX, the arm moved the same 15
    `par-*` cells at one device as at two (measured 2026-09-20), because a
    Python-sharded driver cuts as many shards as its `*_per_shard` argument
    asks for whatever the device count. It is gated on the shard's WORKER RANK
    (`index % len(devices)`) instead. On the CPU route a device index is one
    worker process, so the two-device case runs here as two processes.
    """
    import os
    source = _read("bindings/parallel_shard_common.mojo")
    shift = source[source.index("def shard_read_shift("):source.index("struct ShardTrace")]
    assert "if first < 1 or n_devices < 2 or index % n_devices == 0:" in shift
    assert '"MOJOLEARN_PAR_DRIVER_SABOTAGE"' in shift and '"MOJOLEARN_HOST_ALLOW_SABOTAGE"' in shift
    if not _cpu_only_with("_mojolearn_preprocessing_host"):
        return
    import numpy as np
    from mojolearn.parallel_preprocessing import fit_scaler
    X = np.random.default_rng(5).standard_normal((40, 10)).astype(np.float32)
    plain = np.asarray(mojolearn.StandardScaler().fit(X).mean_).tobytes()

    def mean(devices):
        return np.asarray(fit_scaler(mojolearn.StandardScaler(), X, devices=devices,
                                     columns_per_shard=4).mean_).tobytes()

    saved = {k: os.environ.get(k) for k in
             ("MOJOLEARN_PAR_DRIVER_SABOTAGE", "MOJOLEARN_HOST_ALLOW_SABOTAGE")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert mean((0, 1)) == plain, "off by default"
        os.environ["MOJOLEARN_PAR_DRIVER_SABOTAGE"] = "1"
        assert mean((0, 1)) == plain, "one variable is not enough"
        os.environ["MOJOLEARN_HOST_ALLOW_SABOTAGE"] = "1"
        assert mean((0,)) == plain, "THE CONTROL: the arm must be inert at one device, at every shard count"
        assert mean((0, 1)) != plain, "the arm must fire on every shard whose worker rank is above 0"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
