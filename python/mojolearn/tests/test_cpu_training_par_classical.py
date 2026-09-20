# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the par-* lanes whose driver shards in Python
(lane/cpu-training-par-classical, 2026-09-15): par-scaler, par-arima and
par-holtwinters; since lane/cpu-training-par-wave2 (the same day) also the
query-sharded and reference-sharded neighbor drivers, and since
lane/cpu-verifier-par-samba (wave 3, 2026-09-16) the Samba stack's gradient
shards (par-samba and par-samba-clip), the same ParallelNeuralTrainer route
par-mlp takes.

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

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, _parallel_pool, host_surface

ROOT = Path(__file__).resolve().parents[3]

LANES = {"par-scaler": "preprocessing", "par-arima": "arima", "par-holtwinters": "tsa",
         # lane/lm-attention-fallback (2026-09-19): the prediction half of the
         # same series partition, the four parallel_forecasting drivers
         "par-forecast-arima": "arima", "par-forecast-holtwinters": "tsa",
         # wave 2 (lane/cpu-training-par-wave2, 2026-09-15): the neighbor drivers
         "par-queries-knn": "core", "par-queries-radius": "core", "par-queries-kde": "estimators",
         "par-reference-knn": "core", "par-reference-knn-reg": "core",
         "par-forest": "rf", "par-forest-et": "trees", "par-mlp": "training",
         # wave 3 (lane/cpu-verifier-par-samba, 2026-09-16): the Samba stack's
         # gradient shards, the same driver as par-mlp on the same family
         "par-samba": "training", "par-samba-clip": "training",
         "par-rbf-sampler": "kernel_methods",
         # lane/unlaned-public-algorithms (2026-09-20): the GaussianProcess
         # CLASSIFIER's class-level shards, the two public entries
         # tools/verification_matrix.py reported with no identity lane at all
         "par-gpc-fit": "gp", "par-gpc-predict": "gp"}
DRIVERS = {
    "python/mojolearn/parallel_preprocessing.py": ("scaler_fit", "scaler_transform"),
    "python/mojolearn/parallel_classical.py": ("arima_fit", "holtwinters_fit", "rbf_sampler_rows"),
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
    # par-mlp's and par-samba's gradient shards: ParallelNeuralTrainer builds a
    # plain pool for mlp_gradient/samba_gradient and a cooperative one for the
    # matching update in the same __init__, so their admission is checked by
    # name here and below, not by body.
    trainer = _read("python/mojolearn/parallel_training.py")
    assert "self._operation = 'mlp_gradient'" in trainer and "self._pool = DevicePool(devices)" in trainer
    assert "self._operation = 'samba_gradient'" in trainer
    assert "self._update_pool = DevicePool(self._pool.devices, cooperative=True)" in trainer
    # the one request per logical shard, and the single update that folds them
    assert "results = self._pool.map(requests)" in trainer
    assert "self._operation.replace('_gradient', '_update')" in trainer
    # the fold is the driver's own Python, in shard order
    worker = _read("python/mojolearn/_parallel_worker.py")
    assert "if operation in ('samba_gradient', 'samba_update'):" in worker
    assert "gradients = ordered_sum_gradients(args)" in worker
    wanted.update(("mlp_gradient", "samba_gradient"))
    # DistributedIVFIndex's disjoint shards (lane/laneless-public-classes,
    # 2026-09-19). `from_index` builds ONE non-cooperative pool and sends an
    # `ivf_store` per shard; `search` and the Euclidean root then go through
    # that SAME pool, so their sends sit in a different function from the
    # construction and are checked by name here, as par-mlp's are above. The
    # merge itself is the driver's own Python, in shard order, like the
    # gradient fold.
    ivf = _read("python/mojolearn/parallel_ivf.py")
    assert "pool = DevicePool(devices)" in ivf and "cooperative=True" not in ivf
    assert "requests.append(('ivf_store', shard, ()))" in ivf
    assert "self._pool.map([('ivf_search_stored', None, (q,)) for _ in self.devices])" in ivf
    assert "self._pool.map([('ivf_finalize', result, (self.metric_code_,))])" in ivf
    assert "candidates.sort()" in ivf and "mapping[local]" in ivf
    wanted.update(("ivf_store", "ivf_search_stored", "ivf_finalize"))
    # parallel_forecasting's four drivers (lane/lm-attention-fallback,
    # 2026-09-19). `_run` builds the ONE non-cooperative pool for every
    # request the two predict drivers append, so the send sits in a different
    # function from the pool and is checked by name here, as par-mlp's and
    # DistributedIVFIndex's are. The split is `_ranges` over a range of
    # SERIES and the merge is the driver's own memcopy in `zip(ranges, parts)`
    # order: both are the driver's Python, and the worker runs the bare
    # `state.predict` with no capability call and no device-count env var, so
    # no binding restates the partition at any device count.
    forecasting = _read("python/mojolearn/parallel_forecasting.py")
    assert "pool = DevicePool(devices)" in forecasting and "cooperative=True" not in forecasting
    assert forecasting.count("requests.append(('forecast_predict', part, ('predict', (start, end))))") == 2
    assert forecasting.count("for (lo, hi), value in zip(ranges, parts):") == 2
    assert "MOJOLEARN_" not in forecasting
    assert "    if operation == 'forecast_predict':\n        method, positional = args\n" in worker
    assert "        return state.predict(*positional)\n" in worker
    wanted.add("forecast_predict")
    # parallel_gaussian_process's two class-level drivers
    # (lane/unlaned-public-algorithms, 2026-09-20). `_run` builds the ONE
    # non-cooperative pool for every request the fit and predict drivers
    # build, so the send sits in a different function from the pool and is
    # checked by name here, as par-mlp's, DistributedIVFIndex's and the
    # forecasting drivers' are. The split is one-vs-rest CLASSES, built by a
    # Python comprehension over `columns`, and the merge is the driver's own
    # Python in class order; no device-count environment variable appears, so
    # no binding restates the partition at any device count. The worker's own
    # work is the plain `_fit_binary` / `_latent`, the gp family's single
    # `gpc_fit` and `gpc_predict` host entries.
    gpc = _read("python/mojolearn/parallel_gaussian_process.py")
    assert "pool = DevicePool(devices)" in gpc and "cooperative=True" not in gpc
    assert "columns = [1] if len(classes) == 2 else range(len(classes))" in gpc
    assert "requests = [('gpc_class_fit', _fresh(estimator)," in gpc
    assert "requests.append(('gpc_class_predict', part, (fit, q, want_proba)))" in gpc
    assert "columns = [value.tolist() for value in _run(requests, devices)]" in gpc
    assert "MOJOLEARN_" not in gpc
    assert "    if operation == 'gpc_class_fit':\n" in worker
    assert "        return state._fit_binary(state._extension(), x, y01, *_kernel_arrays(state.kernel))\n" in worker
    assert "    if operation == 'gpc_class_predict':\n" in worker
    assert "        mean, _, probability = state._latent(state._extension(), fit, q, want_proba)\n" in worker
    wanted.update(("gpc_class_fit", "gpc_class_predict"))
    assert set(_parallel_pool.CPU_OPERATIONS) == wanted, sorted(_parallel_pool.CPU_OPERATIONS)
    assert set(_parallel_pool.CPU_SINGLE_DEVICE_COOPERATIVE) == {"mlp_update", "samba_update"}
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
    for devices, cooperative, op, words in (
            ((0,), True, "glm_fit", "cooperative multi-GPU driver glm_fit"),
            ((0,), False, "forest_prepare", "parallel worker operation forest_prepare"),
            ((0, 1), True, "mlp_update", "cooperative multi-GPU driver mlp_update across 2 devices"),
            ((0, 1), True, "samba_update", "cooperative multi-GPU driver samba_update across 2 devices")):
        pool = _parallel_pool.DevicePool(devices, cooperative=cooperative)
        try:
            pool.map([(op, None, None)])
        except NotImplementedError as exc:
            assert "no CPU implementation of the " + words in str(exc), str(exc)
        else:
            raise AssertionError(f"{op} was admitted on a CPU-only install")
        assert pool._workers == [], "a worker started before the refusal"


def test_the_gpc_class_fit_worker_refuses_outside_the_verifier():
    """THE HOLE ADMITTING `gpc_class_fit` OPENED, AND THE GUARD THAT CLOSES IT
    (lane/unlaned-public-algorithms, 2026-09-20).

    Every other admitted fit operation runs its estimator's PUBLIC `fit`, so
    it inherits `_mode._guard_cpu_training` and refuses on a CPU-only install
    outside `reference_training()` -- which is what
    `_parallel_pool.CPU_OPERATIONS`'s docstring promises of the whole set.
    The GPC shard calls `_fit_binary`, which sits BELOW that decorator.
    MEASURED before the guard was added: the plain
    `GaussianProcessClassifier(...).fit(X, y)` raised "fit/training is
    reserved for the internal bitwise verifier" and
    `fit_gaussian_process_classifier` on the same data, in the same process,
    trained and returned a fitted model. The two must refuse with the SAME
    words, which is what this asserts."""
    if not _cpu_only_with("_mojolearn_gp_host"):
        return
    import numpy as np
    from mojolearn.parallel_gaussian_process import fit_gaussian_process_classifier
    X = np.ascontiguousarray(np.random.default_rng(11).standard_normal((48, 3)).astype(np.float32))
    y = (X[:, 0] > 0).astype(np.int64)
    kernel = mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0)
    words = "fit/training is reserved for the internal bitwise verifier"
    try:
        mojolearn.GaussianProcessClassifier(kernel=kernel).fit(X, y)
    except NotImplementedError as exc:
        assert words in str(exc), str(exc)
    else:
        raise AssertionError("the plain CPU fit did not refuse; this box is not the CPU-only route")
    try:
        fit_gaussian_process_classifier(mojolearn.GaussianProcessClassifier(kernel=kernel), X, y,
                                        devices=(0,))
    except Exception as exc:
        assert words in str(exc), str(exc)
    else:
        raise AssertionError(
            "fit_gaussian_process_classifier TRAINED on a CPU-only install outside "
            "reference_training(); the _fit_binary shard bypassed the public fit's guard")


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
    sent = []
    call = _parallel_pool.DevicePool._call

    def spy(worker, request):
        sent.append(request[2][0] if request[0] == "cpu_reference" else request[0])
        return call(worker, request)

    for labels, n_fits in (((X[:, 0] > 0).astype(np.int64), 1),
                           ((X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0).astype(np.int64), 3)):
        sent.clear()
        _parallel_pool.DevicePool._call = staticmethod(spy)
        try:
            par = fit_gaussian_process_classifier(
                mojolearn.GaussianProcessClassifier(kernel=kernel), X, labels, devices=(0,))
            proba = predict_gaussian_process_classifier(par, q, devices=(0,), method="predict_proba")
        finally:
            _parallel_pool.DevicePool._call = staticmethod(call)
        assert sent == ["gpc_class_fit"] * n_fits + ["gpc_class_predict"] * n_fits, sent
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


@reference_training()
def test_sharded_samba_sends_two_gradients_and_one_ordered_update_when_built():
    """par-samba's shards are real: two `samba_gradient` requests per step
    from the non-cooperative pool and ONE `samba_update`, and the gradient
    the update publishes is the driver's own ordered sum of the two shard
    gradients, byte for byte (wave 3, lane/cpu-verifier-par-samba).

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
    from mojolearn.parallel_training import ParallelNeuralTrainer, ordered_sum_gradients
    config = mojolearn.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"),
                                   n_heads=2, intermediate=64)
    model = mojolearn.SambaStack(config, generator=mojolearn.training.Generator(1), lr=1e-3)
    ids = np.arange(4 * 17, dtype=np.int32).reshape(4, 17) % 256
    sent, shard_gradients = [], []
    call = _parallel_pool.DevicePool._call

    def spy(worker, request):
        operation = request[2][0] if request[0] == "cpu_reference" else request[0]
        sent.append(operation)
        result = call(worker, request)
        if operation == "samba_gradient":
            shard_gradients.append(result[1])
        return result

    _parallel_pool.DevicePool._call = staticmethod(spy)
    try:
        with ParallelNeuralTrainer(model, devices=(0,), logical_shards=2) as trainer:
            trainer.train_step([(ids[2 * k: 2 * k + 2, :-1], ids[2 * k: 2 * k + 2, 1:])
                                for k in range(2)])
            published = trainer.export_gradients()
        assert sent == ["samba_gradient", "samba_gradient", "samba_update"], sent
        assert len(shard_gradients) == 2
    finally:
        _parallel_pool.DevicePool._call = staticmethod(call)
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


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
