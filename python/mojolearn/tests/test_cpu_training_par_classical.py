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
         "par-gpc-fit": "gp", "par-gpc-predict": "gp",
         # lane/cpu-routes-gpu-only-four (2026-09-20): the two drivers that
         # checked `_backend.vendor()` before they built a layer or prepared
         # a fold. On a CPU-only install a device index is one worker
         # PROCESS, and everything either driver does with it is its own
         # Python; the shards are the neural and gbdt host families'.
         "par-causal-lm": "neural", "par-cross-val": "gbdt"}
DRIVERS = {
    "python/mojolearn/parallel_preprocessing.py": ("scaler_fit", "scaler_transform"),
    "python/mojolearn/parallel_classical.py": ("arima_fit", "holtwinters_fit", "rbf_sampler_rows"),
    "python/mojolearn/parallel_neighbors.py": ("neighbor_state", "neighbor_query"),
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
    # the requests carry a resident-model cache header since A7 (6c1b338ed)
    assert 'process_pool_map(owner._pool, _neural_requests(owner, _cache_header(' in coordinator
    assert '"samba_update" if samba else "mlp_update"' in coordinator
    assert 'var grads = neural_ordered_sum(args)' in coordinator
    # The legacy shared worker retains these private operations for its other
    # callers; this is not evidence that its Python execution is compliant.
    worker = _read("python/mojolearn/_parallel_worker.py")
    assert "if operation in ('samba_gradient', 'samba_update'):" in worker
    assert "gradients = ordered_sum_gradients(args)" in worker
    wanted.update(("mlp_gradient", "samba_gradient"))
    # DistributedIVFIndex's disjoint shards (lane/laneless-public-classes,
    # 2026-09-19). `from_index` builds ONE non-cooperative pool and sends an
    # `ivf_store` per shard; `search` and the Euclidean root then go through
    # that SAME pool, so their sends sit in a different function from the
    # construction and are checked by name here, as par-mlp's are above. The
    # merge itself is the driver's own call, in shard order (Python until
    # lane/py-dn-ann, 2026-09-28; now `ivf_merge_shards` in every IVF binding).
    ivf = _read("python/mojolearn/parallel_ivf.py")
    assert "pool = DevicePool(devices)" in ivf and "cooperative=True" not in ivf
    assert "requests.append(('ivf_store', shard, ()))" in ivf
    assert "self._pool.map([('ivf_search_stored', None, (q,)) for _ in self.devices])" in ivf
    assert "self._pool.map([('ivf_finalize', result, (self.metric_code_,))])" in ivf
    # the merge is ONE native call since lane/py-dn-ann (2026-09-28), the
    # driver's own statement, served by the same binding on every column
    assert "self._native.ivf_merge_shards(" in ivf and "candidates.sort()" not in ivf
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
    # Since lane migrate-neural-2 the four drivers are one native call each
    # (`parallel_forecast_call`, bindings/parallel_forecast_native.mojo, on
    # the native process pool); the Python split and merge are gone and the
    # shared worker keeps `forecast_predict` for its other callers.
    forecasting = _read("python/mojolearn/parallel_forecasting.py")
    assert "_binding()._module.parallel_forecast_call" in forecasting
    assert "DevicePool" not in forecasting and "MOJOLEARN_" not in forecasting
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
    # args are (fit, q, want_proba) plus the binary out_kind when set
    assert "requests.append(('gpc_class_predict', part, args))" in gpc
    assert "args = (fit, q, want_proba, out_kind) if out_kind else (fit, q, want_proba)" in gpc
    # the class columns stay Arrays and the one-vs-rest combine is the
    # binding's (`_gpc_impl._ovr_combine`, cpu-gpu-cleanup c-gp-kernel)
    assert "arrays = list(_run(requests, devices))" in gpc
    assert "proba, codes32 = _ovr_combine(estimator._extension(), arrays, int(q.shape[0]))" in gpc
    assert "MOJOLEARN_" not in gpc
    assert "    if operation == 'gpc_class_fit':\n" in worker
    # the kernel is the fit-time working copy (`_kernel`, 2c44cf25d: the
    # constructor stores its parameters verbatim)
    assert "        return state._fit_binary(state._extension(), x, y01, *_kernel_arrays(state._kernel))\n" in worker
    assert "    if operation == 'gpc_class_predict':\n" in worker
    assert "        mean, _, probability = state._latent(ext, fit, q, want_proba)\n" in worker
    wanted.update(("gpc_class_fit", "gpc_class_predict"))
    # ParallelCausalLM (lane/cpu-routes-gpu-only-four, 2026-09-20) now enters
    # one native owner (0edc97a26): bindings/parallel_causal_native.mojo builds
    # ONE non-cooperative native process pool over the distinct layer owners,
    # configures each worker with the route the driver resolved (the worker
    # refuses anything before it is told and a route that differs from its
    # connected backend), then builds each layer only on its assigned worker,
    # in layer order. The partition is `layer_devices`, one index per layer;
    # the merge is the sequential hand-off through the embedding owner first
    # and the norm/head owner last.
    causal = _read("python/mojolearn/models/parallel_causal_lm.py")
    assert "DevicePool" not in causal and "MOJOLEARN_" not in causal
    assert 'return _native_parallel_causal("route", (layer_devices,))' in causal
    assert 'return _native_parallel_causal("init", (self, plan, weights, layer_devices, kwargs), _CAUSAL_CONTEXT)' in causal
    owner = _read("bindings/parallel_causal_native.mojo")
    # balanced attribute store (py_balanced) since 2026-10-21
    assert "py_setattr(owner, \"_pool\", process_pool_create(py_call(bi.tuple, unique), False, child_module))" in owner
    assert owner.count("process_pool_create(") == 1  # the one pool, never a cooperative one
    assert '_ = _parallel_causal_rpc(owner, unique[i], "configure", Python.list(metadata, wire_plan))' in owner
    assert 'py_call(owner._blocks.append, _parallel_causal_block(owner, i, weights["layers"][i]))' in owner
    assert 'var metadata = _parallel_causal_rpc(owner, owner.layer_devices[index], "block", Python.list(index, weights))' in owner
    assert 'raise Error("ParallelCausalLM: worker was not told its route")' in owner
    assert 'raise Error("ParallelCausalLM: worker route differs from its connected backend")' in owner
    assert 'var x = _parallel_causal_rpc(owner, owner.layer_devices[0], "embedding", ids)' in owner
    assert 'var result = _parallel_causal_rpc(owner, owner.layer_devices[len(owner.layer_devices) - 1], "final", x)' in owner
    # The legacy shared worker retains `causal_lm_layer` for its other
    # callers; this is not evidence that its Python execution is compliant.
    layer_worker = _read("python/mojolearn/_causal_lm_worker.py")
    assert "block = _block_classes(_route)[kind](weights, **kwargs)" in layer_worker
    assert "raise RuntimeError('loaded-model worker was not told its route')" in layer_worker
    wanted.add("causal_lm_layer")
    # cross_val_score's fold dispatch and its process witness. The split is
    # `_prepare_folds`/`_take_rows` -- the SERIAL API's own fold code -- and
    # the merge is `scores.extend(results)` in fold order. The device
    # inventory is the one piece with no CPU counterpart, so the host route
    # asks a DIFFERENT question through a DIFFERENT function; the GPU check
    # must still be the one the GPU route calls.
    # Since b5807df03 ("Move parallel cross-validation into native workers")
    # cross_val_score is one native call; the native owner keeps the same
    # shape: one NON-cooperative pool, a pre-fit identity wave checked
    # against the spawned PIDs, whole-fold waves, and scores consumed in
    # logical fold order. The legacy worker keeps `cross_val_fold` for its
    # other callers.
    cv = _read("python/mojolearn/parallel_model_selection.py")
    assert "DevicePool" not in cv and "cooperative=True" not in cv and "MOJOLEARN_" not in cv
    assert "_selection._NATIVE(None, 'family_msel_parallel_score'," in cv
    cv_owner = _read("bindings/parallel_cv_native.mojo")
    assert 'var pool = process_pool_create(devices, False, "mojolearn._native_parallel_cv_worker")' in cv_owner
    assert cv_owner.count("process_pool_create(") == 1
    assert '_ = py_call(requests.append, Python.tuple("identity"))' in cv_owner
    assert "native_require_worker_inventory(identities, " in cv_owner
    assert "# driver-loop: consume in logical fold order, never completion order" in cv_owner
    assert "scores.append(Float64(py=result[i][1]))" in cv_owner
    assert "    if operation == 'cross_val_fold':\n" in worker
    assert "        return _fit_score_fold(state, *args)\n" in worker
    wanted.update(("cross_val_fold", "worker_identity"))
    assert set(_parallel_pool.CPU_OPERATIONS) == wanted, sorted(_parallel_pool.CPU_OPERATIONS)
    assert set(_parallel_pool.CPU_SINGLE_DEVICE_COOPERATIVE) == {"mlp_update", "samba_update", "resample"}
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
    sent = []
    # `DevicePool.map` writes each wave's requests with `_send` and reads the
    # replies with `_receive` (dd0b63678); `_call` is no longer on its path.
    send = _parallel_pool.DevicePool._send

    def spy(worker, request):
        sent.append(request[2][0] if request[0] == "cpu_reference" else request[0])
        return send(worker, request)

    for labels, n_fits in (((X[:, 0] > 0).astype(np.int64), 1),
                           ((X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0).astype(np.int64), 3)):
        sent.clear()
        _parallel_pool.DevicePool._send = staticmethod(spy)
        try:
            par = fit_gaussian_process_classifier(
                mojolearn.GaussianProcessClassifier(kernel=kernel), X, labels, devices=(0,))
            proba = predict_gaussian_process_classifier(par, q, devices=(0,), method="predict_proba")
        finally:
            _parallel_pool.DevicePool._send = staticmethod(send)
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
    # `DevicePool.map` writes each wave's requests with `_send` and reads the
    # replies with `_receive` (dd0b63678); `_call` is no longer on its path.
    send = _parallel_pool.DevicePool._send

    def spy(worker, request):
        sent.append(request[2][0] if request[0] == "cpu_reference" else request[0])
        return send(worker, request)

    _parallel_pool.DevicePool._send = staticmethod(spy)
    try:
        par = fit_scaler(mojolearn.StandardScaler(), X, columns_per_shard=4)
        t = transform_scaler(par, X, columns_per_shard=4)
    finally:
        _parallel_pool.DevicePool._send = staticmethod(send)
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
    """`_parallel_pool.driver_read_shift`, the non-cooperative drivers' arm.

    Three properties, and the third is the one that was measured WRONG first.
    A switch that quietly returns wrong answers on ONE env var is a footgun
    (`model_selection._sabotage_fold_order`, 2026-09-17). A switch never seen
    to fire is not a switch, so the ON case is here too. And an arm that is not
    INERT AT ONE DEVICE cannot attribute a moved cell to itself rather than to
    the second device: gated on the shard INDEX, this function moved the same
    15 `par-*` cells at one device as at two (measured 2026-09-20), because a
    Python-sharded driver cuts as many shards as its `*_per_shard` argument
    asks for whatever the device count. It is gated on the shard's WORKER RANK
    (`index % len(devices)`) instead, and this is where that is enforced.
    """
    import os
    from mojolearn._parallel_pool import driver_read_shift, par_driver_sabotage
    saved = {k: os.environ.get(k) for k in
             ("MOJOLEARN_PAR_DRIVER_SABOTAGE", "MOJOLEARN_HOST_ALLOW_SABOTAGE")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert not par_driver_sabotage()
        assert [driver_read_shift(i, 4 * i, (0, 1)) for i in range(4)] == [0, 0, 0, 0], \
            "off by default"
        os.environ["MOJOLEARN_PAR_DRIVER_SABOTAGE"] = "1"
        assert not par_driver_sabotage(), "one variable is not enough"
        assert driver_read_shift(1, 4, (0, 1)) == 0
        os.environ["MOJOLEARN_HOST_ALLOW_SABOTAGE"] = "1"
        assert par_driver_sabotage()
        assert [driver_read_shift(i, 4 * i, (0, 1)) for i in range(4)] == [0, 1, 0, 1], \
            "the arm must fire on every shard whose worker rank is above 0"
        assert [driver_read_shift(i, 4 * i, (0,)) for i in range(4)] == [0, 0, 0, 0], \
            "THE CONTROL: the arm must be inert at one device, at every shard count"
        assert driver_read_shift(1, 0, (0, 1)) == 0, "a shard starting at 0 cannot shift"
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
