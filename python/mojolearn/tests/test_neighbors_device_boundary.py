# SPDX-License-Identifier: Apache-2.0
"""`device=` on the neighbors estimators at the Python boundary: the exact k-NN search, classifier and regressor,
the radius search, LocalOutlierFactor, NearestCentroid and the approximate indexes (IVF flat, PQ, SQ, RaBitQ, CAGRA).

The construction-level rules of every class (default None, refusals at construction and `set_params`, clone, pickle)
are in test_device_parameter.py. This file holds what is specific to these classes:

* each estimator's fit and its queries run on the estimator's device, and two estimators on two devices keep their
  own compiled module across interleaved calls in one process;
* in `identical` mode the CPU estimator returns the bits of the GPU estimator;
* a GPU estimator keeps its fitted index on the device under a handle. An estimator on the other device that is
  handed that fitted state in memory (what `model.check` builds) reads through the handle where it must, then
  builds its own state; it never releases the owner's handle. The owner searches again afterwards, and again after
  the other estimator is collected.
"""
import gc

import numpy as np
import pytest

import mojolearn

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def blobs(rows=320, cols=6, centers=4, seed=11):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 6.0, size=(centers, cols)).astype(np.float32)
    x = means[np.arange(rows) % centers] + rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32), (np.arange(rows) % centers).astype(np.int64)


def _flat(value):
    if isinstance(value, tuple):
        return [np.asarray(item) for item in value]
    return [np.asarray(value)]


class Case:
    def __init__(self, name, build, query, supervised=False, target=None):
        self.name, self.build, self.query, self.supervised, self.target = name, build, query, supervised, target
        self.data = lambda x: x

    def __repr__(self):
        return self.name

    def fit(self, model, x, y):
        if self.supervised:
            return model.fit(self.data(x), y if self.target is None else self.target(y))
        return model.fit(self.data(x))

    def answers(self, model, x):
        return _flat(self.query(model, self.data(x)))

    def on(self, data):
        self.data = data
        return self


CASES = [
    Case("NearestNeighbors", lambda **kw: mojolearn.NearestNeighbors(n_neighbors=5, **kw),
         lambda m, x: m.kneighbors(x[:64])),
    Case("KNeighborsClassifier", lambda **kw: mojolearn.KNeighborsClassifier(n_neighbors=5, **kw),
         lambda m, x: (m.predict(x[:64]), m.predict_proba(x[:64])), supervised=True),
    Case("KNeighborsClassifier-kneighbors", lambda **kw: mojolearn.KNeighborsClassifier(n_neighbors=5, **kw),
         lambda m, x: m.kneighbors(x[:64]), supervised=True),
    Case("KNeighborsRegressor", lambda **kw: mojolearn.KNeighborsRegressor(n_neighbors=5, **kw),
         lambda m, x: m.predict(x[:64]), supervised=True, target=lambda y: y.astype(np.float32)),
    Case("LocalOutlierFactor", lambda **kw: mojolearn.LocalOutlierFactor(n_neighbors=10, novelty=True, **kw),
         lambda m, x: m.score_samples(x[:64])),
    Case("NearestCentroid", lambda **kw: mojolearn.NearestCentroid(**kw),
         lambda m, x: m.predict(x[:64]), supervised=True),
    Case("IVFIndex", lambda **kw: mojolearn.IVFIndex(8, 4, n_neighbors=5, **kw), lambda m, x: m.search(x[:64])),
    Case("IVFPQIndex", lambda **kw: mojolearn.IVFPQIndex(8, 4, pq_dim=3, n_neighbors=5, **kw),
         lambda m, x: m.search(x[:64])),
    Case("IVFSQIndex", lambda **kw: mojolearn.IVFSQIndex(8, 4, n_neighbors=5, **kw), lambda m, x: m.search(x[:64])),
    Case("IVFRaBitQIndex", lambda **kw: mojolearn.IVFRaBitQIndex(8, 4, n_neighbors=5, **kw),
         lambda m, x: m.search(x[:64])),
    Case("CagraIndex", lambda **kw: mojolearn.CagraIndex(graph_degree=8, intermediate_graph_degree=16,
                                                          n_neighbors=5, **kw), lambda m, x: m.search(x[:64])),
    # the kernel-feature, semi-supervised and imputation classes of the same binary (PageRank and Louvain take a
    # graph; their construction-level rules are test_device_parameter.py's)
    Case("PolynomialCountSketch", lambda **kw: mojolearn.PolynomialCountSketch(n_components=16, random_state=0, **kw),
         lambda m, x: m.transform(x[:64])),
    Case("AdditiveChi2Sampler", lambda **kw: mojolearn.AdditiveChi2Sampler(**kw),
         lambda m, x: m.transform(x[:64])).on(np.abs),
    Case("SkewedChi2Sampler", lambda **kw: mojolearn.SkewedChi2Sampler(n_components=16, random_state=0, **kw),
         lambda m, x: m.transform(x[:64])).on(np.abs),
    Case("LabelPropagation", lambda **kw: mojolearn.LabelPropagation(kernel="knn", n_neighbors=7, **kw),
         lambda m, x: m.predict(x[:64]), supervised=True),
    Case("LabelSpreading", lambda **kw: mojolearn.LabelSpreading(kernel="knn", n_neighbors=7, **kw),
         lambda m, x: m.predict(x[:64]), supervised=True),
    Case("KNNImputer", lambda **kw: mojolearn.KNNImputer(n_neighbors=3, **kw), lambda m, x: m.transform(x[:64])),
]
ids = [case.name for case in CASES]


def test_radius_neighbors_takes_device_and_queries_on_it():
    x, _ = blobs()
    model = mojolearn.RadiusNeighbors(radius=3.0, device="cpu").fit(x)
    assert model.get_params()["device"] == "cpu" and model.vendor_used() == "cpu"
    rows = model.radius_neighbors(x[:16], sort_results=True)
    assert len(rows[0]) == 16
    if TWO_DEVICES:
        other = mojolearn.RadiusNeighbors(radius=3.0).fit(x).radius_neighbors(x[:16], sort_results=True)
        for mine, theirs in zip(rows[1], other[1]):
            assert np.asarray(mine).tobytes() == np.asarray(theirs).tobytes()


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_the_class_declares_device_and_runs_on_the_cpu_device(case):
    x, y = blobs()
    cls = type(case.build())
    assert cls._mojolearn_device is True and "device" in cls._get_param_names()
    model = case.fit(case.build(device="cpu"), x, y)
    assert model.vendor_used() == "cpu"
    assert case.answers(model, x)[0].shape[0] == 64


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_two_devices_keep_their_own_module_across_interleaved_fits_and_queries(case):
    x, y = blobs()
    cpu = case.fit(case.build(device="cpu"), x, y)
    gpu = case.fit(case.build(), x, y)
    for _ in range(2):  # fit and query doors share one route per estimator: no call changes the other's device
        assert gpu.vendor_used() == mojolearn.vendor() != "cpu" and cpu.vendor_used() == "cpu"
        assert case.answers(gpu, x)[0].shape == case.answers(cpu, x)[0].shape
        case.fit(cpu, x, y)


# The CPU-equals-GPU bits of the other classes are the two-sided verify lanes'
# (knn, knn-clf, psd-queries-knn, knn-reg, x-neighbors-lof, x-neighbors-nearest-centroid,
# ivf, x-ann-*, x-neighbors-poly-sketch, x-neighbors-*-chi2, x-neighbors-knn-imputer).
# No lane digests predict of the kernel="knn" label propagation and spreading.
_IDENTICAL_CASES = [case for case in CASES if case.name in ("LabelPropagation", "LabelSpreading")]


@two_devices
@pytest.mark.parametrize("case", _IDENTICAL_CASES, ids=[case.name for case in _IDENTICAL_CASES])
def test_identical_mode_cpu_device_returns_the_gpu_bits(case):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = blobs()
    gpu = case.fit(case.build(numeric_mode="identical"), x, y)
    cpu = case.fit(case.build(numeric_mode="identical", device="cpu"), x, y)
    for got, expected in zip(case.answers(cpu, x), case.answers(gpu, x)):
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_the_other_devices_estimator_borrows_the_fitted_index_and_never_releases_it(case):
    from mojolearn import _check_both
    x, y = blobs()
    gpu = case.fit(case.build(numeric_mode="identical"), x, y)
    before = case.answers(gpu, x)                         # the device index is resident now
    twin = _check_both.on_device(gpu, "cpu")             # the same fitted attributes, in memory, on the CPU
    assert twin is not gpu and twin.vendor_used() == "cpu"
    for got, expected in zip(case.answers(twin, x), before):
        assert got.tobytes() == expected.tobytes()
    for got, expected in zip(case.answers(gpu, x), before):   # the owner's handle is still its own
        assert got.tobytes() == expected.tobytes()
    del twin
    gc.collect()                                          # the CPU estimator's finalizer released nothing of the owner's
    for got, expected in zip(case.answers(gpu, x), before):
        assert got.tobytes() == expected.tobytes()


@two_devices
def test_a_cloned_ivf_index_runs_on_its_sources_device():
    x, _ = blobs()
    model = mojolearn.IVFIndex(8, 4, n_neighbors=5, device="cpu").fit(x)
    clone = model._clone()
    assert clone.get_params()["device"] == "cpu" and clone.vendor_used() == "cpu"
