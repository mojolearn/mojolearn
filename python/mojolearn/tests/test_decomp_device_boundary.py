# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on PCA, TruncatedSVD, IncrementalPCA and NMF, at the
Python boundary.

The device is a constructor parameter with the Ridge pilot's rule and words. None is this installation's device: the
GPU where the installation has a GPU set, else the CPU. `get_params` returns what was passed (None stays None). An
explicit `device="gpu"` on a CPU-only installation is refused in words that name `device="cpu"`; `device="cpu"`
always runs. These four classes have a field-only constructor and `set_params`, so the value is stored as given and
checked where it is given (the constructor, `set_params`) and read once, when the estimator's route is resolved,
into the one native route owner (providers/python/device_route_boundary_door.mojo); `get_params`, `set_params`,
clone and pickle carry it, and `set_params(device=...)` resolves again. No public call reads the device. On a GPU installation both images exist in one process and `device="cpu"` runs the CPU
binaries (`vendor_used() == "cpu"`); in `identical` mode it returns the bits the GPU returns. These tests run on a GPU
installation and on a CPU-only one; what needs two devices is skipped by name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import NMF, PCA, IncrementalPCA, TruncatedSVD

CLASSES = [PCA, TruncatedSVD, IncrementalPCA, NMF]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=240, cols=8, seed=5):
    # non-negative (NMF), well conditioned, more rows than columns
    return np.random.default_rng(seed).random((rows, cols), dtype=np.float32) + np.float32(0.25)


def build(cls, **params):
    params.setdefault("n_components", 3)
    if cls is NMF:
        params.setdefault("random_state", 0)
        params.setdefault("init", "random")
    return cls(**params)


def fitted_arrays(model):
    names = ("components_", "singular_values_", "explained_variance_", "explained_variance_ratio_", "mean_", "var_")
    return {name: np.asarray(getattr(model, name)) for name in names if hasattr(model, name)}


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_constructor_parameter_with_default_none(cls):
    assert cls().get_params()["device"] is None
    assert cls(device=None).get_params()["device"] is None
    assert cls(device="cpu").get_params()["device"] == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction_and_by_set_params(cls, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        build(cls, device=bad)
    model = build(cls)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.set_params(device=bad)
    assert model.get_params()["device"] is None               # refused before anything changed


@pytest.mark.parametrize("cls", CLASSES)
def test_a_carried_bad_device_is_refused_when_the_route_is_resolved(cls):
    # an instance that carries a bad value without its constructor's check (a direct assignment is not a supported
    # way to set the device) is refused by the one read of the device, at its first operation
    model = build(cls)
    model.device = "tpu"
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.fit(data())


@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_is_this_installations_device(cls):
    x = data()
    model = build(cls).fit(x)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert np.asarray(model.transform(x)).shape == (len(x), 3)


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(cls):
    x = data()
    if TWO_DEVICES:
        model = build(cls, device="gpu").fit(x)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        build(cls, device="gpu")


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_cpu_always_runs(cls):
    x = data()
    model = build(cls, device="cpu").fit(x)
    assert model.vendor_used() == "cpu"
    assert np.asarray(model.transform(x)).shape == (len(x), 3)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_returns_the_gpu_bits_in_identical_mode(cls):
    x = data()
    gpu = build(cls, device="gpu", numeric_mode="identical").fit(x)
    cpu = build(cls, device="cpu", numeric_mode="identical").fit(x)
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    left, right = fitted_arrays(gpu), fitted_arrays(cpu)
    assert left.keys() == right.keys() and "components_" in left
    for name in left:
        assert left[name].tobytes() == right[name].tobytes(), name
    assert np.asarray(gpu.transform(x)).tobytes() == np.asarray(cpu.transform(x)).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_two_estimators_on_two_devices_interleave_in_one_process(cls):
    x = data()
    gpu, cpu = build(cls, device="gpu"), build(cls, device="cpu")
    for _ in range(3):
        gpu.fit(x), cpu.fit(x)
        assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
        assert np.asarray(gpu.transform(x)).tobytes() == np.asarray(cpu.transform(x)).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_a_changed_device_resolves_again(cls):
    x = data()
    model = build(cls).fit(x)
    assert model.vendor_used() != "cpu"
    first = np.asarray(model.transform(x)).tobytes()
    model.set_params(device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert model.vendor_used() == "cpu"                       # the fitted model now answers on the CPU image
    assert np.asarray(model.transform(x)).tobytes() == first
    model.set_params(device=None)
    assert model.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_refuses_fast_in_words(cls):
    model = build(cls, device="cpu", numeric_mode="fast")
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        model.fit(data())


@pytest.mark.parametrize("cls", CLASSES)
def test_clone_and_pickle_carry_the_device(cls):
    x = data()
    model = build(cls, device="cpu").fit(x)
    clone = cls(**model.get_params())
    assert clone.get_params() == model.get_params() and clone.get_params()["device"] == "cpu"
    assert clone.fit(x).vendor_used() == "cpu"
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded.get_params()["device"] == "cpu" and loaded.vendor_used() == "cpu"
    assert np.asarray(loaded.transform(x)).tobytes() == np.asarray(model.transform(x)).tobytes()


@pytest.mark.parametrize("cls", [PCA, TruncatedSVD])
def test_a_loaded_model_runs_this_installations_device(cls, tmp_path):
    x = data()
    model = build(cls, device="cpu").fit(x)
    path = str(tmp_path / "model.npz")
    model.save(path)
    loaded = cls.load(path)
    assert loaded.device is None                               # a constructor parameter, not model state
    assert loaded.vendor_used() == mojolearn.vendor()
    assert np.asarray(loaded.transform(x)).tobytes() == np.asarray(model.transform(x)).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_check_runs_the_same_estimator_on_both_devices_in_this_process(cls, capsys):
    from mojolearn import _check_both
    x = data()
    model = build(cls, numeric_mode="identical").fit(x)
    assert _check_both.takes_device(model)
    twin = _check_both.on_device(model, "cpu")
    assert type(twin) is cls and twin is not model and twin.get_params()["device"] == "cpu"
    assert twin.components_ is model.components_               # the same fitted arrays, in memory: no file
    assert twin.vendor_used() == "cpu" and model.vendor_used() != "cpu"
    result = model.check(x)
    capsys.readouterr()
    assert result["ran"] and result["transfer"] == "device"   # no save, no host_model, no child process
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"]["vendor_used"] != "cpu" and cpu["route"]["vendor_used"] == "cpu"
    assert result["verdict"]["match"]


@pytest.mark.parametrize("cls", CLASSES)
def test_python_events_per_call_do_not_grow_with_rows(cls):
    # The route is read by pointer compares: a transform of 8,192 rows runs the same number of Python-level events as
    # a transform of 64 rows (no per-row or per-call Python work was added by the device).
    counts = []
    for rows in (64, 8192):
        x = data(rows=rows)
        model = build(cls, device="cpu").fit(x)
        model.transform(x)                                     # the route is resolved before the count
        events = [0]

        def tick(frame, event, arg, events=events):
            events[0] += 1

        sys.setprofile(tick)
        try:
            model.transform(x)
        finally:
            sys.setprofile(None)
        counts.append(events[0])
    assert counts[0] == counts[1], counts


# ---------------------------------------------------------------------------------------------------------------------
# Fit-level rows of the lane's other classes (the construction, refusal, set_params, clone and pickle rows of every
# class that takes `device=` are in test_device_parameter.py). Each row: a name, a factory taking the device keywords,
# and the outputs compared across devices after a fit (fitted arrays by name, then one query on the fitted model).
def _fit_rows():
    from mojolearn import (
        DictionaryLearning, EllipticEnvelope, FactorAnalysis, FastICA, GaussianRandomProjection, KernelPCA,
        LatentDirichletAllocation, MinCovDet, MiniBatchDictionaryLearning, MiniBatchSparsePCA, PolynomialCountSketch,
        SparseCoder, SparsePCA, SparseRandomProjection,
    )
    atoms = np.random.default_rng(11).random((5, 8), dtype=np.float32)
    transform = lambda model, x: model.transform(x)  # noqa: E731
    return [
        ("FastICA", lambda **kw: FastICA(n_components=3, random_state=0, max_iter=50, **kw),
         ("components_", "mixing_", "mean_", "whitening_"), transform),
        ("FactorAnalysis", lambda **kw: FactorAnalysis(n_components=3, max_iter=20, **kw),
         ("components_", "noise_variance_", "mean_"), transform),
        ("LatentDirichletAllocation", lambda **kw: LatentDirichletAllocation(n_components=3, max_iter=3,
                                                                              random_state=0, **kw),
         ("components_", "exp_dirichlet_component_"), transform),
        ("GaussianRandomProjection", lambda **kw: GaussianRandomProjection(n_components=3, random_state=0, **kw),
         ("components_",), transform),
        ("SparseRandomProjection", lambda **kw: SparseRandomProjection(n_components=3, random_state=0, **kw),
         ("components_",), transform),
        ("MinCovDet", lambda **kw: MinCovDet(random_state=0, **kw),
         ("location_", "covariance_", "support_", "dist_"), lambda model, x: model.mahalanobis(x)),
        ("EllipticEnvelope", lambda **kw: EllipticEnvelope(random_state=0, **kw),
         ("location_", "covariance_", "support_", "dist_"), lambda model, x: model.decision_function(x)),
        ("DictionaryLearning", lambda **kw: DictionaryLearning(n_components=3, max_iter=5, random_state=0, **kw),
         ("components_",), transform),
        ("MiniBatchDictionaryLearning", lambda **kw: MiniBatchDictionaryLearning(n_components=3, max_iter=3,
                                                                                  random_state=0, **kw),
         ("components_",), transform),
        ("SparsePCA", lambda **kw: SparsePCA(n_components=3, max_iter=5, random_state=0, **kw),
         ("components_", "mean_"), transform),
        ("MiniBatchSparsePCA", lambda **kw: MiniBatchSparsePCA(n_components=3, max_iter=3, random_state=0, **kw),
         ("components_", "mean_"), transform),
        ("SparseCoder", lambda **kw: SparseCoder(atoms, transform_n_nonzero_coefs=2, **kw), (), transform),
        ("KernelPCA", lambda **kw: KernelPCA(n_components=3, kernel="rbf", **kw),
         ("eigenvalues_", "eigenvectors_"), transform),
        ("PolynomialCountSketch", lambda **kw: PolynomialCountSketch(n_components=16, random_state=0, **kw),
         (), transform),
    ]


FIT_ROWS = _fit_rows()
FIT_IDS = [row[0] for row in FIT_ROWS]


def _fit_outputs(row, x, **device):
    _, factory, names, query = row
    model = factory(**device).fit(x)
    arrays = {name: np.asarray(getattr(model, name)).tobytes() for name in names}
    arrays["query"] = np.asarray(query(model, x)).tobytes()
    return model, arrays


@pytest.mark.parametrize("row", FIT_ROWS, ids=FIT_IDS)
def test_fit_row_default_device_is_this_installations_device(row):
    model, arrays = _fit_outputs(row, data())
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()
    assert len(arrays["query"]) > 0


@pytest.mark.parametrize("row", FIT_ROWS, ids=FIT_IDS)
def test_fit_row_explicit_cpu_always_runs(row):
    model, arrays = _fit_outputs(row, data(), device="cpu")
    assert model.get_params()["device"] == "cpu" and model.vendor_used() == "cpu"
    assert len(arrays["query"]) > 0


@two_devices
@pytest.mark.parametrize("row", FIT_ROWS, ids=FIT_IDS)
def test_fit_row_cpu_device_returns_the_gpu_bits_in_identical_mode(row):
    x = data()
    gpu, left = _fit_outputs(row, x, device="gpu", numeric_mode="identical")
    cpu, right = _fit_outputs(row, x, device="cpu", numeric_mode="identical")
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    assert left.keys() == right.keys()
    for name in left:
        assert left[name] == right[name], (row[0], name)


@two_devices
@pytest.mark.parametrize("row", FIT_ROWS, ids=FIT_IDS)
def test_fit_row_a_changed_device_resolves_again(row):
    x = data()
    model, first = _fit_outputs(row, x)
    assert model.vendor_used() != "cpu"
    model.set_params(device="cpu")
    assert model.vendor_used() == "cpu"                       # the fitted model now answers on the CPU image
    assert np.asarray(row[3](model, x)).tobytes() == first["query"]
    model.set_params(device=None)
    assert model.vendor_used() == mojolearn.vendor()
