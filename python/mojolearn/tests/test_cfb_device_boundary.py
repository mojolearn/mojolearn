# SPDX-License-Identifier: Apache-2.0
"""`device=` on AgglomerativeClustering, SpectralClustering and SpectralEmbedding at the Python boundary.

The construction-level rules of every class (default None, refusals at construction and `set_params`, clone, pickle)
are in test_device_parameter.py. This file holds what is specific to these classes:

* a fit and a predict run on the estimator's device, and two estimators on two devices keep their own compiled
  modules across interleaved calls in one process;
* in `identical` mode the CPU estimator returns the bits of the GPU estimator;
* the fitted state handed to the other device's estimator in memory (what `model.check` builds) answers there, and
  the owner answers again afterwards: AgglomerativeClustering's resident reference copy stays with the estimator
  that minted it.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _check_both

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def blobs(rows=240, cols=5, centers=3, seed=7):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 8.0, size=(centers, cols)).astype(np.float32)
    x = means[np.arange(rows) % centers] + rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _agglo_single(**kw):
    return mojolearn.AgglomerativeClustering(n_clusters=3, linkage="single", prediction_data=True, **kw)


def _agglo_ward(**kw):
    return mojolearn.AgglomerativeClustering(n_clusters=3, linkage="ward", **kw)


def _spectral(**kw):
    return mojolearn.SpectralClustering(n_clusters=3, affinity="nearest_neighbors", n_neighbors=12, random_state=0,
                                        prediction_data=True, **kw)


def _embedding(**kw):
    return mojolearn.SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=12,
                                       random_state=0, **kw)


# name, constructor, fitted attributes compared, has predict
CASES = [
    ("agglo-single", _agglo_single, ("labels_", "children_"), True),
    ("agglo-ward", _agglo_ward, ("labels_", "children_"), False),
    ("spectral-clustering", _spectral, ("labels_",), True),
    ("spectral-embedding", _embedding, ("embedding_",), False),
]
case = pytest.mark.parametrize("spec", CASES, ids=[c[0] for c in CASES])


def _bits(value):
    return np.ascontiguousarray(np.asarray(value)).tobytes()


@case
def test_device_is_stored_as_passed(spec):
    assert spec[1]().get_params()["device"] is None
    assert spec[1](device="cpu").get_params()["device"] == "cpu"


@case
def test_unknown_device_is_refused_at_construction(spec):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        spec[1](device="tpu")


@case
def test_cpu_estimator_fits_and_answers(spec):
    x = blobs()
    model = spec[1](device="cpu").fit(x)
    for name in spec[2]:
        assert np.asarray(getattr(model, name)).shape[0] > 0
    if spec[3]:
        assert np.asarray(model.predict(x[:32])).shape[0] == 32


@two_devices
@case
def test_two_devices_return_the_same_bits(spec):
    x = blobs()
    gpu = spec[1](device="gpu").fit(x)
    cpu = spec[1](device="cpu").fit(x)
    for name in spec[2]:
        assert _bits(getattr(gpu, name)) == _bits(getattr(cpu, name)), name
    if spec[3]:
        # interleaved: each estimator keeps its own modules
        first = _bits(gpu.predict(x[:48]))
        assert _bits(cpu.predict(x[:48])) == first
        assert _bits(gpu.predict(x[:48])) == first


@two_devices
@case
def test_fitted_state_answers_on_the_other_device(spec):
    if not spec[3]:
        pytest.skip("no predict on this estimator")
    x = blobs()
    gpu = spec[1]().fit(x)
    before = _bits(gpu.predict(x[:48]))
    twin = _check_both.on_device(gpu, "cpu")
    assert twin is not gpu and twin.get_params()["device"] == "cpu"
    assert _bits(twin.predict(x[:48])) == before
    assert _bits(gpu.predict(x[:48])) == before      # the owner's own state was not taken or released
    del twin
    assert _bits(gpu.predict(x[:48])) == before
