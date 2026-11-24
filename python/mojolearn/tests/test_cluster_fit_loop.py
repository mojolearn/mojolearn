# SPDX-License-Identifier: Apache-2.0
"""`fit_loop_` at the public boundary of KMeans and MiniBatchKMeans (lane
ab-loop-mode): present after fit as a tuple (int, float), absent before fit
like the other fitted attributes, and not carried by a saved KMeans.

The iteration counts themselves are checked natively
(checks/native_tests/kmeans_native.mojo `test_fit_loop`,
checks/native_tests/kmeans_variants_native.mojo `fit_loop_case`)."""
import os
import tempfile

import numpy as np
import pytest

import mojolearn as ml


def _rows():
    return np.array([[-1, 0], [0, 0], [1, 0], [9, 0], [10, 0], [11, 0]], dtype=np.float32)


def _line():
    return np.array([[i, 0] for i in range(10)], dtype=np.float32)


def _kmeans():
    # tol must be positive (cuVS: "invalid parameter (tol<=0)"). 1e30 is above any
    # squared shift of these rows, so every restart stops at its first Lloyd step.
    return ml.KMeans(n_clusters=2, init="random", n_init=2, max_iter=3, tol=1e30, random_state=7)


def _kmeans_capped():
    # One explicit start (0, 1) on x = 0..9: the first three steps move the centers
    # by squared shifts 16, 2 and 0.5, all above tol, so the fit exhausts max_iter.
    start = np.array([[0, 0], [1, 0]], dtype=np.float32)
    return ml.KMeans(n_clusters=2, init=start, n_init=1, max_iter=3, tol=1e-4, random_state=7)


def _minibatch():
    return ml.MiniBatchKMeans(n_clusters=2, max_iter=3, batch_size=6, tol=0.0,
                              max_no_improvement=None, random_state=7)


@pytest.mark.parametrize("make", [_kmeans, _minibatch], ids=["KMeans", "MiniBatchKMeans"])
def test_fit_loop_is_a_tuple_of_int_and_float_after_fit(make):
    model = make().fit(_rows())
    loop = model.fit_loop_
    assert type(loop) is tuple and len(loop) == 2
    iterations, milliseconds = loop
    assert type(iterations) is int and type(milliseconds) is float
    assert iterations > 0 and milliseconds >= 0.0


def test_kmeans_fit_loop_sums_the_restarts():
    model = _kmeans().fit(_rows())
    # n_iter_ is the kept restart's converged step; the loop counts every
    # executed step of both restarts
    assert model.n_iter_ == 1
    assert model.fit_loop_[0] == 2 * 1


def test_kmeans_fit_loop_counts_an_exhausted_fit():
    model = _kmeans_capped().fit(_line())
    # n_iter_ is max_iter + 1 on exhaustion, as cuVS reports it; the loop
    # counts the max_iter executed steps
    assert model.n_iter_ == 4
    assert model.fit_loop_[0] == 3


def test_minibatch_fit_loop_counts_the_steps():
    model = _minibatch().fit(_rows())
    assert model.fit_loop_[0] == model.n_steps_ == 3


@pytest.mark.parametrize("make", [_kmeans, _minibatch], ids=["KMeans", "MiniBatchKMeans"])
def test_fit_loop_raises_before_fit_like_the_other_fitted_attributes(make):
    model = make()
    with pytest.raises(AttributeError):
        model.n_iter_
    with pytest.raises(AttributeError):
        model.fit_loop_


def test_loaded_kmeans_has_no_fit_loop():
    model = _kmeans().fit(_rows())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.npz")
        model.save(path)
        back = ml.KMeans.load(path)
    assert back.n_iter_ == model.n_iter_
    with pytest.raises(AttributeError):
        back.fit_loop_
