# SPDX-License-Identifier: Apache-2.0
"""A forest fit on a Python-list y costs about what it costs on a NumPy y.

The parallel drivers ship labels as the flat Python list `flatten_labels`
returns. Admission re-validated every connected builtins export (and looked
up the pandas classes) once PER LABEL, so a 20000-label list added ~35 s to
each shard on the L4 (`tools/repro_par_forest.py --direct --inputs pool`:
47 s against 10.8 s with NumPy inputs). One run per arm, after a warm-up.
"""
import time

import numpy as np

import mojolearn as ml
from mojolearn._buffer import as_f32_c
from mojolearn._labels import flatten_labels


def _fit_seconds(X, y):
    model = ml.RandomForestClassifier(n_estimators=2, max_depth=4, random_state=7, numeric_mode="identical")
    started = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - started, model


def test_list_labels_fit_within_twice_the_numpy_fit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 16)).astype(np.float32)
    y = (X[:, 3] > 0).astype(np.int64)
    _fit_seconds(X, y)  # warm-up: binding load and first launch
    numpy_seconds, numpy_model = _fit_seconds(X, y)
    xs, _ = as_f32_c(X, ndim=2, name="X")
    list_seconds, list_model = _fit_seconds(xs, flatten_labels(y))
    assert np.asarray(list_model.predict_proba(X)).tobytes() == np.asarray(numpy_model.predict_proba(X)).tobytes()
    assert list_seconds <= 2 * numpy_seconds + 0.5, (list_seconds, numpy_seconds)
