# SPDX-License-Identifier: Apache-2.0
"""Sequential, the model-description trainer over the autograd tape
(autograd/model.mojo), at the Python boundary: inputs as they are (dtypes,
orders, strides, labels), the refusals of bad descriptions and options, and
the input_shape admission. The training values (loss falls, shapes, the same
bits for the same seed, early stopping, dropout resume, rank-N models) are
the native check checks/native_tests/training_autograd_model.mojo. Runs on whichever sequence binding the
install resolves (CPU, Metal, NVIDIA or AMD); skipped when the binding
predates the model entries."""
import hashlib

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend


def _has_model_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "model_fit")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_model_entries(),
                                reason="the sequence binding has no model_fit entry (rebuild it)")


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)
    yb = (X[:, 0] + X[:, 3] > 0).astype(np.int64)
    return X, yr, y3, yb


def _digest(model):
    return hashlib.sha256(np.ascontiguousarray(model.params_).tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("build, message", [
    (lambda: ml.Sequential().relu(), "Linear"),
    (lambda: ml.Sequential().linear(4).residual(0).linear(2).residual(-1), "width"),
    (lambda: ml.Sequential().linear(4).residual(3), "earlier layer"),
])
def test_bad_descriptions(build, message):
    X, yr, _, _ = _data()
    m = build().compile(loss="mse")
    with pytest.raises(Exception, match=message):
        m.fit(X, yr, epochs=1)


def test_bad_options():
    X, yr, y3, _ = _data()
    with pytest.raises(ValueError):
        ml.Sequential().linear(1).compile(loss="nope")
    with pytest.raises(ValueError):
        ml.Sequential().linear(0)
    with pytest.raises(ValueError):
        ml.Sequential().linear(1).fit(X, yr)  # not compiled
    with pytest.raises(ValueError, match="outputs"):
        ml.Sequential().linear(2).compile(loss="mse").fit(X, yr, epochs=1)
    with pytest.raises(Exception, match="logits"):
        ml.Sequential().linear(3).softmax().compile(loss="cross_entropy").fit(X, y3, epochs=1)
    with pytest.raises(ValueError, match="classes"):
        ml.Sequential().linear(2).compile(loss="cross_entropy").fit(X, y3, epochs=1)


# ---------------------------------------------------------------- inputs as they are
# The entries take the user's array as it is and cast it to float32 in Mojo
# (core/cast_in.mojo); the cast gives the float32 values numpy's astype
# gives, so every accepted dtype and order trains to the float32 C-order bits.
def _has_gather():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "model_gather_labels")
    except Exception:
        return False


needs_gather = pytest.mark.skipif(not _has_gather(),
                                  reason="the sequence binding has no model_gather_labels entry (rebuild it)")


def _fit(loss, n_out, X, y):
    m = ml.Sequential().linear(6).tanh().linear(n_out).compile(loss=loss, optimizer="adam", lr=0.02)
    return m.fit(X, y, epochs=3, batch_size=32, seed=0)


@needs_gather
def test_classifier_inputs_as_they_are_give_the_same_bits():
    X, _, y3, _ = _data()
    ref = _fit("cross_entropy", 3, X, y3)
    ref_pred = ref.predict(X)
    cases = [
        (X.astype(np.float64), y3),
        (X, y3.astype(np.int8)),
        (np.asfortranarray(X), y3),
        (np.asfortranarray(X.astype(np.float64)), y3.astype(np.uint16)),
        (X, y3.astype(np.float32)),
    ]
    for Xin, yin in cases:
        m = _fit("cross_entropy", 3, Xin, yin)
        assert _digest(m) == _digest(ref)
        assert m.classes_.dtype == yin.dtype
        assert np.array_equal(m.classes_, np.unique(yin))
        p = m.predict(Xin)
        assert p.dtype == yin.dtype and p.shape == (120,)
        assert np.array_equal(p, ref_pred.astype(yin.dtype))
        assert np.array_equal(m.predict_proba(Xin), ref.predict_proba(X))


@needs_gather
def test_binary_bool_labels_and_integer_inputs():
    X, _, _, yb = _data()
    Xi = np.round(X * 4).astype(np.int16)
    ref = _fit("bce_with_logits", 1, Xi.astype(np.float32), yb)
    m = _fit("bce_with_logits", 1, Xi, yb.astype(bool))
    assert _digest(m) == _digest(ref)
    assert m.classes_.dtype == np.bool_ and list(m.classes_) == [False, True]
    p = m.predict(Xi)
    assert p.dtype == np.bool_
    assert np.array_equal(p, ref.predict(Xi.astype(np.float32)).astype(bool))


def test_regressor_inputs_as_they_are_give_the_same_bits():
    X, yr, _, _ = _data()
    ref = _fit("mse", 1, X, yr)
    for Xin, yin in [(X.astype(np.float64), yr.astype(np.float64)),
                     (np.asfortranarray(X), yr),
                     (X, yr.reshape(-1, 1).copy(order="F").reshape(-1))]:
        m = _fit("mse", 1, Xin, yin)
        assert _digest(m) == _digest(ref)
        assert np.array_equal(m.predict(Xin), ref.predict(X))


def test_reversed_and_float16_inputs_give_the_contiguous_float32_bits():
    X, yr, _, _ = _data()
    for Xin in (X[::-1], X[:, ::-1], X[::-1, ::-1], np.asfortranarray(X)[::-1]):
        ref = _fit("mse", 1, np.ascontiguousarray(Xin), yr)
        m = _fit("mse", 1, Xin, yr)
        assert _digest(m) == _digest(ref)
        assert np.array_equal(m.predict(Xin), ref.predict(np.ascontiguousarray(Xin)))
    ref = _fit("mse", 1, X, np.ascontiguousarray(yr[::-1]))
    m = _fit("mse", 1, X, yr[::-1])
    assert _digest(m) == _digest(ref)
    X16 = X.astype(np.float16)
    ref = _fit("mse", 1, X16.astype(np.float32), yr)
    m = _fit("mse", 1, X16, yr)
    assert _digest(m) == _digest(ref)
    assert np.array_equal(m.predict(X16), ref.predict(X16.astype(np.float32)))


@needs_gather
def test_reversed_labels_give_the_contiguous_bits():
    X, _, y3, _ = _data()
    ref = _fit("cross_entropy", 3, X, np.ascontiguousarray(y3[::-1]))
    m = _fit("cross_entropy", 3, X, y3[::-1])
    assert _digest(m) == _digest(ref)
    assert np.array_equal(m.classes_, ref.classes_)
    assert np.array_equal(m.predict(X), ref.predict(X))


def test_inputs_refused_by_name():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(1).compile(loss="mse")
    with pytest.raises(TypeError, match="numeric"):
        m.fit(X.astype(object), yr, epochs=1)
    with pytest.raises(TypeError, match="numeric"):
        m.fit(X.astype(np.complex64), yr, epochs=1)


def _has_ops():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


@pytest.mark.skipif(not _has_ops(), reason="the sequence binding has no ops_session_call entry (rebuild it)")
def test_session_tensor_takes_inputs_as_they_are():
    from mojolearn import ops
    X, _, y3, _ = _data()
    with ops.Session() as s:
        for x in (X, X.astype(np.float64), np.asfortranarray(X), np.asfortranarray(X.astype(np.float64))):
            assert np.array_equal(s.numpy(s.tensor(x)), X)
        assert np.array_equal(s.numpy(s.tensor(y3.astype(np.int8))), y3.astype(np.float32).reshape(-1, 1))
        assert np.array_equal(s.numpy(s.tensor(y3 > 0)), (y3 > 0).astype(np.float32).reshape(-1, 1))
        assert np.array_equal(s.numpy(s.tensor(2.5)), np.full((1, 1), 2.5, dtype=np.float32))
        big = np.array([2 ** 40 + 1, -(2 ** 31) - 3, 16777217], dtype=np.int64)
        assert np.array_equal(s.numpy(s.tensor(big)), big.astype(np.float32).reshape(-1, 1))
        # a strided view is cast by byte stride on the device, no host copy
        assert np.array_equal(s.numpy(s.tensor(X[:, ::2])), np.ascontiguousarray(X[:, ::2]))
        # reversed views: the span starts at the lowest corner byte
        for v in (X[::-1], X[:, ::-1], X[::-2, ::-3], np.asfortranarray(X)[::-1], X.T[::-1]):
            assert np.array_equal(s.numpy(s.tensor(v)), np.ascontiguousarray(v))
        assert np.array_equal(s.numpy(s.tensor(y3[::-1].astype(np.int64))),
                              y3[::-1].astype(np.float32).reshape(-1, 1))
        # float16: the exact bit expansion (normals, subnormals, zeros, infinities)
        h = np.array([1.5, -2.0 ** -24, 2.0 ** -14, 6.1e-5, 65504.0, -np.inf, np.inf, 0.0, -0.0, 3.14],
                     dtype=np.float16)
        assert np.array_equal(s.numpy(s.tensor(h)).view(np.uint32), h.astype(np.float32).reshape(-1, 1).view(np.uint32))
        assert np.array_equal(s.numpy(s.tensor(X.astype(np.float16)[::-1])),
                              np.ascontiguousarray(X.astype(np.float16)[::-1]).astype(np.float32))
        # a field of a structured array: strides that are not a multiple of the item
        rec = np.zeros(7, dtype=[("a", "u1"), ("b", "<f4"), ("c", "<i8")])
        rec["b"] = np.arange(7, dtype=np.float32) * 1.25 - 3.0
        rec["c"] = np.array([2 ** 40 + 1, -5, 7, 0, -(2 ** 33), 9, 16777217], dtype=np.int64)
        assert np.array_equal(s.numpy(s.tensor(rec["b"])), rec["b"].astype(np.float32).reshape(-1, 1))
        assert np.array_equal(s.numpy(s.tensor(rec["c"])), rec["c"].astype(np.float32).reshape(-1, 1))
        with pytest.raises(TypeError, match="numeric"):
            s.tensor(X.astype(object))


@pytest.mark.skipif(not _has_ops(), reason="the sequence binding has no ops_session_call entry (rebuild it)")
def test_session_embedding_takes_ids_as_they_are():
    from mojolearn import ops
    table = (np.arange(40, dtype=np.float32).reshape(10, 4) * 0.5 - 3.0)
    ids32 = np.array([3, 1, 3, 0, 9, 7], dtype=np.int32)
    with ops.Session() as s:
        w = s.tensor(table)
        ref = s.numpy(s.embedding(w, ids32))
        assert np.array_equal(ref, table[ids32])
        for ids in (ids32.astype(np.int64), ids32.astype(np.uint8), ids32.astype(np.int16),
                    ids32.astype(np.uint64), ids32.reshape(2, 3), ids32.reshape(3, 2).T.T,
                    np.asfortranarray(ids32.reshape(2, 3))):
            assert np.array_equal(s.numpy(s.embedding(w, ids)), ref)
        rev = ids32.astype(np.int64)[::-1]
        assert np.array_equal(s.numpy(s.embedding(w, rev)), table[rev])
        assert np.array_equal(s.numpy(s.embedding(w, ids32.reshape(1, 2, 3))), ref)
        with pytest.raises(Exception, match="outside"):
            s.embedding(w, np.array([1, 2 ** 40], dtype=np.int64))
        with pytest.raises(Exception, match="outside"):
            s.embedding(w, np.array([1, -(2 ** 40)], dtype=np.int64))
        with pytest.raises(TypeError, match="integers"):
            s.embedding(w, np.array([1.0, 2.0]))


def _has_rankn_model():
    try:
        m = ml.Sequential(input_shape=(1, 4, 4)).conv2d(1, 3).flatten().linear(1).compile(loss="mse")
        m.fit(np.zeros((2, 1, 4, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), epochs=1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_rankn_model(), reason="the sequence binding has no rank-N models (rebuild it)")
def test_rank4_conv_model_flat_rows_and_refusals():
    # The training values (same bits, loss, parameter count, probabilities) are
    # the native check training_autograd_model; this keeps the input_shape
    # admission: flat rows of a model with an input_shape train as the samples.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((48, 2, 7, 9)).astype(np.float32)
    y = (X[:, 0, :3].mean(axis=(1, 2)) > X[:, 1, 4:].mean(axis=(1, 2))).astype(np.int64)
    m = (ml.Sequential().conv2d(3, 3, padding=1).relu().max_pool2d(2).flatten().linear(2)
         .compile(loss="cross_entropy", optimizer="adam", lr=0.02))
    m.fit(X, y, epochs=12, batch_size=16, seed=5)
    flat = (ml.Sequential(input_shape=(2, 7, 9)).conv2d(3, 3, padding=1).relu().max_pool2d(2).flatten()
            .linear(2).compile(loss="cross_entropy", optimizer="adam", lr=0.02))
    flat.fit(X.reshape(48, -1), y, epochs=12, batch_size=16, seed=5)
    assert _digest(flat) == _digest(m)
    assert np.array_equal(flat.predict_proba(X), m.predict_proba(X))
    with pytest.raises(ValueError, match="input_shape"):
        flat.fit(np.ascontiguousarray(X[:, :, :6]), y, epochs=1)
    with pytest.raises(Exception, match="one vector per sample"):
        ml.Sequential().conv2d(2, 3).linear(1).compile(loss="mse").fit(X, y.astype(np.float32), epochs=1)


@pytest.mark.skipif(not _has_rankn_model(), reason="the sequence binding has no rank-N models (rebuild it)")
def test_rank3_sequence_model_layer_listing():
    m = ml.Sequential().linear(4).tanh().permute((1, 0)).reshape((-1,)).linear(1)
    assert m.layers[2] == ("permute", (1, 0, -1, -1, -1, -1), 0.0)
