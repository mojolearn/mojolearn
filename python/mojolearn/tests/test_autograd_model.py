# SPDX-License-Identifier: Apache-2.0
"""Sequential, the model-description trainer over the autograd tape
(autograd/model.mojo): small models (scikit-learn-test sizes) whose loss
falls, prediction shapes, the same bits for the same seed across two fits,
and the refusals of bad descriptions. Runs on whichever sequence binding the
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


def test_regression_loss_falls_and_shapes():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(8).tanh().linear(1).compile(loss="mse", optimizer="adam", lr=0.02)
    m.fit(X, yr, epochs=30, batch_size=32, seed=0)
    assert m.n_iter_ == 30 and len(m.loss_curve_) == 30
    assert m.loss_curve_[-1] < 0.5 * m.loss_curve_[0]
    assert m.predict(X).shape == (120,)
    y2 = np.stack([yr, -yr], axis=1)
    m2 = ml.Sequential().linear(4).relu().linear(2).compile(loss="huber", optimizer="momentum", lr=0.01)
    m2.fit(X, y2, epochs=10, batch_size=40, seed=1)
    assert m2.predict(X).shape == (120, 2)
    assert m2.loss_curve_[-1] < m2.loss_curve_[0]


def test_layer_norm_and_residual():
    X, yr, _, _ = _data()
    m = (ml.Sequential().linear(6).layer_norm().relu().linear(6).residual(2).linear(1)
         .compile(loss="mse", optimizer="adamw", lr=0.01, weight_decay=0.01, schedule="step", step_size=5,
                  gamma=0.5))
    m.fit(X, yr, epochs=15, batch_size=30, seed=2)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert m.predict(X).shape == (120,)


def test_classifiers():
    X, _, y3, yb = _data()
    c = ml.Sequential().linear(8).relu().linear(3).compile(loss="cross_entropy", optimizer="adam", lr=0.02)
    c.fit(X, y3, epochs=20, batch_size=32, seed=0)
    assert c.loss_curve_[-1] < c.loss_curve_[0]
    p = c.predict_proba(X)
    assert p.shape == (120, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(c.predict(X))) <= set(c.classes_)
    b = ml.Sequential().linear(4).tanh().linear(1).compile(loss="bce_with_logits", optimizer="sgd", lr=0.1)
    b.fit(X, yb, epochs=20, batch_size=20, seed=0, shuffle=False)
    assert b.loss_curve_[-1] < b.loss_curve_[0]
    assert b.predict_proba(X).shape == (120, 2)
    assert b.predict(X).shape == (120,)


def test_same_seed_same_bits():
    X, yr, y3, _ = _data()
    for loss, y, width in (("mse", yr, 1), ("cross_entropy", y3, 3)):  # two small models
        digests, curves = [], []
        for _ in range(2):  # two fits with one seed
            m = ml.Sequential().linear(6).relu().linear(width).compile(loss=loss, lr=0.01)
            m.fit(X, y, epochs=5, batch_size=16, seed=7)
            digests.append(_digest(m))
            curves.append(m.loss_curve_)
            print(f"MODEL_DIGEST {loss} {digests[-1]}")
        assert digests[0] == digests[1] and curves[0] == curves[1]


def test_early_stop():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(1).compile(loss="mse", optimizer="sgd", lr=1e-6, tol=10.0, n_iter_no_change=2)
    m.fit(X, yr, epochs=50, batch_size=60, seed=0)
    assert m.n_iter_ < 50


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


def test_dropout_trains_or_is_refused():
    X, yr, _, _ = _data()
    m = ml.Sequential().linear(8).relu().dropout(0.2).linear(1).compile(loss="mse", lr=0.01)
    try:
        m.fit(X, yr, epochs=5, batch_size=32, seed=0)
    except Exception as exc:  # a build without the tape's dropout refuses by name
        assert "Dropout is not available" in str(exc)
        return
    assert m.predict(X).shape == (120,)


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
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_open")
    except Exception:
        return False


@pytest.mark.skipif(not _has_ops(), reason="the sequence binding has no ops_open entry (rebuild it)")
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


@pytest.mark.skipif(not _has_ops(), reason="the sequence binding has no ops_open entry (rebuild it)")
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
def test_rank4_conv_model():
    # a small image classifier: conv2d -> relu -> max_pool2d -> flatten -> linear
    rng = np.random.default_rng(3)
    X = rng.standard_normal((48, 2, 7, 9)).astype(np.float32)
    y = (X[:, 0, :3].mean(axis=(1, 2)) > X[:, 1, 4:].mean(axis=(1, 2))).astype(np.int64)
    digests = []
    for _ in range(2):  # two fits with one seed
        m = (ml.Sequential().conv2d(3, 3, padding=1).relu().max_pool2d(2).flatten().linear(2)
             .compile(loss="cross_entropy", optimizer="adam", lr=0.02))
        m.fit(X, y, epochs=12, batch_size=16, seed=5)
        digests.append(_digest(m))
        print(f"RANKN_MODEL_DIGEST conv2d {digests[-1]}")
    assert digests[0] == digests[1]
    assert np.isfinite(m.params_).all() and m.loss_curve_[-1] < m.loss_curve_[0]
    # 3 x (2 x 3 x 3) weights + 3 biases, then (3 * 3 * 4) x 2 + 2
    assert m.params_.shape == (3 * 2 * 9 + 3 + 36 * 2 + 2,)
    p = m.predict_proba(X)
    assert p.shape == (48, 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    # the same samples as flat rows of a model with an input_shape
    flat = (ml.Sequential(input_shape=(2, 7, 9)).conv2d(3, 3, padding=1).relu().max_pool2d(2).flatten()
            .linear(2).compile(loss="cross_entropy", optimizer="adam", lr=0.02))
    flat.fit(X.reshape(48, -1), y, epochs=12, batch_size=16, seed=5)
    assert _digest(flat) == digests[0]
    assert np.array_equal(flat.predict_proba(X), p)
    with pytest.raises(ValueError, match="input_shape"):
        flat.fit(X[:, :, :6], y, epochs=1)
    with pytest.raises(Exception, match="one vector per sample"):
        ml.Sequential().conv2d(2, 3).linear(1).compile(loss="mse").fit(X, y.astype(np.float32), epochs=1)


@pytest.mark.skipif(not _has_rankn_model(), reason="the sequence binding has no rank-N models (rebuild it)")
def test_rank3_sequence_model_with_permute_and_reshape():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((40, 5, 6)).astype(np.float32)
    y = X[:, :, 0].sum(axis=1) - X[:, 0, :].mean(axis=1)
    m = (ml.Sequential().linear(4).tanh().permute((1, 0)).reshape((-1,)).linear(1)
         .compile(loss="mse", optimizer="adam", lr=0.02))
    m.fit(X, y, epochs=10, batch_size=8, seed=1)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert m.predict(X).shape == (40,)
    assert m.layers[2] == ("permute", (1, 0, -1, -1, -1, -1), 0.0)
