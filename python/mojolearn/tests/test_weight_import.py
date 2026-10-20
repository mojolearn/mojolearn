# SPDX-License-Identifier: Apache-2.0
"""Weight import (autograd/weight_import.mojo): Sequential.load_state_dict
and Session.load_weights from safetensors files the test writes itself
(known values, float32 / float16 / bfloat16 tensors, no torch anywhere) and
from mappings of NumPy arrays (C order, transposed and strided views).

Checks: the imported params_ hold the expected words (the torch (out, in)
linear weight transposed to (in, out), everything else in place), and the
imported model's forward equals the forward of a reference model holding
those words, word for word; the session import equals uploading the same
words. Refusals by name: a missing tensor, a shape mismatch, an unused
tensor, a bias given to a layer without one. Prints WEIGHT_IMPORT_DIGEST
lines; run once with MOJOLEARN_VENDOR=cpu and once on the GPU and diff them
(the digests must be equal on CPU, Metal, NVIDIA and AMD)."""
import hashlib

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend, ops
from mojolearn.models.safetensors import write_safetensors


def _has_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "model_import_params") and hasattr(b, "ops_load_weights")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no weight import entries (rebuild it)")


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, *arrays):
    print(f"WEIGHT_IMPORT_DIGEST {name} {_digest(*arrays)}")


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _known(shape, salt):
    # exact in float16 and bfloat16: multiples of 1/32 in (-1, 1)
    n = int(np.prod(shape))
    k = (np.arange(n) * 7 + salt * 5) % 63 - 31
    return (k.astype(np.float32) / 32).reshape(shape)


def _st(dtype, a):
    """(safetensors dtype, shape, bytes) of float32 array `a` as `dtype`."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    if dtype == "F32":
        raw = a.tobytes()
    elif dtype == "F16":
        raw = a.astype(np.float16).tobytes()
    elif dtype == "BF16":
        raw = (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    else:
        raise AssertionError(dtype)
    return dtype, a.shape, raw


# ---------------------------------------------------------------- linear + layer norm + linear
D, H, O = 4, 8, 3


def _mlp_state():
    return {"0.weight": _known((H, D), 1), "0.bias": _known((H,), 2),
            "1.weight": _known((H,), 3) + 1, "1.bias": _known((H,), 4),
            "2.weight": _known((O, H), 5), "2.bias": _known((O,), 6)}


def _mlp():
    return ml.Sequential().linear(H).layer_norm().linear(O)


def _mlp_expected(s):
    # params_' layout: linear weight (in, out), its bias; layer norm weight, bias
    return np.concatenate([s["0.weight"].T.ravel(), s["0.bias"], s["1.weight"], s["1.bias"],
                           s["2.weight"].T.ravel(), s["2.bias"]]).astype(np.float32)


def _reference(layers_model, params, n_in, n_out, sample=None):
    # a model holding the expected words, as fit would leave it
    ref = layers_model
    ref.params_ = np.ascontiguousarray(params, dtype=np.float32)
    ref.n_features_in_ = n_in
    ref.n_outputs_ = n_out
    ref._sample_shape = sample
    return ref


def _mlp_file(tmp_path, dtypes):
    s = _mlp_state()
    path = tmp_path / "mlp.safetensors"
    write_safetensors(path, {k: _st(dtypes.get(k, "F32"), v) for k, v in s.items()})
    return path, s


@pytest.mark.parametrize("dtypes", [
    {},
    {"0.weight": "BF16", "2.weight": "BF16", "0.bias": "F16", "1.weight": "F16"},
    {k: "F16" for k in _mlp_state()},
], ids=["f32", "mixed", "f16"])
def test_mlp_safetensors(tmp_path, dtypes):
    path, s = _mlp_file(tmp_path, dtypes)
    model = _mlp().load_state_dict(path)
    expected = _mlp_expected(s)
    assert model.params_.shape == expected.shape
    assert np.array_equal(_bits(model.params_), _bits(expected))
    assert model.n_features_in_ == D and model.n_outputs_ == O
    X = np.random.default_rng(0).standard_normal((32, D)).astype(np.float32)
    pred = model.predict(X)
    ref = _reference(_mlp(), expected, D, O).predict(X)
    assert pred.shape == (32, O) and np.isfinite(pred).all()
    assert np.array_equal(_bits(pred), _bits(ref))
    again = _mlp().load_state_dict(path, source="safetensors")
    assert np.array_equal(_bits(again.params_), _bits(model.params_))
    _emit(f"mlp_{'_'.join(sorted(set(dtypes.values()))) or 'F32'}_params", model.params_)
    _emit(f"mlp_{'_'.join(sorted(set(dtypes.values()))) or 'F32'}_pred", pred)


def test_mlp_mapping_views():
    # the same weights as NumPy arrays of other layouts: a transposed view, a
    # Fortran-order copy, a strided slice and float64
    s = _mlp_state()
    views = dict(s)
    views["0.weight"] = np.ascontiguousarray(s["0.weight"].T).T
    views["2.weight"] = np.asfortranarray(s["2.weight"])
    wide = np.zeros(2 * H, dtype=np.float32)
    wide[::2] = s["0.bias"]
    views["0.bias"] = wide[::2]
    views["1.bias"] = s["1.bias"].astype(np.float64)
    model = _mlp().load_state_dict(views)
    assert np.array_equal(_bits(model.params_), _bits(_mlp_expected(s)))
    model2 = _mlp().load_state_dict(s, source="torch")
    assert np.array_equal(_bits(model2.params_), _bits(model.params_))


def test_names_and_prefix(tmp_path):
    s = _mlp_state()
    renamed = {"model.fc1.weight": s["0.weight"], "model.fc1.bias": s["0.bias"],
               "model.norm.weight": s["1.weight"], "model.norm.bias": s["1.bias"],
               "model.fc2.weight": s["2.weight"], "model.fc2.bias": s["2.bias"]}
    model = _mlp().load_state_dict(renamed, prefix="model.", names={0: "fc1", 1: "norm", 2: "fc2"})
    assert np.array_equal(_bits(model.params_), _bits(_mlp_expected(s)))


def test_fit_starts_from_imported(tmp_path):
    path, s = _mlp_file(tmp_path, {})
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, D)).astype(np.float32)
    y = rng.standard_normal((64, O)).astype(np.float32)
    a = _mlp().compile(loss="mse", optimizer="sgd", lr=0.01).load_state_dict(path)
    a = a.fit(X, y, epochs=1, batch_size=64, shuffle=False)
    b = _mlp().compile(loss="mse", optimizer="sgd", lr=0.01).load_state_dict(path)
    b = b.fit(X, y, epochs=1, batch_size=64, shuffle=False)
    assert np.array_equal(_bits(a.params_), _bits(b.params_))
    # one full-batch SGD step from the imported words, not from a fresh init
    fresh = _mlp().compile(loss="mse", optimizer="sgd", lr=0.01).fit(X, y, epochs=1, batch_size=64,
                                                                         shuffle=False)
    assert not np.array_equal(_bits(a.params_), _bits(fresh.params_))
    assert np.max(np.abs(a.params_ - _mlp_expected(s))) < 0.5
    _emit("mlp_fit_from_import", a.params_)


# ---------------------------------------------------------------- conv2d
def _cnn():
    return (ml.Sequential(input_shape=(1, 6, 6)).conv2d(4, 3).relu().flatten().linear(2))


def _cnn_state():
    return {"0.weight": _known((4, 1, 3, 3), 7), "0.bias": _known((4,), 8),
            "3.weight": _known((2, 64), 9), "3.bias": _known((2,), 10)}


def test_conv2d_safetensors(tmp_path):
    s = _cnn_state()
    path = tmp_path / "cnn.safetensors"
    write_safetensors(path, {"0.weight": _st("BF16", s["0.weight"]), "0.bias": _st("F32", s["0.bias"]),
                             "3.weight": _st("F16", s["3.weight"]), "3.bias": _st("F32", s["3.bias"])})
    model = _cnn().load_state_dict(path)
    expected = np.concatenate([s["0.weight"].ravel(), s["0.bias"], s["3.weight"].T.ravel(),
                               s["3.bias"]]).astype(np.float32)
    assert np.array_equal(_bits(model.params_), _bits(expected))
    X = np.random.default_rng(2).standard_normal((16, 1, 6, 6)).astype(np.float32)
    pred = model.predict(X)
    ref = _reference(_cnn(), expected, 36, 2, (1, 6, 6)).predict(X)
    assert pred.shape == (16, 2) and np.array_equal(_bits(pred), _bits(ref))
    _emit("cnn_params", model.params_)
    _emit("cnn_pred", pred)


def test_conv2d_needs_input_shape():
    model = ml.Sequential().conv2d(4, 3).relu().flatten().linear(2)
    with pytest.raises(ValueError, match="input_shape"):
        model.load_state_dict(_cnn_state())


# ---------------------------------------------------------------- refusals by name
def test_refuses_missing_name(tmp_path):
    s = _mlp_state()
    s["fc.weight"] = s.pop("2.weight")
    path = tmp_path / "bad.safetensors"
    write_safetensors(path, {k: _st("F32", v) for k, v in s.items()})
    with pytest.raises(ValueError, match=r"no tensor '2\.weight'"):
        _mlp().load_state_dict(path)


def test_refuses_shape_mismatch(tmp_path):
    s = _mlp_state()
    s["2.weight"] = _known((O, H - 1), 5)
    path = tmp_path / "bad.safetensors"
    write_safetensors(path, {k: _st("F32", v) for k, v in s.items()})
    with pytest.raises(Exception, match=r"'2\.weight' has shape \(3, 7\); layer 2 \(linear\) takes a weight "
                                        r"of shape \(3, 8\)"):
        _mlp().load_state_dict(path)


def test_refuses_conv_shape_mismatch():
    s = _cnn_state()
    s["0.weight"] = _known((4, 1, 5, 5), 7)
    with pytest.raises(Exception, match=r"'0\.weight' has shape \(4, 1, 5, 5\)"):
        _cnn().load_state_dict(s)


def test_refuses_unused_tensor():
    s = _mlp_state()
    s["extra.weight"] = _known((2, 2), 11)
    with pytest.raises(ValueError, match="extra.weight"):
        _mlp().load_state_dict(s)
    model = _mlp().load_state_dict(s, strict=False)
    assert np.array_equal(_bits(model.params_), _bits(_mlp_expected(_mlp_state())))


def test_refuses_integer_tensor():
    s = _mlp_state()
    s["0.bias"] = np.arange(H, dtype=np.int32)
    with pytest.raises(Exception, match=r"'0\.bias'.*float16, bfloat16, float32 or float64"):
        _mlp().load_state_dict(s)


# ---------------------------------------------------------------- session
def test_session_load_weights(tmp_path):
    path, s = _mlp_file(tmp_path, {"0.weight": "BF16", "2.bias": "F16"})
    X = np.random.default_rng(3).standard_normal((16, D)).astype(np.float32)

    def forward(sess, w0, b0, g, beta, w2, b2):
        h = sess.layer_norm(sess.linear(sess.tensor(X), w0, b0), g, beta)
        return sess.numpy(sess.linear(h, w2, b2))

    with ops.Session() as sess:
        w0 = sess.tensor(np.zeros((D, H), dtype=np.float32))
        b0 = sess.tensor(np.zeros((1, H), dtype=np.float32))
        g = sess.tensor(np.zeros((1, H), dtype=np.float32))
        beta = sess.tensor(np.zeros((1, H), dtype=np.float32))
        w2 = sess.tensor(np.zeros((H, O), dtype=np.float32))
        b2 = sess.tensor(np.zeros((1, O), dtype=np.float32))
        hs = sess.load_weights(path, [(w0, "0.weight", "transpose"), (b0, "0.bias"), (g, "1.weight"),
                                      (beta, "1.bias"), (w2, "2.weight", "transpose"), (b2, "2.bias")])
        assert hs == [w0, b0, g, beta, w2, b2]
        assert np.array_equal(_bits(sess.numpy(w0)), _bits(s["0.weight"].T))
        assert np.array_equal(_bits(sess.numpy(b2)), _bits(s["2.bias"][None, :]))
        out = forward(sess, w0, b0, g, beta, w2, b2)
    with ops.Session() as sess:
        ref = forward(sess, sess.tensor(np.ascontiguousarray(s["0.weight"].T)), sess.tensor(s["0.bias"][None, :]),
                      sess.tensor(s["1.weight"][None, :]), sess.tensor(s["1.bias"][None, :]),
                      sess.tensor(np.ascontiguousarray(s["2.weight"].T)), sess.tensor(s["2.bias"][None, :]))
    assert np.array_equal(_bits(out), _bits(ref))
    _emit("session_forward", out)


def test_session_refusals(tmp_path):
    path, _ = _mlp_file(tmp_path, {})
    with ops.Session() as sess:
        w = sess.tensor(np.zeros((D, H), dtype=np.float32))
        with pytest.raises(ValueError, match=r"no tensor 'missing\.weight'"):
            sess.load_weights(path, [(w, "missing.weight")])
        with pytest.raises(Exception, match=r"'0\.weight' has shape \(8, 4\); the tensor has shape \(4, 8\)"):
            sess.load_weights(path, [(w, "0.weight")])
        computed = sess.relu(w)
        with pytest.raises(Exception, match="computed tensor"):
            sess.load_weights(path, [(computed, "0.weight", "transpose")])
