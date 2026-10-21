# SPDX-License-Identifier: Apache-2.0
"""Whole-operation import/checkpoint audits with independent byte/layout oracles."""
import hashlib
import json
import struct
from types import BuiltinFunctionType

import numpy as np
import pytest

from mojolearn import Sequential, _backend
from mojolearn._cast_in import describe
from mojolearn._verify_native_runtime import audit_native_call


def binding():
    value = _backend.binding("_mojolearn_x_sequence")
    assert isinstance(value.model_import_params, BuiltinFunctionType)
    assert value.x_sequence_vendor() == _backend.vendor()
    assert value.x_sequence_numeric_mode() == {"fast": 0, "identical": 1}[_backend.default_mode()]
    return value


def model():
    return Sequential().linear(4).tanh().linear(2)


def weights(dtype="f4"):
    return {"0.weight": (np.arange(12).reshape(4, 3) / 16 - 0.2).astype(dtype),
            "0.bias": (np.arange(4) / 32 - 0.1).astype(dtype),
            "2.weight": (np.arange(8).reshape(2, 4) / 32 - 0.3).astype(dtype),
            "2.bias": np.array([-0.125, 0.25], dtype=dtype)}


def expected_params(state):
    return np.concatenate([np.asarray(state["0.weight"]).T.ravel(), state["0.bias"],
                           np.asarray(state["2.weight"]).T.ravel(), state["2.bias"]]).astype("f4")


def raw_prediction(params, x):
    output = np.zeros((len(x), 2), "f4")
    binding().model_predict([describe(x, "X"), params.ctypes.data, output.ctypes.data],
                            [1, 3, 3, 1, 4, 1, 0, 4, 0, 0, 0, 1, 2, 1, 0],
                            [0.0, 0.0, 0.0], [len(x), 4096, 0])
    return output


def write_safetensors(path, state):
    # Independent test-only safetensors writer, with no package loader/writer.
    header, payload = {}, bytearray()
    for name, value in state.items():
        value = np.ascontiguousarray(value)
        begin = len(payload)
        payload.extend(value.tobytes())
        header[name] = {"dtype": {"f2": "F16", "f4": "F32", "f8": "F64"}[value.dtype.str[1:]],
                        "shape": list(value.shape), "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@pytest.mark.parametrize("layout", ["float16", "float64", "fortran", "strided", "lists"])
def test_native_sequential_import_preserves_torch_layout_and_dtype_cast(layout):
    state = weights("f2" if layout == "float16" else "f8" if layout == "float64" else "f4")
    if layout == "fortran":
        state = {name: np.asfortranarray(value) if value.ndim == 2 else value for name, value in state.items()}
    if layout == "strided":
        state = {name: value[..., ::-1] for name, value in state.items()}
    if layout == "lists":
        state = {name: value.tolist() for name, value in state.items()}
    expected = expected_params(state)
    out = model()
    assert audit_native_call(out.load_state_dict, state).assert_native() is out
    assert out.params_.tobytes() == expected.tobytes()
    assert out.n_features_in_ == 3 and out.n_outputs_ == 2
    assert out._imported and out._fit_state is None
    x = np.arange(15, dtype="f4").reshape(5, 3) / 10
    actual = audit_native_call(out.predict, x).assert_native()
    assert actual.tobytes() == raw_prediction(expected, x).tobytes()


@pytest.mark.parametrize("sharded", [False, True])
def test_native_sequential_safetensors_path_and_shards(sharded, tmp_path):
    state = weights("f2")
    if sharded:
        first = {name: value for name, value in state.items() if name.startswith("0.")}
        second = {name: value for name, value in state.items() if name.startswith("2.")}
        write_safetensors(tmp_path / "first.safetensors", first)
        write_safetensors(tmp_path / "second.safetensors", second)
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            name: "first.safetensors" if name in first else "second.safetensors" for name in state}}))
        source = tmp_path
    else:
        source = tmp_path / "model.safetensors"
        write_safetensors(source, state)
    out = model()
    assert audit_native_call(out.load_state_dict, source).assert_native() is out
    assert out.params_.tobytes() == expected_params(state).tobytes()


def test_native_sequential_import_names_prefix_and_warm_fit():
    state = weights()
    named = {name.replace("0.", "module.first.").replace("2.", "module.last."): value
             for name, value in state.items()}
    out = model().compile(optimizer="sgd", lr=0.03)
    audit_native_call(out.load_state_dict, named, names={0: "first", 2: "last"}, prefix="module.").assert_native()
    expected = expected_params(state)
    x = np.arange(18, dtype="f4").reshape(6, 3) / 10
    # Exact current outputs give zero MSE gradients. This checks that fit
    # starts from imported words instead of randomly reinitializing them.
    y = raw_prediction(expected, x)
    audit_native_call(out.fit, x, y, epochs=1, seed=91).assert_native()
    assert out.params_.tobytes() == expected.tobytes()


def dataset(loss):
    x = np.arange(30, dtype="f4").reshape(10, 3) / 17 - 0.5
    if loss == "mse":
        y = (x[:, 0] - x[:, 1] * 0.25).astype("f4")
    elif loss == "cross_entropy":
        y = np.array([4, 9, 20, 4, 9, 20, 4, 9, 20, 4], "u2")
    else:
        y = np.array([False, True, False, True, True, False, False, True, False, True])
    return x, y


def training_model(loss):
    return (Sequential().linear(4).tanh().linear(3 if loss == "cross_entropy" else 1)
            .compile(loss=loss, optimizer="adamw", lr=0.03, beta1=0.8, beta2=0.95,
                     eps=1e-6, weight_decay=0.002, schedule="step", step_size=2, gamma=0.7))


def state_bytes(value):
    return tuple(getattr(value, name).tobytes() for name in ("params_", "_opt_m", "_opt_v", "_fit_state")) + (
        np.asarray(value.loss_curve_, "f4").tobytes(), value.n_iter_)


def assert_native_refusal(trace, exception=ValueError):
    assert isinstance(trace.exception, exception), trace.report()
    report = trace.report()
    assert not report["python_frames"], report
    assert not report["forbidden_native_calls"], report
    assert not report["wrapper_work"], report
    assert not report["unobserved_existing_threads"], report


@pytest.mark.parametrize("loss", ["mse", "cross_entropy", "bce_with_logits"])
def test_native_sequential_checkpoint_roundtrip_resume_and_verifier(loss, tmp_path):
    from mojolearn._verify_model import verify, compare
    x, y = dataset(loss)
    straight = training_model(loss).fit(x, y, epochs=4, batch_size=4, seed=(1 << 63) + 5)
    first = training_model(loss).fit(x, y, epochs=2, batch_size=4, seed=(1 << 63) + 5)
    path = tmp_path / "model.ckpt"
    assert audit_native_call(first.save_checkpoint, path).assert_native() is None
    retained = json.loads(path.with_suffix(".ckpt.verify.json").read_text())
    assert retained["model"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert retained["install"]["vendor"] == _backend.vendor()
    assert retained["install"]["numeric_mode"] == _backend.default_mode()
    assert retained["inputs"][0]["shape"] == [16, 3]
    actual = verify(path, expect=retained)
    assert compare(actual, retained)[0] == "MATCH"
    restored = audit_native_call(Sequential.from_checkpoint, path).assert_native()
    assert state_bytes(restored) == state_bytes(first)
    if loss != "mse":
        assert restored.classes_.dtype == first.classes_.dtype
        assert restored.classes_.tobytes() == first.classes_.tobytes()
    back = tmp_path / "back.ckpt"
    audit_native_call(restored.save_checkpoint, back).assert_native()
    assert back.read_bytes() == path.read_bytes()
    audit_native_call(restored.fit, x, y, epochs=2, resume=True).assert_native()
    assert state_bytes(restored) == state_bytes(straight)
    assert restored.predict(x).tobytes() == straight.predict(x).tobytes()


@pytest.mark.parametrize("loss", ["mse", "cross_entropy", "bce_with_logits"])
def test_native_sequential_export_prediction_and_training_state_refusals(loss, tmp_path):
    x, y = dataset(loss)
    first = training_model(loss).fit(x, y, epochs=2, seed=4)
    path = tmp_path / "model.export"
    audit_native_call(first.export, path).assert_native()
    restored = audit_native_call(Sequential.load, path).assert_native()
    assert restored._fit_state is None and restored.n_iter_ == 0
    assert restored.params_.tobytes() == first.params_.tobytes()
    assert audit_native_call(restored.predict, x).assert_native().tobytes() == first.predict(x).tobytes()
    if loss != "mse":
        assert audit_native_call(restored.predict_proba, x).assert_native().tobytes() == first.predict_proba(x).tobytes()
    for operation, args, kwargs in [(restored.fit, (x, y), {"resume": True}),
                                     (restored.save_checkpoint, (tmp_path / "bad.ckpt",), {})]:
        trace = audit_native_call(operation, *args, **kwargs)
        assert_native_refusal(trace)


def test_native_sequential_checkpoint_and_import_do_not_call_python_helpers(monkeypatch, tmp_path):
    from mojolearn import _verify_model, _weight_import
    x, y = dataset("mse")
    fitted = training_model("mse").fit(x, y, epochs=1)
    imported = model()
    state = weights()
    path = tmp_path / "poison.ckpt"

    def forbidden(*args, **kwargs):
        raise AssertionError("Python helper reached in native Sequential operation")

    with monkeypatch.context() as patch:
        patch.setattr(_verify_model, "record_sequential", forbidden)
        patch.setattr(_weight_import, "WeightSource", forbidden)
        patch.setattr(_backend, "binding", forbidden)
        for name in ("empty", "zeros", "asarray", "ascontiguousarray", "array"):
            patch.setattr(np, name, forbidden)
        audit_native_call(imported.load_state_dict, state).assert_native()
        audit_native_call(fitted.save_checkpoint, path).assert_native()
        restored = audit_native_call(Sequential.load, path).assert_native()
    assert state_bytes(restored) == state_bytes(fitted)


@pytest.mark.parametrize("case", ["missing", "unused", "integer", "shape", "prefix", "names", "source"])
def test_native_sequential_weight_refusals_are_named_and_audited(case):
    state = weights()
    kwargs = {}
    expected = ValueError
    if case == "missing":
        del state["2.bias"]
    elif case == "unused":
        state["unused"] = np.ones(1, "f4")
    elif case == "integer":
        state["2.bias"] = np.ones(2, "i4")
    elif case == "shape":
        state["2.weight"] = np.ones((2, 5), "f4")
    elif case == "prefix":
        kwargs["prefix"] = 1
        expected = TypeError
    elif case == "names":
        kwargs["names"] = []
        expected = TypeError
    else:
        kwargs["source"] = "unknown"
    trace = audit_native_call(model().load_state_dict, state, **kwargs)
    assert_native_refusal(trace, expected)
    if case in ("missing", "shape"):
        assert "2." in str(trace.exception)


def test_native_sequential_checkpoint_corruption_and_shape_refusal(tmp_path):
    x, y = dataset("mse")
    fitted = training_model("mse").fit(x, y, epochs=1)
    path = tmp_path / "corrupt.ckpt"
    fitted.save_checkpoint(path)
    damaged = bytearray(path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    trace = audit_native_call(Sequential.load, path)
    assert_native_refusal(trace)
    shaped = Sequential(input_shape=(1, 3)).flatten().linear(1).compile()
    shaped.fit(x.reshape(10, 1, 3), y, epochs=1)
    trace = audit_native_call(shaped.save_checkpoint, tmp_path / "shape.ckpt")
    assert_native_refusal(trace)
    assert "2-D inputs" in str(trace.exception)
