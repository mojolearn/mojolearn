# SPDX-License-Identifier: Apache-2.0
"""Whole-operation audits against an independent legacy numerical ABI driver."""
from types import BuiltinFunctionType

import numpy as np
import pytest

from mojolearn import Array, Sequential, _backend
from mojolearn._cast_in import describe
from mojolearn._verify_native_runtime import audit_native_call


def binding():
    value = _backend.binding("_mojolearn_x_sequence")
    assert isinstance(value.model_fit, BuiltinFunctionType)
    assert value.x_sequence_vendor() == _backend.vendor()
    assert value.x_sequence_numeric_mode() == {"fast": 0, "identical": 1}[_backend.default_mode()]
    return value


def raw_fit(x, y, width, *, loss="mse", epochs=4, validation=None, restore=False,
            patience=0, tolerance=0.0, shaped=False):
    """Test-only direct model ABI, with NumPy independently encoding labels."""
    b = binding()
    if shaped:
        program = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2,
                   11, 0, 0, 0, 0, 0, 0, 0, 1, width, 1, 0, 0, 0, 0, 0]
        table = [0.0, 0.0]
    else:
        program = [1, 3, 3, 1, 4, 1, 0, 4, 0, 0, 0, 1, width, 1, 0]
        table = [0.0, 0.0, 0.0]
    count, _ = b.model_n_params(program, table)
    params = np.zeros(count, "f4")
    moment = np.zeros(count, "f4")
    variance = np.zeros(count, "f4")
    state = np.zeros(12, "i8")
    curve = np.zeros(epochs, "f4")
    classes = None
    target = 0
    labels = y
    if loss != "mse":
        classes, labels = np.unique(y, return_inverse=True)
        labels = labels.astype("i4")
        target = 1 if loss == "cross_entropy" else 2
    matrix = x.reshape(x.shape[0], -1) if shaped else x
    seed = (1 << 63) + 19
    ip = [len(x), {"mse": 0, "cross_entropy": 1, "bce_with_logits": 2}[loss],
          target, 2, 0, 1, 2, 4, epochs, 1, seed & 0xffffffff, seed >> 32,
          patience, 1, 0]
    fp = [0.03, 0.8, 0.95, 1e-6, 0.002, 0.0, tolerance, 0.7, 1.0] + table
    addrs = [describe(matrix, "X"), describe(labels, "y"), params.ctypes.data,
             curve.ctypes.data, moment.ctypes.data, variance.ctypes.data, state.ctypes.data, 0]
    val_curve = val_info = None
    if validation is not None:
        vx, vy = validation
        val_curve = np.empty(epochs, "f4")
        val_info = np.empty(2, "i8")
        addrs += [describe(vx, "validation X"), describe(vy, "validation y"),
                  None if classes is None else describe(classes, "classes"),
                  val_curve.ctypes.data, val_info.ctypes.data]
        ip.append(int(restore))
    ran = b.model_fit(addrs, program, ip, fp)
    return dict(params=params, moment=moment, variance=variance, state=state, curve=curve,
                ran=ran, program=program, table=table, classes=classes,
                validation_curve=val_curve, validation_info=val_info)


def build(width, loss="mse", *, shaped=False, patience=0, tolerance=0.0):
    if shaped:
        out = Sequential(input_shape=(2, 2)).flatten().linear(width)
    else:
        out = Sequential().linear(4).tanh().linear(width)
    return out.compile(loss=loss, optimizer="adamw", lr=0.03, beta1=0.8, beta2=0.95,
                       eps=1e-6, weight_decay=0.002, schedule="step", step_size=2,
                       gamma=0.7, tol=tolerance, n_iter_no_change=patience)


def same_state(out, expected):
    for actual, wanted in ((out.params_, expected["params"]), (out._opt_m, expected["moment"]),
                           (out._opt_v, expected["variance"]), (out._fit_state, expected["state"])):
        assert actual.dtype == wanted.dtype
        assert actual.shape == wanted.shape
        assert actual.tobytes() == wanted.tobytes()
    assert out.n_iter_ == expected["ran"]
    np.testing.assert_array_equal(np.asarray(out.loss_curve_, "f4").view("u4"),
                                  expected["curve"][:out.n_iter_].view("u4"))


def assert_prediction(out, x, expected, probabilities=False):
    b = binding()
    n = len(x)
    classes = expected["classes"]
    if classes is None:
        transform, width = 0, 1
    elif probabilities:
        transform, width = (3, 2) if len(classes) == 2 else (1, len(classes))
    else:
        transform, width = (5, 1) if len(classes) == 2 else (4, 1)
    raw = np.zeros((n, width), "f4")
    matrix = x.reshape(n, -1) if x.ndim > 2 else x
    b.model_predict([describe(matrix, "X"), expected["params"].ctypes.data, raw.ctypes.data],
                    expected["program"], expected["table"], [n, 4096, transform])
    operation = out.predict_proba if probabilities else out.predict
    actual = audit_native_call(operation, x).assert_native()
    wanted = raw if probabilities else (raw[:, 0] if classes is None else classes[raw[:, 0].astype("i4")])
    assert actual.dtype == wanted.dtype
    assert actual.shape == wanted.shape
    assert actual.tobytes() == wanted.tobytes()


@pytest.mark.parametrize("loss,width", [("mse", 1), ("cross_entropy", 3), ("bce_with_logits", 1)])
@pytest.mark.parametrize("layout", ["strided", "fortran", "list"])
def test_native_sequential_fit_predict_and_resume_match_independent_abi(loss, width, layout):
    base = np.arange(60, dtype="f8").reshape(10, 6) / 20 - 1
    x = base[:, ::2]
    if layout == "fortran":
        x = np.asfortranarray(x)
    y = (x[:, 0] * 2 - x[:, 1]).astype("f4") if loss == "mse" else (
        np.array([-7, 4, 19, -7, 4, 19, -7, 4, 19, -7], "i8") if width == 3 else
        np.array([False, True, False, False, True, True, False, True, False, True]))
    # List conversion is independent NumPy test preparation; native product
    # code receives actual nested Python scalar containers in the list case.
    supplied_x = x.tolist() if layout == "list" else x
    supplied_y = y.tolist() if layout == "list" else y
    expected_x = np.asarray(supplied_x)
    expected_y = np.asarray(supplied_y)
    expected = raw_fit(expected_x, expected_y, width, loss=loss)
    out = build(width, loss)
    assert audit_native_call(out.fit, supplied_x, supplied_y, epochs=2, batch_size=4,
                             seed=(1 << 63) + 19).assert_native() is out
    assert audit_native_call(out.fit, supplied_x, supplied_y, epochs=2, resume=True,
                             seed=(1 << 63) + 19).assert_native() is out
    same_state(out, expected)
    if loss != "mse":
        np.testing.assert_array_equal(out.classes_, expected["classes"])
    assert_prediction(out, expected_x, expected)
    if loss != "mse":
        assert_prediction(out, expected_x, expected, probabilities=True)


@pytest.mark.parametrize("classification", [False, True])
def test_native_sequential_validation_early_stop_best_state_matches_independent_abi(classification):
    x = np.linspace(-1, 1, 30, dtype="f4").reshape(10, 3)
    y = np.arange(10, dtype="i8") % 3 if classification else x[:, 0].copy()
    loss, width = ("cross_entropy", 3) if classification else ("mse", 1)
    vx = x[::-1].copy()
    vy = y[::-1].copy()
    expected = raw_fit(x, y, width, loss=loss, epochs=8, validation=(vx, vy), restore=True,
                       patience=1, tolerance=1e6)
    out = build(width, loss, patience=1, tolerance=1e6)
    audit_native_call(out.fit, x, y, epochs=8, batch_size=4, seed=(1 << 63) + 19,
                      validation_data=(vx, vy), restore_best_weights=True).assert_native()
    same_state(out, expected)
    assert (out.epochs_run_, out.best_epoch_) == tuple(expected["validation_info"])
    np.testing.assert_array_equal(np.asarray(out.validation_loss_curve_, "f4").view("u4"),
        expected["validation_curve"][:out.epochs_run_].view("u4"))


def test_native_sequential_shape_model_and_array_input_match_existing_description():
    x = np.linspace(-1, 1, 40, dtype="f4").reshape(10, 2, 2)
    y = x[:, 0, 0].copy()
    expected = raw_fit(x, y, 1, shaped=True)
    out = build(1, shaped=True)
    supplied_x = Array.from_list(x.tolist(), "<f4")
    supplied_y = Array.from_list(y.tolist(), "<f4")
    audit_native_call(out.fit, supplied_x, supplied_y, epochs=4, batch_size=4,
                      seed=(1 << 63) + 19).assert_native()
    same_state(out, expected)
    assert_prediction(out, x, expected)


def test_native_sequential_does_not_call_backend_or_numpy_preparation(monkeypatch):
    x = np.linspace(-1, 1, 30, dtype="f4").reshape(10, 3)
    y = x[:, 0].copy()
    out = build(1)
    def poisoned(*args, **kwargs):
        raise AssertionError("Python preparation entered the native operation")
    with monkeypatch.context() as patch:
        patch.setattr(_backend, "binding", poisoned)
        for name in ("asarray", "array", "zeros", "empty", "ascontiguousarray"):
            patch.setattr(np, name, poisoned)
        audit_native_call(out.fit, x, y, epochs=2, batch_size=4).assert_native()
        predicted = audit_native_call(out.predict, x).assert_native()
    assert predicted.shape == (10,)


@pytest.mark.parametrize("case,error", [("seed", ValueError), ("resume", ValueError),
    ("validation", ValueError), ("shape", ValueError), ("labels", ValueError),
    ("dtype", TypeError), ("byteorder", TypeError), ("ragged", ValueError)])
def test_native_sequential_fit_refusals_have_no_python_runtime(case, error):
    x = np.arange(30, dtype="f4").reshape(10, 3)
    y = x[:, 0].copy()
    options = dict(epochs=1)
    out = build(1)
    if case == "seed": options["seed"] = True
    elif case == "resume": options["resume"] = True
    elif case == "validation": options["restore_best_weights"] = True
    elif case == "shape": y = y[:5]
    elif case == "labels":
        out = build(3, "cross_entropy")
        y = np.zeros(10, "i4")
    elif case == "dtype": x = x.astype("complex64")
    elif case == "byteorder": x = x.astype(">f4")
    elif case == "ragged": x = [[1, 2], [3]]
    result = audit_native_call(out.fit, x, y, **options)
    assert isinstance(result.exception, error), result.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
