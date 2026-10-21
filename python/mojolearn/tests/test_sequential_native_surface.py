# SPDX-License-Identifier: Apache-2.0
"""Independent description oracles for the native Sequential public owner."""
import numpy as np
import pytest

from mojolearn import Sequential
from mojolearn._verify_native_runtime import audit_native_call


def model(input_shape=None):
    out = Sequential.__new__(Sequential)
    audit_native_call(out.__init__, input_shape=input_shape).assert_native()
    return out


def call(model, name, *args, **kwargs):
    result = audit_native_call(getattr(model, name), *args, **kwargs).assert_native()
    assert result is model


def test_native_sequential_ordinary_description_matches_existing_abi():
    out = model()
    call(out, "linear", np.int64(8), bias=False)
    call(out, "relu")
    call(out, "layer_norm", 0.002)
    call(out, "dropout", 0.25)
    call(out, "residual", 0)
    call(out, "sigmoid")
    call(out, "tanh")
    call(out, "softmax")
    expected = [
        ("linear", 8, 0, 0.0), ("relu", 0, 0, 0.0),
        ("layer_norm", 0, 0, 0.002), ("dropout", 0, 0, 0.25),
        ("residual", 0, 0, 0.0), ("sigmoid", 0, 0, 0.0),
        ("tanh", 0, 0, 0.0), ("softmax", 0, 0, 0.0),
    ]
    assert audit_native_call(type(out).layers.fget, out).assert_native() == expected
    expected_program = [1, 4, 8, 1, 8, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0,
                        7, 0, 0, 0, 8, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0,
                        5, 0, 0, 0]
    assert audit_native_call(out._program, 4).assert_native() == expected_program
    assert audit_native_call(out._table).assert_native() == [0, 0, 0.002, 0.25, 0, 0, 0, 0]


def test_native_sequential_shape_layers_and_grouped_convolution_description():
    out = model((np.int32(4), 8, 8))
    call(out, "conv2d", 8, 3, stride=2, padding=1, bias=False, groups=2)
    call(out, "max_pool2d", 2)
    call(out, "avg_pool2d", 2, stride=1)
    call(out, "reshape", (8, -1))
    call(out, "permute", (1, 0))
    call(out, "flatten")
    expected_layers = [
        ("conv2d", (8, 3, 2, 1, 0, 2), 0.0),
        ("max_pool2d", (2, 2, 0, 0, 0, 0), 0.0),
        ("avg_pool2d", (2, 1, 0, 0, 0, 0), 0.0),
        ("reshape", (8, -1, 0, 0, 0, 0), 0.0),
        ("permute", (1, 0, -1, -1, -1, -1), 0.0),
        ("flatten", (0, 0, 0, 0, 0, 0), 0.0),
    ]
    assert audit_native_call(type(out).layers.fget, out).assert_native() == expected_layers
    sample = audit_native_call(out._sample_of, np.zeros((2, 4, 8, 8), "f4")).assert_native()
    assert sample == (4, 8, 8)
    expected_program = [2, 3, 4, 8, 8, 1, 1, 1, 1, 1, 6,
                        12, 8, 3, 2, 1, 0, 2, 0,
                        13, 2, 2, 0, 0, 0, 0, 0,
                        14, 2, 1, 0, 0, 0, 0, 0,
                        9, 8, -1, 0, 0, 0, 0, 0,
                        10, 1, 0, -1, -1, -1, -1, 0,
                        11, 0, 0, 0, 0, 0, 0, 0]
    assert audit_native_call(out._program, 256, sample).assert_native() == expected_program


@pytest.mark.parametrize("optimizer, f1, f2, flags, code", [
    ("sgd", 0.0, 0.0, 0, 0), ("momentum", 0.7, 0.0, 1, 0),
    ("adam", 0.8, 0.95, 0, 1), ("adamw", 0.8, 0.95, 0, 2),
])
def test_native_sequential_compile_retains_optimizer_and_schedule_options(optimizer, f1, f2, flags, code):
    out = model()
    call(out, "compile", "huber", optimizer, 0.02, momentum=0.7, nesterov=True,
         beta1=0.8, beta2=0.95, eps=0.003, weight_decay=0.004, schedule="exponential",
         step_size=3, gamma=0.5, huber_delta=0.2, tol=0.001, n_iter_no_change=4)
    assert out._options == dict(loss="huber", opt=code, flags=flags, lr=0.02,
        f1=f1, f2=f2, eps=0.003, weight_decay=0.004, sched=2, step_size=3,
        gamma=0.5, huber_delta=0.2, tol=0.001, n_iter_no_change=4)


def test_native_sequential_add_keyword_binding_and_builtin_iterators():
    out = model(iter([2, 3]))
    call(out, "add", "LINEAR", units=7, bias=False)
    call(out, "add", "reshape", shape=iter([7, -1]))
    call(out, "add", "permute", axes=iter([1, 0]))
    assert out.input_shape == (2, 3)
    assert out.layers == [("linear", 7, 0, 0.0),
                          ("reshape", (7, -1, 0, 0, 0, 0), 0.0),
                          ("permute", (1, 0, -1, -1, -1, -1), 0.0)]


@pytest.mark.parametrize("method,args,kwargs,error", [
    ("linear", (True,), {}, ValueError), ("linear", (0,), {}, ValueError),
    ("dropout", (1.0,), {}, ValueError), ("layer_norm", (0,), {}, ValueError),
    ("reshape", ([2, 0],), {}, ValueError), ("permute", ([-1, 0],), {}, ValueError),
    ("conv2d", (4, 3), {"groups": 0}, ValueError),
    ("add", ("linear", 2), {"units": 3}, TypeError),
    ("add", ("linear",), {}, TypeError), ("add", ("relu", 1), {}, TypeError),
    ("add", ("unknown",), {}, ValueError),
    ("compile", (), {"loss": "unknown"}, ValueError),
    ("compile", (), {"optimizer": "unknown"}, ValueError),
    ("compile", (), {"schedule": "unknown"}, ValueError),
])
def test_native_sequential_invalid_build_options_are_native_and_do_not_append(method, args, kwargs, error):
    out = model()
    observed = audit_native_call(getattr(out, method), *args, **kwargs)
    assert isinstance(observed.exception, error), observed.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not observed.report()[key], observed.report()
    assert out._layers == []
    assert out._options is None
