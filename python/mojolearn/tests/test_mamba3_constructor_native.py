# SPDX-License-Identifier: Apache-2.0
"""Mamba-3 native constructor metadata and independent parameter-byte oracle."""
import numpy as np
import pytest

from mojolearn import Mamba3Block
from mojolearn.neural_inference import Mamba3BlockInference
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_causal_model_native_loading import _poison
from mojolearn.tests.test_mamba23_backward_surface import host_fixture


@pytest.mark.parametrize("cls", [Mamba3Block, Mamba3BlockInference])
def test_mamba3_construction_has_no_package_callbacks(monkeypatch, cls):
    weights, _ = host_fixture(3)
    original = {name: value.tobytes() for name, value in weights.items()}
    instance = cls.__new__(cls)
    constructor = cls.__init__
    with monkeypatch.context() as patch:
        _poison(patch)
        audit_native_call(constructor, instance, weights).assert_native()
    assert (instance.d_model, instance.d_inner, instance.nheads, instance.d_in_proj) == (32, 64, 1, 419)
    assert instance.weight_format == "float32"
    for name, value in zip(instance._W_NAMES, instance._w):
        assert value.shape == weights[name].shape
        assert value.tobytes() == original[name]


def test_mamba3_dt_bias_is_a_head_vector():
    weights, _ = host_fixture(3)
    weights["dt_bias"] = np.zeros((1, 4), np.float32)
    result = audit_native_call(Mamba3Block.__init__, Mamba3Block.__new__(Mamba3Block), weights)
    assert isinstance(result.exception, ValueError), result.report()
    assert "dt_bias" in str(result.exception)
