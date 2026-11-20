# SPDX-License-Identifier: Apache-2.0
"""Resident GP storage admission refuses Python descriptor/finalizer hooks."""
import numpy as np
import pytest

from mojolearn._array import Array
from mojolearn._gp_impl import GaussianProcessRegressor


@pytest.fixture
def fitted_gp():
    x = np.array([[-2, 0], [-1, 1], [0, -1], [1, 1], [2, 0], [3, -1]], dtype=np.float32)
    y = np.array([-2, 0, -1, 2, 2, 2], dtype=np.float32)
    return GaussianProcessRegressor(optimizer=None).fit(x, y), x


@pytest.mark.parametrize("name", ["_base", "_readonly", "__setattr__"])
def test_gp_freeze_refuses_rebound_array_protocol_without_invoking_it(fitted_gp, monkeypatch, name):
    model, x = fitted_gp
    calls = []

    def hook(*args):
        calls.append(args)
        raise AssertionError("Python storage hook invoked")

    replacement = hook if name == "__setattr__" else property(hook, hook)
    monkeypatch.setattr(Array, name, replacement)
    with pytest.raises(Exception, match="native|storage|layout|descriptor"):
        model.predict(x)
    assert calls == []


def test_gp_freeze_does_not_replace_an_unadmitted_old_flag(fitted_gp):
    model, x = fitted_gp
    calls = []

    class Flag:
        def __bool__(self):
            calls.append("bool")
            raise AssertionError("Python flag hook invoked")

        def __del__(self):
            calls.append("del")

    value = model.alpha_
    original = object.__getattribute__(value, "_readonly")
    flag = Flag()
    object.__setattr__(value, "_readonly", flag)
    try:
        with pytest.raises(Exception, match="native|storage|owner|protocol|readonly|read.only"):
            model.predict(x)
        assert object.__getattribute__(value, "_readonly") is flag
        assert calls == []
    finally:
        object.__setattr__(value, "_readonly", original)
