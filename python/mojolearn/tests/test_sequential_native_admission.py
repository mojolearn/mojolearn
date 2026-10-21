# SPDX-License-Identifier: Apache-2.0
"""Native admission refuses Python hooks before invoking any of them."""
import numpy as np
import pytest

from mojolearn import Array, Sequential
from mojolearn._verify_native_runtime import audit_native_call


def refused(operation, *args, **kwargs):
    trace = audit_native_call(operation, *args, **kwargs)
    assert isinstance(trace.exception, TypeError), trace.report()
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not trace.report()[key], trace.report()


@pytest.mark.parametrize("kind", ["array_protocol", "ndarray_subclass", "scalar_subclass", "list_subclass",
                                    "mapping_subclass", "path_protocol", "iterator_protocol", "string_subclass"])
def test_native_sequential_refuses_python_protocols_without_entering_them(kind, tmp_path):
    entered = []

    def hook(*args, **kwargs):
        entered.append(kind)
        raise AssertionError("Python-defined protocol entered native Sequential execution")

    x = np.arange(12, dtype="f4").reshape(4, 3) / 20
    y = np.arange(4, dtype="f4")
    out = Sequential().linear(1).compile(optimizer="sgd", lr=0.03)
    if kind == "array_protocol":
        class Foreign:
            __array__ = hook
        refused(out.fit, Foreign(), y, epochs=1)
    elif kind == "ndarray_subclass":
        class Foreign(np.ndarray):
            dtype = property(hook)
            shape = property(hook)
            __array_interface__ = property(hook)
        supplied = x.view(Foreign)
        entered.clear()
        refused(out.fit, supplied, y, epochs=1)
    elif kind == "scalar_subclass":
        class Foreign(np.float64):
            __float__ = hook
        refused(out.compile, lr=Foreign(0.03))
    elif kind == "list_subclass":
        class Foreign(list):
            __getitem__ = hook
            __iter__ = hook
            __len__ = hook
        refused(out.fit, Foreign(x.tolist()), y, epochs=1)
    elif kind == "mapping_subclass":
        class Foreign(dict):
            keys = hook
            items = hook
            __getitem__ = hook
        refused(out.load_state_dict, Foreign({"0.weight": np.ones((1, 3), "f4"), "0.bias": np.zeros(1, "f4")}))
    elif kind == "path_protocol":
        class Foreign:
            __fspath__ = hook
        out.fit(x, y, epochs=1)
        refused(out.save_checkpoint, Foreign())
        refused(Sequential.load, Foreign())
        refused(out.load_state_dict, Foreign())
    elif kind == "iterator_protocol":
        class Foreign:
            __iter__ = hook
        refused(Sequential, input_shape=Foreign())
        refused(out.reshape, Foreign())
    else:
        class Foreign(str):
            __str__ = hook
            strip = hook
        refused(Sequential, numeric_mode=Foreign("identical"))
    assert entered == []


def test_native_sequential_refuses_custom_metadata_from_safe_builtin_iterator():
    entered = []

    class Integer:
        def __index__(self):
            entered.append("index")
            return 3

        def __int__(self):
            entered.append("int")
            return 3

    refused(Sequential, input_shape=iter([2, Integer()]))
    assert entered == []


def test_native_sequential_refuses_mutated_model_metadata_before_property_hooks():
    entered = []

    class Layers(list):
        def __len__(self):
            entered.append("len")
            return 0

    out = Sequential().linear(1).compile()
    out._layers = Layers(out._layers)
    refused(out.fit, np.ones((3, 2), "f4"), np.zeros(3, "f4"), epochs=1)
    assert entered == []


def test_native_sequential_refuses_model_subclasses_without_attribute_hooks():
    entered = []

    class Foreign(Sequential):
        def __getattribute__(self, name):
            entered.append(name)
            return object.__getattribute__(self, name)

    refused(Foreign)
    refused(Foreign.load, "unused.ckpt")
    assert entered == []


def test_native_sequential_uses_cached_native_array_slots_when_properties_are_poisoned(monkeypatch):
    x = np.arange(12, dtype="f4").reshape(4, 3) / 20
    y = np.arange(4, dtype="f4")
    supplied = Array.from_list(x.tolist(), "<f4")
    expected = Sequential().linear(1).compile(optimizer="sgd", lr=0.03).fit(x, y, epochs=1, seed=7)
    out = Sequential().linear(1).compile(optimizer="sgd", lr=0.03)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python Array metadata property entered native execution")

    with monkeypatch.context() as patch:
        for name in ("_mv", "shape", "dtype", "order"):
            patch.setattr(Array, name, property(forbidden))
        audit_native_call(out.fit, supplied, y, epochs=1, seed=7).assert_native()
    assert out.params_.tobytes() == expected.params_.tobytes()
