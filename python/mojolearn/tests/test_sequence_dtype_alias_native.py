"""Private native dtype candidate: versioned alias metadata and callback refusal.

These checks do not qualify the public SequenceDeviceTensor.__array__ wrapper.
"""
import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


@pytest.fixture
def resident():
    values = np.asarray([0., -0., 1.25, -2.5], dtype="f4")
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("candidate dtype qualification requires a device tensor")
    return values, tensor


def _candidate(tensor, dtype):
    return audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)


@pytest.mark.parametrize("dtype", ["float32", "float64", "i2", ">f4"])
def test_exact_integer_dtype_alias_keys_preserve_native_conversion(resident, monkeypatch, dtype):
    values, tensor = resident
    expected = values.astype(dtype, copy=False)
    aliases = dict(np.sctypeDict)
    # Include a deliberate native int/string hash collision. Their equality
    # stays native, just like NumPy1.26's ordinary integer type-number keys.
    aliases[hash(dtype)] = np.float64
    with monkeypatch.context() as patch:
        patch.setattr(np, "sctypeDict", aliases)
        actual = _candidate(tensor, dtype).assert_native()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ["int_subclass", "str_subclass", "object"])
def test_foreign_dtype_alias_keys_refuse_before_hash_or_equality_hooks(resident, monkeypatch, kind):
    _, tensor = resident
    calls = []

    class Hooks:
        def __hash__(self):
            calls.append("hash")
            return hash("float64")

        def __eq__(self, other):
            calls.append("eq")
            return False

    class Integer(Hooks, int):
        pass

    class Text(Hooks, str):
        pass

    key = Integer(999) if kind == "int_subclass" else Text("foreign") if kind == "str_subclass" else Hooks()
    aliases = {key: np.float32, "float64": np.float64}
    calls.clear()  # Test setup may hash/compare; the native call must not.
    with monkeypatch.context() as patch:
        patch.setattr(np, "sctypeDict", aliases)
        audit = _candidate(tensor, "float64")
    assert type(audit.exception) is TypeError, audit.report()
    assert "dtype aliases require native" in str(audit.exception)
    assert not calls
    report = audit.report()
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[field], report
