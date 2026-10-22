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


def _assert_no_runtime_callbacks(audit):
    report = audit.report()
    for field in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not report[field], report


@pytest.mark.parametrize("kind", ["S", "U", "V"])
def test_negative_flexible_size_preserves_native_exception_fields(resident, kind):
    values, tensor = resident
    with pytest.raises(Exception) as oracle:
        values.astype((kind, -1), copy=False)
    audit = _candidate(tensor, (kind, -1))
    _assert_no_runtime_callbacks(audit)
    actual, expected = audit.exception, oracle.value
    assert type(actual) is type(expected)
    assert actual.args == expected.args
    if isinstance(expected, MemoryError):
        assert vars(actual) == vars(expected)
        assert actual.shape == values.shape
        assert actual.dtype.itemsize < 0
    else:
        assert type(actual) is ValueError


def _legacy_allocation_exception():
    if int(np.__version__.split(".")[0]) != 1:
        pytest.skip("NumPy 1.x alone admits negative flexible descriptors")
    import numpy.core._exceptions as module
    return module, module._ArrayMemoryError


def test_negative_storage_bypasses_python_exception_initializer(resident, monkeypatch):
    _, tensor = resident
    _, category = _legacy_allocation_exception()
    calls = []

    def initializer(*args, **kwargs):
        calls.append("init")
        raise AssertionError("Python exception initializer ran")

    with monkeypatch.context() as patch:
        patch.setattr(category, "__init__", initializer)
        audit = _candidate(tensor, ("S", -1))
    _assert_no_runtime_callbacks(audit)
    assert type(audit.exception) is category
    assert audit.exception.args == ((4,), np.dtype("S-1"))
    assert vars(audit.exception) == {"shape": (4,), "dtype": np.dtype("S-1")}
    assert not calls


@pytest.mark.parametrize("hook", ["__new__", "__del__", "__getattribute__", "__getattr__", "__setattr__", "__dict__", "foreign_module_name", "foreign_qualname", "foreign_class_key", "metaclass"])
def test_negative_storage_refuses_exception_hooks_before_allocation(resident, monkeypatch, hook):
    _, tensor = resident
    module, _ = _legacy_allocation_exception()
    calls = []

    def callback(*args, **kwargs):
        calls.append(hook)
        raise AssertionError("Python exception hook ran")

    class ForeignName:
        def __eq__(self, other):
            return callback()

    class ForeignQualname(str):
        def __ne__(self, other):
            return callback()

    class ForeignKey(str):
        def __hash__(self):
            calls.append("key hash")
            return hash("__bases__")

        def __eq__(self, other):
            calls.append("key equality")
            return False

    class Meta(type):
        def __getattribute__(self, name):
            return callback()

    fields = {"__module__": "numpy.core._exceptions", "__qualname__": "_ArrayMemoryError"}
    if hook == "foreign_module_name":
        fields["__module__"] = ForeignName()
    elif hook == "foreign_qualname":
        fields["__qualname__"] = ForeignQualname("_ArrayMemoryError")
    elif hook == "foreign_class_key":
        fields[ForeignKey("foreign_metadata")] = None
    elif hook == "__dict__":
        fields[hook] = property(callback)
    elif hook != "metaclass":
        fields[hook] = callback
    metaclass = Meta if hook == "metaclass" else type
    category = metaclass("_ArrayMemoryError", (MemoryError,), fields)
    calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_ArrayMemoryError", category)
        audit = _candidate(tensor, ("S", -1))
    _assert_no_runtime_callbacks(audit)
    assert type(audit.exception) is TypeError, audit.report()
    assert not calls
