"""Native output honors NumPy's scalar notation version and print settings."""
import warnings

import numpy as np
import pytest

from mojolearn._training_impl import _OPT_NATIVE, _OPT_CONTEXT
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device


RELEASE = tuple(int(part) for part in np.__version__.split(".")[:2])
LEGACY = [False, "1.13", "1.21"] + (["1.25", "2.2"] if RELEASE >= (2, 3) else [])
DTYPES = ["U32", "S8", np.dtype([("text", "U32"), ("bytes", "S8")])]
if hasattr(np.dtypes, "StringDType"):
    DTYPES.append(np.dtypes.StringDType())


@pytest.fixture(scope="module")
def resident():
    words = np.random.default_rng(42).integers(0, 2 ** 32, size=4096, dtype="u4")
    edges = []
    for center in (1e-4, 1e6, 1e16):
        middle = np.float32(center)
        lower, upper = np.nextafter(middle, np.float32(-np.inf)), np.nextafter(middle, np.float32(np.inf))
        edges.extend((lower, middle, upper, -lower, -middle, -upper))
    values = np.concatenate((words.view("f4"), np.array(edges, dtype="f4")))
    tensor = to_device(values, numeric_mode="identical")
    if type(tensor) is not SequenceDeviceTensor:
        pytest.skip("text output qualification requires a device tensor")
    return values, tensor


def call(tensor, dtype):
    return audit_native_call(_OPT_NATIVE, "seqdev_array", tensor, (dtype, None), None, _OPT_CONTEXT)


@pytest.mark.parametrize("legacy", LEGACY)
@pytest.mark.parametrize("dtype", DTYPES, ids=["unicode", "bytes", "record"] + (["string"] if len(DTYPES) == 4 else []))
def test_native_text_matches_versioned_and_legacy_scalar_notation(resident, dtype, legacy):
    # The change in2.3 is documented, and <=2.2 print mode preserves the
    # previous threshold. Legacy1.13 preserves its six-digit str contract.
    # https://numpy.org/doc/2.3/release/2.3.0-notes.html#changes
    values, tensor = resident
    with np.printoptions(legacy=legacy), np.errstate(all="ignore"):
        with warnings.catch_warnings(record=True) as old:
            warnings.simplefilter("always")
            expected = values.astype(dtype, copy=False)
        with warnings.catch_warnings(record=True) as new:
            warnings.simplefilter("always")
            actual = call(tensor, dtype).assert_native()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.flags.writeable == expected.flags.writeable
    if actual.dtype.kind == "T":
        assert actual.tolist() == expected.tolist()
    else:
        assert actual.tobytes() == expected.tobytes()
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]


def test_native_text_reads_configuration_without_python_helpers(resident, monkeypatch):
    values, tensor = resident
    arrayprint = np.__dict__.get("_core", np.__dict__.get("core")).arrayprint
    calls = []
    def hook(*args):
        calls.append("printoptions")
        raise AssertionError("Python print helper called")
    with np.printoptions(legacy="1.21"):
        expected = values.astype("U32")
        with monkeypatch.context() as patch:
            patch.setattr(np, "get_printoptions", hook)
            patch.setattr(arrayprint, "_get_legacy_print_mode", hook)
            actual = call(tensor, "U32").assert_native()
    assert actual.tobytes() == expected.tobytes()
    assert not calls


@pytest.mark.parametrize("field", ["container", "legacy", "key"])
def test_native_text_refuses_python_configuration_hooks(resident, monkeypatch, field):
    _, tensor = resident
    arrayprint = np.__dict__.get("_core", np.__dict__.get("core")).arrayprint
    calls = []
    class Foreign:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("Python print configuration hook called")
        def __int__(self):
            calls.append("int")
            return 202
        def __index__(self):
            calls.append("index")
            return 202
    options = {"legacy": 202}
    if field == "container":
        options = Foreign()
    elif field == "legacy":
        options["legacy"] = Foreign()
    else:
        options[Foreign()] = 1
    with monkeypatch.context() as patch:
        # This is the actual native dictionary on1.26--2.0. On versions
        # using ContextVar, poison its value while retaining the native var.
        variable = vars(arrayprint).get("format_options")
        if variable is None:
            patch.setattr(arrayprint, "_format_options", options)
            result = call(tensor, "U32")
        else:
            token = variable.set(options)
            try:
                result = call(tensor, "U32")
            finally:
                variable.reset(token)
    assert type(result.exception) is TypeError
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()


def test_native_text_refuses_replaced_contextvar_before_get(resident, monkeypatch):
    _, tensor = resident
    arrayprint = np.__dict__.get("_core", np.__dict__.get("core")).arrayprint
    calls = []
    class Foreign:
        def get(self):
            calls.append("get")
            return {"legacy": 202}
    with monkeypatch.context() as patch:
        patch.delattr(arrayprint, "_format_options", raising=False)
        patch.setattr(arrayprint, "format_options", Foreign(), raising=False)
        result = call(tensor, "U32")
    assert type(result.exception) is TypeError
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
