# SPDX-License-Identifier: Apache-2.0
"""CPU KDE scores preserve the saved numerical contract without Python work."""
import ast
import gc
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _archive, _state
from mojolearn.tests.test_host_inference_native import _input, _refused, test_application_subclass_keeps_constructor_fit_and_score_guards as _subclass_guards
from mojolearn.tests.test_host_kernel_inference_native import FORMATS

KERNELS = ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine")
METHODS = ("score_samples", "score")


def _model(tmp_path, kernel="gaussian", weighted=True):
    model = host.HostKernelDensity.load(_archive(tmp_path, "KernelDensity", override={"kernel": kernel}))
    if not weighted:
        model._w = None
    return model


def _check(model, x, method):
    expected = legacy.kde(model, x, method == "score")
    actual = audit_native_call(getattr(model, method), x).assert_native()
    if method == "score":
        assert type(actual) is float
    else:
        assert type(actual) is np.ndarray and actual.dtype == np.dtype("<f4")
    assert _state(actual) == _state(expected)
    return actual


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["float32", "list", "fortran", "strided", "big_endian", "half", "int16", "bool", "uint64"])
def test_kde_dense_scores_preserve_complete_output_bits(tmp_path, kernel, weighted, method, kind):
    model = _model(tmp_path, kernel, weighted)
    before = _state(vars(model))
    _check(model, _input(2, kind), method)
    assert _state(vars(model)) == before


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("method", METHODS)
def test_kde_sparse_inputs_keep_original_dense_expansion(tmp_path, format, method):
    sp = pytest.importorskip("scipy.sparse")
    _check(_model(tmp_path), getattr(sp, format + "_matrix")(_input(2, "float32")), method)


@pytest.mark.parametrize("metric", ["euclidean", "l2", "sqeuclidean", "l1", "cityblock", "manhattan", "chebyshev", "cosine", "minkowski"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_metric_aliases_preserve_results(tmp_path, metric, method):
    model = _model(tmp_path)
    model.metric = metric
    _check(model, np.asarray([[1., 2.], [2., 3.], [6., 7.]], dtype="<f4"), method)


@pytest.mark.parametrize("field", ["_x", "_w"])
@pytest.mark.parametrize("layout", ["c", "fortran", "readonly"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_saved_native_buffer_layouts_preserve_physical_storage(tmp_path, field, layout, method):
    model = _model(tmp_path)
    value = np.array(vars(model)[field], order="F" if layout == "fortran" else "C", copy=True)
    if layout == "readonly":
        value.flags.writeable = False
    setattr(model, field, value)
    _check(model, _input(2, "float32"), method)


@pytest.mark.parametrize("kind", ["float", "string", "bytes", "numpy32", "zero_dim", "infinity"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_bandwidth_native_scalar_forms_preserve_results(tmp_path, kind, method):
    model = _model(tmp_path)
    model.bandwidth = {"float": .75, "string": ".75", "bytes": b".75", "numpy32": np.float32(.75), "zero_dim": np.asarray(.75), "infinity": np.inf}[kind]
    _check(model, _input(2, "float32"), method)


_PRIMITIVE_BANDWIDTHS = [
    ("?", True, "scalar"), ("i1", 1, "scalar"), ("u1", 1, "scalar"),
    ("i2", 1, "array"), ("u2", 1, "array"), ("i4", 1, "swapped"),
    ("u4", 1, "swapped"), ("i8", 1, "scalar"), ("u8", 2**64 - 1, "scalar"),
    ("q", 1, "scalar"), ("Q", 1, "scalar"), ("f2", .75, "scalar"),
    ("f4", .75, "scalar"), ("f8", .75, "swapped"),
    ("g", .75, "scalar"), ("g", .75, "swapped"),
    ("f4", 0., "array"), ("f8", -0., "scalar"), ("f4", -1., "scalar"),
    ("f4", float("nan"), "scalar"), ("f8", float("inf"), "array"),
    ("f4", np.nextafter(np.float32(0), np.float32(1)), "scalar"),
]


def _primitive_bandwidth(code, number, form):
    value = np.array(number, dtype=code)
    if form == "scalar":
        return value[()]
    if form == "swapped":
        value = value.astype(value.dtype.newbyteorder(">"))
    value.flags.writeable = False
    return value


def _check_primitive_bandwidth(model, x, method, value):
    original = value.tobytes()
    state = _state(vars(model))
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter("always")
        try:
            expected = legacy.kde(model, x, method == "score")
        except Exception as error:
            failure = type(error), str(error)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter("always")
        audit = audit_native_call(getattr(model, method), x)
    if failure:
        _refused(audit, failure[0])
        assert str(audit.exception) == failure[1]
    else:
        actual = audit.assert_native()
        assert _state(actual) == _state(expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
    assert model.bandwidth is value and value.tobytes() == original
    assert _state(vars(model)) == state


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("code,number,form", _PRIMITIVE_BANDWIDTHS)
def test_kde_primitive_bandwidth_buffer_decode_preserves_consumed_bits(tmp_path, method, code, number, form):
    model = _model(tmp_path)
    value = _primitive_bandwidth(code, number, form)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), method, value)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("code,form", [("q", "scalar"), ("Q", "scalar"),
                                       ("g", "scalar"), ("g", "swapped")])
def test_kde_primitive_bandwidth_native_slots_preserve_alias_ownership(tmp_path, method, code, form):
    class NativeBandwidthSlot(host.HostKernelDensity):
        __slots__ = ("bandwidth",)
    model = object.__new__(NativeBandwidthSlot)
    vars(model).update(vars(_model(tmp_path)))
    vars(model).pop("bandwidth")
    value = _primitive_bandwidth(code, 1, form)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), method, value)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("code", ["F", "D", "G"])
@pytest.mark.parametrize("form", ["scalar", "array", "swapped", "singleton", "empty", "multiple"])
def test_kde_complex_bandwidth_preserves_consumed_output_and_warnings(tmp_path, method, code, form):
    value = _primitive_bandwidth(code, .75 + 2j, form)
    if form in ("singleton", "empty", "multiple"):
        value = np.full({"singleton": (1, 1), "empty": (0,), "multiple": (2,)}[form], .75 + 2j, dtype=code)
    model = _model(tmp_path)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), method, value)


_BANDWIDTH_WARNING_CHILD = r'''
import json, sys, warnings
import numpy as np
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call

method, archive, module_mode = sys.argv[1:]
model = host.HostKernelDensity.load(archive)
model.bandwidth = .75
x = np.array([[.5, 1.], [1.5, 2.], [3., 4.]], dtype='f4')
operation = getattr(model, method)
expected = operation(x)
expected_bits = np.asarray(expected).tobytes()
bandwidth = np.complex128(.75 + 2j)
model.bandwidth = bandwidth
category = np.exceptions.ComplexWarning
message = 'Casting complex values to real discards the imaginary part'
active, events, retained = [False], [], []
def forbidden_pointer(*args):
    retained.append(args)
    raise AssertionError('KDE called the retained frame pointer')
host._CLASSICAL['host_inference_context']['classical_getframe'] = forbidden_pointer

# Profile the actual public score operation separately from the audit-hook
# arm: the testing profiler's own frame inspection is outside the product.
with warnings.catch_warnings(record=True) as profiled_warnings:
    warnings.simplefilter('always', category)
    profiled = audit_native_call(operation, x).assert_native()
assert np.asarray(profiled).tobytes() == expected_bits
assert len(profiled_warnings) == 1 and profiled_warnings[0].category is category

namespace = {'__name__': 'application.kde_warning_probe' if module_mode == 'named' else 42}
exec(compile('def invoke(call, values):\n    return call(values)\n',
             'native_kde_warning_caller.py', 'exec'), namespace)
invoke = namespace['invoke']
def hook(event, args):
    frame_event = event in ('sys._getframe', 'sys._getframemodulename')
    frame_event = frame_event or (event == 'object.__getattr__' and len(args) > 1
                                  and args[1] in ('f_code', 'f_back', 'f_globals'))
    if active[0] and frame_event:
        events.append(event)
        raise AssertionError('KDE warning attribution invoked a Python audit hook')
sys.addaudithook(hook)
active[0] = True
try:
    sys._getframe()
except AssertionError:
    pass
finally:
    active[0] = False
assert events == ['sys._getframe'], events
events.clear()

for action in ('always', 'ignore', 'error', 'module_error'):
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        if action == 'module_error':
            warnings.simplefilter('ignore', category)
            module_pattern = r'^application\.kde_warning_probe$' if module_mode == 'named' else r'^$'
            warnings.filterwarnings('error', category=category, module=module_pattern)
        else:
            warnings.simplefilter(action, category)
        active[0] = True
        try:
            actual = invoke(operation, x)
        except category as error:
            failure = error
        finally:
            active[0] = False
    if action in ('error', 'module_error'):
        assert type(failure) is category and str(failure) == message, (action, failure)
        assert caught == []
    else:
        assert failure is None
        assert type(actual) is type(expected)
        assert np.asarray(actual).dtype == np.asarray(expected).dtype
        assert np.asarray(actual).tobytes() == expected_bits
        if action == 'always':
            assert len(caught) == 1
            warning = caught[0]
            assert warning.category is category and str(warning.message) == message
            assert warning.filename == 'native_kde_warning_caller.py', warning.filename
            assert warning.lineno == 2, warning.lineno
        else:
            assert caught == []
    assert model.bandwidth is bandwidth
    assert events == [] and retained == [], (events, retained)
print(json.dumps({'method': method, 'module_mode': module_mode, 'filter_cases': 4,
                  'audit_events': events, 'retained_calls': retained}))
'''


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('module_mode', ['named', 'nonstring'])
def test_kde_bandwidth_warning_has_native_attribution_and_no_frame_callbacks(tmp_path, method, module_mode):
    # Process isolation keeps the permanent Python audit hook out of pytest.
    archive = _archive(tmp_path, 'KernelDensity')
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = package_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    result = subprocess.run([sys.executable, '-c', _BANDWIDTH_WARNING_CHILD, method, str(archive), module_mode],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


@pytest.mark.parametrize("code,form", [("F", "scalar"), ("G", "array"), ("G", "swapped")])
@pytest.mark.parametrize("action", ["always", "ignore", "error"])
def test_kde_complex_warning_filters_preserve_categories_and_native_slots(tmp_path, code, form, action):
    class NativeBandwidthSlot(host.HostKernelDensity):
        __slots__ = ("bandwidth",)
    model = object.__new__(NativeBandwidthSlot)
    vars(model).update(vars(_model(tmp_path)))
    vars(model).pop("bandwidth")
    value = _primitive_bandwidth(code, .75 + 2j, form)
    model.bandwidth = value
    x = _input(2, "float32")
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter(action)
        try:
            expected = legacy.kde(model, x)
        except Exception as error:
            failure = type(error), str(error)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter(action)
        result = audit_native_call(model.score_samples, x)
    if failure:
        _refused(result, failure[0])
        assert str(result.exception) == failure[1]
    else:
        assert _state(result.assert_native()) == _state(expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
    assert model.bandwidth is value


@pytest.mark.parametrize("code,form", [("F", "scalar"), ("G", "array"), ("G", "swapped")])
@pytest.mark.parametrize("number", [complex(-0., 2), complex(float("nan"), 2), complex(float("inf"), 2)])
def test_kde_complex_special_values_preserve_consumed_bits(tmp_path, code, form, number):
    model = _model(tmp_path)
    value = _primitive_bandwidth(code, number, form)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), "score_samples", value)


@pytest.mark.parametrize("kind", ["function", "init", "finalizer"])
def test_kde_complex_warning_category_refuses_python_hooks(tmp_path, monkeypatch, kind):
    model = _model(tmp_path)
    model.bandwidth = np.complex128(.75 + 2j)
    touched = []
    if kind == "function":
        def replacement(*args):
            touched.append("function")
    else:
        class replacement(RuntimeWarning):
            pass
        def callback(self, *args):
            touched.append(kind)
        setattr(replacement, "__init__" if kind == "init" else "__del__", callback)
    monkeypatch.setattr(np.exceptions, "ComplexWarning", replacement)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = audit_native_call(model.score_samples, _input(2, "float32"))
    _refused(result, TypeError)
    assert touched == []


_TEMPORAL_UNITS = [None, "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "3ns"]
_TEMPORAL_BOUNDARIES = [
    ("M8[Y]", -1969), ("M8[Y]", -1970), ("M8[Y]", 8029), ("M8[Y]", 8030),
    ("M8[M]", -23628), ("M8[M]", -23629),
    ("M8[D]", -719162), ("M8[D]", -719163),
    ("M8[D]", 2932896), ("M8[D]", 2932897),
    ("M8[us]", 253402300799999999), ("M8[us]", 253402300800000000),
    ("m8[D]", 999999999), ("m8[D]", 1000000000),
    ("m8[D]", -999999999), ("m8[D]", -1000000000),
    ("m8[W]", 2**63 - 1), ("m8[D]", 2**63 - 1),
    ("m8[m]", 2**63 - 1), ("m8[s]", -(2**63) + 1),
    ("M8[3D]", 2**63 - 1), ("m8[7h]", -(2**63) + 1),
    ("M8[ns]", 2**53 + 1), ("m8[ns]", 2**63 - 1),
]


def _temporal_bandwidth(code, ticks, form):
    value = np.array(ticks, dtype="i8").view(code)
    if form == "scalar":
        return value[()]
    if form == "swapped":
        value = value.astype(value.dtype.newbyteorder(">"))
    elif form == "singleton":
        value = value.reshape(1, 1)
    value.flags.writeable = False
    return value


@pytest.mark.parametrize("kind", ["M8", "m8"])
@pytest.mark.parametrize("unit", _TEMPORAL_UNITS)
@pytest.mark.parametrize("ticks", [1, -(2**63)])
def test_kde_temporal_bandwidth_preserves_units_nat_and_consumed_results(tmp_path, kind, unit, ticks):
    code = kind if unit is None else f"{kind}[{unit}]"
    value = _temporal_bandwidth(code, ticks, "scalar" if ticks == 1 else "array")
    model = _model(tmp_path)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), "score_samples", value)


@pytest.mark.parametrize("code,ticks", _TEMPORAL_BOUNDARIES)
@pytest.mark.parametrize("form", ["scalar", "swapped"])
def test_kde_temporal_calendar_range_and_versioned_day_width_preserve_contract(tmp_path, code, ticks, form):
    value = _temporal_bandwidth(code, ticks, form)
    model = _model(tmp_path)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), "score", value)


@pytest.mark.parametrize("code,ticks", [
    ("M8[2D]", (2**63 - 1) // 2), ("M8[2D]", 2**62),
    ("M8[2D]", -((2**63 - 1) // 2)), ("M8[2D]", -(2**62)),
    ("M8[3Y]", (2**63 - 1) // 3 + 1),
    ("M8[3W]", -((2**63 - 1) // 3) - 1),
    ("M8[3us]", (2**63 - 1) // 3 + 1),
    ("M8[3ns]", 2**63 - 1), ("M8[2D]", -(2**63)),
    ("m8[2D]", 2**62), ("m8[2D]", -(2**62)),
    ("m8[3us]", (2**63 - 1) // 3 + 1),
])
@pytest.mark.parametrize("form", ["scalar", "swapped"])
def test_kde_temporal_unit_overflow_preserves_versioned_boundaries(tmp_path, code, ticks, form):
    value = _temporal_bandwidth(code, ticks, form)
    model = _model(tmp_path)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), "score_samples", value)


@pytest.mark.parametrize("code", ["M8[ns]", "m8[us]", "m8[3ns]"])
@pytest.mark.parametrize("action", ["always", "ignore", "error"])
def test_kde_temporal_singleton_rank_warning_precedes_conversion(tmp_path, code, action):
    value = _temporal_bandwidth(code, 1, "singleton")
    model = _model(tmp_path)
    model.bandwidth = value
    failure = None
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter(action)
        try:
            expected = legacy.kde(model, _input(2, "float32"))
        except Exception as error:
            failure = type(error), str(error)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter(action)
        result = audit_native_call(model.score_samples, _input(2, "float32"))
    if failure:
        _refused(result, failure[0])
        assert str(result.exception) == failure[1]
    else:
        assert _state(result.assert_native()) == _state(expected)
    assert [(type(w.message), str(w.message)) for w in new] == [(type(w.message), str(w.message)) for w in old]
    assert model.bandwidth is value


@pytest.mark.parametrize("code", ["f4", "g", "q", "u8"])
@pytest.mark.parametrize("shape", [(1,), (1, 1), (0,), (2,)])
def test_kde_numeric_array_rank_preserves_versioned_conversion_contract(tmp_path, code, shape):
    value = np.ones(shape, dtype=code)
    model = _model(tmp_path)
    model.bandwidth = value
    _check_primitive_bandwidth(model, _input(2, "float32"), "score_samples", value)


@pytest.mark.parametrize("code", ["M8[ns]", "m8[us]"])
def test_kde_temporal_array_subclass_refuses_hooks_before_dtype_or_buffer(tmp_path, code):
    touched = []
    class Foreign(np.ndarray):
        @property
        def dtype(self):
            touched.append("dtype")
            return np.dtype(code)
        def __float__(self):
            touched.append("float")
            return 1.
    value = _temporal_bandwidth(code, 1, "array").view(Foreign)
    model = _model(tmp_path)
    model.bandwidth = value
    result = audit_native_call(model.score_samples, _input(2, "float32"))
    _refused(result, TypeError)
    assert touched == []


@pytest.mark.parametrize("code,ticks", [("M8[ns]", 1), ("m8[us]", 1), ("M8[ns]", -(2**63))])
def test_kde_temporal_buffer_owners_release_after_results_and_refusals(tmp_path, code, ticks):
    value = _temporal_bandwidth(code, ticks, "swapped")
    model = _model(tmp_path)
    model.bandwidth = value
    x = _input(2, "float32")
    method = model.score_samples
    for _ in range(2):
        audit_native_call(method, x)
    gc.collect()
    before = sys.getrefcount(value)
    for _ in range(5):
        result = audit_native_call(method, x)
        if code == "M8[ns]" and ticks == 1:
            result.assert_native()
        else:
            _refused(result, TypeError)
        del result
    gc.collect()
    assert sys.getrefcount(value) == before


@pytest.mark.parametrize("field", ["n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_kde_native_scalar_metadata_preserves_results(tmp_path, field):
    model = _model(tmp_path)
    setattr(model, field, np.asarray(vars(model).get(field, False)))
    _check(model, _input(2, "float32"), "score_samples")


@pytest.mark.parametrize("field", ["bandwidth", "n_features_in_", "_mojolearn_fitted", "_mojolearn_fitting"])
def test_kde_scalar_state_refuses_python_conversion_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return .75
        def __int__(self):
            touched.append("int")
            return 2
        def __bool__(self):
            touched.append("bool")
            return True
    setattr(model, field, np.asarray(Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["_x", "_w"])
def test_kde_saved_object_buffers_refuse_element_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Scalar:
        def __float__(self):
            touched.append("float")
            return 1.
    setattr(model, field, np.full(vars(model)[field].shape, Scalar(), dtype=object))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["kernel", "metric"])
def test_kde_string_options_refuse_python_subclass_hooks(tmp_path, field):
    model = _model(tmp_path)
    touched = []
    class Name(str):
        def __str__(self):
            touched.append("str")
            return "gaussian"
    setattr(model, field, Name(vars(model)[field]))
    _refused(audit_native_call(model.score_samples, [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("method", METHODS)
def test_kde_protocol_inputs_refused_before_hooks(tmp_path, method):
    model = _model(tmp_path)
    touched = []
    class Input:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return np.ones((2, 2), dtype="<f4")
    _refused(audit_native_call(getattr(model, method), Input()), TypeError)
    assert touched == []


def test_kde_score_does_not_inspect_ignored_y(tmp_path):
    model = _model(tmp_path)
    touched = []
    class Ignored:
        def __array__(self, *args, **kwargs):
            touched.append("array")
        def __bool__(self):
            touched.append("bool")
            return True
    x = _input(2, "float32")
    result = audit_native_call(model.score, x, Ignored()).assert_native()
    assert result == legacy.kde(model, x, True) and touched == []


@pytest.mark.parametrize("method", METHODS)
def test_kde_fixed_numerical_entry_refuses_python_replacement(tmp_path, method, monkeypatch):
    model = _model(tmp_path)
    touched = []
    def callback(*args):
        touched.append("entry")
        return 0
    monkeypatch.setitem(host._CLASSICAL["host_inference_context"]["host_inference_entries"], "kde_score_samples", callback)
    _refused(audit_native_call(getattr(model, method), [[1., 2.]]), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["_x", "_w", "query"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_kde_nonfinite_data_preserves_existing_native_refusals(tmp_path, field, value):
    model = _model(tmp_path)
    x = _input(2, "float32")
    if field == "query":
        x[0, 0] = value
    else:
        data = np.asarray(vars(model)[field]).copy()
        data.flat[0] = value
        setattr(model, field, data)
    with pytest.raises(Exception) as old:
        legacy.kde(model, x)
    assert type(old.value) is Exception
    audit = audit_native_call(model.score_samples, x)
    # The established public wrapper converts nonfinite diagnostics to
    # ValueError while the positive-weight check remains builtin Exception.
    normalized = field != "_w" or value == np.inf
    _refused(audit, ValueError if normalized else Exception)
    expected = f"Input X contains NaN or infinity (HostKernelDensity: {old.value})" if normalized else str(old.value)
    assert type(audit.exception) is (ValueError if normalized else Exception)
    assert str(audit.exception) == expected


@pytest.mark.parametrize("field,value", [("bandwidth", 0.), ("bandwidth", -1.), ("bandwidth", np.nan), ("kernel", "bogus"), ("metric", "bogus")])
def test_kde_invalid_kernel_configuration_preserves_refusal(tmp_path, field, value):
    model = _model(tmp_path)
    setattr(model, field, value)
    with pytest.raises(Exception) as old:
        legacy.kde(model, [[1., 2.]])
    assert type(old.value) is Exception
    audit = audit_native_call(model.score_samples, [[1., 2.]])
    _refused(audit, Exception)
    assert type(audit.exception) is Exception and str(audit.exception) == str(old.value)


@pytest.mark.parametrize("shape", [(0, 2), (2, 3), (3,)])
def test_kde_invalid_query_shape_is_refused_natively(tmp_path, shape):
    _refused(audit_native_call(_model(tmp_path).score_samples, np.ones(shape, dtype="<f4")))


def test_kde_unfitted_and_numeric_mode_guards_remain_native(tmp_path):
    model = object.__new__(host.HostKernelDensity)
    _refused(audit_native_call(model.score_samples, [[1., 2.]]))
    model = _model(tmp_path)
    model.numeric_mode = "fast"
    _refused(audit_native_call(model.score_samples, [[1., 2.]]))


def test_kde_ordinary_subclass_keeps_native_attributes_and_scores(tmp_path):
    cls = type("OrdinaryHostKDE", (host.HostKernelDensity,), {})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    for method in METHODS:
        audit_native_call(getattr, model, method).assert_native()
        _check(model, _input(2, "float32"), method)
    audit_native_call(cls.__del__, model).assert_native()


@pytest.mark.parametrize("field", ["bandwidth", "kernel", "metric"])
@pytest.mark.parametrize("storage", ["slot", "inherited", "instance_override"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_native_option_attributes_preserve_bits_and_state(tmp_path, field, storage, method):
    value = {"bandwidth": .75, "kernel": "exponential", "metric": "manhattan"}[field]
    members = {"__slots__": (field,)} if storage == "slot" else {field: value}
    parent = type("NativeKDEOptions", (host.HostKernelDensity,), members)
    cls = type("InheritedKDEOptions", (parent,), {})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    if storage == "slot":
        vars(model).pop(field)
        setattr(model, field, value)
    elif storage == "inherited":
        vars(model).pop(field)
    before = _state(vars(model))
    original = getattr(model, field)
    _check(model, _input(2, "float32"), method)
    assert _state(vars(model)) == before and getattr(model, field) is original


@pytest.mark.parametrize("field", ["bandwidth", "kernel", "metric"])
@pytest.mark.parametrize("storage", ["property", "descriptor", "slot_scalar"])
def test_kde_resolved_options_refuse_python_hooks(tmp_path, field, storage):
    touched = []
    def hook(*args):
        touched.append("get")
        return .75 if field == "bandwidth" else "gaussian"
    class Descriptor:
        __get__ = hook
    class Scalar:
        def __float__(self):
            touched.append("float")
            return .75
        def __str__(self):
            touched.append("str")
            return "gaussian"
    members = {field: property(hook) if storage == "property" else Descriptor()}
    if storage == "slot_scalar":
        members = {"__slots__": (field,)}
    cls = type("RefusedKDEOption", (host.HostKernelDensity,), members)
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    vars(model).pop(field)
    if storage == "slot_scalar":
        setattr(model, field, Scalar())
    before = _state(vars(model))
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), TypeError)
    assert touched == [] and _state(vars(model)) == before


_OPTION_FORMS = [(field, form) for field in ("kernel", "metric", "bandwidth")
                 for form in ("str", "numpy_str", "bytes", "numpy_bytes", "bytearray", "plain_subclass")]
_OPTION_FORMS += [("bandwidth", form) for form in ("numpy_float32", "numpy_float64", "zero_dim32", "zero_dim64")]


@pytest.mark.parametrize("field,form", _OPTION_FORMS)
@pytest.mark.parametrize("storage", ["dict", "slot"])
@pytest.mark.parametrize("method", METHODS)
def test_kde_native_option_scalar_forms_preserve_original_contract(tmp_path, field, form, storage, method):
    text = {"bandwidth": ".75", "kernel": "gaussian", "metric": "euclidean"}[field]
    class Plain(str):
        pass
    value = {"str": text, "numpy_str": np.str_(text), "bytes": text.encode(),
             "numpy_bytes": np.bytes_(text), "bytearray": bytearray(text.encode()),
             "plain_subclass": Plain(text), "numpy_float32": np.float32(.75),
             "numpy_float64": np.float64(.75), "zero_dim32": np.asarray(.75, dtype="<f4"),
             "zero_dim64": np.asarray(.75, dtype="<f8")}[form]
    cls = host.HostKernelDensity if storage == "dict" else type("NativeOptionSlot", (host.HostKernelDensity,), {"__slots__": (field,)})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    if storage == "slot":
        vars(model).pop(field)
    setattr(model, field, value)
    before = _state(vars(model))
    x = _input(2, "float32")
    if field != "bandwidth" and form in ("bytes", "numpy_bytes", "bytearray"):
        with pytest.raises(Exception) as old:
            legacy.kde(model, x, method == "score")
        audit = audit_native_call(getattr(model, method), x)
        _refused(audit, type(old.value))
        assert type(audit.exception) is type(old.value) is Exception
        assert str(audit.exception) == str(old.value)
    else:
        _check(model, x, method)
    assert _state(vars(model)) == before and getattr(model, field) is value


@pytest.mark.parametrize("field", ["bandwidth", "kernel", "metric"])
@pytest.mark.parametrize("storage", ["dict", "slot"])
def test_kde_option_string_subclass_conversion_hooks_are_never_called(tmp_path, field, storage):
    touched = []
    def hook(*args):
        touched.append("conversion")
        return .75 if field == "bandwidth" else "gaussian"
    cls_value = type("HookOption", (str,), {"__float__" if field == "bandwidth" else "__str__": hook})
    value = cls_value(".75" if field == "bandwidth" else "gaussian")
    cls = host.HostKernelDensity if storage == "dict" else type("HookOptionSlot", (host.HostKernelDensity,), {"__slots__": (field,)})
    model = object.__new__(cls)
    vars(model).update(vars(_model(tmp_path)))
    if storage == "slot":
        vars(model).pop(field)
    setattr(model, field, value)
    _refused(audit_native_call(model.score_samples, _input(2, "float32")), TypeError)
    assert touched == []


def test_kde_native_finalizer_masks_device_base_python_finalizer(tmp_path, monkeypatch):
    model = _model(tmp_path)
    touched = []
    def finalizer(self):
        touched.append("del")
    monkeypatch.setattr(host.KernelDensity, "__del__", finalizer)
    _check(model, _input(2, "float32"), "score_samples")
    audit_native_call(host.HostKernelDensity.__del__, model).assert_native()
    assert touched == []


@pytest.mark.parametrize("name", ["KernelDensity", "ElasticNet", "KMeans", "KernelRidge"])
def test_resolved_model_python_finalizer_is_refused_before_attribute_binding(tmp_path, name, monkeypatch):
    cls = getattr(host, "Host" + name)
    model = cls.load(_archive(tmp_path, name))
    touched = []
    def finalizer(self):
        touched.append("del")
    with monkeypatch.context() as patch:
        patch.setattr(cls, "__del__", finalizer, raising=False)
        _refused(audit_native_call(getattr, model, "score_samples" if name == "KernelDensity" else "predict"), TypeError)
        assert touched == []


def test_kde_internal_output_owner_survives_model_and_input(tmp_path):
    model = _model(tmp_path)
    x = _input(2, "float32")
    expected = _state(legacy.kde(model, x))
    out = audit_native_call(model.score_samples, x).assert_native()
    # Enter the public boundary explicitly; pytest module names must not
    # select the package-internal Array output policy.
    # Keep every value, subsequent-call and owner-lifetime check below.
    assert type(out) is np.ndarray and _state(out) == expected
    audit_native_call(model.score_samples, x + 1).assert_native()
    del model, x
    gc.collect()
    assert _state(out) == expected


def test_kde_dataframe_feature_name_warning_and_torch_inputs(tmp_path):
    pd = pytest.importorskip("pandas")
    model = _model(tmp_path)
    x = pd.DataFrame(_input(2, "float32"), columns=["a", "b"])
    _check(model, x, "score_samples")
    model.feature_names_in_ = np.asarray(["a", "b"], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_native_call(model.score_samples, _input(2, "float32")).assert_native()
    assert len(caught) == 1
    del model.feature_names_in_
    torch = pytest.importorskip("torch")
    _check(model, torch.tensor(_input(2, "float32")), "score_samples")


def test_kde_inherited_application_guards_remain_intact():
    _subclass_guards("KernelDensity")


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("weighted", [False, True])
def test_kde_inherited_fit_preserves_saved_state_and_scoring(tmp_path, kernel, weighted):
    x = np.asarray([[1., 2.], [3., 4.], [5., 6.]], dtype="<f4")
    model = host.HostKernelDensity(bandwidth=.75, kernel=kernel, numeric_mode="identical")
    assert model.fit(x, sample_weight=np.asarray([1., 2., 1.], dtype="<f4") if weighted else None) is model
    assert vars(model)["_mojolearn_fitted"] is True
    assert "_mojolearn_fitting" not in vars(model)
    for method in METHODS:
        _check(model, _input(2, "float32"), method)


def test_kde_public_methods_are_single_native_calls():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostKernelDensity")
    for name in METHODS:
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = method.body[1:]
        assert len(body) == 1 and isinstance(body[0], ast.Return)
        assert isinstance(body[0].value, ast.Call) and body[0].value.func.id == "_native_classical"


_NUMPY_IMPORT_CHILD = r'''
import importlib.abc, json, sys
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            # Model a missing optional dependency; a broken import must still propagate.
            raise ModuleNotFoundError("NumPy deliberately unavailable in this test", name="numpy")
blocker = BlockNumPy()
sys.meta_path.insert(0, blocker)
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
payload = json.load(sys.stdin)
if payload["order"] == "late":
    sys.meta_path.remove(blocker)
    import numpy
rows = []
for record in payload["records"]:
    model = host.HostKernelDensity.load(record["path"])
    result = audit_native_call(model.score_samples, payload["source"]).assert_native()
    assert type(result).__name__ == ("ndarray" if payload["order"] == "late" else "Array")
    total = audit_native_call(model.score, payload["source"]).assert_native()
    assert type(total) is float
    rows.append({"kernel":record["kernel"], "bytes":result.tobytes().hex(), "shape":list(result.shape), "total":total})
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps(rows))
'''


@pytest.mark.parametrize("order", ["absent", "late"])
def test_kde_numpy_absent_and_late_import_preserve_consumed_results(tmp_path, order):
    records, expected = [], []
    source = _input(2, "list")
    for kernel in KERNELS:
        folder = tmp_path / kernel
        folder.mkdir()
        model = _model(folder, kernel)
        records.append(dict(kernel=kernel, path=str(folder / "KernelDensity.npz")))
        result = legacy.kde(model, source)
        expected.append(dict(kernel=kernel, bytes=result.tobytes().hex(), shape=list(result.shape), total=legacy.kde(model, source, True)))
    done = subprocess.run([sys.executable, "-c", _NUMPY_IMPORT_CHILD],
                          input=json.dumps(dict(order=order, records=records, source=source)), text=True,
                          capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
    assert json.loads(done.stdout) == expected


def test_kde_bandwidth_conversion_never_calls_replaced_builtin_float(tmp_path, monkeypatch):
    import builtins
    model = _model(tmp_path)
    model.bandwidth = ".75"
    x = _input(2, "float32")
    expected = _state(legacy.kde(model, x))
    touched = []
    def callback(value):
        touched.append("float")
        return .75
    score = model.score_samples  # bind first: the native attribute read consults the connected builtins
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "float", callback)
        audit = audit_native_call(score, x)
    # A replaced connected builtins export refuses (see _connected_exports).
    assert_connected_export_refused(audit, "builtins", "float")
    assert touched == []
    audit = audit_native_call(model.score_samples, x)
    assert _state(audit.assert_native()) == expected
    assert touched == []
