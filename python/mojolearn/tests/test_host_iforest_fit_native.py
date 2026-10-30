# SPDX-License-Identifier: Apache-2.0
"""Complete explicit CPU IsolationForest fit against its frozen public contract."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_iforest_fit as legacy
from mojolearn.tests._legacy_host_inference import iforest as old_score
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_inference_native import _refused
from mojolearn.tests.test_host_kde_fit_native import _x


def _model(**kwargs):
    options = dict(n_estimators=7, max_samples=3, max_depth=2, max_features=1., random_state=11, numeric_mode="identical")
    options.update(kwargs)
    return host.HostIsolationForest(**options)


def _pair(x, y=None, **options):
    old, model = _model(**options), _model(**options)
    legacy.fit(old, x, y)
    assert audit_native_call(model.fit, x, y).assert_native() is model
    assert _state(vars(model)) == _state(vars(old))
    assert type(vars(model)["_x"]) is Array
    for want, method in enumerate(("score_samples", "decision_function", "predict")):
        expected = old_score(old, _x(), want)
        actual = audit_native_call(getattr(model, method), _x()).assert_native()
        assert _state(actual) == _state(expected)
    return old, model


@pytest.mark.parametrize("kind", ["c", "list", "fortran", "strided", "negative_stride", "readonly", "float64", "half", "int16", "big_endian", "array", "memoryview", "object", "pandas", "torch"])
@pytest.mark.parametrize("options", [{}, {"max_samples": "auto", "max_depth": None, "bootstrap": True, "contamination": .25},
                                     {"max_samples": .75, "max_features": 1, "random_state": None}])
def test_iforest_fit_preserves_complete_state_and_consumed_outputs(kind, options):
    x = _x(kind)
    if kind == "object":
        x = x.astype(object)
    elif kind == "pandas":
        x = pytest.importorskip("pandas").DataFrame(x, columns=["left", "right"])
    elif kind == "torch":
        x = pytest.importorskip("torch").from_numpy(x)
    _pair(x, **options)


@pytest.mark.parametrize("field,value", [("n_estimators", np.int32(4)), ("n_estimators", np.asarray(4)),
    ("max_samples", np.str_("AUTO")), ("max_samples", np.int64(2)), ("max_samples", np.float32(.75)),
    ("max_samples", np.asarray(.75)), ("max_depth", np.asarray(2)), ("max_features", np.float64(.75)),
    ("bootstrap", np.asarray(True)), ("contamination", np.str_("AUTO")), ("contamination", np.asarray(.25)),
    ("random_state", np.uint64(4294967295)), ("verbose", np.asarray(False)), ("warm_start", np.asarray(False)),
    ("max_samples", np.longlong(2)), ("max_samples", np.ulonglong(2)), ("max_features", np.longlong(1)),
    ("max_samples", np.longdouble(.75)), ("contamination", np.longdouble(.25))])
def test_iforest_fit_native_parameter_scalar_forms_keep_bits(field, value):
    _pair(_x(), **{field: value})


@pytest.mark.parametrize("dtype,value", [("<f2", -0.), ("<f2", .5), ("<f4", 0.), ("<f8", 1e-300), ("<i4", 1)])
def test_iforest_fit_array_truth_controls_remain_native(dtype, value):
    _pair(_x(), bootstrap=Array.from_buffer(np.asarray([value], dtype=dtype)))


_PARAMETERS = {"trees_zero": ("n_estimators", 0), "samples_zero": ("max_samples", 0),
    "samples_bool": ("max_samples", True), "samples_fraction": ("max_samples", 2.),
    "samples_name": ("max_samples", "bogus"), "depth_zero": ("max_depth", 0),
    "features_zero": ("max_features", 0), "features_bool": ("max_features", True),
    "features_fraction": ("max_features", 2.), "features_large": ("max_features", 7),
    "contamination_zero": ("contamination", 0.), "contamination_name": ("contamination", "bogus"),
    "seed_negative": ("random_state", -1), "warm_start": ("warm_start", True), "n_jobs": ("n_jobs", 1),
    "verbose": ("verbose", True), "output_type": ("output_type", "numpy"),
    "samples_np": ("max_samples", np.float32(2)), "samples_0d": ("max_samples", np.asarray(2.)),
    "fast_mode": ("numeric_mode", "fast")}
_CASES = tuple(_PARAMETERS) + ("x_nan", "x_inf", "x_empty", "x_width0", "x_rank1", "x_rank3", "x_complex", "x_object_bad", "x_string", "sparse_csr", "sparse_lil", "y_short", "y_nan", "y_complex", "y_column", "y_scalar", "y_strings", "y_rank3", "frame_names", "frame_mixed", "bad_params_and_nan", "bad_params_and_rank3", "old_token", "depth0", "depth1", "depth2")


def _case(case, model):
    x, y = _x().copy(), None
    if case in _PARAMETERS:
        setattr(model, *_PARAMETERS[case])
    elif case in ("x_nan", "x_inf"):
        x[0, 0] = np.nan if case == "x_nan" else np.inf
    elif case == "x_empty": x = x[:0]
    elif case == "x_width0": x = x[:, :0]
    elif case == "x_rank1": x = x[0]
    elif case == "x_rank3": x = x.reshape(3, 2, 1)
    elif case == "x_complex": x = x.astype("complex64")
    elif case == "x_object_bad": x = x.astype(object); x[0, 0] = "bad"
    elif case == "x_string": x = x.astype("U")
    elif case.startswith("sparse_"): x = getattr(pytest.importorskip("scipy.sparse"), case[7:] + "_matrix")(x)
    elif case == "y_short": y = [1., 2.]
    elif case == "y_nan": y = [1., np.nan, 3.]
    elif case == "y_complex": y = np.ones(3, dtype="complex64")
    elif case == "y_column": y = np.ones((3, 1))
    elif case == "y_scalar": y = 5
    elif case == "y_strings": y = ["a", "b", "c"]
    elif case == "y_rank3": y = np.ones((3, 2, 2))
    elif case.startswith("frame_"): x = pytest.importorskip("pandas").DataFrame(x, columns=["a", "b" if case == "frame_names" else 1])
    elif case == "bad_params_and_nan": model.max_samples = 0; x[0, 0] = np.nan
    elif case == "bad_params_and_rank3": model.max_samples = 0; x = x.reshape(3, 2, 1)
    elif case == "old_token": model._if_token = 23
    elif case.startswith("depth"): model._mojolearn_fitting = int(case[-1])
    return x, y


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("refit", [False, True])
def test_iforest_fit_preserves_exact_errors_warnings_and_partial_state(case, refit):
    old, model = _model(), _model()
    if refit:
        legacy.fit(old, _x() + 10)
        legacy.fit(model, _x() + 10)
    old_args, args = _case(case, old), _case(case, model)
    error = None
    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        try:
            legacy.fit(old, *old_args)
        except Exception as caught:
            error = caught
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        audit = audit_native_call(model.fit, *args)
    if error is None:
        assert audit.assert_native() is model
    else:
        _refused(audit, type(error))
        assert type(audit.exception) is type(error) and str(audit.exception) == str(error)
    assert [(type(w.message), str(w.message)) for w in actual_warnings] == [(type(w.message), str(w.message)) for w in expected_warnings]
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("depth", [True, False, 1., .5, np.int32(1), np.float32(1), np.float16(.1),
    np.asarray(1, dtype="i4"), np.asarray(1, dtype="f4"), np.asarray([[1.]], dtype="f4")])
@pytest.mark.parametrize("fails", [False, True])
def test_iforest_fit_preserves_native_nested_depth_state(depth, fails):
    old, model = _model(), _model()
    for item in (old, model):
        vars(item)["_mojolearn_fitting"] = depth
        if fails:
            vars(item)["max_samples"] = 0
    if fails:
        with pytest.raises(ValueError) as expected:
            legacy.fit(old, _x())
        audit = audit_native_call(model.fit, _x())
        _refused(audit, ValueError)
        assert str(audit.exception) == str(expected.value)
    else:
        legacy.fit(old, _x())
        assert audit_native_call(model.fit, _x()).assert_native() is model
    actual, expected = vars(model).get("_mojolearn_fitting"), vars(old).get("_mojolearn_fitting")
    assert type(actual) is type(expected)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("field", ["n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state", "warm_start", "verbose"])
def test_iforest_fit_parameter_hooks_are_refused_before_conversion(field):
    model = _model()
    touched = []
    class Scalar:
        def __int__(self): touched.append("int"); return 1
        def __float__(self): touched.append("float"); return .5
        def __bool__(self): touched.append("bool"); return False
        def __repr__(self): touched.append("repr"); return "scalar"
    setattr(model, field, Scalar())
    _refused(audit_native_call(model.fit, _x()), TypeError)
    assert touched == []


@pytest.mark.parametrize("field,value", [("max_samples", np.asarray(2., dtype="<f4")),
    ("max_samples", np.asarray(2., dtype="<f8")), ("max_samples", np.asarray(0, dtype="<i4")),
    ("max_samples", np.asarray(False)), ("max_samples", np.float64(np.inf)),
    ("max_features", np.asarray(2., dtype="<f4")), ("contamination", np.asarray(1., dtype="<f8")),
    ("max_depth", np.asarray(0, dtype="<i4")), ("random_state", np.asarray(-1, dtype="<i8")),
    ("max_samples", np.str_("bogus")), ("max_samples", np.asarray(2., dtype=">f4")),
    ("max_features", np.asarray(2., dtype=">f8")), ("random_state", np.asarray(-1, dtype=">i8"))])
def test_iforest_fit_invalid_native_scalars_preserve_exact_error_text(field, value):
    old, model = _model(), _model()
    setattr(old, field, value)
    setattr(model, field, value)
    with pytest.raises(ValueError) as expected:
        legacy.fit(old, _x())
    audit = audit_native_call(model.fit, _x())
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("dtype,value,options", [
    ("f8", 2.675123459, {}), ("f4", 2.675123459, {}), ("f2", 2.675123459, {}),
    ("g", 2.675123459, {}), ("q", -1, {}), ("Q", 2, {}),
    ("f8", 2.675, {"precision": 2}), ("f4", 2.675, {"precision": 2}),
    ("f8", 2.1, {"floatmode": "fixed", "precision": 4}),
    ("f8", 2.675123459, {"floatmode": "unique"}),
    ("f8", 2.675123459, {"floatmode": "maxprec_equal", "precision": 3}),
    ("f8", 2.1, {"sign": "+"}), ("f8", 2.1, {"sign": " "}),
    ("f8", np.nan, {"sign": " "}), ("f8", np.inf, {"sign": "+", "infstr": "infinity"}),
    ("f8", -1e-7, {"suppress": True}), ("f8", -1e-7, {}),
    ("f4", 1e6, {}), ("f4", 2.1, {"legacy": "1.13"}),
    ("g", 2.675, {"legacy": "1.13"}),
    ("f4", 2.1, {"formatter": {"longfloat": lambda value: "unused"}}),
    ("i4", -1, {"linewidth": 12}), ("f4", 2.1, {"threshold": 0}),
    ("f4", 2.1, {"formatter": {"int": lambda value: "unused"}}),
])
def test_iforest_fit_scalar_error_preserves_native_print_configuration(dtype, value, options):
    old, model = _model(max_samples=np.asarray(value, dtype=dtype)), _model(max_samples=np.asarray(value, dtype=dtype))
    with np.printoptions(**options):
        with pytest.raises(ValueError) as expected:
            legacy.fit(old, _x())
        audit = audit_native_call(model.fit, _x())
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("dtype,value,options", [
    ("f8", 2.1, {"threshold": 0}), ("i4", -1, {"threshold": 0}),
    ("?", False, {"threshold": 0}), ("f4", 2.1, {"threshold": 1}),
    ("f4", 2.1, {"threshold": 0, "legacy": "1.21"}),
])
def test_iforest_scalar_shape_annotation_matches_numpy_release(dtype, value, options):
    old, model = _model(max_samples=np.asarray(value, dtype=dtype)), _model(max_samples=np.asarray(value, dtype=dtype))
    with np.printoptions(**options):
        with pytest.raises(ValueError) as expected:
            legacy.fit(old, _x())
        audit = audit_native_call(model.fit, _x())
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("dtype,value", [("f4", 2.), ("i4", -1), ("g", 2.), (">f4", 2.)])
def test_iforest_fit_scalar_error_never_calls_numpy_dtype_formatters(monkeypatch, dtype, value):
    import importlib
    module = importlib.import_module("numpy._core._dtype" if np.__version__.split(".")[0] != "1" else "numpy.core._dtype")
    old, model = _model(max_samples=np.asarray(value, dtype=dtype)), _model(max_samples=np.asarray(value, dtype=dtype))
    with pytest.raises(ValueError) as expected:
        legacy.fit(old, _x())
    touched = []
    def formatter(*args, **kwargs):
        touched.append("dtype formatter")
        return "foreign"
    # Native dtype descriptors can transitively call these Python functions.
    # Preserve the real poisoned module through the whole public operation.
    with monkeypatch.context() as patch:
        patch.setattr(module, "_name_get", formatter)
        patch.setattr(module, "__str__", formatter)
        audit = audit_native_call(model.fit, _x())
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))
    assert touched == []


@pytest.mark.parametrize("scalar,value", [(np.str_, "AUTO"), (np.longlong, 2), (np.ulonglong, 2), (np.longdouble, .75)])
def test_iforest_fitted_original_options_refuse_numpy_scalar_subclass_hooks(scalar, value):
    model = _model()
    legacy.fit(model, _x())
    touched = []
    def conversion(*args):
        touched.append("conversion")
        return .75
    foreign = type("ForeignOption", (scalar,), {"__int__": conversion, "__float__": conversion,
                                               "__str__": conversion, "__bool__": conversion})
    model.max_samples = foreign(value)
    _refused(audit_native_call(model.score_samples, _x()), TypeError)
    assert touched == []


@pytest.mark.parametrize("key", ["all", "float", "float_kind"])
def test_iforest_fit_error_refuses_python_print_formatters(key):
    touched = []
    def formatter(value):
        touched.append(value)
        return "custom"
    model = _model(max_samples=np.asarray(2.))
    with np.printoptions(formatter={key: formatter}):
        audit = audit_native_call(model.fit, _x())
    _refused(audit, TypeError)
    assert touched == []


@pytest.mark.parametrize("field,value", [("n_estimators", None), ("n_estimators", "bad"),
    ("n_estimators", np.nan), ("n_estimators", np.inf), ("max_samples", None),
    ("max_samples", b"bad"), ("max_features", "bad"), ("random_state", "bad"),
    ("random_state", np.inf), ("max_depth", np.inf)])
def test_iforest_fit_native_scalar_conversion_errors_preserve_exception_class(field, value):
    old, model = _model(), _model()
    setattr(old, field, value)
    setattr(model, field, value)
    with pytest.raises(Exception) as expected:
        legacy.fit(old, _x())
    audit = audit_native_call(model.fit, _x())
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("kind", ["input_protocol", "object_input", "object_parameter", "parameter_repr", "parameter_property", "assignment", "after_fit", "resident_finalizer"])
def test_iforest_fit_transitive_python_hooks_are_refused(kind):
    touched = []
    def hook(*args, **kwargs):
        touched.append(kind)
        return 1
    model, x = _model(), _x()
    if kind == "input_protocol":
        x = type("ForeignInput", (), {"__array__": hook})()
    elif kind in ("object_input", "object_parameter"):
        value = type("ForeignScalar", (), {"__float__": hook})()
        if kind == "object_input":
            x = x.astype(object); x[0, 0] = value
        else:
            model.max_samples = np.asarray(value, dtype=object)
    elif kind == "parameter_repr":
        model.max_samples = type("ForeignRepr", (str,), {"__repr__": hook})("bogus")
    elif kind in ("parameter_property", "assignment", "after_fit"):
        members = {"max_samples": property(hook)} if kind == "parameter_property" else {"__setattr__" if kind == "assignment" else "_mojolearn_after_fit": hook}
        cls = type("ForeignFitClass", (host.HostIsolationForest,), members)
        supplied = object.__new__(cls); vars(supplied).update(vars(model)); model = supplied
    else:
        model._if_finalizer = hook
    _refused(audit_native_call(model.fit, x), TypeError)
    assert touched == []


@pytest.mark.parametrize("field", ["n_estimators", "max_samples", "max_depth", "max_features", "bootstrap", "contamination", "random_state"])
def test_iforest_fit_preserves_native_constructor_option_slots(field):
    cls = type("SlottedHostIForest", (host.HostIsolationForest,), {"__slots__": (field,)})
    defaults = vars(_model())
    old, model = object.__new__(cls), object.__new__(cls)
    for item in (old, model):
        vars(item).update(defaults)
        value = vars(item).pop(field)
        setattr(item, field, value)
    legacy.fit(old, _x())
    assert audit_native_call(model.fit, _x()).assert_native() is model
    assert _state(vars(model)) == _state(vars(old))
    assert getattr(model, field) is getattr(old, field)
    assert _state(audit_native_call(model.predict, _x()).assert_native()) == _state(old_score(old, _x(), 2))


@pytest.mark.parametrize("mode", [None, "identical", "fast"])
def test_iforest_fit_preserves_inherited_mode_before_finite_check(mode):
    cls = type("InheritedModeIForest", (host.HostIsolationForest,), {"numeric_mode": mode})
    old, model = object.__new__(cls), object.__new__(cls)
    for item in (old, model):
        vars(item).update(vars(_model()))
        vars(item).pop("numeric_mode")
    x = _x(); x[0, 0] = np.nan
    with pytest.raises(ValueError) as expected:
        legacy.fit(old, x)
    audit = audit_native_call(model.fit, x)
    _refused(audit, ValueError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


_NUMPY_CHILD = r'''
import json,sys,importlib.abc
# -S disables site initialization, but PYTHONPATH intentionally exposes the
# installed package directory and therefore its optional NumPy neighbor too.
# Model genuine absence until the explicit late-import arm enables NumPy.
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise ModuleNotFoundError('NumPy intentionally unavailable', name='numpy')
block_numpy = BlockNumPy()
sys.meta_path.insert(0, block_numpy)
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
payload=json.loads(sys.argv[1])
assert "numpy" not in sys.modules
if payload["numpy"]:
    sys.meta_path.remove(block_numpy)
    sys.path.extend(payload["site_paths"])
    import numpy
model=host.HostIsolationForest(n_estimators=7,max_samples=3,max_depth=2,random_state=11,numeric_mode="identical")
audit_native_call(model.fit,payload["x"]).assert_native()
outputs=[audit_native_call(getattr(model,name),payload["x"]).assert_native().tobytes().hex()
         for name in ("score_samples","decision_function","predict")]
print(json.dumps(dict(x=vars(model)["_x"].tobytes().hex(),offset=model.offset_,max_samples=model.max_samples_,outputs=outputs)))
'''


@pytest.mark.parametrize("numpy", [False, True])
def test_iforest_fit_without_numpy_and_with_numpy_imported_later(numpy):
    old = _model()
    x = _x().tolist()
    legacy.fit(old, x)
    expected = dict(x=vars(old)["_x"].tobytes().hex(), offset=old.offset_, max_samples=old.max_samples_,
                    outputs=[old_score(old, x, want).tobytes().hex() for want in range(3)])
    site_paths = [path for path in sys.path if "site-packages" in path]
    env = dict(os.environ, PYTHONPATH=str(Path(host.__file__).resolve().parents[1]))
    payload = dict(x=x, numpy=numpy, site_paths=site_paths)
    completed = subprocess.run([sys.executable, "-S", "-c", _NUMPY_CHILD, json.dumps(payload)], env=env,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == expected


def test_iforest_fit_public_wrapper_is_one_native_call():
    source = Path(host.__file__).read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HostIsolationForest")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    body = method.body[1:]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    assert sum(isinstance(n, ast.Call) for n in ast.walk(body[0])) == 1
