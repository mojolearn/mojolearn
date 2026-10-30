# SPDX-License-Identifier: Apache-2.0
"""Whole CPU KDE fit preserves independent state, aliases and refusal ordering."""
import ast
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
import warnings

import numpy as np
import pytest

from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests._connected_exports import assert_connected_export_refused
from mojolearn.tests import _legacy_host_inference as legacy
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_inference_native import _refused
from mojolearn.tests.test_host_kernel_inference_native import FORMATS


def _x(kind="c"):
    x = np.asarray([[1., 2.], [3., 4.], [5., 6.]], dtype="<f4")
    if kind == "list":
        return x.tolist()
    if kind == "fortran":
        return np.asfortranarray(x)
    if kind == "strided":
        return np.repeat(x, 2, axis=1)[:, ::2]
    if kind == "negative_stride":
        return x[::-1]
    if kind == "readonly":
        x.flags.writeable = False
    if kind in ("float64", "half", "int16", "big_endian"):
        return x.astype({"float64": "<f8", "half": "<f2", "int16": "<i2", "big_endian": ">f4"}[kind])
    if kind == "array":
        return Array.from_buffer(x)
    if kind == "memoryview":
        return memoryview(x)
    return x


def _weight(kind):
    w = np.asarray([1., 2., 1.], dtype="<f4")
    if kind == "none":
        return None
    if kind == "scalar":
        return 2.
    if kind == "zero_dim":
        return np.asarray(2.)
    if kind == "numpy_bool":
        return np.bool_(True)
    if kind == "list":
        return w.tolist()
    if kind == "float64":
        return w.astype("<f8")
    if kind == "big_endian":
        return w.astype(">f4")
    if kind == "strided":
        return np.repeat(w, 2)[::2]
    if kind == "readonly":
        w.flags.writeable = False
    return w


def _model():
    return host.HostKernelDensity(bandwidth=.75, numeric_mode="identical")


def _pair(x, weight=None, y=None, *, old=None, actual=None):
    old = _model() if old is None else old
    actual = _model() if actual is None else actual
    legacy.kde_fit(old, x, y, weight)
    assert audit_native_call(actual.fit, x, y, sample_weight=weight).assert_native() is actual
    assert _state(vars(actual)) == _state(vars(old))
    assert type(vars(actual)["_x"]) is Array
    assert vars(actual)["_w"] is None or type(vars(actual)["_w"]) is Array
    assert _state(actual.score_samples(_x())) == _state(legacy.kde(old, _x()))
    return old, actual


@pytest.mark.parametrize("kind", ["c", "list", "fortran", "strided", "negative_stride", "readonly", "float64", "half", "int16", "big_endian", "array", "memoryview"])
@pytest.mark.parametrize("weight", ["none", "c", "scalar", "zero_dim", "numpy_bool", "list", "float64", "big_endian", "strided", "readonly"])
def test_kde_fit_preserves_complete_state_and_consumed_predictions(kind, weight):
    _pair(_x(kind), _weight(weight))


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("dtype", ["<f4", "<f8", "<i2", "?"])
@pytest.mark.parametrize("container", ["matrix", "array"])
def test_kde_fit_sparse_formats_preserve_original_copy_metadata(format, dtype, container):
    sp = pytest.importorskip("scipy.sparse")
    x = getattr(sp, format + "_" + container)(_x().astype(dtype))
    _pair(x, _weight("c"))


@pytest.mark.parametrize("kind", ["frame_c", "frame_dict", "frame_f64", "frame_mixed", "frame_integer", "torch", "torch_strided"])
def test_kde_fit_dataframe_and_torch_storage_preserve_state(kind):
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    x = _x()
    if kind == "frame_c":
        x = pd.DataFrame(x, columns=["a", "b"])
    elif kind == "frame_dict":
        x = pd.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    elif kind == "frame_f64":
        x = pd.DataFrame(x.astype("<f8"))
    elif kind == "frame_mixed":
        x = pd.DataFrame(x, columns=["a", 1])
    elif kind == "frame_integer":
        x = pd.DataFrame(x, columns=[0, 1])
    elif kind == "torch":
        x = torch.from_numpy(x)
    else:
        x = torch.from_numpy(np.repeat(x, 2, axis=1))[:, ::2]
    _pair(x, pd.Series([1., 2., 1.], dtype="float32"))


@pytest.mark.parametrize("kind", ["none", "scalar", "strings", "string_array", "object_array", "list_column", "matrix", "column", "sparse", "finite_large", "rank3"])
def test_kde_fit_preserves_optional_target_validation(kind):
    y = {"none": None, "scalar": 5, "strings": ["a", "b", "c"], "string_array": np.asarray(["a", "b", "c"]),
         "object_array": np.asarray(["a", "b", "c"], dtype=object), "list_column": [[1.], [2.], [3.]],
         "matrix": np.ones((3, 2)), "column": np.ones((3, 1)), "finite_large": np.full(3, 1e300), "rank3": np.ones((3, 2, 2))}.get(kind)
    if kind == "sparse":
        y = pytest.importorskip("scipy.sparse").csr_matrix(np.ones((3, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _pair(_x(), y=y)


BAD_CASES = ("weight_short", "weight_empty", "weight_column", "weight_zero", "weight_negative", "weight_nan", "weight_inf", "weight_bool", "weight_complex",
             "x_nan", "x_inf", "x_complex", "x_empty", "x_rank1", "x_width0", "x_rank3", "y_short", "y_nan", "y_complex", "fast_mode")


def _bad(case, model):
    x, y, w = _x(), None, None
    weights = {"weight_short": [1., 2.], "weight_empty": [], "weight_column": np.ones((3, 1)), "weight_zero": [0., 0., 0.],
               "weight_negative": [1., -1., 1.], "weight_nan": [1., np.nan, 1.], "weight_inf": [1., np.inf, 1.],
               "weight_bool": True, "weight_complex": np.ones(3, dtype="complex64")}
    if case in weights:
        w = weights[case]
    elif case == "x_nan":
        x[0, 0] = np.nan
    elif case == "x_inf":
        x[0, 0] = np.inf
    elif case == "x_complex":
        x = x.astype("complex64")
    elif case == "x_empty":
        x = x[:0]
    elif case == "x_rank1":
        x = x[0]
    elif case == "x_width0":
        x = x[:, :0]
    elif case == "x_rank3":
        x = x.reshape(3, 2, 1)
    elif case == "y_short":
        y = [1., 2.]
    elif case == "y_nan":
        y = [1., np.nan, 1.]
    elif case == "y_complex":
        y = np.ones(3, dtype="complex64")
    elif case == "fast_mode":
        model.numeric_mode = "fast"
    return x, y, w


@pytest.mark.parametrize("case", BAD_CASES)
@pytest.mark.parametrize("refit", [False, True])
def test_kde_fit_refusals_preserve_exact_error_and_partial_state(case, refit):
    old, model = _model(), _model()
    if refit:
        legacy.kde_fit(old, _x() + 10, sample_weight=_weight("c"))
        legacy.kde_fit(model, _x() + 10, sample_weight=_weight("c"))
    args = _bad(case, model)
    old_args = _bad(case, old)
    with pytest.raises(Exception) as expected:
        legacy.kde_fit(old, *old_args)
    audit = audit_native_call(model.fit, *args)
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("kind", ["tuple", "range", "ndarray", "array", "memoryview"])
@pytest.mark.parametrize("refit", [False, True])
def test_kde_empty_weight_storage_preserves_shape_error_and_partial_state(kind, refit):
    empty = np.empty(0, dtype="<f4")
    weight = {"tuple": (), "range": range(0), "ndarray": empty,
              "array": Array.from_buffer(empty), "memoryview": memoryview(empty)}[kind]
    old, model = _model(), _model()
    if refit:
        legacy.kde_fit(old, _x() + 10, sample_weight=_weight("c"))
        legacy.kde_fit(model, _x() + 10, sample_weight=_weight("c"))
    with pytest.raises(ValueError) as expected:
        legacy.kde_fit(old, _x(), sample_weight=weight)
    audit = audit_native_call(model.fit, _x(), sample_weight=weight)
    _refused(audit, type(expected.value))
    assert type(audit.exception) is type(expected.value)
    assert str(audit.exception) == str(expected.value) == "sample_weight.shape == (0,), expected (3,)!"
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("name", ["list", "tuple", "range"])
def test_kde_weight_admission_ignores_replaced_container_types(name, monkeypatch):
    import builtins
    import dis
    import threading
    from mojolearn import _verify_native_runtime as verifier
    touched = []
    class Foreign:
        def __len__(self):
            touched.append("len")
            return 0
        def __iter__(self):
            touched.append("iter")
            return iter(())
        def __float__(self):
            touched.append("float")
            return 1.
    model, x, weight = _model(), _x(), Foreign()
    fit = model.fit
    with monkeypatch.context() as patch:
        # The verifier is a test controller; preserve its bytecode inspection
        # constructors while poisoning the product's live builtin lookup.
        for controller in (verifier, dis, threading):
            patch.setattr(controller, name, getattr(builtins, name), raising=False)
        patch.setattr(builtins, name, Foreign)
        audit = audit_native_call(fit, x, sample_weight=weight)
    _refused(audit, TypeError)
    assert touched == []


@pytest.mark.parametrize("mode", [None, "identical", "fast"])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_kde_fit_preserves_inherited_mode_and_instance_precedence(mode, indirect, override):
    class ModeKDE(host.HostKernelDensity):
        numeric_mode = mode
    class DescendantKDE(ModeKDE):
        pass
    cls = DescendantKDE if indirect else ModeKDE
    options = {"numeric_mode": "identical"} if override else {}
    old, model = cls(bandwidth=.75, **options), cls(bandwidth=.75, **options)
    assert ("numeric_mode" in vars(model)) is override
    if mode == "fast" and not override:
        with pytest.raises(ValueError) as expected:
            legacy.kde_fit(old, _x())
        audit = audit_native_call(model.fit, _x())
        _refused(audit, ValueError)
        assert str(audit.exception) == str(expected.value)
        assert "_x" not in vars(model) and model.input_copied_ is False
    else:
        legacy.kde_fit(old, _x())
        assert audit_native_call(model.fit, _x()).assert_native() is model
        assert _state(model.score_samples(_x())) == _state(legacy.kde(old, _x()))
    assert _state(vars(model)) == _state(vars(old))
    assert ("numeric_mode" in vars(model)) is override


@pytest.mark.parametrize("kind", ["property", "descriptor", "data_descriptor"])
@pytest.mark.parametrize("instance_mode", [False, True])
def test_kde_fit_class_mode_descriptors_never_execute_hooks(kind, instance_mode):
    touched = []
    class ModeKDE(host.HostKernelDensity):
        pass
    model = ModeKDE(bandwidth=.75, numeric_mode="identical")
    if not instance_mode:
        del vars(model)["numeric_mode"]
    class Descriptor:
        def __get__(self, instance, owner):
            touched.append("get")
            return "identical"
    class DataDescriptor(Descriptor):
        def __set__(self, instance, value):
            touched.append("set")
    def get_mode(self):
        touched.append("property")
        return "identical"
    ModeKDE.numeric_mode = {"property": property(get_mode), "descriptor": Descriptor(),
                            "data_descriptor": DataDescriptor()}[kind]
    audit = audit_native_call(model.fit, _x())
    if instance_mode and kind == "descriptor":
        assert audit.assert_native() is model
        assert model.n_samples_fit_ == 3
    else:
        _refused(audit, TypeError)
        assert "_x" not in vars(model)
    assert touched == []


@pytest.mark.parametrize("field,value", [("bandwidth", "scott"), ("algorithm", "kd_tree"), ("kernel", "bogus"), ("metric", "bogus"), ("leaf_size", -1)])
def test_kde_fit_does_not_add_inference_parameter_validation(field, value):
    old, model = _model(), _model()
    setattr(old, field, value)
    setattr(model, field, value)
    legacy.kde_fit(old, _x())
    assert audit_native_call(model.fit, _x()).assert_native() is model
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("readonly", [False, True])
def test_kde_fit_retains_zero_copy_input_and_weight_owners(readonly):
    x, w = _x(), _weight("c")
    if readonly:
        x.flags.writeable = w.flags.writeable = False
    model = _model()
    audit_native_call(model.fit, x, sample_weight=w).assert_native()
    assert vars(model)["_x"]._addr == x.ctypes.data
    assert vars(model)["_w"]._addr == w.ctypes.data
    assert vars(model)["_x"]._readonly is readonly
    assert vars(model)["_w"]._readonly is readonly
    if not readonly:
        x[0, 0], w[0] = 8., 5.
    expected = _state(legacy.kde(model, _x()))
    del x, w
    gc.collect()
    assert _state(audit_native_call(model.score_samples, _x()).assert_native()) == expected


def test_kde_fit_refit_clears_old_cache_and_feature_names():
    old, model = _model(), _model()
    for item in (old, model):
        legacy.kde_fit(item, _x(), sample_weight=_weight("c"))
        # Warm a real owned cache when this backend supports one. A fabricated
        # integer handle is not a valid native residency ownership record.
        item.score_samples(_x())
        item.feature_names_in_ = np.asarray(["old_a", "old_b"], dtype=object)
    _pair(_x() + 1, old=old, actual=model)
    assert vars(model).get("_resident") is None and "feature_names_in_" not in vars(model)


@pytest.mark.parametrize("slot", ["X", "y", "sample_weight", "_mojolearn_fitting", "numeric_mode", "_mojolearn_after_fit"])
def test_kde_fit_refuses_python_protocols_before_hooks(slot):
    model = _model()
    touched = []
    class Callback:
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return _x()
        def __int__(self):
            touched.append("int")
            return 0
        def __bool__(self):
            touched.append("bool")
            return False
        def __call__(self, *args, **kwargs):
            touched.append("call")
    x, y, w = _x(), None, None
    if slot == "X":
        x = Callback()
    elif slot == "y":
        y = Callback()
    elif slot == "sample_weight":
        w = Callback()
    else:
        setattr(model, slot, Callback())
    _refused(audit_native_call(model.fit, x, y, sample_weight=w), TypeError)
    assert touched == []


def test_kde_fit_never_calls_binding_selection_or_residency_python(monkeypatch):
    model = _model()
    model._resident = ((123,), 7)
    touched = []
    def forbidden(*args, **kwargs):
        touched.append("callback")
        raise AssertionError("Python product orchestration")
    monkeypatch.setattr(host._HostBound, "_bind", forbidden)
    monkeypatch.setattr(host.KernelDensity, "_release_resident_fit", forbidden)
    audit_native_call(model.fit, _x(), sample_weight=_weight("c")).assert_native()
    assert touched == [] and model._resident is None


def test_kde_fit_public_wrapper_is_one_native_call():
    tree = ast.parse(Path(host.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HostKernelDensity")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    assert len(method.body) == 2 and isinstance(method.body[1], ast.Return)
    assert sum(isinstance(n, ast.Call) for n in ast.walk(method.body[1])) == 1
    assert getattr(vars(host.HostKernelDensity)["fit"], "_mojolearn_tracks_fit", False)
    assert not hasattr(vars(host.HostKernelDensity)["fit"], "__wrapped__")


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("failure", [False, True])
def test_kde_fit_preserves_nested_fitting_marker_cleanup(depth, failure):
    old, model = _model(), _model()
    old._mojolearn_fitting = model._mojolearn_fitting = depth
    weight = [1., -1., 1.] if failure else None
    if failure:
        with pytest.raises(ValueError) as expected:
            legacy.kde_fit(old, _x(), sample_weight=weight)
        audit = audit_native_call(model.fit, _x(), sample_weight=weight)
        _refused(audit)
        assert str(audit.exception) == str(expected.value)
    else:
        legacy.kde_fit(old, _x())
        audit_native_call(model.fit, _x()).assert_native()
    assert _state(vars(model)) == _state(vars(old))


@pytest.mark.parametrize("target", ["X", "sample_weight"])
def test_kde_fit_uint64_signed_storage_refusal_preserved(target):
    old, model = _model(), _model()
    x, weight = _x(), None
    if target == "X":
        x = np.full((3, 2), 2**63, dtype="<u8")
    else:
        weight = np.full(3, 2**63, dtype="<u8")
    with pytest.raises(OverflowError) as expected:
        legacy.kde_fit(old, x, sample_weight=weight)
    audit = audit_native_call(model.fit, x, sample_weight=weight)
    _refused(audit, OverflowError)
    assert str(audit.exception) == str(expected.value)
    assert _state(vars(model)) == _state(vars(old))


_FIT_CHILD = r'''
import importlib.abc,json,sys
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            # Model a missing optional dependency; a broken import must still propagate.
            raise ModuleNotFoundError("NumPy deliberately unavailable", name="numpy")
blocker=BlockNumPy()
sys.meta_path.insert(0,blocker)
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
payload=json.load(sys.stdin)
if payload["order"] == "late":
    sys.meta_path.remove(blocker)
    import numpy
model=host.HostKernelDensity(bandwidth=.75,numeric_mode="identical")
assert audit_native_call(model.fit,payload["x"],sample_weight=payload["weight"]).assert_native() is model
assert type(vars(model)["_x"]).__name__=="Array"
result=audit_native_call(model.score_samples,payload["x"]).assert_native()
assert type(result).__name__ == ("Array" if payload["order"]=="absent" else "ndarray")
print(json.dumps(dict(shape=list(result.shape),bytes=result.tobytes().hex(),copied=model.input_copied_)))
if payload["order"]=="absent":
    assert "numpy" not in sys.modules
'''


@pytest.mark.parametrize("order", ["absent", "late"])
@pytest.mark.parametrize("weight", [None, 2., [1., 2., 1.]])
def test_kde_fit_numpy_absent_and_late_import_preserve_outputs(order, weight):
    x = _x().tolist()
    old = _model()
    legacy.kde_fit(old, x, sample_weight=weight)
    result = legacy.kde(old, x)
    expected = dict(shape=list(result.shape), bytes=result.tobytes().hex(), copied=old.input_copied_)
    done = subprocess.run([sys.executable, "-c", _FIT_CHILD], input=json.dumps(dict(order=order, weight=weight, x=x)),
                          text=True, capture_output=True, env=dict(os.environ), timeout=90)
    assert done.returncode == 0, done.stderr[-4000:]
    assert json.loads(done.stdout) == expected


@pytest.mark.parametrize("slot", ["X", "sample_weight", "_x", "_w", "feature_names_in_"])
def test_kde_fit_refuses_object_storage_before_conversion_or_finalizer_hooks(slot):
    model = _model()
    touched = []
    class Value:
        def __float__(self):
            touched.append("float")
            return 1.
        def __del__(self):
            touched.append("del")
    objects = np.empty((3, 2) if slot in ("X", "_x") else (3,), dtype=object)
    owners = []
    for index in np.ndindex(objects.shape):
        value = Value()
        owners.append(value)
        objects[index] = value
    x, weight = _x(), None
    if slot == "X":
        x = objects
    elif slot == "sample_weight":
        weight = objects
    else:
        setattr(model, slot, objects)
    _refused(audit_native_call(model.fit, x, sample_weight=weight), TypeError)
    assert touched == []
    # Test cleanup is outside the operation; retain every owner through audit.
    del model, x, weight, objects, value, owners


def test_kde_fit_refuses_class_after_fit_callback_without_invoking_it():
    touched = []
    class CallbackKDE(host.HostKernelDensity):
        def _mojolearn_after_fit(self):
            touched.append("after")
    model = CallbackKDE(bandwidth=.75, numeric_mode="identical")
    _refused(audit_native_call(model.fit, _x()), TypeError)
    assert touched == []


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly"])
@pytest.mark.parametrize("kind", ["integer", "float", "numpy_scalar"])
def test_kde_fit_preserves_native_numeric_object_buffers(layout, kind):
    x = _x(layout if layout != "c" else "c")
    if kind == "integer":
        x = x.astype("<i4").astype(object)
    elif kind == "float":
        x = x.astype(object)
    else:
        data = np.empty(x.shape, dtype=object)
        for index in np.ndindex(x.shape):
            data[index] = np.float32(x[index])
        x = data
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "strided":
        x = np.repeat(x, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        x.flags.writeable = False
    _pair(x, _weight("c").astype(object), y=[1., 2.])


@pytest.mark.parametrize("kind", ["numpy_uint64", "zero_dim_uint64"])
def test_kde_fit_large_unsigned_scalar_weights_keep_scalar_contract(kind):
    weight = np.uint64(2**63 + 2**40)
    if kind == "zero_dim_uint64":
        weight = np.asarray(weight)
    _pair(_x(), weight)


@pytest.mark.parametrize("branch", ["borrow", "cast", "pandas_borrow", "pandas_numpy_objects", "csr", "lil", "numpy_scalar", "target_rank3", "complex_weight"])
def test_kde_fit_never_calls_replaced_memoryview_constructor(branch, monkeypatch):
    import builtins
    x, weight, y = _x(), _weight("c"), None
    if branch == "cast":
        x = x.astype("<f8")
    elif branch == "pandas_borrow":
        x = pytest.importorskip("pandas").DataFrame(x, copy=False)
    elif branch == "pandas_numpy_objects":
        data = np.empty(x.shape, dtype=object)
        for index in np.ndindex(x.shape):
            data[index] = np.float64(x[index])
        x = pytest.importorskip("pandas").DataFrame(data, copy=False)
    elif branch in ("csr", "lil"):
        x = getattr(pytest.importorskip("scipy.sparse"), branch + "_matrix")(x)
    elif branch == "numpy_scalar":
        weight = np.float64(2.)
    elif branch == "target_rank3":
        y = np.ones((3, 2, 2))
    elif branch == "complex_weight":
        weight = np.ones(3, dtype="complex64")
    old, model = _model(), _model()
    expected_error = None
    try:
        legacy.kde_fit(old, x, y, weight)
    except Exception as error:
        expected_error = (type(error), str(error))
    expected = _state(vars(old))
    touched = []
    def poisoned(*args, **kwargs):
        touched.append("memoryview")
        raise AssertionError("mutable buffer constructor executed")
    # Preserve the native descriptor identity read by pandas admission;
    # the exercised constructor replacement itself remains a Python function.
    poisoned.obj = builtins.memoryview.obj
    fit = model.fit  # bind first: the native attribute read consults the connected builtins
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "memoryview", poisoned)
        audit = audit_native_call(fit, x, y, sample_weight=weight)
    # A replaced connected builtins export refuses before fit touches the
    # model (see _connected_exports); the restored export then fits.
    assert_connected_export_refused(audit, "builtins", "memoryview")
    assert touched == [] and _state(vars(model)) == _state(vars(_model()))
    audit = audit_native_call(model.fit, x, y, sample_weight=weight)
    if expected_error is None:
        assert audit.assert_native() is model
    else:
        _refused(audit, expected_error[0])
        assert str(audit.exception) == expected_error[1]
    assert _state(vars(model)) == expected
    assert touched == []


@pytest.mark.parametrize("branch", ["borrow", "cast"])
def test_kde_fit_never_calls_replaced_object_allocator(branch, monkeypatch):
    import builtins
    old, model = _model(), _model()
    x = _x() if branch == "borrow" else _x().astype("<f8")
    legacy.kde_fit(old, x)
    expected = _state(vars(old))
    touched = []
    class PoisonObject:
        def __new__(cls, *args, **kwargs):
            touched.append("object.__new__")
            raise AssertionError("mutable allocator executed")
    fit = model.fit  # bind first: the native attribute read consults the connected builtins
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "object", PoisonObject)
        audit = audit_native_call(fit, x)
    assert_connected_export_refused(audit, "builtins", "object")
    assert touched == [] and _state(vars(model)) == _state(vars(_model()))
    audit = audit_native_call(model.fit, x)
    assert audit.assert_native() is model
    assert _state(vars(model)) == expected and touched == []
