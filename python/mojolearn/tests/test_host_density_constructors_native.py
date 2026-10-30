# SPDX-License-Identifier: Apache-2.0
"""Public density constructors retain independent state and native boundaries."""
import ast
import gc
import inspect
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_host_density_constructors as legacy
from mojolearn.tests.test_host_archive_native import _state
from mojolearn.tests.test_host_inference_native import _refused

CLASSES = {"kde": host.HostKernelDensity, "iforest": host.HostIsolationForest}
DEFAULTS = {"kde": legacy.KDE_DEFAULTS, "iforest": legacy.IFOREST_DEFAULTS}
INVALID = {
    "kde": dict(bandwidth="scott", algorithm="kd_tree", kernel="bogus", metric="bogus", atol=-2,
                rtol=3, breadth_first=False, leaf_size=-9, metric_params={"p": np.float32(2)}),
    "iforest": dict(n_estimators=-1, max_samples="BOGUS", max_depth=-2, max_features=-1,
                    bootstrap=True, contamination="BOGUS", random_state=123, warm_start=True,
                    n_jobs=2, verbose=True, output_type="numpy"),
}


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("mode", [None, "identical", "fast", "custom-retained-mode"])
@pytest.mark.parametrize("invalid", [False, True])
def test_density_constructors_preserve_deferred_validation_and_exact_state(family, mode, invalid):
    options = dict(INVALID[family] if invalid else {}, numeric_mode=mode)
    cls = CLASSES[family]
    expected = legacy.construct(cls, family, **options)
    actual = audit_native_call(cls, **options).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    for name, value in options.items():
        assert getattr(actual, name) is value


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("kind", ["numpy_scalar", "complex_scalar", "numpy_string", "list", "mapping", "array", "ndarray", "memoryview"])
def test_density_constructor_retains_native_parameter_identity(family, kind):
    values = {"numpy_scalar": np.float32(1.25), "complex_scalar": np.complex64(1+2j),
              "numpy_string": np.str_("unchanged"), "list": [1., 2.], "mapping": {"p": np.int64(2)},
              "array": Array.from_buffer(np.asarray([1, 2], dtype="<f4")),
              "ndarray": np.asarray([1., 2.]), "memoryview": memoryview(bytearray(b"abcd"))}
    name = "metric_params" if family == "kde" else "random_state"
    value = values[kind]
    actual = audit_native_call(CLASSES[family], **{name: value}).assert_native()
    assert getattr(actual, name) is value
    expected = legacy.construct(CLASSES[family], family, **{name: value})
    assert _state(vars(actual)) == _state(vars(expected))


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("previous", [None, "fast"])
@pytest.mark.parametrize("supplied", [None, "identical", "custom"])
def test_density_reinitialize_preserves_unrelated_state_and_mode(family, previous, supplied):
    cls = CLASSES[family]
    expected, actual = cls(numeric_mode=previous), cls(numeric_mode=previous)
    stored = np.asarray([[1., 2.]], dtype="<f4")
    for model in (expected, actual):
        model._x = stored
        model.n_features_in_ = 2
        model._mojolearn_fitted = True
        model.marker = {"keep": [1, 2]}
    legacy.initialize(expected, family, numeric_mode=supplied)
    assert audit_native_call(cls.__init__, actual, numeric_mode=supplied).assert_native() is None
    assert _state(vars(actual)) == _state(vars(expected))
    assert vars(actual)["_x"] is stored


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("mode", [None, "identical", "fast"])
@pytest.mark.parametrize("indirect", [False, True])
def test_density_constructor_preserves_inherited_mode_without_instance_field(family, mode, indirect):
    class Inherited(CLASSES[family]):
        numeric_mode = mode
    class Descendant(Inherited):
        pass
    cls = Descendant if indirect else Inherited
    expected = legacy.construct(cls, family)
    actual = audit_native_call(cls).assert_native()
    assert _state(vars(actual)) == _state(vars(expected))
    assert "numeric_mode" not in vars(actual) and actual.numeric_mode is mode


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mode", [None, "identical", "fast"])
def test_density_constructor_preserves_native_slot_parameters_and_mode(family, existing, mode):
    parameter = "bandwidth" if family == "kde" else "n_estimators"
    class Slotted(CLASSES[family]):
        __slots__ = (parameter, "numeric_mode")
    actual, expected = object.__new__(Slotted), object.__new__(Slotted)
    if existing:
        actual.numeric_mode = expected.numeric_mode = "fast"
    legacy.initialize(expected, family, numeric_mode=mode)
    assert audit_native_call(Slotted.__init__, actual, numeric_mode=mode).assert_native() is None
    assert _state(vars(actual)) == _state(vars(expected))
    assert getattr(actual, parameter) == getattr(expected, parameter)
    assert actual.numeric_mode == expected.numeric_mode
    assert parameter not in vars(actual) and "numeric_mode" not in vars(actual)


@pytest.mark.parametrize("family", CLASSES)
def test_density_constructor_public_signature_keeps_keyword_defaults(family):
    signature = inspect.signature(CLASSES[family])
    expected = dict(DEFAULTS[family], numeric_mode=None)
    assert set(signature.parameters) == set(expected)
    for name, parameter in signature.parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == expected[name]


@pytest.mark.parametrize("family", CLASSES)
def test_density_constructor_public_method_is_one_native_call(family):
    tree = ast.parse(Path(host.__file__).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == CLASSES[family].__name__)
    method = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    body = method.body[1:] if isinstance(method.body[0], ast.Expr) and isinstance(method.body[0].value, ast.Constant) else method.body
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_native_classical"
    assert call.args[1].value == "host_density_initialize"


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("hook", ["setattr", "property", "descriptor", "mode_property"])
def test_density_constructor_refuses_python_assignment_hooks(family, hook):
    touched = []
    class Custom(CLASSES[family]):
        pass
    def setter(self, *args):
        touched.append(hook)
    def getter(self):
        touched.append(hook)
        return None
    class Descriptor:
        def __get__(self, instance, owner):
            touched.append("get")
            return None
        def __set__(self, instance, value):
            touched.append("set")
    parameter = "bandwidth" if family == "kde" else "n_estimators"
    if hook == "setattr":
        Custom.__setattr__ = setter
    elif hook == "mode_property":
        Custom.numeric_mode = property(getter, setter)
    elif hook == "property":
        setattr(Custom, parameter, property(getter, setter))
    else:
        setattr(Custom, parameter, Descriptor())
    _refused(audit_native_call(Custom), TypeError)
    assert touched == []


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("nested", [False, True])
def test_density_constructor_refuses_protocol_options_without_hooks(family, nested):
    touched = []
    class Callback:
        def __float__(self):
            touched.append("float")
            return 1.
        def __array__(self, *args, **kwargs):
            touched.append("array")
            return np.ones(1)
    value = Callback()
    option = {"p": [value]} if nested else value
    parameter = "metric_params" if family == "kde" else "random_state"
    _refused(audit_native_call(CLASSES[family], **{parameter: option}), TypeError)
    assert touched == []


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("owner", ["object", "container", "object_array", "array_slot", "weakref"])
def test_density_reinitialize_refuses_callback_bearing_replaced_owners(family, owner):
    touched = []
    class Finalizer:
        def __del__(self):
            touched.append("finalizer")
    cls = CLASSES[family]
    model = cls()
    parameter = "metric_params" if family == "kde" else "random_state"
    value = np.ones(1) if owner == "weakref" else Finalizer()
    if owner == "container":
        value = {"p": [value]}
    elif owner == "object_array":
        values = np.empty(1, dtype=object)
        values[0] = value
        value = values
    elif owner == "array_slot":
        array = Array.from_buffer(np.ones(1, dtype="<f4"))
        array._base = value
        value = array
    elif owner == "weakref":
        ref = weakref.ref(value, lambda _: touched.append("weakref"))
    setattr(model, parameter, value)
    identity = id(value)
    del value
    if owner == "object_array":
        del values
    elif owner == "array_slot":
        del array
    audit = audit_native_call(cls.__init__, model)
    _refused(audit, TypeError)
    assert id(getattr(model, parameter)) == identity and touched == []
    del model
    gc.collect()


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("kind", ["array", "numeric", "object", "structured", "structured_object", "dtype_metadata", "subarray_dtype"])
def test_density_reinitialize_preserves_safe_native_owner_release(family, kind):
    cls = CLASSES[family]
    model = cls()
    parameter = "metric_params" if family == "kde" else "random_state"
    if kind == "array":
        value = Array.from_buffer(np.asarray([1., 2.], dtype="<f4"))
    elif kind == "numeric":
        value = np.asarray([1., 2.])[::-1]
    elif kind == "object":
        value = np.asarray([np.int64(2), "native", 1+2j], dtype=object)
    elif kind == "structured":
        value = np.asarray([(1, [2., 3.])], dtype=[("code", "<i4"), ("x", "<f4", (2,))])
    elif kind == "structured_object":
        value = np.empty(1, dtype=[("code", "<i4"), ("tag", object)])
        value["code"] = 1
        value["tag"][0] = {"name": ["native", np.float32(2)]}
    elif kind == "dtype_metadata":
        value = np.dtype("f4", metadata={"label": ["native", np.int64(2)]})
    else:
        value = np.dtype((np.dtype("f4"), (2, 3)))
    setattr(model, parameter, value)
    del value
    assert audit_native_call(cls.__init__, model).assert_native() is None
    assert getattr(model, parameter) is None
    assert _state(vars(model)) == _state(vars(legacy.construct(cls, family)))


_CONSTRUCTOR_CHILD = r'''
import importlib.abc, json, sys
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            # Model a missing optional dependency; a broken import must still propagate.
            raise ModuleNotFoundError("NumPy deliberately unavailable", name="numpy")
blocker = BlockNumPy()
sys.meta_path.insert(0, blocker)
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
payload = json.load(sys.stdin)
cls = host.HostKernelDensity if payload["family"] == "kde" else host.HostIsolationForest
parameter = "metric_params" if payload["family"] == "kde" else "random_state"
value = {"p": [2, 3]}
if payload["order"] == "late":
    sys.meta_path.remove(blocker)
    import numpy as np
    value = np.asarray([1., 2.])
model = audit_native_call(cls, numeric_mode="fast", **{parameter: value}).assert_native()
assert getattr(model, parameter) is value
assert audit_native_call(cls.__init__, model).assert_native() is None
assert model.numeric_mode == "fast" and getattr(model, parameter) is None
if payload["order"] == "absent":
    assert "numpy" not in sys.modules
print(json.dumps({"mode": model.numeric_mode, "keys": sorted(vars(model))}))
'''


@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("order", ["absent", "late"])
def test_density_constructor_numpy_absent_and_late_import_are_native(family, order):
    done = subprocess.run([sys.executable, "-c", _CONSTRUCTOR_CHILD], text=True, capture_output=True,
                          input=json.dumps(dict(family=family, order=order)), env=dict(os.environ), timeout=60)
    assert done.returncode == 0, done.stderr[-4000:]
    assert json.loads(done.stdout) == {"mode": "fast", "keys": sorted((*DEFAULTS[family], "numeric_mode"))}


@pytest.mark.parametrize("family", ["HostKernelDensity", "HostIsolationForest", "HostElasticNet", "HostKMeans", "HostStandardScaler", "HostLinearRegression"])
def test_native_unset_member_keeps_attribute_error_and_default_lookup(family):
    cls = type("UnsetNativeSlot", (getattr(host, family),), {"__slots__": ("native_probe",)})
    model = object.__new__(cls)
    with pytest.raises(AttributeError) as expected:
        object.__getattribute__(model, "native_probe")
    audit = audit_native_call(getattr, model, "native_probe")
    _refused(audit, AttributeError)
    assert str(audit.exception) == str(expected.value)
    sentinel = object()
    assert audit_native_call(getattr, model, "native_probe", sentinel).assert_native() is sentinel
    assert audit_native_call(hasattr, model, "native_probe").assert_native() is False
    model.native_probe = 7
    assert audit_native_call(getattr, model, "native_probe").assert_native() == 7
    assert audit_native_call(hasattr, model, "native_probe").assert_native() is True
