"""Real native manifold import, output and subclass contracts.

Run against an installed current build. Native owners are required: missing
bindings fail rather than skip. This is bounded correctness, not a benchmark.
The NumPy-free case runs in a new process without hiding or patching NumPy.
"""
import importlib
import inspect
import json
import subprocess
import sys
import types
import numpy as np
import pytest
import mojolearn as ml
from mojolearn import Array

__mojolearn_user_boundary__ = True
CASES = [("ClassicalMDS", {}), ("Isomap", {"n_neighbors": 2}),
         ("LocallyLinearEmbedding", {"n_neighbors": 3}),
         ("MDS", {"max_iter": 2, "n_init": 1}),
         ("TSNE", {"perplexity": 2, "max_iter": 2, "init": "random"})]


def _x():
    return np.random.default_rng(17).normal(size=(8, 4)).astype(np.float32)


def _array_result(result):
    assert type(result) is np.ndarray, type(result)
    assert result.shape == (8, 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


@pytest.mark.parametrize("name, settings", CASES)
def test_import_constructor_and_passive_class_wiring(name, settings):
    cls = getattr(ml, name)
    assert getattr(importlib.import_module("mojolearn.manifold"), name) is cls
    assert inspect.signature(cls).parameters["numeric_mode"].default is None
    assert not hasattr(cls.__init__, "__wrapped__")
    assert vars(cls())["numeric_mode"] is None
    assert vars(cls(numeric_mode="identical"))["numeric_mode"] == "identical"
    passive = types.new_class("Passive", (cls,), {}, lambda ns: ns.update(__module__=__name__))
    assert passive.__init__ is cls.__init__
    for method in ("fit", "fit_transform"):
        assert getattr(passive, method) is getattr(cls, method)
        assert not hasattr(getattr(cls, method), "__wrapped__")
        assert getattr(cls, method)._mojolearn_native_public_operation


@pytest.mark.parametrize("name, settings", CASES)
def test_native_methods_return_self_and_numpy_and_inherited_metadata(name, settings):
    cls = getattr(ml, name)
    x = _x()
    model = cls(**settings, numeric_mode="identical")
    assert model.fit(x) is model
    assert type(model.get_params()) is dict
    assert model.set_params(n_components=2) is model
    # set_params may invalidate fitted state; fit_transform is a new operation.
    _array_result(model.fit_transform(x))
    assert vars(model)["_mojolearn_fitted"] is True
    assert type(model.embedding_) is np.ndarray
    if name in ("Isomap", "LocallyLinearEmbedding"):
        _array_result(model.transform(x))
    if name == "Isomap":
        assert type(model.reconstruction_error()) is float
    passive = types.new_class("Passive", (cls,), {}, lambda ns: ns.update(__module__=__name__))
    _array_result(passive(**settings, numeric_mode="identical").fit_transform(x))


@pytest.mark.parametrize("name, settings", CASES)
def test_application_overrides_execute_with_fitted_tracking_and_numpy_output(name, settings):
    cls = getattr(ml, name)
    def fit(self, value, y=None):
        return self
    def transform(self, value):
        return Array.from_buffer(value)
    application = types.new_class("Application", (cls,), {}, lambda ns: ns.update(
        __module__=__name__, fit=fit, transform=transform))
    model = application(**settings, numeric_mode="identical")
    x = _x()
    assert model.fit(x) is model
    assert vars(model)["_mojolearn_fitted"] is True
    result = model.transform(x)
    assert type(result) is np.ndarray and result.shape == x.shape
    np.testing.assert_array_equal(result, x)
    assert np.shares_memory(result, x)


@pytest.mark.parametrize("name, settings", CASES)
def test_numpy_free_native_result_in_fresh_installed_process(name, settings):
    # No fake modules or import blocking. Prove NumPy was never imported, use
    # the package's own public Array input, and assert its default result type.
    code = """
import json, sys
import mojolearn as ml
assert 'numpy' not in sys.modules
name, settings = json.loads(sys.argv[1])
x = ml.Array.from_list([[((i * 7 + j * 3) % 17 - 8) / 8 for j in range(4)] for i in range(8)], '<f4')
result = getattr(ml, name)(**settings, numeric_mode='identical').fit_transform(x)
assert type(result) is ml.Array, type(result)
assert result.shape == (8, 2)
assert 'numpy' not in sys.modules
print(json.dumps({'class': name, 'result': type(result).__name__, 'shape': result.shape}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, json.dumps([name, settings])],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
