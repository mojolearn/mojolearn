# SPDX-License-Identifier: Apache-2.0
"""Source-only ABI checks; no package import, native build or GPU required."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/mojolearn"
MODULES = ("_expansion_linear", "_solver_impl", "_svm_impl", "svm", "_gp_impl", "kernel_methods", "neighbors",
           "linear_model", "_expansion_trees")


def _signature(node):
    args = node.args
    parameters = []
    positional = args.posonlyargs + args.args
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults)) + args.defaults
    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                            default=None if isinstance(default, ast.AST) else default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                            default=None if default is not None else inspect.Parameter.empty))
    return inspect.Signature(parameters)


@pytest.fixture
def plans(monkeypatch):
    package = types.ModuleType("_plan_fixture")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    sources = {}
    classes = {}
    for name in MODULES:
        module = types.ModuleType(package.__name__ + "." + name)
        tree = ast.parse((PACKAGE / (name + ".py")).read_text())
        sources[name] = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for node in sources[name].values():
            cls = type(node.name, (), {})
            init = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
            cls.__signature__ = _signature(init) if init is not None else inspect.Signature()
            setattr(module, node.name, cls)
            classes[cls] = (name, node.name)
        for entry in ("_XL_CALL", "_LINEAR_SCORE", "_CD_CALL", "_NATIVE", "_QN_CALL", "_TREES"):
            setattr(module, entry, len)
        for owner in ("_XL_CONTEXT", "_LINEAR_SCORE_CONTEXT", "_CD_CONTEXT", "_NATIVE_CONTEXT", "_QN_CONTEXT", "_TREES_CONTEXT"):
            setattr(module, owner, {})
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = types.ModuleType(package.__name__ + "._backend")
    backend.binding = lambda name, mode: (name, mode)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    package._backend = backend
    spec = importlib.util.spec_from_file_location(package.__name__ + ".plans", PACKAGE / "_native_msel_prep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = {}
    def register(cls, entry, owner, parameters, init, methods):
        registry[cls] = dict(entry=entry, context=owner, parameters=parameters, init=init, methods=methods, forest=None)
    context = {}
    module.connect(context, register, registry)
    return registry, classes, sources, context


def _method(sources, module, cls, method):
    node = sources[module].get(cls)
    if node is None:
        return None
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method:
            return item
    for base in node.bases:
        if isinstance(base, ast.Name):
            found = _method(sources, module, base.id, method)
            if found:
                return found
    return None


def test_restored_fit_plans_match_public_native_argument_order(plans):
    registry, classes, sources, _ = plans
    checked = []
    for cls, plan in registry.items():
        if "fit" not in plan["methods"]:
            continue  # native kernel value schemas
        module, name = classes[cls]
        fit = _method(sources, module, name, "fit")
        assert fit is not None, name
        native = next(node.value for node in ast.walk(fit)
                      if isinstance(node, ast.Return) and isinstance(node.value, ast.Call))
        assert isinstance(native.args[1], ast.Constant), name
        operation, recipe = plan["methods"]["fit"]
        assert operation == native.args[1].value, name
        expected = native.args[2].elts
        assert len(recipe) == len(expected), name
        for source, item in zip(expected, recipe):
            if isinstance(source, ast.Name):
                assert item == {"X": ("arg", 0), "y": ("arg", 1),
                                "sample_weight": ("keyword", "sample_weight", None)}[source.id], name
            elif isinstance(source, ast.Constant):
                assert item == ("constant", source.value), name
            else:
                assert item[0] == "binding", name
        checked.append(name)
    assert len(checked) == 36


def test_native_plans_preserve_posterior_and_weight_semantics(plans):
    registry, _, _, context = plans
    named = {cls.__name__: plan for cls, plan in registry.items()}
    assert "predict_std" in named["GaussianProcessRegressor"]["methods"]
    assert "predict_std" not in named["BayesianRidge"]["methods"]  # historically unsupported
    assert named["BayesianRidge"]["keywords"]["fit"] == ("sample_weight",)
    assert named["ARDRegression"]["keywords"]["fit"] == ()
    assert named["Lasso"]["initial_fields"] == {"l1_ratio": 1.0}
    assert named["_Combined"]["parameters"] == ("a", "b", "op", "sym")
    assert named["_SavedKernel"]["parameters"] == ("_SavedKernel__nodes",)
    for name in ("SVC", "SVR"):
        assert named[name]["methods"]["coef_"][0].endswith("_coef")
    assert "_mojolearn_metrics" in context["family_msel_bindings"]


def test_rfe_passes_fit_parameters_to_every_fit_without_python_attribute_hooks():
    source = (ROOT / "bindings/prep_select_native.mojo").read_text()
    assert source.count("y, fit_params, context)") == 2
    assert 'context, fit_params)' in source
    assert 'len(fit_params) != 0' not in source
    assert 'attrgetter' not in source
    assert '_call(b.getattr, Python.tuple(est' not in source
    helper = (ROOT / "bindings/prep_native_estimator.mojo").read_text()
    assert 'plan["fields_get"]' in helper


def test_posterior_is_native_prediction_and_existing_draw_algorithm():
    source = (ROOT / "bindings/prep_iterimp_native.mojo").read_text()
    assert 'prep_estimator_call(est, "predict_std"' in source
    assert 'truncnorm_draws_binding(' in source
    assert 'inspect.signature' not in source
    assert '_method(est' not in source
    dispatch = (ROOT / "bindings/msel_estimator_native.mojo").read_text()
    assert 'sequential_admit_value(keywords[keys[k]], context)' in dispatch
    assert 'kind == "keyword"' in dispatch and 'kind == "binding"' in dispatch
