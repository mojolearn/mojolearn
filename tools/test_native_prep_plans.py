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
sys.path.insert(0, str(ROOT / "tools"))
from boundary_keys_plain import plain  # noqa: E402
PACKAGE = ROOT / "python/mojolearn"
MODULES = ("_expansion_linear", "_solver_impl", "_svm_impl", "svm", "_gp_impl", "kernel_methods", "neighbors",
           "linear_model", "_expansion_trees", "model_selection", "_metrics_impl")


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
    # The composed call reads its bindings from the family connection (_native_family_bridge
    # connect_bindings; lane imputer-gp-native-module, 2026-11-16), not from _backend.binding.
    bridge = types.ModuleType(package.__name__ + "._native_family_bridge")
    shared = {"family_binding_connections": {}}
    bridge.shared_context = lambda: shared
    def connect_bindings(*names):
        for name in names:
            shared["family_binding_connections"][name] = {mode: ((name, mode),) for mode in ("identical", "fast")}
            # The real bridge connects the CPU identical module of each name beside the device
            # one (`family_binding_connections_cpu`: module or refusal text, read by the plans).
            shared.setdefault("family_binding_connections_cpu", {})[name] = ((name, "cpu"),)
    bridge.connect_bindings = connect_bindings
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    package._native_family_bridge = bridge
    spec = importlib.util.spec_from_file_location(package.__name__ + ".plans", PACKAGE / "_native_msel_prep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = {}
    def register(cls, entry, owner, parameters, init, methods):
        registry[cls] = dict(entry=entry, context=owner, parameters=parameters, init=init, methods=methods, forest=None)
    context = {'family_msel_splitter_layouts': {}}
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
        # The operation is a string constant or a named code on the family's operation table
        # (`_OP.trees_public_boost_fit`, the integer codes of `_operation_codes()`).
        op_node = native.args[1]
        if isinstance(op_node, ast.Attribute) and isinstance(op_node.value, ast.Name) and op_node.value.id == '_OP':
            public_operation = op_node.attr
        else:
            assert isinstance(op_node, ast.Constant), name
            public_operation = op_node.value
        operation, recipe = plan["methods"]["fit"]
        if public_operation == 'trees_public_meta_fit':
            # Public doors own lifecycle/output conversion; composed estimators
            # intentionally target the internal Array ABI. Verify the bridge
            # explicitly rather than demanding two distinct doors have one name.
            assert name in {'VotingClassifier', 'VotingRegressor', 'StackingClassifier',
                            'StackingRegressor', 'MultiOutputRegressor', 'MultiOutputClassifier',
                            'OneVsRestClassifier', 'CalibratedClassifierCV'}
            assert operation == 'trees_meta_fit', name
            lifecycle = (ROOT / 'bindings/trees_public_meta_lifecycle_pack.mojo').read_text()
            assert 'if operation == "trees_public_meta_fit":\n        method = "fit"' in lifecycle
            # Admission runs once per configuration and the admitted kind is passed down (cb376df254).
            assert 'meta_admitted_operation(model, kind, "trees_meta_" + method, call_args, context)' in lifecycle
        elif public_operation == 'trees_public_boost_fit':
            # Boost/bagging/DART/RTE public doors (2c61cfe498) own the
            # lifecycle and forward to the internal trees_boost_* ABI, the
            # same split as the meta doors above.
            assert operation == 'trees_boost_fit', name
            lifecycle = (ROOT / 'providers/python/tree_boost_public_lifecycle_pyapi_pack.mojo').read_text()
            assert 'var method = String(operation[byte=String("trees_public_boost_").byte_length():])' in lifecycle
            # The boost kind is resolved once per call and passed down (boost_operation_of, fed1b4e384).
            assert 'boost_operation_of(model, kind, "trees_boost_" + method, call_args, context)' in lifecycle
            dispatch = (ROOT / 'bindings/trees_public_native_door.mojo').read_text()
            # The dispatcher routes by the leaf code's range (lane bw3-trees,
            # bindings/operation_names_trees.mojo); a name the table lacks keeps the prefix.
            assert ('if operation >= TREES_FIRST_PUBLIC_BOOST and operation < TREES_END_PUBLIC_BOOST:\n'
                    '        return tree_boost_public_operation(model, name, args, context)') in dispatch
            assert 'if operation.startswith("trees_public_boost_"):\n        return tree_boost_public_operation(' in dispatch
        else:
            assert operation == public_operation, name
        expected = native.args[2].elts
        assert len(recipe) == len(expected), name
        for source, item in zip(expected, recipe):
            if isinstance(source, ast.Name):
                assert item == {"X": ("arg", 0), "y": ("arg", 1),
                                "sample_weight": ("keyword", "sample_weight", None)}[source.id], name
            elif isinstance(source, ast.Constant) and source.value is None and item[0] == "binding":
                # The public door passes None in the binding slot and resolves its own
                # connected module natively; the composed plan hands the binding in.
                pass
            elif isinstance(source, ast.Constant):
                assert item == ("constant", source.value), name
            else:
                assert item[0] == "binding", name
        checked.append(name)
    # Every fit plan was checked against its public call; the registry grows as families gain
    # native plans (44 when this was written, 48 at f776cb552c), never shrinks.
    assert len(checked) == sum(1 for plan in registry.values() if "fit" in plan["methods"]) >= 48


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
    source = (ROOT / "bindings/prep_select_native_door.mojo").read_text()
    assert source.count("y, fit_params, context)") == 2
    assert 'context, fit_params)' in source
    # The member route (member ABI, 2026-11-21) carries no fit parameters, so
    # `_rfe_member` declines it when there are any and the estimator's own
    # doors run with them (the two calls above). That route choice is the only
    # place fit parameters are tested; no fit drops them.
    member = source.split("def _rfe_member(", 1)[1].split("\ndef ", 1)[0]
    assert source.count('len(fit_params) != 0') == member.count('len(fit_params) != 0') == 1
    assert 'return member^' in member.split('len(fit_params) != 0', 1)[1].split('\n', 2)[1]
    assert 'attrgetter' not in source
    assert '_call(b.getattr, Python.tuple(est' not in source
    helper = plain("bindings/prep_native_estimator_door.mojo",
                   (ROOT / "bindings/prep_native_estimator_door.mojo").read_text())
    assert 'plan["fields_get"]' in helper


def test_posterior_is_native_prediction_and_existing_draw_algorithm():
    source = (ROOT / "bindings/prep_iterimp_native_door.mojo").read_text()
    assert 'prep_estimator_call(est, "predict_std"' in source
    assert 'truncnorm_draws_binding(' in source
    assert 'inspect.signature' not in source
    assert '_method(est' not in source
    dispatch = (ROOT / "bindings/msel_estimator_native_door.mojo").read_text()
    # The native dispatch context now admits the full classical buffer/pandas
    # surface. NumPy must be connected before first-call keyword admission;
    # key/type checks still precede the exact compiled entry invocation.
    admission = 'classical_admit_value(keywords[keys[k]], context)'
    assert admission in dispatch
    assert dispatch.index('classical_connect_numpy(context)') < dispatch.index(admission)
    assert dispatch.index('if py_type(keywords) is not b.dict:') < dispatch.index(admission)
    assert dispatch.index('if py_type(keys[k]) is not b.str:') < dispatch.index(admission)
    assert dispatch.index('if not py_contains(allowed, keys[k]):') < dispatch.index(admission)
    assert dispatch.index(admission) < dispatch.index('return family_call(')
    assert 'kind == "keyword"' in dispatch and 'kind == "binding"' in dispatch


def test_meta_owners_do_not_call_member_methods_or_python_constructors():
    source = (ROOT / 'bindings/trees_public_meta_door.mojo').read_text()
    forbidden = ('py_call(member.fit', 'py_call(e.fit', '.predict,', '.predict_proba,',
                 '.decision_function,', 'est.get_params', 'c.split,', 'est._validate_params',
                 'py_call(_import(', '_call_kw(r2,', 'target.tolist', 'col.tolist',
                 'context["array_class"].from_list')
    for token in forbidden:
        assert token not in source, token
    assert 'prep_native_split(c, X, y, context)' in source
    # Constructor admission runs once per configuration in its own door (cb376df254).
    admission = (ROOT / 'bindings/trees_meta_admission_door.mojo').read_text()
    assert 'prep_estimator_call(model, "__admit__"' in admission
    assert 'msel_admit(args, prep_native_context(context))' in source
    # The r2 score is passed as the metrics owner's leaf code (lane bw2-metrics).
    assert 'metrics_cls_operation(PythonObject(None), MCLS_OPN_FAMILY_MCLS_R2,' in source
    assert 'native_estimator_defaults' in source
    # The entry is connected when the family loads (b6aa87e41d), not in __init__.
    assert 'native_estimator_entry=selection._NATIVE' in (PACKAGE / '_families.py').read_text()
    # The cross-validated linear fits' UNPACK (lane dw-xlinear): the splitter runs natively there.
    linear = (ROOT / 'bindings/x_linear_cv_unpack.mojo').read_text()
    assert 'py_call(cv.split' not in linear
    assert 'py_call(cv.split' not in (ROOT / 'bindings/x_linear_native_door.mojo').read_text()
    assert 'prep_native_split(cv, x_source, y_source, context)' in linear


def test_native_clone_bounds_recursive_configurations():
    source = (ROOT / 'bindings/msel_estimator_native_door.mojo').read_text()
    assert source.count('if depth > 64:') == 2
    assert 'msel_native_clone(value, context, depth + 1)' in source
    assert 'msel_clone_parameter(given[names[k]], context, depth + 1)' in source
