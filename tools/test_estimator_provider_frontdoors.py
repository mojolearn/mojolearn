"""Enforce one-call frontdoors; these source checks do not qualify runtime paths."""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_selected_frontdoors_delegate_binding_admission_after_native_entry():
    expected = {
        'svm.py': {'LinearSVC': {'fit','predict','decision_function','score'},
                   'LinearSVR': {'fit','predict','score'}},
        '_svm_impl.py': {'SVC': {'fit','predict','predict_proba','predict_log_proba','decision_function','score'},
                        'SVR': {'fit','predict','score'}},
        'cluster.py': {'KMeans': {'fit','predict','transform','fit_predict','fit_transform'}},
    }
    checked = set()
    for filename, classes in expected.items():
        module = ast.parse((ROOT / 'python/mojolearn' / filename).read_text())
        for cls in (n for n in module.body if isinstance(n, ast.ClassDef) and n.name in classes):
            for method in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in classes[cls.name]):
                calls = [n for statement in method.body for n in ast.walk(statement) if isinstance(n, ast.Call)]
                assert len(calls) == 1, (filename, cls.name, method.name)
                call = calls[0]
                assert isinstance(call.func, ast.Name) and call.func.id == '_NATIVE'
                assert isinstance(call.args[2].elts[0], ast.Constant) and call.args[2].elts[0].value is None
                checked.add((filename, cls.name, method.name))
    assert len(checked) == sum(len(methods) for classes in expected.values() for methods in classes.values())

def test_one_shared_admission_helper_keeps_explicit_internal_arguments():
    source = (ROOT / 'providers/python/estimator_binding_pyapi.mojo').read_text()
    assert 'if len(args) == 0 or args[0] is not None:' in source
    assert 'estimator_binding(model, name, context)' in source
    family = (ROOT / 'bindings/family_native.mojo').read_text()
    for name in ('_mojolearn_estimators', '_mojolearn_svm', '_mojolearn'):
        assert f'estimator_arguments(model, "{name}", args, context)' in family
