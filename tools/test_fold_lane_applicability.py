"""Fold descriptors use native host helpers even on a GPU installation."""
import ast
import numbers
from pathlib import Path

import pytest

import lane_applicability as applicability
import nvidia_baseline_qualification as qualification

ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_default_fold_route_reaches_native_helpers():
    # Execute the actual route functions without importing a GPU package.
    # The sentinel is the native boundary, not a duplicate implementation.
    tree = ast.parse((ROOT / 'python/mojolearn/model_selection.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    descriptor_calls = {n.func.id for n in ast.walk(functions['split_descriptor'])
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert '_folds' in descriptor_calls

    class NativeRoute(Exception):
        pass

    def native(y, splits, classifier):
        assert y == [0, 1, 0, 1] and splits == 2 and classifier is True
        raise NativeRoute

    namespace = dict(numbers=numbers, _sabotage_requested=lambda: False,
                     _classifier=lambda estimator: True, _native_default_folds=native)
    selected = ast.Module(body=[functions['_folds'], functions['_default_fold_arrays']], type_ignores=[])
    exec(compile(selected, 'actual-model-selection-routing', 'exec'), namespace)
    with pytest.raises(NativeRoute):
        list(namespace['_folds'](2, object(), object(), [0, 1, 0, 1], None))
    native_calls = {n.args[0].value for n in ast.walk(functions['_native_default_folds'])
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_native'
                    and n.args and isinstance(n.args[0], ast.Constant)}
    assert {'fold_ids', 'select_fold_i64'} <= native_calls
    core = next(f for f in applicability.host_surface().FAMILIES if f['family'] == 'core')
    assert native_calls <= set(core['exports'])


def test_fold_lane_is_cpu_applicable_and_never_gpu_coverage():
    scopes = applicability.scopes()
    scope = scopes['cross-val-folds']
    assert not scope.pure_python
    assert scope.has_cpu_route and '_mojolearn_core_host' in scope.host_only
    assert scope.applicable('cpu-host')[0]
    for column in ('nvidia-1gpu', 'nvidia-2gpu', 'apple-metal', 'amd-1gpu'):
        ok, reason = scope.applicable(column)
        assert not ok and 'CPU host route' in reason
    assert not applicability._SCOPES_DISAGREE


def test_baseline_inventory_excludes_host_work_without_a_metadata_gap():
    inventory = qualification.inventory(applicability.identity_break())
    assert 'cross-val-folds' not in inventory['lanes']
    assert 'CPU host route' in inventory['excluded']['cross-val-folds']
    assert not inventory['applicability_gaps']
