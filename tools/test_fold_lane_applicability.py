"""Fold descriptors use native host helpers even on a GPU installation."""
import ast
from pathlib import Path


import lane_applicability as applicability

ROOT = Path(__file__).resolve().parents[1]


def test_descriptor_default_fold_route_reaches_native_helpers():
    # The Python fold shims (`_folds`, `_default_fold_arrays`) were deleted by
    # f4d37e5eec; the default folds are the native model_selection owner's
    # `msel_default_folds`. split_descriptor and the public default_folds
    # (which the verify lane calls) reach it through the one owner entry, and
    # it calls core's fold helpers.
    tree = ast.parse((ROOT / 'python/mojolearn/model_selection.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert '_default_fold_arrays' not in functions and '_folds' not in functions
    descriptor_ops = {n.args[1].value for n in ast.walk(functions['split_descriptor'])
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_NATIVE'
                      and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)}
    assert descriptor_ops == {'family_msel_split_descriptor'}
    owner = (ROOT / 'bindings/msel_split_door.mojo').read_text()
    body = owner[owner.index('def msel_default_folds('):owner.index('# ---', owner.index('def msel_default_folds('))]
    assert 'msel_fold_ids(' in body and '_select_folds(' in body and 'msel_sabotage()' in body
    door = (ROOT / 'bindings/model_selection_native_door.mojo').read_text()
    route = door[door.index('if operation == "family_msel_default_folds":'):]
    assert 'return msel_default_folds(' in route.split('raise Error', 1)[0]
    cv = (ROOT / 'bindings/msel_cv_door.mojo').read_text()
    source = cv[cv.index('def msel_fold_source('):]
    assert 'msel_default_folds(' in source.split('\ndef ', 1)[0]
    harness = (ROOT / 'tools/identity_break.py').read_text()
    lane = harness[harness.index('@lane("cross-val-folds")'):]
    lane = lane[:lane.index('\n@lane(')]
    assert 'ms.default_folds(' in lane and 'ms._NATIVE' not in lane and 'ms._default_fold_arrays(' not in lane
    public = [n for n in ast.walk(functions['default_folds'])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_NATIVE']
    assert [n.args[1].value for n in public] == ['family_msel_default_folds']


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


def test_nvidia_inventory_excludes_host_work_without_a_metadata_gap():
    applicability.use_harness(applicability.identity_break())
    ok, reason = applicability.scopes()['cross-val-folds'].applicable('nvidia-1gpu')
    assert not ok and 'CPU host route' in reason
    assert not applicability._SCOPES_DISAGREE
