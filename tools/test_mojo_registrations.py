# SPDX-License-Identifier: Apache-2.0
"""Coverage and lane selection must retain every supported native door."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_select as ls
import verify_coverage_audit as audit
from mojo_registrations import registered_functions, sole_registration


@pytest.mark.parametrize('statement', [
    'module.def_function[local_alias]("exported_name")',
    'module.def_py_function[local_alias]("exported_name")',
    'module.def_py_c_function[local_alias]("exported_name")',
    'module.def_py_c_function(local_alias, "exported_name")',
    "module.def_py_c_function(\n local_alias,\n 'exported_name'\n)",
])
def test_all_supported_doors_keep_explicit_export_and_local_alias(statement):
    expected = {'exported_name': ('local_alias', [])}
    assert registered_functions(statement) == expected
    assert audit.registrations(statement) == expected
    assert sole_registration(statement).name == 'exported_name'


@pytest.mark.parametrize('statement', [
    'module.def_function[local_alias[DevExec, True]]("name")',
    'module.def_py_function[local_alias[DevExec, True]]("name")',
    'module.def_py_c_function[local_alias[DevExec, True]]("name")',
    'module.def_py_c_function(local_alias[DevExec, True], "name")',
])
def test_generic_executor_dependencies_are_retained(statement):
    assert registered_functions(statement) == {'name': ('local_alias', ['DevExec', 'True'])}


def test_comments_docstrings_and_example_strings_are_not_exports():
    text = '''
# module.def_py_c_function(fake, "comment")
"""module.def_py_c_function(fake, "docstring")"""
var example = 'module.def_py_c_function(fake, "string")'
module.def_py_c_function(real, "actual") # module.def_function[no]("tail")
'''
    assert registered_functions(text) == {'actual': ('real', [])}


@pytest.mark.parametrize('statement', [
    'module.def_py_c_function(entry, "name"); perform_work()',
    'var ignored = module.def_py_c_function(entry, "name")',
    'module.def_py_c_function(entry, "name", extra_option)',
])
def test_additive_detection_refuses_other_changed_work(statement):
    assert sole_registration(statement) is None


def test_binding_export_and_graph_follow_an_imported_c_alias(monkeypatch):
    binding, impl = 'bindings/_registration_test.mojo', 'core/_registration_test.mojo'
    texts = {
        binding: 'from core._registration_test import owner as local_alias\n'
                 'def build__mojolearn_probe():\n'
                 '    module.def_py_c_function(local_alias, "public_door")\n',
        impl: 'def owner():\n    return 1\n',
    }
    monkeypatch.setattr(ls, '_read', lambda path: texts[path])
    monkeypatch.setattr(ls, '_mojo_imports', lambda path: {impl} if path == binding else set())
    monkeypatch.setattr(ls, '_mojo_module_files', lambda module, rel, names: {impl})
    ls.reset_caches()
    try:
        assert ls._binding_exports(binding) == {'public_door': 'local_alias'}
        assert ls._MojoDefGraph().roots(binding, 'public_door') == {('t', impl, 'owner')}
    finally:
        ls.reset_caches()


def test_delegated_registrar_retains_c_and_python_exports(monkeypatch):
    binding, impl = 'bindings/_registration_test.mojo', 'core/_registration_test.mojo'
    monkeypatch.setattr(ls, '_mojo_import_targets', lambda path: ({'install': {(impl, 'register')}}, set()))
    monkeypatch.setattr(ls, '_mojo_defs_for', lambda path: ({'register':
        'def register(module):\n'
        '    module.def_py_c_function(owner, "native_door")\n'
        '    module.def_py_function[other[DevExec]]("python_door")\n'}, {}))
    ls.reset_caches()
    try:
        assert ls._delegated_exports(binding) == {'native_door': (impl, 'owner'), 'python_door': (impl, 'other')}
    finally:
        ls.reset_caches()


def test_new_c_registration_is_additive_without_losing_existing_exports(monkeypatch):
    path = 'bindings/_registration_test.mojo'
    old = 'def old_owner():\n    return 1\n\ndef build__mojolearn_probe():\n    module.def_function[old_owner]("old")\n'
    new = old.replace('def build__mojolearn_probe():', 'def new_owner():\n    return 2\n\ndef build__mojolearn_probe():')
    new += '    module.def_py_c_function(new_owner, "new")\n'
    monkeypatch.setattr(ls, '_git_show', lambda ref, rel: old)
    monkeypatch.setattr(ls, '_read', lambda rel: new)
    assert ls.binding_additions('baseline', path) == {'new'}
    new = new.replace('new_owner, "new"', 'new_owner, "old"')
    assert ls.binding_additions('baseline', path) is None


def test_real_scaler_native_doors_are_in_both_censuses():
    source = 'bindings/_mojolearn_tokenizer_host.mojo'
    ls.reset_caches()
    exports = ls._binding_exports(source)
    census = audit.registrations(ls._read(source))
    expected = {'standard_scaler_fit_native': 'scaler_fit_c',
                'standard_scaler_native_call': 'scaler_lifecycle_c',
                'minmax_scaler_native_call': 'minmax_lifecycle_c',
                'maxabs_scaler_native_call': 'maxabs_lifecycle_c'}
    for name, implementation in expected.items():
        assert exports[name] == implementation
        assert census[name] == (implementation, [])
    graph = ls._MojoDefGraph()
    assert ('t', 'bindings/minmax_native.mojo', 'minmax_lifecycle_c') in graph.roots(source, 'minmax_scaler_native_call')


def test_scaler_route_annotations_preserve_host_infer_execution_debt():
    for entry in ('standard_fit', 'standard_transform', 'minmax_fit', 'minmax_transform'):
        status, lanes, reason = audit.ANNOTATIONS_BY_SRC[entry, 'bindings/_mojolearn_preprocessing_host.mojo']
        assert status == 'covered' and len(lanes) == 1
        assert 'MOJOLEARN_VENDOR=cpu' in reason and 'not execution' in reason
    for entry in ('standard_transform', 'minmax_transform'):
        status, lanes, reason = audit.ANNOTATIONS_BY_SRC[entry, 'bindings/_mojolearn_estimators_host.mojo']
        assert status == 'owed' and len(lanes) == 1
        assert 'MOJOLEARN_IDENTITY_HOST_INFER=' in reason and 'still owed' in reason
