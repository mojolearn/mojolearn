# SPDX-License-Identifier: Apache-2.0
"""Source/connection tests; no product imports or compiled runtime required."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from boundary_keys_plain import plain  # noqa: E402  (the doors read through held keys: bw2-classical)


def connection():
    # The layouts module registers itself through the sibling
    # `._native_loaded_modules` (standard library only) at import, so load
    # both from the source tree inside a throwaway package. No mojolearn
    # import runs and nothing stays in sys.modules.
    package = types.ModuleType('_family_layout_pkg')
    package.__path__ = [str(ROOT / 'python/mojolearn')]
    name = package.__name__ + '._native_family_layouts'
    path = ROOT / 'python/mojolearn/_native_family_layouts.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved = {key: sys.modules.get(key) for key in (package.__name__, name, package.__name__ + '._native_loaded_modules')}
    sys.modules[package.__name__] = package
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


def test_connected_getter_reads_real_fields_and_records_descriptor_identity(monkeypatch):
    package = types.ModuleType('_family_fixture')
    module = types.ModuleType('_family_fixture.owner')
    cls = type('Estimator', (), {'__module__': module.__name__, 'score': lambda self: 1})
    module.Estimator = cls
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    bridge = connection()
    bridge.connect(package)
    plan = bridge.LAYOUTS[cls]
    model = cls()
    model.alpha = 3
    assert plan['fields_get'](model, cls) is vars(model)
    assert plan['descriptors'][0][1]['score'] is vars(cls)['score']
    assert plan['names']['__getattribute__'] is object.__getattribute__


def test_later_import_adds_classes_without_repinning_mutated_existing_class(monkeypatch):
    package = types.ModuleType('_family_fixture')
    module = types.ModuleType('_family_fixture.owner')
    first = type('First', (), {'__module__': module.__name__, 'score': lambda self: 1})
    module.First = first
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    bridge = connection()
    bridge.connect(package)
    old = bridge.LAYOUTS[first]['descriptors'][0][1]['score']
    first.score = property(lambda self: (_ for _ in ()).throw(AssertionError('callback')))
    second = type('Second', (), {'__module__': module.__name__})
    module.Second = second
    bridge.connect(package)
    assert bridge.LAYOUTS[first]['descriptors'][0][1]['score'] is old
    assert second in bridge.LAYOUTS


def test_application_classes_are_not_registered_as_trusted_package_owners(monkeypatch):
    package = types.ModuleType('_family_fixture')
    module = types.ModuleType('_family_fixture.owner')
    foreign = type('UserEstimator', (), {'__module__': 'application'})
    module.Foreign = foreign
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    bridge = connection()
    bridge.connect(package)
    assert foreign not in bridge.LAYOUTS


def test_native_family_door_admits_storage_before_dispatch_and_operation_coercion():
    # The family operations moved to their own host door (d24e390da9); the classical door only routes them.
    source = (ROOT / 'bindings/classical_native_pack.mojo').read_text()
    body = source.split('def classical_c(', 1)[1]
    # bw2-classical: the str is resolved by one table lookup (classical_operation_leaf); only a name
    # the table does not hold is copied into a String. Both follow the exact-str admission.
    assert body.index('py_type(request[1]) is not b.str') < body.index('var operation = classical_operation_leaf(request[1])')
    assert body.index('py_type(request[1]) is not b.str') < body.index('var name = String(request[1])')
    door = (ROOT / 'bindings/classical_family_door.mojo').read_text()
    entry = door.split('def classical_family_c(', 1)[1]
    assert entry.index('py_type(request[1]) is not b.str') < entry.index('var operation = String(request[1])')
    assert entry.index('classical_family_admit(model)') < entry.index('family_operation(model,')
    coded = door.split('def _family_coded(', 1)[1].split('\ndef ', 1)[0]
    assert coded.index('classical_family_admit(model)') < coded.index('family_operation_routed(model,')


def test_guard_keeps_none_passive_subclasses_and_native_dictionary_boundary():
    rel = 'providers/python/classical_family_admission_pyapi_unpack.mojo'
    source = plain(rel, (ROOT / rel).read_text())
    assert source.index('if model is None:') < source.index('mods.module("mojolearn._native_family_layouts")')
    assert 'len(ancestry) - start != len(connected)' in source
    assert 'String(name) == "__init__" or not py_contains(names, name)' in source
    assert 'py_call(plan["fields_get"], model, cls)' in source
    assert 'py_fields(model)' not in source and 'object.__getattribute__' not in source
    assert '"__dict__"]' in source
    assert source.index('descriptor changed after connection') < source.index('py_call(plan["fields_get"]')


def test_connections_follow_installers_and_refresh_only_on_new_module_import():
    source = (ROOT / 'python/mojolearn/_families.py').read_text()  # the package's class connections
    assert source.index('_connect_family_layouts(') > source.index('_connect_declared_constructors(_declared_constructor_setters)')
    assert source.index('_connect_family_layouts(') < source.index('_connect_estimator_defaults()')
    lazy = (ROOT / 'python/mojolearn/_expansion_sequence.py').read_text()
    assert 'connect_family_layouts(sys.modules[__package__])' in lazy
