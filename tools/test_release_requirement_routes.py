from pathlib import Path
from types import SimpleNamespace
import textwrap
import pytest

import release_requirement_dependencies as D
import release_requirement_routes as R
from test_release_requirement_dependencies import binding, snapshot


def fixture(tmp_path):
    package = tmp_path / 'python/mojolearn'
    package.mkdir(parents=True)
    backend = ('_HOST_MODULES = host_surface.routed_modules()\n'
               '_HOST_INFERENCE_MODULES = host_surface.inference_routes()\n'
               'def _select_cpu_only(pkg, mode, reason):\n' + textwrap.indent(R.CPU_SELECTOR, '    ')
               + '\ndef default_mode():\n' + textwrap.indent(R.DEFAULT_MODE, '    '))
    (package / '_backend.py').write_text(backend)
    (package / 'host_surface.py').write_text('def routed_modules():\n    return {f["routes"]: f["binding"] for f in FAMILIES if f["routes"]}\n')
    (package / 'tokenizer.py').write_text('''
from . import _backend
_FAST_EXTENSION = "_mojolearn_tokenizer_fast"
def _fast_binding():
    if _backend.default_mode() != "fast":
        return None
    return _backend._load_member("fast", _FAST_EXTENSION, "path")
''')
    surface = SimpleNamespace(routed_modules=lambda: {'logical_gpu': 'explicit_cpu_owner'},
                              inference_routes=lambda: {}, ADAPTED_MODULES={})
    return package, surface


def test_proven_unrelated_fast_loader_does_not_block_identical(tmp_path):
    _, surface = fixture(tmp_path)
    routes = R.loader_routes(tmp_path, surface)
    witness = routes['mode_exclusions']['identical']['_mojolearn_tokenizer_fast']
    assert witness['active_mode'] == 'fast'
    assert witness['source'] in routes['sources']
    snap = snapshot()
    snap['loader_routes'] = routes
    lane = dict(bindings=['one', '_mojolearn_tokenizer_fast'])
    environment, missing = D._binding_context(snap, 'nvidia', lane, {})
    assert not missing
    assert '_mojolearn_tokenizer_fast' in environment['loader_routes']['excluded']
    assert len(environment['bindings']) == 1


def test_unguarded_or_additional_loader_use_stays_blocked(tmp_path):
    package, surface = fixture(tmp_path)
    path = package / 'tokenizer.py'
    path.write_text(path.read_text() + '\ndef unguarded():\n    return _backend._load_member("fast", _FAST_EXTENSION, "other")\n')
    routes = R.loader_routes(tmp_path, surface)
    assert not routes['mode_exclusions']
    snap = snapshot()
    snap['loader_routes'] = routes
    _, missing = D._binding_context(snap, 'nvidia', {'bindings': ['one', '_mojolearn_tokenizer_fast']}, {})
    assert any('_mojolearn_tokenizer_fast' in item for item in missing)


def test_cpu_alias_resolves_only_from_actual_loader_map(tmp_path):
    _, surface = fixture(tmp_path)
    snap = snapshot()
    host = binding('explicit_cpu_owner', 'bytes', vendor='host', arch='', target='osx-arm64')
    host['tier'] = 'host'
    snap['bindings'] = [host]
    snap['toolchains']['osx-arm64'] = {'compiler': 'actual'}
    snap['loader_routes'] = R.loader_routes(tmp_path, surface)
    environment, missing = D._binding_context(snap, 'cpu', {'bindings': ['logical_gpu']}, {'cpu_target': 'osx-arm64'})
    assert not missing
    assert list(environment['bindings']) == [host['key']]
    assert environment['loader_routes']['aliases']['logical_gpu']['bindings'] == ['explicit_cpu_owner']
    _, missing = D._binding_context(snap, 'cpu', {'bindings': ['logical_unmapped']}, {'cpu_target': 'osx-arm64'})
    assert any('logical_unmapped' in item for item in missing)


def test_cpu_loader_reassignment_cannot_be_hidden_by_old_map_reference(tmp_path):
    package, surface = fixture(tmp_path)
    path = package / '_backend.py'
    path.write_text(path.read_text().replace('        if basename:\n', '        basename = "unproven_owner"\n        if basename:\n'))
    routes = R.loader_routes(tmp_path, surface)
    assert not routes['cpu_aliases']
    assert routes['unresolved']


def test_witness_changes_with_loader_source_bytes(tmp_path):
    package, surface = fixture(tmp_path)
    first = R.loader_routes(tmp_path, surface)
    path = package / 'tokenizer.py'
    path.write_text(path.read_text() + '\n# retained loader source change\n')
    second = R.loader_routes(tmp_path, surface)
    assert first['sources'] != second['sources']


@pytest.mark.parametrize('other', [
    'from . import _backend\nx = _backend._load_member("fast", "_mojolearn_tokenizer_fast", "path")',
    'from .tokenizer import _FAST_EXTENSION\nx = load(_FAST_EXTENSION)',
    'from . import tokenizer\nx = load(tokenizer._FAST_EXTENSION)',
])
def test_an_unrelated_guard_cannot_hide_another_modules_loader_use(tmp_path, other):
    package, surface = fixture(tmp_path)
    (package / 'other.py').write_text(other)
    assert not R.loader_routes(tmp_path, surface)['mode_exclusions']


def optional_fixture(tmp_path):
    package, surface = fixture(tmp_path)
    (package / 'tokenizer.py').write_text('''
import os
from . import _backend
_FAST_EXTENSION = "_mojolearn_tokenizer_fast"
_NO_FAST = (None, False, False, 0)
def _fast_binding(*, connect_all=False):
    selected = _backend.default_mode()
    if not connect_all and selected != "fast":
        return _NO_FAST
    try:
        path = os.path.join(_backend.tier_dir("fast"), _FAST_EXTENSION + ".so")
    except ImportError:
        return _NO_FAST
    if not os.path.exists(path):
        return _NO_FAST
    return _backend._load_member("fast", _FAST_EXTENSION, path)
_CONNECTED = _fast_binding(connect_all=True)
''')
    return package, surface


def test_optional_absence_needs_source_predicate_and_explicit_layout_contract(tmp_path):
    _, surface = optional_fixture(tmp_path)
    snap = snapshot(); snap['loader_routes'] = R.loader_routes(tmp_path, surface)
    assert not snap['loader_routes']['mode_exclusions']  # connect_all is not mode-exclusive
    name = '_mojolearn_tokenizer_fast'
    assert snap['loader_routes']['optional_files'][name]['basename'] == name + '.so'
    lane = dict(bindings=['one', name])
    _, missing = D._binding_context(snap, 'nvidia', lane, {})
    assert any(name in x for x in missing)
    absent = {'optional_native_files': {'linux-64': {name: False}}}
    environment, missing = D._binding_context(snap, 'nvidia', lane, absent)
    assert not missing
    witness = environment['loader_routes']['optional_files'][name]
    assert witness['present'] is False and witness['selected_layout_sha256']
    # A required missing binding never inherits another binding's optionality.
    _, missing = D._binding_context(snap, 'nvidia', {'bindings':['required_missing', name]}, absent)
    assert any('required_missing' in x for x in missing)
    assert not any(name in x for x in missing)


def test_optional_presence_changes_inputs_and_cannot_hide_missing_native_context(tmp_path):
    _, surface = optional_fixture(tmp_path)
    snap = snapshot(); snap['loader_routes'] = R.loader_routes(tmp_path, surface)
    name = '_mojolearn_tokenizer_fast'; lane = dict(bindings=['one', name])
    absent, _ = D._binding_context(snap, 'nvidia', lane, {'optional_native_files':{'linux-64':{name:False}}})
    present, missing = D._binding_context(snap, 'nvidia', lane, {'optional_native_files':{'linux-64':{name:True}}})
    assert any(name in x for x in missing)
    assert D.digest(absent) != D.digest(present)
    snap['bindings'].append(binding(name, 'actual-fast-code'))
    mapped, missing = D._binding_context(snap, 'nvidia', lane, {'optional_native_files':{'linux-64':{name:True}}})
    assert not missing and len(mapped['bindings']) == 2
    assert D.digest(mapped) != D.digest(present)
    _, conflicts = D._binding_context(snap, 'nvidia', lane, {'optional_native_files':{'linux-64':{name:False}}})
    assert any('contradicts selected native layout' in x for x in conflicts)


@pytest.mark.parametrize('change', [
    lambda s:s.replace('os.path.exists(path)', 'user_exists(path)'),
    lambda s:s.replace('import os', 'import os\nos.path.exists = user_hook'),
    lambda s:s.replace('if not os.path.exists(path):', 'if not user.exists(path):'),
    lambda s:s.replace('return _NO_FAST\n    return _backend', 'return callback()\n    return _backend'),
    lambda s:s.replace('    selected =', '    user_callback(_FAST_EXTENSION)\n    selected ='),
    lambda s:s.replace('    if not os.path.exists(path):', '    path = user_path\n    if not os.path.exists(path):'),
])
def test_unknown_predicate_hooks_or_extra_binding_uses_stay_unresolved(tmp_path, change):
    package, surface = optional_fixture(tmp_path)
    p = package / 'tokenizer.py'; p.write_text(change(p.read_text()))
    assert not R.loader_routes(tmp_path, surface)['optional_files']
