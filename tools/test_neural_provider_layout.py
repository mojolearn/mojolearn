"""Architecture guard negative controls and lossless boundary relocation."""
import hashlib
import json
from pathlib import Path

from check_neural_provider_layout import ROOT, check


def test_six_native_trees_have_no_local_provider_or_python_backedges():
    roots, _, failures = check()
    assert len(roots) > 600
    assert not failures, failures


def test_guard_rejects_python_import_even_if_file_is_named_native(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from std.python import PythonObject as Hidden\n')
    _, _, failures = check(tmp_path)
    assert any(reason == 'provider/Python dependency' for _, reason in failures)


def test_guard_follows_transitive_edges_outside_the_six_directories(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'solver').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from solver.bridge import execute\n')
    (tmp_path / 'solver/bridge.mojo').write_text('from providers.python.hidden_pyapi import execute\n')
    _, _, failures = check(tmp_path)
    assert (('core.native', 'solver.bridge', 'providers.python.hidden_pyapi'), 'provider/Python dependency') in failures


def test_guard_rejects_native_to_bindings_even_when_target_looks_native(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from bindings.bridge import execute\n')
    _, _, failures = check(tmp_path)
    assert any(chain[-1] == 'bindings.bridge' for chain, _ in failures)


def test_guard_rejects_direct_cpython_c_call_without_python_import(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from std.ffi import external_call\nexternal_call["PyObject_Call", Int](a, b, c)\n')
    _, _, failures = check(tmp_path)
    assert any(reason == 'Python/CPython helper token in native code' for _, reason in failures)


def test_all_boundary_moves_preserve_original_body_modulo_module_references():
    report = json.loads((ROOT / 'experiments/neural_provider_layout_20261027/migration.json').read_text())
    assert len(report['moves']) == 36
    assert len(report['testing_exclusions']) == 3
    for row in report['moves']:
        assert not (ROOT / row['old']).exists()
        assert row['new'].endswith('_pyapi.mojo')
        text = (ROOT / row['new']).read_text()
        for moved in report['moves']:
            text = text.replace(moved['new_module'], moved['old_module']).replace(moved['new'], moved['old'])
        assert hashlib.sha256(text.encode()).hexdigest() == row['sha256_before'], row['new']
