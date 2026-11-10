"""Dependency inventory negative controls; historical moves are not live gates."""

from check_neural_provider_layout import check


def test_six_native_tree_dependencies_resolve():
    roots, _, failures = check()
    assert len(roots) > 600
    assert not failures, failures


def test_guard_rejects_python_import_even_if_file_is_named_native(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from std.python import PythonObject as Hidden\n')
    _, _, failures = check(tmp_path, require_native_only=True)
    assert any(reason == 'provider/Python dependency' for _, reason in failures)


def test_guard_follows_transitive_edges_outside_the_six_directories(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'solver').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from solver.bridge import execute\n')
    (tmp_path / 'solver/bridge.mojo').write_text('from providers.python.hidden_pyapi import execute\n')
    _, _, failures = check(tmp_path, require_native_only=True)
    assert (('core.native', 'solver.bridge', 'providers.python.hidden_pyapi'), 'provider/Python dependency') in failures


def test_guard_rejects_native_to_bindings_even_when_target_looks_native(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from bindings.bridge import execute\n')
    _, _, failures = check(tmp_path)
    assert any(chain[-1] == 'bindings.bridge' for chain, _ in failures)


def test_guard_rejects_direct_cpython_c_call_without_python_import(tmp_path):
    (tmp_path / 'core').mkdir()
    (tmp_path / 'core/native.mojo').write_text('from std.ffi import external_call\nexternal_call["PyObject_Call", Int](a, b, c)\n')
    _, _, failures = check(tmp_path, require_native_only=True)
    assert any(reason == 'Python/CPython helper token in native code' for _, reason in failures)


