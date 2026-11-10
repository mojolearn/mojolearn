"""Current dependency inventory checks, not runtime qualification.

The 2026-10-27 relocation manifest is historical evidence, not a restriction
on subsequent changes to provider bodies or ownership directories.
"""

from check_library_provider_layout import check


def test_all_native_family_dependencies_resolve():
    roots, _, problems = check()
    assert len(roots) > 2400
    assert not problems, problems


def test_guard_rejects_python_import_in_non_neural_family(tmp_path):
    (tmp_path / 'svm').mkdir()
    (tmp_path / 'svm/native.mojo').write_text('from std.python import PythonObject\n')
    _, _, problems = check(tmp_path, require_native_only=True)
    assert any(reason == 'provider/Python dependency' for _, reason in problems)


def test_guard_follows_family_dependencies_into_a_hidden_provider(tmp_path):
    (tmp_path / 'x_decomp').mkdir()
    (tmp_path / 'outside').mkdir()
    (tmp_path / 'x_decomp/native.mojo').write_text('from outside.bridge import run\n')
    (tmp_path / 'outside/bridge.mojo').write_text('from providers.python.hidden_pyapi import run\n')
    _, _, problems = check(tmp_path, require_native_only=True)
    assert (('x_decomp.native', 'outside.bridge', 'providers.python.hidden_pyapi'), 'provider/Python dependency') in problems


def test_missing_generated_dependency_is_not_silently_exempted(tmp_path):
    (tmp_path / 'tokenizer').mkdir()
    (tmp_path / 'tokenizer/native.mojo').write_text('from tokenizer.impl.unicode_table_generated import TABLE\n')
    _, _, problems = check(tmp_path)
    assert any(reason == 'unresolved local dependency' for _, reason in problems)


def test_present_generated_dependency_is_audited_for_backedges(tmp_path):
    (tmp_path / 'tokenizer/impl').mkdir(parents=True)
    (tmp_path / 'tokenizer/native.mojo').write_text('from tokenizer.impl.unicode_table_generated import TABLE\n')
    (tmp_path / 'tokenizer/impl/unicode_table_generated.mojo').write_text('from bindings.sneaky import TABLE\n')
    _, _, problems = check(tmp_path)
    assert any(chain[-1] == 'bindings.sneaky' for chain, _ in problems)
