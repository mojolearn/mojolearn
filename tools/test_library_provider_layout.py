"""Source architecture and lossless relocation checks, not runtime qualification."""
import hashlib
import json
from pathlib import Path

from check_library_provider_layout import ROOT, FAMILY_TREES, check


def test_all_native_family_trees_have_no_python_provider_or_bindings_dependencies():
    roots, _, problems = check()
    assert len(roots) > 2400
    assert not problems, problems


def test_family_guard_covers_every_recorded_non_neural_tree():
    report = json.loads((ROOT / 'experiments/library_provider_layout_20261027/migration.json').read_text())
    assert set(FAMILY_TREES) == set(report['family_roots'])
    assert len(FAMILY_TREES) == 40


def test_all_forty_three_boundary_bodies_are_unchanged_except_import_paths():
    report = json.loads((ROOT / 'experiments/library_provider_layout_20261027/migration.json').read_text())
    assert len(report['moves']) == 43
    assert len(report['affected_groups']) == 18
    for row in report['moves']:
        assert not (ROOT / row['old']).exists()
        assert row['new'].startswith('providers/python/') and row['new'].endswith('_pyapi.mojo')
        text = (ROOT / row['new']).read_text()
        for move in report['moves']:
            text = text.replace(move['new_module'], move['old_module']).replace(move['new'], move['old'])
        assert hashlib.sha256(text.encode()).hexdigest() == row['sha256_before'], row['new']


def test_guard_rejects_python_import_in_non_neural_family(tmp_path):
    (tmp_path / 'svm').mkdir()
    (tmp_path / 'svm/native.mojo').write_text('from std.python import PythonObject\n')
    _, _, problems = check(tmp_path)
    assert any(reason == 'provider/Python dependency' for _, reason in problems)


def test_guard_follows_family_dependencies_into_a_hidden_provider(tmp_path):
    (tmp_path / 'x_decomp').mkdir()
    (tmp_path / 'outside').mkdir()
    (tmp_path / 'x_decomp/native.mojo').write_text('from outside.bridge import run\n')
    (tmp_path / 'outside/bridge.mojo').write_text('from providers.python.hidden_pyapi import run\n')
    _, _, problems = check(tmp_path)
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
