"""Runtime partition dependency boundaries; these checks never compile Mojo."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache
from development_build import SourceGraph

ROOT = Path(__file__).resolve().parent.parent


def test_shared_host_allocator_has_no_archive_or_tree_training_dependency():
    closure = SourceGraph(ROOT).closure(['host_array_native'])
    assert closure['scope'] == 'closure', closure['unresolved']
    assert 'bindings/host_archive_native.mojo' not in closure['files']
    assert 'core/host_archive_gbdt.mojo' not in closure['files']
    assert not any(path.startswith('gbdt/') for path in closure['files'])


def test_model_selection_does_not_pull_archive_readers_through_sparse_allocation():
    closure = SourceGraph(ROOT).closure(['model_selection_native'])
    assert closure['scope'] == 'closure', closure['unresolved']
    assert 'bindings/host_sparse_native.mojo' in closure['files']
    assert 'bindings/host_array_native.mojo' in closure['files']
    assert 'bindings/host_archive_native.mojo' not in closure['files']
    assert not any(path.startswith('gbdt/') for path in closure['files'])


def test_storage_only_imports_use_the_leaf_instead_of_the_archive_facade():
    obsolete = []
    for path in (ROOT/'bindings').glob('*.mojo'):
        for module, names in bincache.parse_imports(path.read_text()):
            if module == 'bindings.host_archive_native' and set(names) == {'host_archive_typed'}:
                obsolete.append(str(path.relative_to(ROOT)))
    assert not obsolete, obsolete


def test_model_selection_edits_invalidate_only_the_independent_development_owner():
    groups = json.loads((ROOT/'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    owners = set()
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            if 'bindings/msel_split.mojo' in closure['files']:
                owners.add((mode, name))
    assert owners == {('host', 'runtime_model_selection')}


def test_release_keeps_one_host_file_and_includes_the_new_logical_entry():
    release = json.loads((ROOT/'bindings/merged_groups.json').read_text())
    groups = release['dirs']['host']['groups']
    assert set(groups) == {'_mojolearn_native_host'}
    assert groups['_mojolearn_native_host'].count('_mojolearn_runtime_model_selection_host') == 1
    entry = ROOT/'bindings/entry/entry__mojolearn_runtime_model_selection_host.mojo'
    assert 'PyInit__mojolearn_runtime_model_selection_host' in entry.read_text()
