"""Runtime partition dependency boundaries; these checks never compile Mojo."""
from pathlib import Path
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
