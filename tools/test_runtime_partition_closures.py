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


def release_host_members():
    """Every member of the release host groups, in group order (0.8.59: several
    host files, one compile each; a member lives in exactly one of them)."""
    groups = json.loads((ROOT/'bindings/merged_groups.json').read_text())['dirs']['host']['groups']
    assert groups and all(stem.endswith('_host') for stem in groups)
    return [name for members in groups.values() for name in members]


def test_release_host_groups_hold_each_member_once_and_include_the_new_logical_entry():
    assert release_host_members().count('_mojolearn_runtime_model_selection_host') == 1
    entry = ROOT/'bindings/entry/entry__mojolearn_runtime_model_selection_host.mojo'
    assert 'PyInit__mojolearn_runtime_model_selection_host' in entry.read_text()


def test_manifold_implementation_edits_invalidate_only_the_manifold_owner():
    groups = json.loads((ROOT/'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    paths = ('bindings/mani_kit.mojo', 'bindings/mani_decomp_native.mojo',
             'bindings/mani_tsne_native.mojo', 'bindings/mani_umap_native.mojo')
    owners = {path: set() for path in paths}
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            for path in paths:
                if path in closure['files']:
                    owners[path].add((mode, name))
    assert all(found == {('host', 'runtime_manifold')} for found in owners.values()), owners


def test_manifold_owner_has_no_tokenizer_type_registration_or_archive_dependency():
    closure = SourceGraph(ROOT).closure(['_mojolearn_runtime_manifold_host'])
    assert closure['scope'] == 'closure', closure['unresolved']
    assert 'bindings/mani_kit.mojo' in closure['files']
    assert 'bindings/family_native.mojo' not in closure['files']
    assert 'bindings/host_archive_native.mojo' not in closure['files']
    assert 'bindings/_mojolearn_tokenizer_host.mojo' not in closure['files']
    assert all('.add_type' not in (ROOT/path).read_text() for path in closure['files'])
    assert release_host_members().count('_mojolearn_runtime_manifold_host') == 1


def test_sequential_implementation_edits_do_not_invalidate_tokenizer():
    groups = json.loads((ROOT/'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    paths = ('bindings/sequential_execution.mojo', 'bindings/sequential_checkpoint.mojo')
    owners = {path: set() for path in paths}
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            for path in paths:
                if path in closure['files']:
                    owners[path].add((mode, name))
    assert all(found == {('host', 'runtime_sequential')} for found in owners.values()), owners


def test_sequential_adds_no_registered_type_and_existing_types_each_keep_one_owner():
    graph = SourceGraph(ROOT)
    owner = graph.closure(['_mojolearn_runtime_sequential_host'])
    tokenizer = graph.closure(['_mojolearn_tokenizer_host'])
    assert owner['scope'] == tokenizer['scope'] == 'closure'
    assert 'bindings/sequential_execution.mojo' in owner['files']
    assert 'bindings/sequential_execution.mojo' not in tokenizer['files']
    assert all('.add_type' not in (ROOT/path).read_text() for path in owner['files'])
    source = (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert source.count('module.add_type[') == 6
    classical = (ROOT/'bindings/_mojolearn_runtime_classical_host.mojo').read_text()
    assert classical.count('module.add_type[') == 1
    assert 'module.add_type[TreesResident]("_TreesResident")' in classical
    assert 'module.add_type[TreesResident]' not in source
    assert release_host_members().count('_mojolearn_runtime_sequential_host') == 1


def test_classical_resident_allocation_and_lifetime_are_owned_by_one_development_unit():
    groups = json.loads((ROOT/'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    paths = ('bindings/trees_lifetime.mojo', 'bindings/trees_public_forest.mojo',
             'bindings/host_svr_native.mojo', 'bindings/host_svc_native.mojo',
             'bindings/classical_native.mojo', 'bindings/host_archive_native.mojo')
    owners = {path: set() for path in paths}
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            for path in paths:
                if path in closure['files']:
                    owners[path].add((mode, name))
    assert all(found == {('host', 'runtime_classical')} for found in owners.values()), owners


def test_classical_legacy_exports_are_declared_native_aliases_with_one_type_registration():
    import ast
    import re
    manifest = ast.parse((ROOT/'python/mojolearn/host_surface.py').read_text())
    aliases = next(ast.literal_eval(node.value) for node in manifest.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                       and t.id == 'HOST_IMPORT_ALIASES' for t in node.targets))
    spec, = aliases['_mojolearn_tokenizer_host']
    assert spec['binding'] == '_mojolearn_runtime_classical_host'
    assert spec['types'] == ('_TreesResident',)
    assert len(spec['functions']) == 29
    legacy = (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    owner = (ROOT/'bindings/_mojolearn_runtime_classical_host.mojo').read_text()
    names = lambda text: set(re.findall(r'module\.def_py_c_function\([^,]+, "([^"]+)"\)', text))
    assert set(spec['functions']) <= names(legacy)
    assert set(spec['functions']) == names(owner)
    registrations = [str(path.relative_to(ROOT)) for path in (ROOT/'bindings').glob('*.mojo')
                     if 'module.add_type[TreesResident]' in path.read_text()]
    assert registrations == ['bindings/_mojolearn_runtime_classical_host.mojo']
    assert release_host_members().count(spec['binding']) == 1
