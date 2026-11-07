"""Runtime partition dependency boundaries; these checks never compile Mojo."""
from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache
from development_build import SourceGraph

ROOT = Path(__file__).resolve().parent.parent


ADD_TYPE = re.compile(r"\.add_type\[(.+?)\]\(\"(\w+)\"\)")
REGISTRAR = re.compile(r"^def (\w+)\(mut module: PythonModuleBuilder\)[^\n]*:\n((?:[ \t]+[^\n]*\n|\n)*)", re.M)


def registered_types(member):
    """(names the entry registers, names it attaches through a lookup-guarded shared registrar).

    One process loads every binding, and a Mojo type registered by two modules aborts the import, so a
    type has one owner. Registrations are what the entry itself calls: its own module.add_type lines and
    the registrar functions (def f(mut module: PythonModuleBuilder) with an add_type) it calls with its
    module. A file in the closure for another export registers nothing. A registrar that looks the type up
    first (native_identity_type) is shared by design and reported apart.
    """
    graph = SourceGraph(ROOT)
    entry = f"bindings/{member}.mojo"
    closure = graph.closure([member])
    assert closure['scope'] == 'closure', closure['unresolved']
    source = (ROOT/entry).read_text()
    own = {name for _, name in ADD_TYPE.findall(source)}
    shared = set()
    for path in closure['files']:
        if path == entry:
            continue
        for match in REGISTRAR.finditer((ROOT/path).read_text()):
            body = match.group(2)
            if ".add_type[" in body and re.search(rf"\b{match.group(1)}\(module\)", source):
                names = {name for _, name in ADD_TYPE.findall(body)}
                (shared if "lookup_py_type_object" in body else own).update(names)
    return own, shared


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
    own, shared = registered_types('_mojolearn_runtime_manifold_host')
    assert own and all(name.startswith('_Manifold') for name in own), own
    assert not shared, shared
    assert not own & registered_types('_mojolearn_tokenizer_host')[0]
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
    # The sequential owner registers only its own types and attaches the shared identity key.
    own, shared = registered_types('_mojolearn_runtime_sequential_host')
    assert own and all(name.startswith('_Sequential') for name in own), own
    assert shared <= {'_NativeIdentityKey'}, shared
    # Every type registered by a release host entry has exactly one owner.
    owners = {}
    for member in release_host_members():
        for name in registered_types(member)[0]:
            owners.setdefault(name, []).append(member)
    assert not {name: found for name, found in owners.items() if len(found) > 1}
    assert owners['_TreesResident'] == ['_mojolearn_runtime_classical_host']
    source = (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    assert 'module.add_type[TreesResident]' not in source
    assert release_host_members().count('_mojolearn_runtime_sequential_host') == 1


#: The twelve development units the classical and tokenizer owners became
#: (2026-11-07 host split, docs/build/HOT_GROUPS.md section 6).
SPLIT_UNITS = ('runtime_classical', 'runtime_trees', 'runtime_family', 'runtime_prep', 'runtime_decomp',
               'runtime_linear', 'tokenizer', 'runtime_catalog', 'runtime_xseq', 'runtime_byte_trainer',
               'runtime_parallel', 'runtime_neural')


def test_classical_resident_allocation_and_lifetime_are_owned_by_one_development_unit():
    groups = json.loads((ROOT/'bindings/development_groups.json').read_text())['groups']
    graph = SourceGraph(ROOT)
    # The classical door and the two doors it forwards to are each in exactly
    # one unit (the import graph at file level).
    expected = {'bindings/classical_native.mojo': {('host', 'runtime_classical')},
                'bindings/classical_trees_door.mojo': {('host', 'runtime_trees')},
                'bindings/trees_public_native.mojo': {('host', 'runtime_trees')},
                'bindings/classical_family_door.mojo': {('host', 'runtime_family')},
                'bindings/family_native.mojo': {('host', 'runtime_family')}}
    owners = {path: set() for path in expected}
    for mode, partitions in groups.items():
        for name, members in partitions.items():
            closure = graph.closure(members)
            assert closure['scope'] == 'closure', closure['unresolved']
            for path in expected:
                if path in closure['files']:
                    owners[path].add((mode, name))
    assert owners == expected, owners
    # Files such as trees_lifetime.mojo and host_archive_native.mojo are
    # imported by several units for a type or one helper, so the resident
    # allocator's owner is checked on the export graph instead: of the split
    # units' exports, only the tree door reaches trees_lifetime.mojo, and the
    # others reach host_archive_native.mojo only for its metric and layout
    # helpers.
    from lane_select import _MojoDefGraph, _binding_exports
    defs = _MojoDefGraph()
    helpers = {'host_archive_rbc_metric_object', 'host_archive_radius_refusals',
               'host_archive_check_layout', 'host_archive_class_values'}
    reached = set()
    for unit in SPLIT_UNITS:
        assert groups['host'][unit] == [f'_mojolearn_{unit}_host']
        source = f'bindings/_mojolearn_{unit}_host.mojo'
        exports = _binding_exports(source)
        assert exports, unit
        for name in exports:
            closure = defs.closure(source, name)
            assert closure is not None, (unit, name)
            if any(node[1] == 'bindings/trees_lifetime.mojo' for node in closure):
                reached.add((unit, name))
            if unit != 'runtime_trees':
                archive = {node[2] for node in closure if node[1] == 'bindings/host_archive_native.mojo'}
                assert archive <= helpers, (unit, name, archive - helpers)
    assert reached == {('runtime_trees', 'trees_native_call')}, reached


def test_classical_legacy_exports_are_declared_native_aliases_with_one_type_registration():
    import ast
    import re
    manifest = ast.parse((ROOT/'python/mojolearn/host_surface.py').read_text())
    aliases = next(ast.literal_eval(node.value) for node in manifest.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                       and t.id == 'HOST_IMPORT_ALIASES' for t in node.targets))
    specs = aliases['_mojolearn_tokenizer_host']
    # One spec per owner (2026-11-07 host split); the classical owner, which
    # registers the resident tree types, is connected first.
    assert [spec['binding'] for spec in specs] == [f'_mojolearn_{unit}_host' for unit in SPLIT_UNITS
                                                   if unit != 'tokenizer']
    assert specs[0]['types'] == ('_TreesResident', '_TreesResidentDeath')
    assert all(spec['types'] == () for spec in specs[1:])
    assert {spec['marker'] for spec in specs} == {'tokenizer_host_runtime_classical_abi'}
    assert {spec['abi'] for spec in specs} == {2}
    legacy = (ROOT/'bindings/_mojolearn_tokenizer_host.mojo').read_text()
    names = lambda text: set(re.findall(r'module\.def_py_c_function\([^,]+, "([^"]+)"\)', text))
    refusals = set(re.findall(r'module\.def_py_c_function\(runtime_classical_unavailable_c, "([^"]+)"\)', legacy))
    seen = set()
    for spec in specs:
        owner = (ROOT/f"bindings/{spec['binding']}.mojo").read_text()
        assert set(spec['functions']) <= names(legacy), spec['binding']
        assert set(spec['functions']) == names(owner), spec['binding']
        assert not seen & set(spec['functions'])
        seen |= set(spec['functions'])
        assert release_host_members().count(spec['binding']) == 1
    # Every native refusal the legacy module registers has exactly one owner.
    assert refusals <= seen, sorted(refusals - seen)
    registrations = [str(path.relative_to(ROOT)) for path in (ROOT/'bindings').glob('*.mojo')
                     if 'module.add_type[TreesResident]' in path.read_text()]
    assert registrations == ['bindings/_mojolearn_runtime_classical_host.mojo']
