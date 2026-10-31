"""Synthetic file-proof tests; these do not qualify native execution or compilation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

import release_host_composition as hc

ROOT = Path(__file__).resolve().parents[1]
COMPILER = 'a' * 64
COMMIT = 'b' * 40
#: the merged host files and the identical device files the real plan ships
HOST_FILES = hc.host_files()
sys.path.insert(0, str(ROOT / 'tools'))
import merged_bindings  # noqa: E402
DEVICE_FILES = sorted(stem + '.so' for stem, _c, _m in merged_bindings.plan('identical', 'linux', True))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return path


def build(root, vendor, arch, body, source, rootdir=None, linux_cpu='x86-64-v3'):
    leg = vendor + '/' + arch
    setdir = root / 'sets' / leg
    hosts = {f: setdir / 'host' / f for f in HOST_FILES}
    (setdir / 'host').mkdir(parents=True)
    for f, host in hosts.items():
        host.write_bytes(body + b'-' + f.encode())
    (setdir / 'identical').mkdir()
    gpus = [setdir / 'identical' / f for f in DEVICE_FILES]
    for gpu in gpus:
        gpu.write_bytes(b'device-' + leg.encode() + b'-' + gpu.name.encode())
    library = setdir / '.libs/libRuntime.so'
    library.parent.mkdir()
    library.write_bytes(b'runtime')
    names = sorted(hc.expected_names())
    index = {'format': 1, 'files': hc.host_plan()}
    write(setdir / 'host' / hc.INDEX, index)
    (setdir / 'readback.txt').write_text(''.join(f'host {name} cpu\n' for name in names))
    inventory = [[name, hc.file_sha(source / name)] for name in ['sample.mojo', *hc.HOST_FORMAT_SOURCES]]
    write(root / 'source-inventory.json', inventory)
    flags = dict(compile_jobs='2', gpu_archs=None, linux_cpu='x86-64-v3', numeric_mode='identical', package_byte_lm='1', target_column='cpu')
    identities = {'linux-64/host/-/host/' + name: dict(schema='mojolearn.binding-identity.v2',
        name=name, target='linux-64', vendor='host', tier='host', arch='', flags=flags,
        toolchain={'packages': ['mojo-fixture']}, builders={'builder': 'c' * 64}, image={'cpu': 'fixture'},
        closure={'digest': 'd' * 64, 'scope': 'fixture'}) for name in names}
    capture = dict(schema=hc.CAPTURE_SCHEMA, phase='before-build', source_commit=COMMIT,
        vendor=vendor, arch=arch, compiler_sha256=COMPILER, binding_identities=identities,
        source_inventory_sha256=hc.file_sha(root / 'source-inventory.json'), source_inventory_digest=hc.digest(inventory),
        installed_packages=['mojo-fixture'], toolchain={'packages': ['mojo-fixture']})
    write(root / 'compile-inputs.json', capture)
    manifest = dict(binding_layout='merged', extensions=[dict(path='host/' + f,
        sha256=hc.file_sha(host), needed=['libRuntime.so'], runpath='$ORIGIN/../.libs') for f, host in hosts.items()],
        staged_libs=[dict(name=library.name, sha256=hc.file_sha(library))])
    write(setdir / 'manifest.json', manifest)
    cache_keys = {}
    for f in HOST_FILES:
        key = dict(schema='mojolearn-bincache-v1', script='bindings/merged_build.sh',
            args=['host', '--group', f[:-3]] + (['--rootdir', rootdir] if rootdir else []),
            target_column='cpu', numeric_mode='identical', device_arch='none', gpu_archs='',
            build_env={'MOJOLEARN_TARGET_COLUMN': 'cpu', 'MOJOLEARN_COMPILE_JOBS': '2',
                       'MOJOLEARN_LINUX_CPU': linux_cpu, 'MOJOLEARN_CUDA_CODE_FORMAT': 'ptx' if arch == 'sm_80' else 'native'})
        cache_keys[f] = hc.digest(key)
        write(root / 'bincache/keys' / (cache_keys[f] + '.json'), key)
    prefix = 'mojolearn/' + leg + '/'
    proof = dict(schema=hc.BUILD_SCHEMA, action='build', complete=True, build_exit=0, source_commit=COMMIT,
        binding_layout='merged', release_modes=['identical'], compile_inputs_sha256=hc.file_sha(root / 'compile-inputs.json'),
        host_extension={prefix + 'host/' + f: hc.file_sha(host) for f, host in hosts.items()},
        extensions={prefix + 'identical/' + gpu.name: hc.file_sha(gpu) for gpu in gpus},
        merged_indexes={prefix + 'host/' + hc.INDEX: dict(sha256=hc.file_sha(setdir / 'host' / hc.INDEX), files=index['files'])},
        source_inventory=inventory, source_sha256=hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest(),
        set_manifests={prefix + 'manifest.json': hc.file_sha(setdir / 'manifest.json')},
        bincache={prefix + 'host/' + f: dict(key=cache_keys[f], compiled_from_commit=COMMIT,
            outcome='miss+built-uploaded', unstaged_sha256='e' * 64) for f in HOST_FILES})
    write(root / 'build-provenance.json', proof)
    return root


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'sample.mojo').write_text('fixture source\n')
    for name in hc.HOST_FORMAT_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixed host compiler fixture\n')
    builds = [build(tmp_path / str(i), v, a, b'host-' + a.encode(), source)
              for i, (v, a) in enumerate((('cuda', 'sm_89'), ('cuda', 'sm_80'), ('hip', 'gfx942')))]
    return source, builds


def replace_witness(doc, index, name, change):
    value = hc.decoded(doc['sources'][index][name])
    change(value)
    text = json.dumps(value, sort_keys=True, indent=2) + '\n'
    record = {'utf8': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    doc['sources'][index][name] = record
    return value, record


def test_selects_one_exact_retained_host_preserving_original_variants(fixture, tmp_path):
    source, builds = fixture
    original = {str(p): hc.file_sha(p) for root in builds for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'assembly'
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, out, source)
    sets = {s['leg']: out / 'sets' / s['leg'] for s in doc['sources']}
    report = hc.validate_assembly(doc, sets, [b / 'build-provenance.json' for b in builds], source)
    assert report['selected_leg'] == 'cuda/sm_89'
    assert len({json.dumps(v['host_sha256'], sort_keys=True) for v in report['variants']}) == 3
    assert all(set(v['host_sha256']) == set(HOST_FILES) for v in report['variants'])
    assert all(hc.file_sha(Path(p)) == h for p, h in original.items())
    for f in HOST_FILES:
        assert len({hc.file_sha(s / 'host' / f) for s in sets.values()}) == 1
    assert all((s / 'manifest.json').read_bytes() == (builds[i] / 'sets' / leg / 'manifest.json').read_bytes()
               for i, (leg, s) in enumerate(sets.items()))
    with pytest.raises(ValueError, match='destination already exists'):
        hc.prepare(builds, 'cuda/sm_89', COMPILER, out, source)


@pytest.mark.parametrize('name,change,reason', [
    ('proof', lambda d: d.update(build_exit=1), 'Incomplete'),
    ('proof', lambda d: d.update(build_exit=False), 'Incomplete'),
    ('capture', lambda d: d.update(compiler_sha256='f' * 64), 'Capture|compiler'),
    ('capture', lambda d: d['binding_identities'].pop(next(iter(d['binding_identities']))), 'Capture|identities'),
    ('index', lambda d: d['files'][HOST_FILES[0]].pop(), 'index'),
    ('index', lambda d: d['files'][HOST_FILES[1]].append(d['files'][HOST_FILES[0]].pop()), 'index'),
    ('host_cache_key', lambda d: d[HOST_FILES[-1]]['build_env'].update(MOJOLEARN_COMPILE_JOBS='8'), 'cache witness|flags'),
    ('host_cache_key', lambda d: d[HOST_FILES[0]].update(args=['host']), 'cache witness'),
    ('manifest', lambda d: d['staged_libs'][0].update(sha256='f' * 64), 'manifest'),
])
def test_mutated_raw_producer_inputs_cannot_be_admitted(fixture, name, change, reason):
    source, builds = fixture
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, source_root=source)
    replace_witness(doc, 1, name, change)
    with pytest.raises(ValueError, match=reason):
        hc.validate_document(doc)


def test_even_consistently_rehashed_changed_compiler_is_refused(fixture):
    source, builds = fixture
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, source_root=source)
    _, record = replace_witness(doc, 1, 'capture', lambda d: d.update(compiler_sha256='f' * 64))
    replace_witness(doc, 1, 'proof', lambda d: d.update(compile_inputs_sha256=record['sha256']))
    with pytest.raises(ValueError, match='declared compiler'):
        hc.validate_document(doc)


def test_even_consistently_rehashed_changed_host_flags_are_refused(fixture):
    source, builds = fixture
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, source_root=source)
    keys, _ = replace_witness(doc, 1, 'host_cache_key',
                              lambda d: d[HOST_FILES[0]]['build_env'].update(MOJOLEARN_MOJO_BUILD_FLAGS='--unsafe'))
    replace_witness(doc, 1, 'proof', lambda d: d['bincache']['mojolearn/cuda/sm_80/host/' + HOST_FILES[0]].update(
        key=hc.digest(keys[HOST_FILES[0]])))
    with pytest.raises(ValueError, match='contract differs'):
        hc.validate_document(doc)


@pytest.mark.parametrize('target', ['selected', 'library', 'manifest', 'device', 'index', 'source'])
def test_packing_rechecks_actual_bytes(fixture, tmp_path, target):
    source, builds = fixture
    out = tmp_path / 'assembly'
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, out, source)
    sets = {s['leg']: out / 'sets' / s['leg'] for s in doc['sources']}
    first = sets['cuda/sm_89']
    paths = dict(selected=first / 'host' / HOST_FILES[-1], library=first / '.libs/libRuntime.so',
        manifest=first / 'manifest.json', device=first / 'identical' / DEVICE_FILES[-1],
        index=first / 'host' / hc.INDEX, source=source / 'sample.mojo')
    paths[target].write_bytes(paths[target].read_bytes() + b'changed')
    with pytest.raises(ValueError):
        hc.validate_assembly(doc, sets, [b / 'build-provenance.json' for b in builds], source)


def test_missing_original_bytes_and_wrong_declared_compiler_stop_before_copy(fixture, tmp_path):
    source, builds = fixture
    with pytest.raises(ValueError, match='declared compiler'):
        hc.prepare(builds, 'cuda/sm_89', 'f' * 64, tmp_path / 'bad-compiler', source)
    (builds[1] / 'sets/cuda/sm_80/host' / HOST_FILES[-1]).write_bytes(b'wrong')
    with pytest.raises(ValueError, match='original host bytes'):
        hc.prepare(builds, 'cuda/sm_89', COMPILER, tmp_path / 'bad-host', source)
    assert not (tmp_path / 'bad-compiler').exists() and not (tmp_path / 'bad-host').exists()


def test_unknown_selection_missing_proof_and_witness_hash_tampering_fail(fixture, tmp_path):
    source, builds = fixture
    with pytest.raises(ValueError, match='absent'):
        hc.prepare(builds, 'cuda/sm_90', COMPILER, source_root=source)
    out = tmp_path / 'assembly'
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, out, source)
    sets = {s['leg']: out / 'sets' / s['leg'] for s in doc['sources']}
    with pytest.raises(ValueError, match='Pack proofs'):
        hc.validate_assembly(doc, sets, [builds[0] / 'build-provenance.json'], source)
    doc['sources'][0]['capture']['utf8'] += ' '
    with pytest.raises(ValueError, match='witness hash'):
        hc.validate_document(doc)


def load_packer():
    spec = importlib.util.spec_from_file_location('host_composition_packer', ROOT / 'packaging/linux/pack_wheel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pack_requires_receipt_and_attributes_only_the_selected_host(fixture, tmp_path, monkeypatch):
    source, builds = fixture
    version_file = source / 'python/mojolearn/_version.py'
    version_file.parent.mkdir(parents=True)
    version_file.write_text('__version__ = "0.8.57"\n')
    monkeypatch.setenv('MOJOLEARN_RELEASE_MODES', 'identical')
    out = tmp_path / 'assembly'
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, out, source)
    pw = load_packer()
    sets = []
    for row in doc['sources']:
        vendor, arch = row['leg'].split('/')
        root = out / 'sets' / row['leg']
        sets.append(pw.SetDir(vendor, arch,
            {row['leg'] + '/identical/' + f: root / 'identical' / f for f in DEVICE_FILES},
            {'libRuntime.so': root / '.libs/libRuntime.so'}, hc.decoded(row['manifest']),
            {name: root / 'host' / f for f, names in hc.host_plan().items() for name in names},
            None, 'merged', {'host/' + hc.INDEX: root / 'host' / hc.INDEX}, {}))
    proofs = [b / 'build-provenance.json' for b in builds]
    options = dict(source_root=source, required={tuple(s['leg'].split('/')) for s in doc['sources']},
                   profile=pw.RELEASE_SPLIT_PROFILE)
    with pytest.raises(SystemExit, match='explicit validated host composition'):
        pw.release_inventory(sets, proofs, '0.8.57', **options)
    inventory = pw.release_inventory(sets, proofs, '0.8.57',
        host_composition=out / 'host-composition.json', **options)
    assert inventory['host_composition']['selected_leg'] == 'cuda/sm_89'
    assert len(inventory['host_composition']['variants']) == 3
    for row in inventory['host_native'].values():
        assert row['built_by'] == ['cuda/sm_89']
        assert row['reused_in'] == ['cuda/sm_80', 'hip/gfx942']
        assert row['origin'] == 'retained-host-composition'
    doc['selected_sha256'] = 'f' * 64
    write(out / 'host-composition.json', doc)
    with pytest.raises(SystemExit, match='Invalid retained host composition'):
        pw.release_inventory(sets, proofs, '0.8.57', host_composition=out / 'host-composition.json', **options)


def test_new_pack_tool_reads_shipped_source_from_explicit_frozen_checkout(tmp_path, monkeypatch):
    # Existing packaging fixtures hold inert bytes; this executes no product
    # import, Mojo operation, publication or native qualification gate.
    spec = importlib.util.spec_from_file_location('host_composition_pack_fixtures', ROOT / 'packaging/linux/test_split_wheels.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    pw = fixtures.pw
    set_dirs = fixtures.make_sets(tmp_path / 'inputs', sets=fixtures.NATIVE_SETS)
    monkeypatch.setattr(pw, 'REPO', tmp_path / 'not-the-package-source')
    monkeypatch.setattr(pw, 'PY_DIR', tmp_path / 'not-the-package-python')
    monkeypatch.setattr(pw, 'PKG', tmp_path / 'not-the-package')
    assert pw.main([arg for directory in set_dirs for arg in ('--set', directory)] +
        ['--source-root', str(ROOT), '--profile', 'generic', '--out', str(tmp_path / 'wheel')], _gates=False) == 0
    wheel = next((tmp_path / 'wheel').glob('*.whl'))
    import zipfile
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read('mojolearn/_version.py') == (ROOT / 'python/mojolearn/_version.py').read_bytes()


def test_unknown_host_format_scope_cannot_exempt_a_flag_difference(fixture):
    source, builds = fixture
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, source_root=source)
    doc.pop('host_format_scope')
    with pytest.raises(ValueError, match='scope witness'):
        hc.validate_document(doc)


def _text_witness(text):
    return {'utf8':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}


def test_restored_shard_requires_successful_matching_original_compilation(tmp_path):
    source_dir=tmp_path/'source'; source_dir.mkdir(); (source_dir/'sample.mojo').write_text('source\n')
    for rel in hc.HOST_FORMAT_SOURCES:
        path=source_dir/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('source\n')
    directory=build(tmp_path/'build', 'hip','gfx942',b'host',source_dir)
    source,_=hc.inspect_build(directory)
    proof=hc.decoded(source['proof']); keys=hc.decoded(source['host_cache_key'])
    origins={}
    for filename,key in keys.items():
        cache=proof['bincache']['mojolearn/hip/gfx942/host/'+filename]
        cache['outcome']='hot-hit'
        outcome='miss+built-not-uploaded:no-slot'
        origins[filename]=dict(
            shard=_text_witness('host:merged_build.sh:'+filename[:-3]+'\tok\t'+outcome+'\t1.0\t'+cache['key']+'\n'),
            key=_text_witness(json.dumps(key)),
            provenance=_text_witness('date\tbindings/merged_build.sh\t'+outcome+'\t'+cache['key']+'\t1.0\tpython/mojolearn/host/'+filename+'='+cache['unstaged_sha256']+'\n'),
            archive=_text_witness(cache['key']+'\t'+'a'*64+'\n'),
            assembly_archive=_text_witness(cache['key']+'\t'+'a'*64+'\n'))
    source['proof']=_text_witness(json.dumps(proof)); source['host_shard_origins']=origins
    hc._record(source,COMPILER)
    filename=next(iter(origins))
    for field in ('shard','key','provenance','archive','assembly_archive'):
        bad=copy.deepcopy(source)
        bad['host_shard_origins'][filename][field]=_text_witness('{}' if field=='key' else 'not-the-compiled-group\n')
        with pytest.raises((ValueError, KeyError), match='shard|Shard|Assembly|compiler'):
            hc._record(bad,COMPILER)
    bad=copy.deepcopy(source);bad.pop('host_shard_origins')
    with pytest.raises(ValueError,match='receipt'):
        hc._record(bad,COMPILER)


# ---------------------------------------------------------------- the 0.8.63 fixed root
def _source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'sample.mojo').write_text('fixture source\n')
    for name in hc.HOST_FORMAT_SOURCES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixed host compiler fixture\n')
    return source


LEGS = (('cuda', 'sm_89'), ('hip', 'gfx942'), ('cuda', 'sm_80'))


def test_fixed_generated_root_directory_is_an_accepted_host_invocation(tmp_path):
    source = _source(tmp_path)
    builds = [build(tmp_path / str(i), v, a, b'host-' + a.encode(), source, rootdir='/tmp/mojolearn-merged-roots/host')
              for i, (v, a) in enumerate(LEGS)]
    doc = hc.prepare(builds, 'cuda/sm_89', COMPILER, None, source)
    assert doc['selected_leg'] == 'cuda/sm_89'


def test_a_different_generated_root_directory_is_a_different_contract(tmp_path):
    source = _source(tmp_path)
    builds = [build(tmp_path / str(i), v, a, b'host-' + a.encode(), source,
                    rootdir='/tmp/roots-' + (a if a == 'sm_80' else 'x'))
              for i, (v, a) in enumerate(LEGS)]
    with pytest.raises(ValueError, match='contract differs'):
        hc.prepare(builds, 'cuda/sm_89', COMPILER, None, source)


def test_a_relative_or_extra_argument_is_not_a_host_invocation(tmp_path):
    source = _source(tmp_path)
    builds = [build(tmp_path / str(i), v, a, b'host-' + a.encode(), source, rootdir='relative/root')
              for i, (v, a) in enumerate(LEGS)]
    with pytest.raises(ValueError, match='Cache witness'):
        hc.prepare(builds, 'cuda/sm_89', COMPILER, None, source)


# ---------------------------------------------------------------- release.py's linux-wait composition
class _Leg:
    """The fields of release.Leg compose_host_mismatch reads."""
    def __init__(self, base, vendor, arch):
        self.name, self.vendor, self.arch = f'{vendor}-{arch}', vendor, arch
        self.release_build = base / self.name / 'release-build'


def _release_legs(tmp_path, legs=LEGS, **kw):
    source = _source(tmp_path)
    out = []
    for vendor, arch in legs:
        leg = _Leg(tmp_path / 'legs', vendor, arch)
        body = b'host-owner' if arch != 'sm_80' else b'host-other-cpu'
        build(leg.release_build / 'build', vendor, arch, body, source,
              rootdir='/tmp/mojolearn-merged-roots/host', **(kw if arch == 'sm_80' else {}))
        out.append(leg)
    return source, out


def _release():
    sys.path.insert(0, str(ROOT / 'tools'))
    import release
    return release


def test_linux_wait_composes_a_host_mismatch_from_the_owner_leg(tmp_path):
    release = _release()
    source, legs = _release_legs(tmp_path)
    mismatch = release.host_digest_mismatches(legs)
    assert set(mismatch) == set(HOST_FILES)
    rec = release.compose_host_mismatch(legs, mismatch, source, tmp_path / 'comp')
    assert rec['owner'] == 'cuda-sm_89' and rec['selected_leg'] == 'cuda/sm_89'
    owner = legs[0].release_build / 'build/sets/cuda/sm_89/host'
    for f, row in rec['taken'].items():
        assert row['from_leg'] == 'cuda-sm_89' and row['sha256'] == hc.file_sha(owner / f)
        for vendor, arch in LEGS:
            assert hc.file_sha(tmp_path / 'comp/sets' / vendor / arch / 'host' / f) == row['sha256']
    assert hc.file_sha(Path(rec['receipt'])) == rec['receipt_sha256']
    # a rerun keeps the retained composition when it is still the same one
    again = release.compose_host_mismatch(legs, mismatch, source, tmp_path / 'comp')
    assert again['receipt_sha256'] == rec['receipt_sha256']
    # a device binding changed after composing is refused, never re-composed
    device = tmp_path / 'comp/sets/hip/gfx942/identical' / DEVICE_FILES[0]
    device.write_bytes(b'tampered')
    with pytest.raises(release.StepFailed, match='Device artifact changed'):
        release.compose_host_mismatch(legs, mismatch, source, tmp_path / 'comp')


def test_linux_wait_refuses_without_the_owner_leg(tmp_path):
    release = _release()
    source, legs = _release_legs(tmp_path, legs=(('hip', 'gfx942'), ('cuda', 'sm_80')))
    with pytest.raises(release.StepFailed, match='host owner leg cuda-sm_89'):
        release.compose_host_mismatch(legs, release.host_digest_mismatches(legs), source, tmp_path / 'comp')
    assert not (tmp_path / 'comp').exists()


def test_linux_wait_refuses_a_mismatch_whose_recorded_flags_differ(tmp_path):
    release = _release()
    source, legs = _release_legs(tmp_path, linux_cpu='x86-64-v4')
    with pytest.raises(release.StepFailed, match='refused'):
        release.compose_host_mismatch(legs, release.host_digest_mismatches(legs), source, tmp_path / 'comp')
    assert not (tmp_path / 'comp').exists()


def test_linux_wait_refuses_a_non_host_or_missing_file(tmp_path):
    release = _release()
    source, legs = _release_legs(tmp_path)
    mismatch = release.host_digest_mismatches(legs)
    with pytest.raises(release.StepFailed, match='not planned merged host'):
        release.compose_host_mismatch(legs, dict(mismatch, **{'_stray.so': {'cuda-sm_89': 'a'}}), source, tmp_path / 'c1')
    first = HOST_FILES[0]
    short = dict(mismatch, **{first: {k: v for k, v in mismatch[first].items() if k != 'cuda-sm_80'}})
    with pytest.raises(release.StepFailed, match='missing on a leg'):
        release.compose_host_mismatch(legs, short, source, tmp_path / 'c2')
