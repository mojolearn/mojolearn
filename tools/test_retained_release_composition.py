"""Synthetic artifact/provenance controls; no compilation or product execution."""
import copy
import json
from pathlib import Path

import pytest

import release_host_composition as hc
import release_reuse as rr
import retained_release_composition as rc
import test_release_host_composition as old_fixture

OLD = 'b' * 40
NEW = 'c' * 40
COMPILER = old_fixture.COMPILER


def update(path, operation):
    doc = json.loads(path.read_text())
    operation(doc)
    old_fixture.write(path, doc)
    return doc


def augment(root, vendor, arch, identities):
    leg = vendor + '/' + arch
    prefix = 'mojolearn/' + leg + '/'
    directory = root / 'sets' / leg
    keys = {k: v for k, v in identities.items() if k.startswith('linux-64/' + leg + '/')}
    update(root / 'compile-inputs.json', lambda d: d['binding_identities'].update(keys))
    index = dict(format=1, files={'_mojolearn_native.so': sorted(v['name'] for v in keys.values())})
    old_fixture.write(directory / 'identical' / hc.INDEX, index)
    with (directory / 'readback.txt').open('a') as out:
        out.write(''.join(f'identical {name} {vendor}\n' for name in index['files']['_mojolearn_native.so']))
    gpu = directory / 'identical/_mojolearn_native.so'
    update(directory / 'manifest.json', lambda d: d['extensions'].append(
        dict(path='identical/' + gpu.name, sha256=hc.file_sha(gpu))))
    def revise(proof):
        proof['compile_inputs_sha256'] = hc.file_sha(root / 'compile-inputs.json')
        proof['merged_indexes'][prefix + 'identical/' + hc.INDEX] = dict(
            sha256=hc.file_sha(directory / 'identical' / hc.INDEX), files=index['files'])
        proof['set_manifests'][prefix + 'manifest.json'] = hc.file_sha(directory / 'manifest.json')
    update(root / 'build-provenance.json', revise)
    return root


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ['sample.mojo', *hc.HOST_FORMAT_SOURCES]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original source\n')
    identities = {}
    for binding in rr.bindings(rr.LINUX, ['identical']):
        if binding.host:
            continue
        identities[binding.key] = dict(schema=rr.SCHEMA, name=binding.name, target=binding.target,
            vendor=binding.vendor, arch=binding.arch, tier=binding.tier, script=binding.script,
            flags=binding.flags(), toolchain={'packages': ['mojo-fixture']}, builders={'builder': 'a' * 64},
            image='fixture', closure={'scope': 'closure', 'digest': 'd' * 64})
    monkeypatch.setattr(old_fixture, 'COMMIT', OLD)
    old = [augment(old_fixture.build(tmp_path / ('old-' + arch), vendor, arch, b'old-host', source),
                   vendor, arch, identities) for vendor, arch in (('cuda','sm_89'),('cuda','sm_80'),('hip','gfx942'))]
    old_capture = json.loads((old[0] / 'compile-inputs.json').read_text())
    before = dict(binding_identities={**identities, **old_capture['binding_identities']},
                  runtime_identity={'runtime': 'd' * 64}, adapter_sources={'adapter.mojo': 'e' * 64},
                  adapter_builders={'builder': 'f' * 64}, source_inventory=json.loads((old[0] / 'source-inventory.json').read_text()))
    (source / 'sample.mojo').write_text('new host source\n')
    monkeypatch.setattr(old_fixture, 'COMMIT', NEW)
    host = augment(old_fixture.build(tmp_path / 'new-host', 'cuda', 'sm_89', b'new-host', source),
                   'cuda', 'sm_89', identities)
    update(host / 'compile-inputs.json', lambda d: d['binding_identities']['linux-64/host/-/host/_mojolearn_tokenizer_host']['closure'].update(digest='e' * 64))
    update(host / 'build-provenance.json', lambda d: d.update(compile_inputs_sha256=hc.file_sha(host / 'compile-inputs.json')))
    current = copy.deepcopy(before)
    current['binding_identities']['linux-64/host/-/host/_mojolearn_tokenizer_host']['closure']['digest'] = 'e' * 64
    current['source_inventory'] = json.loads((host / 'source-inventory.json').read_text())
    monkeypatch.setattr(rc, 'current_source', lambda root: (NEW, copy.deepcopy(current['source_inventory']), copy.deepcopy(current)))
    monkeypatch.setattr(rc, 'source_context', lambda root, revision: copy.deepcopy({OLD: before, NEW: current}[revision]))
    return source, old, host, before, current


def assembled(fixture, tmp_path, index=0):
    source, builds, host, _, _ = fixture
    output = tmp_path / ('assembled-' + str(index))
    if index == 1:
        directory = builds[index] / 'sets/cuda/sm_80'
        members = json.loads((directory / 'identical' / hc.INDEX).read_text())['files']['_mojolearn_native.so']
        old_fixture.write(directory / 'PTX_BASELINE.json', dict(
            schema='mojolearn.ptx-set.v2', code_format='ptx', vendor='cuda', target='sm_80',
            min_compute_capability=[8, 0], errors=[], source_commit=OLD, source_dirty=False,
            arithmetic='unchanged', files=[dict(file='identical/_mojolearn_native.so',
                sha256=hc.file_sha(directory / 'identical/_mojolearn_native.so'), numeric_mode='identical',
                members=members, ptx_modules=[dict(target='sm_80', ptx_isa='8.1', approx={}, sha256='f' * 64)])],
            indexes=[dict(file='identical/' + hc.INDEX, sha256=hc.file_sha(directory / 'identical' / hc.INDEX))]))
    proof = rc.prepare(builds[index], host, source, COMPILER, output)
    return proof, output


def test_retains_original_producers_and_independent_new_host(fixture, tmp_path):
    source, old, host, _, _ = fixture
    original = {str(p): hc.file_sha(p) for root in [*old, host] for p in root.rglob('*') if p.is_file()}
    proof, output = assembled(fixture, tmp_path)
    assert rc.validate(proof, source) and rc.validate_bytes(proof, output / 'set')
    assert proof['action'] == rc.ACTION and 'build_exit' not in proof
    assert {r['source_commit'] for r in proof['artifact_origins'].values()} == {OLD, NEW}
    assert (output / 'set/host' / hc.HOST_FILE).read_bytes() == b'new-host'
    assert (output / 'set/identical/_mojolearn_native.so').read_bytes() == b'device-cuda/sm_89'
    assert all(hc.file_sha(Path(path)) == value for path, value in original.items())
    with pytest.raises(ValueError, match='destination already exists'):
        rc.prepare(old[0], host, source, COMPILER, output)


@pytest.mark.parametrize('changed', ['device-closure','device-flags','device-target','host-closure',
                                    'runtime','adapter','current-inventory','invented-build-exit'])
def test_different_inputs_cannot_be_admitted(fixture, tmp_path, changed):
    proof, _ = assembled(fixture, tmp_path)
    current = proof['admitted_inputs']
    device_key = next(k for k in current['binding_identities'] if '/cuda/sm_89/' in k)
    if changed == 'device-closure':
        current['binding_identities'][device_key]['closure']['digest'] = 'f' * 64
    elif changed == 'device-flags':
        current['binding_identities'][device_key]['flags']['compile_jobs'] = '99'
    elif changed == 'device-target':
        current['binding_identities'][device_key]['arch'] = 'sm_80'
    elif changed == 'host-closure':
        current['binding_identities']['linux-64/host/-/host/_mojolearn_tokenizer_host']['closure']['digest'] = 'f' * 64
    elif changed == 'runtime':
        current['runtime_identity'] = {'runtime': 'e' * 64}
    elif changed == 'adapter':
        # A declared retained adapter turns on exact source/build admission.
        device = hc.decoded(proof['original_device']['proof'])
        device['adapters'] = {'mojolearn/cuda/sm_89/_adapters': {'files': {'a': 'a' * 64}}}
        raw = json.dumps(device)
        proof['original_device']['proof'] = {'utf8': raw, 'sha256': hc.hashlib.sha256(raw.encode()).hexdigest()}
        proof.update(rc._projection(proof['original_device'], proof['original_host'], COMPILER))
        current['adapter_sources']['adapter.mojo'] = 'f' * 64
    elif changed == 'current-inventory':
        proof['source_inventory'][0][1] = 'f' * 64
        proof['source_sha256'] = rc.sha_inventory(proof['source_inventory'])
    else:
        proof['build_exit'] = 0
    with pytest.raises(ValueError):
        rc.validate(proof, fixture[0])


@pytest.mark.parametrize('rel', ['identical/_mojolearn_native.so', 'host/' + hc.HOST_FILE,
                                'identical/' + hc.INDEX, '.libs/libRuntime.so'])
def test_mutated_shipped_bytes_fail(fixture, tmp_path, rel):
    proof, output = assembled(fixture, tmp_path)
    (output / 'set' / rel).write_bytes(b'changed')
    with pytest.raises(ValueError):
        rc.validate_bytes(proof, output / 'set')


def test_compiler_change_cannot_be_hidden_by_consistent_capture_hash(fixture, tmp_path):
    source, old, host, _, _ = fixture
    update(old[0] / 'compile-inputs.json', lambda d: d.update(compiler_sha256='f' * 64))
    update(old[0] / 'build-provenance.json', lambda d: d.update(compile_inputs_sha256=hc.file_sha(old[0] / 'compile-inputs.json')))
    with pytest.raises(ValueError, match='compiler'):
        rc.prepare(old[0], host, source, COMPILER, tmp_path / 'refused')


def test_original_source_context_is_rechecked(fixture, tmp_path, monkeypatch):
    proof, _ = assembled(fixture, tmp_path)
    original = rc.source_context
    def changed(root, revision):
        doc = original(root, revision)
        if revision == OLD:
            doc['adapter_sources']['adapter.mojo'] = 'c' * 64
        return doc
    monkeypatch.setattr(rc, 'source_context', changed)
    with pytest.raises(ValueError, match='original source inputs'):
        rc.validate(proof, fixture[0])


def test_ptx_rebind_preserves_arithmetic_metadata_and_exact_bytes(fixture, tmp_path):
    proof, output = assembled(fixture, tmp_path, 1)
    p = output / 'set/PTX_BASELINE.json'
    doc = json.loads(p.read_text())
    assert doc['source_commit'] == NEW and doc['arithmetic'] == 'unchanged'
    assert rc.validate_ptx(doc, output / 'set')
    original = hc.decoded(doc[rc.PTX_FIELD]['original_manifest'])
    assert original['source_commit'] == OLD
    doc['arithmetic'] = 'forged'
    with pytest.raises(ValueError, match='arithmetic metadata'):
        rc.validate_ptx(doc, output / 'set')


def test_new_schema_dispatch_does_not_admit_a_claimed_build(fixture, tmp_path):
    from staged_native_admission import complete_native_proof
    proof, _ = assembled(fixture, tmp_path)
    assert complete_native_proof(proof, fixture[0])
    proof['build_exit'] = 0
    assert not complete_native_proof(proof, fixture[0])


def test_packer_reports_original_sources_and_zero_new_compilations(fixture, tmp_path, monkeypatch):
    source, _, _, _, _ = fixture
    version = source / 'python/mojolearn/_version.py'
    version.parent.mkdir(parents=True)
    version.write_text('__version__ = "0.8.57"\n')
    monkeypatch.setenv('MOJOLEARN_RELEASE_MODES', 'identical')
    proof, output = assembled(fixture, tmp_path)
    pw = old_fixture.load_packer()
    root = output / 'set'
    s = pw.SetDir('cuda', 'sm_89',
        {'cuda/sm_89/identical/_mojolearn_native.so': root / 'identical/_mojolearn_native.so'},
        {'libRuntime.so': root / '.libs/libRuntime.so'}, hc.decoded(proof['original_device']['manifest']),
        {name: root / 'host' / hc.HOST_FILE for name in hc.expected_names()}, None, 'merged',
        {'host/' + hc.INDEX: root / 'host' / hc.INDEX, 'identical/' + hc.INDEX: root / 'identical' / hc.INDEX}, {})
    result = pw.release_inventory([s], [output / 'assembly-proof.json'], '0.8.57', source_root=source,
                                  required={('cuda','sm_89')}, profile=pw.RELEASE_SPLIT_PROFILE)
    assert result['sets']['cuda/sm_89']['origin'] == 'retained-composition'
    assert result['reuse']['built'] == 0 and result['reuse']['retained_composed'] == 2
    assert result['binding_origin']['mojolearn/cuda/sm_89/identical/_mojolearn_native.so']['source_commit'] == OLD
    assert all('built_by' not in row and row['admitted_by'] == ['cuda/sm_89']
               for row in result['binding_origin'].values())
    assert all(r['source_commit'] == NEW and r['origin'] == 'retained-composition' and 'built_by' not in r
               for r in result['host_native'].values())


def test_adapter_closure_resolves_symlinked_temp_root_and_rejects_source_escape(tmp_path):
    root = tmp_path / 'source'
    files = {'adapters/native/c_api.mojo': 'from core.fixture import scalar\n',
             'core/fixture.mojo': 'comptime scalar = 1\n', 'adapters/max_ops/__init__.mojo': '',
             'adapters/torch/registration.cpp': '#include "abi.h"\n', 'adapters/native/abi.h': '// API\n'}
    files.update({p: '# fixture\n' for p in ('packaging/adapters/build.sh', 'packaging/adapters/fetch_torch.py',
                 'packaging/adapters/manifest.py', 'packaging/adapters/pins.json')})
    for name, text in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    link = tmp_path / 'link'
    link.symlink_to(root, target_is_directory=True)
    inputs = rc.adapter_inputs(link)
    assert inputs == rc.adapter_inputs(root)
    assert inputs['core/fixture.mojo'] == hc.file_sha(root / 'core/fixture.mojo')
    outside = tmp_path / 'outside.mojo'
    outside.write_text('comptime scalar = 2\n')
    (root / 'core/fixture.mojo').unlink()
    (root / 'core/fixture.mojo').symlink_to(outside)
    with pytest.raises(ValueError, match='escaping adapter source'):
        rc.adapter_inputs(link)
