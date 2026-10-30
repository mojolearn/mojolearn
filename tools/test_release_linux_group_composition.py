"""Composition boundaries: no compiler, GPU or product numerical work."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import release_linux_group_composition as composition
from staged_native_admission import complete_native_proof


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def recipe():
    source = dict(source='source-tree', declared=['source.mojo'], merged='merged-plan')
    fields = dict(source='source-tree', script='bindings/merged_build.sh',
        args=['gpu', '--group', '_mojolearn_native'], numeric_mode='identical', gpu_archs='gfx942',
        target_column='amd', device_arch='none', repo_path='/root/mojolearn', os=dict(machine='x86_64'),
        declared=dict(inputs=['source.mojo'], outputs=['python/mojolearn/identical/_mojolearn_native.so'], shell='bash'),
        build_env=dict(MODULAR_HOME='/root/mojolearn/.pixi/envs/default/share/max',
            MOJOLEARN_BUILD_PIXI_ENV='default', MOJOLEARN_BUILD_TIERS='identical',
            MOJOLEARN_COMPILE_JOBS='8', MOJOLEARN_CUDA_CODE_FORMAT='native', MOJOLEARN_GPU_ARCHS='gfx942',
            MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_MERGED_KEY='merged-plan',
            MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_PACKAGE_BYTE_LM='1', MOJOLEARN_TARGET_COLUMN='amd'))
    return fields, source


def test_original_recipe_admitted(recipe):
    composition.device_recipe(*recipe, '_mojolearn_native')


@pytest.mark.parametrize('field,value', [
    ('source', 'changed'), ('script', 'another-builder.sh'), ('args', ['gpu']),
    ('numeric_mode', 'fast'), ('gpu_archs', 'gfx950'), ('target_column', 'nvidia'),
    ('device_arch', 'sm_89'), ('variant', 'ptx'), ('repo_path', '/different'),
    ('os', {'machine': 'aarch64'}), ('declared', {'inputs': [], 'outputs': [], 'shell': 'bash'})])
def test_changed_native_recipe_refused(recipe, field, value):
    fields, source = recipe
    fields[field] = value
    with pytest.raises(ValueError):
        composition.device_recipe(fields, source, '_mojolearn_native')


@pytest.mark.parametrize('field,value', [
    ('MOJOLEARN_COMPILE_JOBS', '1'), ('MOJOLEARN_CUDA_CODE_FORMAT', 'ptx'),
    ('MOJOLEARN_LINUX_CPU', 'native'), ('MOJOLEARN_NUMERIC_MODE', 'fast'),
    ('MOJOLEARN_GPU_ARCHS', 'gfx950'), ('MOJOLEARN_MERGED_KEY', 'other'),
    ('MOJOLEARN_EXTRA_FLAGS', '-O0')])
def test_changed_or_unrecorded_flags_refused(recipe, field, value):
    fields, source = recipe
    fields['build_env'][field] = value
    with pytest.raises(ValueError, match='flags/platform'):
        composition.device_recipe(fields, source, '_mojolearn_native')


@pytest.fixture
def proof(tmp_path, monkeypatch):
    retained = tmp_path / 'compiler.log'
    retained.write_bytes(b'original compiler output\n')
    projection = dict(evidence={str(retained): dict(sha256=sha(retained.read_bytes()), bytes=retained.stat().st_size)},
                      host_extension={'host': 'a' * 64}, extensions={'device': 'b' * 64}, readback_complete=True)
    monkeypatch.setattr(composition, 'project', lambda *args: copy.deepcopy(projection))
    monkeypatch.setattr(composition.shutil, 'which', lambda _: '/trusted/patchelf')
    return dict(schema=composition.SCHEMA, action=composition.ACTION, complete=True,
                producer_inputs={}, **copy.deepcopy(projection)), retained


def test_composition_dispatch_does_not_require_whole_build_exit(proof):
    document, _ = proof
    assert complete_native_proof(document, '/source')
    assert 'build_exit' not in document


@pytest.mark.parametrize('change', ['failed', 'fake-build', 'missing-evidence', 'changed-raw', 'changed-projection', 'missing-source', 'missing-readback'])
def test_proof_boundary_refuses_false_success(proof, change):
    document, retained = proof
    source = '/source'
    if change == 'failed':
        document['complete'] = False
    elif change == 'fake-build':
        document['build_exit'] = 0
    elif change == 'missing-evidence':
        retained.unlink()
    elif change == 'changed-raw':
        retained.write_bytes(b'altered compiler output\n')
    elif change == 'changed-projection':
        document['extensions']['device'] = 'c' * 64
    elif change == 'missing-readback':
        document['readback_complete'] = False
    else:
        source = None
    assert not complete_native_proof(document, source)


def test_proof_cannot_select_an_executable(proof, monkeypatch):
    document, _ = proof
    document['patchelf'] = '/untrusted/proof-supplied-program'
    seen = []
    previous = composition.project
    monkeypatch.setattr(composition, 'project', lambda inputs, source, tool: (seen.append(tool), previous())[1])
    assert composition.validate(document, '/source')
    assert seen == ['/trusted/patchelf']


def test_coverage_refuses_missing_and_duplicate_groups(tmp_path):
    for rows in ([], [dict(binding='unknown')] * 46):
        with pytest.raises(ValueError, match='coverage'):
            composition.host_groups(rows, tmp_path, 'a' * 40, tmp_path, 'b' * 64, 'patchelf', composition.Evidence())
    with pytest.raises(ValueError, match='coverage'):
        composition.device_groups([], tmp_path, tmp_path, 'b' * 64, 'patchelf', composition.Evidence())


def test_source_and_runtime_bytes_are_not_replaceable_by_digest_only(tmp_path):
    file = tmp_path / 'raw.so'
    file.write_bytes(b'original')
    evidence = composition.Evidence()
    evidence.file(file)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed during'):
        evidence.file(file)


def test_packed_byte_check_refuses_native_swap(tmp_path):
    (tmp_path / 'host').mkdir()
    native = tmp_path / 'host/one.so'
    native.write_bytes(b'swapped')
    document = dict(extensions={}, host_extension={composition.PREFIX + 'host/one.so': sha(b'original')})
    with pytest.raises(ValueError, match='packed native bytes'):
        composition.validate_bytes(document, tmp_path)


@pytest.fixture
def native_readbacks(tmp_path, monkeypatch):
    import release_ptx_assembly
    monkeypatch.setattr(release_ptx_assembly, 'architectures',
                        lambda p: [] if p.parent.name == 'host' else ['gfx942'])
    source = 'a' * 40
    rows = []
    for tier, name, vendor in [('host', 'one', 'cpu'), ('identical', 'two', 'hip')]:
        raw = dict(name=name, tier=tier, vendor=vendor, binary_sha256=sha(name.encode()))
        if tier == 'host':
            raw.update(numeric_mode=1, column='cpu')
        log = tmp_path / (tier + '-' + name + '.log')
        log.write_bytes(b'original native load output\n')
        execution = dict(name=name, tier=tier, exit_code=0, log_sha256=sha(log.read_bytes()),
                         binary_sha256=raw['binary_sha256'])
        for suffix, value in (('.json', raw), ('.execution.json', execution)):
            (tmp_path / (tier + '-' + name + suffix)).write_text(json.dumps(value))
        rows.append(dict(**raw, architectures=[] if tier == 'host' else ['gfx942'], execution=execution))
    (tmp_path / 'producer.py').write_bytes(b'retained metadata-only producer\n')
    report = dict(schema='mojolearn.amd.package-readback.v1', source_commit=source, failures=[],
                  producer_sha256=sha((tmp_path / 'producer.py').read_bytes()),
                  expected=2, completed=2, rows=rows)
    (tmp_path / 'readback.json').write_text(json.dumps(report))
    (tmp_path / 'readback.txt').write_text('host one cpu\nidentical two hip\n')
    (tmp_path / 'arch_readback.txt').write_text('host one NONE-BY-DESIGN\nidentical two gfx942\n')
    hosts = {composition.PREFIX + 'host/one.so': sha(b'one')}
    extensions = {composition.PREFIX + 'identical/group.so': sha(b'two')}
    indexes = {'index': dict(files={'group.so': ['two']})}
    return tmp_path, source, tmp_path / 'site', hosts, extensions, indexes


def test_actual_readback_receipts_are_required(native_readbacks):
    assert composition.readbacks(*native_readbacks, composition.Evidence())


@pytest.mark.parametrize('change', ['binary', 'source', 'failed', 'coverage', 'typed-marker'])
def test_readback_refuses_unrelated_or_failed_observations(native_readbacks, change):
    directory = native_readbacks[0]
    p = directory / 'readback.json'
    report = json.loads(p.read_text())
    if change == 'binary':
        report['rows'][0]['binary_sha256'] = 'b' * 64
    elif change == 'source':
        report['source_commit'] = 'c' * 40
    elif change == 'failed':
        report['rows'][0]['execution']['exit_code'] = 1
    elif change == 'coverage':
        report['rows'].pop()
    else:
        (directory / 'readback.txt').write_text('host one cpu\nidentical two cuda\n')
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        composition.readbacks(*native_readbacks, composition.Evidence())
