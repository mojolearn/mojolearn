"""Synthetic retained-producer tests only; no compilation or product execution."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

import release_macos_host_family_evidence as host


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def producer(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    files = {'bindings/build_host_family.sh': b'build the named family\n',
             'bindings/build_defines.sh': b'defines\n', 'tools/compile_slot.sh': b'slot\n',
             'pixi.lock': b'lock\n', 'pixi.toml': b'[workspace]\nname="fixture"\n',
             'bindings/_mojolearn_core_host.mojo': b'# module\n',
             'bindings/entry/entry__mojolearn_core_host.mojo': b'# entry\n'}
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '-q')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    manifest = tmp_path / 'source-manifest.json'
    dump(manifest, {k: sha(v) for k, v in files.items()})
    archive = tmp_path / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    binary = tmp_path / '_mojolearn_core_host.so'
    binary.write_bytes(b'fixture raw binary bytes')
    (tmp_path / 'compile.log').write_text('retained complete compile output\n')
    receipt = dict(family='core', source_commit=git('rev-parse', 'HEAD'), cwd=str(root),
                   argv=['bash', 'tools/compile_slot.sh', 'sh', 'bindings/build_host_family.sh', 'core'],
                   exit_code=0, sources_changed_during_build=[], target='CPU identical macOS arm64',
                   environment=dict(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_TARGET_COLUMN='cpu',
                                    MOJOLEARN_COMPILE_JOBS='10', MOJOLEARN_MOJO_BUILD_FLAGS=''),
                   compiler_sha256='a'*64, source_count=len(files), sdk='26.5',
                   source_archive_sha256=sha(archive.read_bytes()), binary=str(binary),
                   binary_sha256=sha(binary.read_bytes()), binary_bytes=binary.stat().st_size)
    path = tmp_path / 'receipt.json'
    dump(path, receipt)
    return root, path, receipt


def test_retains_original_revision_flags_and_missing_witness(producer):
    root, path, _ = producer
    result = host._collect(root, {}, {'receipt': str(path)})
    assert result['spec_ids'] == ['compile-apple']
    assert result['facts']['host_toolchain'] is None
    assert result['facts']['limitations'] == ['Original Apple auxiliary-toolchain witness was not captured']
    context = result['configuration']['dependency_context']
    assert set(context['selected_binding_compile_jobs'].values()) == {'10'}
    assert list(context['selected_binding_build_flags'].values()) == [{'MOJOLEARN_MOJO_BUILD_FLAGS': ''}]
    assert len(result['facts']['evidence']) == 5


@pytest.mark.parametrize('change,match', [
    ({'exit_code': 1}, 'compile failed'),
    ({'exit_code': False}, 'compile failed'),
    ({'sources_changed_during_build': ['changed.mojo']}, 'source changed'),
    ({'binary_sha256': 'f'*64}, 'raw binary'),
    ({'source_archive_sha256': 'f'*64}, 'archive hash'),
    ({'family': 'tokenizer'}, 'family differs'),
    ({'apple_toolchain': {'sdk': '26.5'}}, 'incomplete original Apple'),
])
def test_rejects_inexact_or_failed_originals(producer, change, match):
    root, path, receipt = producer
    receipt.update(change)
    dump(path, receipt)
    with pytest.raises(ValueError, match=match):
        host._facts(root, {'receipt': str(path)})


@pytest.mark.parametrize('key,value', [('MOJOLEARN_COMPILE_JOBS','0'),
    ('MOJOLEARN_NUMERIC_MODE','fast'), ('MOJOLEARN_MOJO_BUILD_FLAGS','--optimization-level 2'),
    ('MOJOLEARN_UNRECORDED_SWITCH','1')])
def test_rejects_changed_or_unknown_environment(producer, key, value):
    root, path, receipt = producer
    receipt['environment'][key] = value
    dump(path, receipt)
    with pytest.raises(ValueError):
        host._facts(root, {'receipt': str(path)})


def test_detects_manifest_and_archive_disagreement(producer):
    root, path, _ = producer
    manifest = path.parent / 'source-manifest.json'
    data = json.loads(manifest.read_text())
    data['pixi.lock'] = 'f'*64
    dump(manifest, data)
    with pytest.raises(ValueError, match='archive bytes differ'):
        host._facts(root, {'receipt': str(path)})


def test_missing_witness_cannot_qualify_an_otherwise_matching_unit(producer, monkeypatch):
    root, path, _ = producer
    result = host._collect(root, {}, {'receipt': str(path)})
    rid, unit = next(iter(result['units'].items()))
    wanted = {'inputs': {'binding': dict(target=host.common.MAC, vendor='host', arch=None,
                                       tier='host', name='_mojolearn_core_host'), 'unit': {}}}
    monkeypatch.setattr(host, 'supports_requirement', lambda _: True)
    with pytest.raises(ValueError, match='missing original Apple auxiliary'):
        host._check(rid, wanted, result['facts'], unit)


def test_original_tokenizer_controller_receipt_shape(producer):
    root, path, receipt = producer
    receipt['command'] = receipt.pop('argv')
    receipt.pop('family')
    receipt.pop('cwd')
    dump(path, receipt)
    assert host._facts(root, {'receipt': str(path)})['family'] == 'core'


def qualified(producer):
    root, path, receipt = producer
    receipt['apple_toolchain'] = dict(build='25F84', developer_dir='/Applications/Xcode.app/Contents/Developer',
                                     metal='fixture Metal', product='26.5.2', sdk='26.5', xcode='Xcode 26.6')
    dump(path, receipt)
    result = host._collect(root, {}, {'receipt': str(path)})
    rid, unit = next(iter(result['units'].items()))
    facts = result['facts']
    tree = host.common.Tree.at(root, facts['source_revision'])
    flags = dict(numeric_mode='identical', target_column='cpu', gpu_archs=None, compile_jobs='10', package_byte_lm='1')
    semantic = {'MOJOLEARN_MOJO_BUILD_FLAGS': ''}
    binding = dict(target=host.common.MAC, vendor='host', arch=None, tier='host', name='_mojolearn_core_host',
                   host_toolchain=receipt['apple_toolchain'], flags=flags, selected_build_flags=semantic,
                   toolchain={'pixi_lock_sha256': facts['source_files']['pixi.lock']},
                   closure={'scope':'closure', 'digest':host.common.closure_digest(tree, facts['source_files'])})
    wanted = dict(kind='compile', inputs=dict(binding=binding, unit=dict(kind='compile', vendor='apple',
                  release_modes=['identical'], build_flags=semantic, compiler_sha256=receipt['compiler_sha256'])),
                  coverage=[host.common.BASE_COVERAGE,unit['binding_key']], provenance_sources=list(facts['source_files']))
    return rid, wanted, facts, unit


def test_exact_original_witness_and_inputs_qualify(producer):
    rid, wanted, facts, unit = qualified(producer)
    rows, evidence = host._check(rid, wanted, facts, unit)
    assert [r['id'] for r in rows] == wanted['coverage']
    assert all(r['completed'] == 1 and r['failed'] == 0 for r in rows)
    assert len(evidence) == 5


@pytest.mark.parametrize('field,value,match', [('compile_jobs','1','worker allocation'),
    ('numeric_mode','fast','routing'), ('package_byte_lm','0','routing')])
def test_requirement_cannot_relabel_original_flags(producer, field, value, match):
    rid, wanted, facts, unit = qualified(producer)
    wanted['inputs']['binding']['flags'][field] = value
    with pytest.raises(ValueError, match=match):
        host._check(rid, wanted, facts, unit)


def test_requirement_cannot_relabel_default_as_o0(producer):
    rid, wanted, facts, unit = qualified(producer)
    wanted['inputs']['unit']['build_flags'] = {'MOJOLEARN_MOJO_BUILD_FLAGS':'--optimization-level 0'}
    with pytest.raises(ValueError, match='compiler flags differ'):
        host._check(rid, wanted, facts, unit)


def test_required_uncaptured_source_is_refused(producer):
    rid, wanted, facts, unit = qualified(producer)
    facts['source_files'].pop('bindings/_mojolearn_core_host.mojo')
    with pytest.raises(ValueError, match='required source was not captured'):
        host._check(rid, wanted, facts, unit)
