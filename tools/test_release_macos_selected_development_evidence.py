"""Scope and provenance failures for selected Mac host compilation admission."""
import copy
import json
from pathlib import Path

import pytest

import development_build as dev
import release_macos_selected_development_evidence as selected
from test_macos_cpu_compile_contract import image_bytes
from test_release_ledger_admission import repo, development, git, put, sha  # noqa: F401


@pytest.fixture
def partial(repo, development):
    path = development['result']
    result = json.loads(path.read_text())
    row = result['plan']['groups'][0]
    row['selected'] = True
    row['fields']['toolchain']['locked']['pixi_lock_sha256'] = sha(repo / 'pixi.lock')
    result.update(state='incomplete', activated=[], missing=['host:_mojolearn_dev_unbuilt_host'])
    other = copy.deepcopy(row)
    other.update(stem='_mojolearn_dev_unbuilt_host', selected=False)
    result['plan']['groups'].append(other)

    def save():
        row['key'] = dev.digest(row['fields'])
        log = put(path.parent / 'selected-compile.log', 'synthetic successful compiler output\n')
        result['results'] = [dict(stem=row['stem'], key=row['key'], exit_code=0, log=str(log))]
        artifact = Path(result['plan']['cache']) / 'artifacts' / row['key']
        artifact.mkdir(parents=True, exist_ok=True)
        binary = artifact / (row['stem'] + '.so')
        binary.write_bytes(image_bytes())
        put(artifact / 'receipt.json', dict(key=row['key'], fields=row['fields'], exit_code=0, sha256=sha(binary)))
        put(path, result)
        return binary

    binary = save()
    return dict(path=path, result=result, row=row, save=save, binary=binary)


def facts(repo, partial):
    return selected.facts(repo, dict(result=partial['path']))


def test_partial_result_qualifies_only_successful_selected_host_group(repo, partial):
    found = facts(repo, partial)
    assert found['original_state'] == 'incomplete'
    assert found['full_set_qualified'] is False
    assert list(found['groups']) == ['_mojolearn_dev_native_host']
    record = selected.collect(repo, {'configuration': {'dependency_context': {'cpu_target': 'osx-arm64'}}},
                              dict(result=partial['path']))
    assert set(record['units']) == {'compile-apple/osx-arm64/host/-/host/_mojolearn_native_host',
                                   'compile-cpu/osx-arm64/host/-/host/_mojolearn_native_host'}
    assert record['facts']['groups']['_mojolearn_dev_native_host']['host_toolchain']['schema'] == 'mojolearn.macos-cpu-compile.v1'
    assert json.loads(partial['path'].read_text())['state'] == 'incomplete'


@pytest.mark.parametrize('mutation', ['failed', 'missing-result', 'wrong-key', 'duplicate-result', 'missing-selected'])
def test_missing_or_failed_selected_evidence_cannot_pass(repo, partial, mutation):
    result = partial['result']
    if mutation == 'failed': result['results'][0]['exit_code'] = 1
    if mutation == 'missing-result': result['results'] = []
    if mutation == 'wrong-key': result['results'][0]['key'] = '0' * 64
    if mutation == 'duplicate-result': result['results'].append(dict(result['results'][0]))
    if mutation == 'missing-selected': result['missing'].append('host:' + partial['row']['stem'])
    put(partial['path'], result)
    with pytest.raises(ValueError, match='selected group'):
        facts(repo, partial)


def test_binary_mismatch_refused(repo, partial):
    partial['binary'].write_bytes(image_bytes() + b'altered')
    with pytest.raises(ValueError, match='binary differs'):
        facts(repo, partial)


def test_complete_compile_output_must_remain_retained(repo, partial):
    Path(partial['result']['results'][0]['log']).unlink()
    with pytest.raises(ValueError, match='compile log is not retained'):
        facts(repo, partial)


def test_changed_frozen_source_requires_actual_compiled_bytes(repo, partial):
    put(repo / 'bindings/native.mojo', 'fn native(): changed()\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'changed numerical input')
    with pytest.raises(ValueError, match='compiled source bytes differ'):
        facts(repo, partial)


def test_precommit_revision_is_provenance_not_a_false_source_match(repo, partial):
    original = partial['result']['source_commit']
    put(repo / 'bindings/native.mojo', 'fn native(): compiled_repair()\n')
    partial['row']['fields']['source']['files']['bindings/native.mojo'] = sha(repo / 'bindings/native.mojo')
    partial['save']()
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'freeze actual compiled repair')
    found = facts(repo, partial)
    assert found['original_source_revision'] == original
    assert found['source_revision'] == git(repo, 'rev-parse', 'HEAD') != original


def test_lock_provenance_cannot_be_reconstructed_from_current_machine(repo, partial):
    partial['row']['fields']['toolchain']['locked']['pixi_lock_sha256'] = '0' * 64
    partial['save']()
    with pytest.raises(ValueError, match='compiled CPU lock differs'):
        facts(repo, partial)


def test_unselected_failure_does_not_invent_selected_failure_or_success(repo, partial):
    partial['result']['results'].append(dict(stem='_mojolearn_dev_unbuilt_host', exit_code=1))
    put(partial['path'], partial['result'])
    assert len(facts(repo, partial)['groups']) == 1


def test_cpu_link_contract_does_not_qualify_device_image(repo, partial):
    partial['row']['mode'] = 'identical'
    put(partial['path'], partial['result'])
    with pytest.raises(ValueError, match='host groups only'):
        facts(repo, partial)
