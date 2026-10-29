"""Actual GHA selected-group compilation; no RunPod or installed-stage fiction."""
import hashlib
from pathlib import Path
import re

import development_build as dev
import release_linux_development_evidence as linux
import release_retained_compile_evidence as common
from release_ledger_admission import load, reference, require

NAME = 'gha-development-build'
SCHEMA = 'mojolearn.gha-development-group.v1'


def facts(root, producer):
    request_path = Path(producer['request']).resolve()
    base, request = request_path.parent, load(request_path)
    revision = request.get('source_commit', '')
    require(request.get('schema') == SCHEMA and re.fullmatch('[0-9a-f]{40}', revision)
            and request.get('dirty_tracked') == '' and request.get('status') == 'selected-group-compiled',
            'not a clean successful GHA selected-group request')
    tree = common.Tree.at(root, revision)
    require(common.HEX64.fullmatch(str(request.get('source_builder_sha256')))
            and request['source_builder_sha256'] == tree.sha256('tools/development_build.py'),
            'GHA source builder differs from frozen source')
    archive = reference(base/'source.tar.gz')
    require(archive['sha256'] == request.get('source_archive_sha256'), 'retained source archive differs')
    # Workflow dotfiles may be omitted by upload-artifact. A separately retained
    # pre-dispatch tooling directory is admitted only by the original hashes.
    tooling = Path(producer.get('tooling', base/'tooling')).resolve()
    expected_tools = {'tools/gha_development_group.py', 'tools/gha_release_box.sh',
                      '.github/workflows/development-group-build.yml'}
    require(set(request.get('tooling_files', {})) == expected_tools, 'GHA tooling inventory differs')
    tool_refs = []
    for name, digest in request['tooling_files'].items():
        item = reference(linux._relative(tooling, name))
        require(item['sha256'] == digest, 'retained GHA tooling differs: '+name)
        tool_refs.append(item)
    require(request['tooling_script_sha256'] == request['tooling_files']['tools/gha_development_group.py'],
            'GHA controller differs from pre-run witness')
    result_path = base/'evidence/result.json'
    result = load(result_path)
    require(result.get('schema') == common.DEV_SCHEMA and result.get('source_commit') == revision
            and result.get('state') in ('activated', 'incomplete')
            and result.get('plan', {}).get('platform') == 'linux'
            and result['plan'].get('layout') == 'development', 'GHA development result differs')
    require((request.get('exit_code'), result['state']) in ((0, 'activated'), (3, 'incomplete')),
            'GHA whole-set exit/state differs')
    rows = [r for r in result['plan']['groups'] if r.get('stem') == request.get('group')]
    require(len(rows) == 1, 'GHA selected group is not unique')
    row = rows[0]
    require(row.get('selected') is True and row['key'] == request.get('retained_key')
            and row['members'] == request.get('members')
            and dev.selected_group(row['mode'], row['stem'], [request.get('requested_group', '')]),
            'GHA request does not name the compiled group')
    require(all(r.get('exit_code') == 0 for r in result.get('results', [])), 'GHA requested build failed')
    group = linux.group_facts(row, result, result_path, base/'retained'/row['key']/'receipt.json', tree)
    workers = request.get('compiler_workers')
    require(type(workers) is int and workers > 0 and workers == group['compiler_workers']
            and request.get('build_jobs') == 1, 'GHA actual worker request differs')
    env = dict(request.get('environment', {}))
    require(env.pop('MOJOLEARN_COMMIT', None) == revision, 'GHA commit environment differs')
    actual = group['environment']
    require(all(actual.get(k) == v for k, v in env.items())
            and set(actual)-set(env) <= set(common.GLUE_INTERPRETERS), 'GHA environment differs from compiled fields')
    require(request.get('arch') == actual.get('MOJOLEARN_GPU_ARCHS'), 'GHA requested target differs')
    manifest_path = base/'retained-inputs.json'
    inputs = load(manifest_path)
    require(isinstance(inputs, dict) and len(inputs) == request.get('retained_input_count'),
            'GHA consumed input inventory differs')
    for name, digest in inputs.items():
        require(reference(linux._relative(base, name))['sha256'] == digest, 'retained input differs: '+name)
    for name, digest in group['source_files'].items():
        require(inputs.get('consumed-source/'+name) == digest, 'compiled input was not retained: '+name)
    require(inputs.get('generated-roots/'+row['stem']+'.mojo') == group['root_sha256'],
            'actual generated root differs from compiled root')
    require(request.get('bootstrap_inputs') and all(inputs.get(k) == v for k,v in request['bootstrap_inputs'].items()),
            'bootstrap runtime bytes are not retained')
    return dict(producer=NAME, source_root=str(Path(root).resolve()), source_revision=revision,
                request=reference(request_path), result=reference(result_path), source_archive=archive,
                retained_inputs=reference(manifest_path), tooling=tool_refs,
                artifact_scope='raw-selected-compilation-only', groups={row['stem']:group})


def collect(root, document, producer):
    return linux.collect_facts(facts(root, producer))


def check_unit(requirement_id, wanted, found, unit):
    coverage = linux.check_group_unit(requirement_id, wanted, found, unit)
    group = found['groups'][unit['group']]
    return coverage, [found[k] for k in ('request','result','source_archive','retained_inputs')] + found['tooling'] + [
        group[k] for k in ('receipt','execution','log','binary')]


ADAPTERS = {NAME: common._Adapter(collect, facts, check_unit)}
