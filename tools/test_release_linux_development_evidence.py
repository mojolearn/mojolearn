"""Retained evidence tests only; no native build or product execution."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import development_build as dev
import merged_bindings as merged
import release_linux_development_evidence as adapter
import release_retained_compile_evidence as common


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return path


class Tree:
    root = "/source"
    files = {"a.mojo": "x"}

    def sha256(self, path):
        return hashlib.sha256(b"source").hexdigest() if path == "a.mojo" else None

    def text(self, path):
        if path == "tools/merged_bindings.py":
            return "\n".join(k + " = " + repr(getattr(merged, k))
                             for k in ("ENTRY", "ROOT_HEAD", "ROOT_MEMBER", "ROOT_TAIL"))
        if path.startswith("bindings/entry/"):
            return merged.ENTRY.format(name="_mojolearn_training")
        return "echo build\n"


@pytest.fixture
def producer(tmp_path, monkeypatch):
    tree = Tree()
    monkeypatch.setattr(common.Tree, 'at', lambda *args: tree)
    leg = tmp_path / 'leg'
    retained = leg / 'remote/leg_out'
    result_path = retained / 'phase/evidence/result.json'
    repo, cache = Path('/original/repo'), Path('/original/cache')
    stem = '_mojolearn_dev_training'
    env = dict(MOJOLEARN_COMPILE_JOBS='32', MOJOLEARN_NUMERIC_MODE='identical',
               MOJOLEARN_GPU_ARCHS='sm_89', MOJOLEARN_TARGET_COLUMN='nvidia',
               MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_PACKAGE_BYTE_LM='1')
    scripts = {p: hashlib.sha256(b'echo build\n').hexdigest()
               for p in ('bindings/merged_build.sh', 'bindings/build_defines.sh')}
    fields = dict(schema=common.DEV_KEY_SCHEMA, repo=str(repo), cache=str(cache), platform='linux',
                  stem=stem, mode='identical', members=['_mojolearn_training'], flags_class='default',
                  source=dict(scope='closure', unresolved=[], files={'a.mojo': tree.sha256('a.mojo')}),
                  toolchain=dict(available=True, compiler_sha256='c'*64,
                                 locked=dict(platform='linux-64', packages=['mojo']), installed=['mojo']),
                  environment=env, link_environment={}, builders=dev.compile_recipe(scripts, 'identical', stem, 'default'))
    fields["root_sha256"] = hashlib.sha256(merged.root_text(stem, fields["members"]).encode()).hexdigest()
    key = dev.digest(fields)
    command = dev.group_command('identical', stem, cache/'work'/key/'output', cache/'work'/key/'roots')
    execution = dict(command=command, complete=True, status='passed', exit_code=0,
                     process_returncode=0, cwd=str(repo))
    binary = put(retained/'development-cache/artifacts'/key/(stem+'.so'), 'actual image')
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    row = dict(fields=fields, key=key, stem=stem, mode='identical', members=fields['members'],
               selected=True, flags_class='default')
    result = dict(schema=common.DEV_SCHEMA, state='incomplete', source_commit='', missing=['othergroup'],
                  plan=dict(platform='linux', layout='development', repo=str(repo), cache=str(cache), groups=[row]),
                  results=[dict(key=key, exit_code=0, execution=execution, sha256=digest)])
    receipt = dict(fields=fields, key=key, stem=stem, mode='identical', exit_code=0, sha256=digest,
                   execution=execution, command=command, execution_receipt='/original/build.log.execution.json',
                   log='/original/build.log')
    receipt_path = put(binary.parent/'receipt.json', receipt)
    put(result_path, result)
    put(result_path.parent/'build.log.execution.json', execution)
    put(result_path.parent/'build.log', 'compiler completed\n')
    stage = put(retained/'native-stage/identical/_mojolearn_training.so', 'actual image')
    put(retained/'native-stage-manifest.json', [dict(source_key=key, group=stem,
         staged_artifact=str(stage.relative_to(retained)), sha256=digest)])
    put(leg/'leg.txt', 'commit='+'a'*40+'\ndirty_tracked=0\n')
    return dict(leg=str(leg), result=str(result_path)), binary, receipt_path, stage


def test_selected_success_from_incomplete_set(producer):
    p, _, _, _ = producer
    found = adapter.collect('/source', {}, p)
    assert list(found['units']) == ['compile-nvidia/linux-64/cuda/sm_89/identical/_mojolearn_training']
    group = found['facts']['groups']['_mojolearn_dev_training']
    assert group['compiler_workers'] == 32
    assert found['configuration']['dependency_context']['compiler_sha256']['linux-64'] == 'c'*64


@pytest.mark.parametrize('target', ['binary', 'stage', 'execution', 'log', 'receipt'])
def test_missing_retained_bytes_refused(producer, target):
    p, binary, receipt, stage = producer
    paths = dict(binary=binary, stage=stage, receipt=receipt, execution=Path(p['result']).parent/'build.log.execution.json',
                 log=Path(p['result']).parent/'build.log')
    paths[target].unlink()
    with pytest.raises((ValueError, OSError)):
        adapter.collect('/source', {}, p)


@pytest.mark.parametrize('target', ['binary', 'stage'])
def test_changed_binary_refused(producer, target):
    p, binary, _, stage = producer
    (binary if target == 'binary' else stage).write_text('different image')
    with pytest.raises(ValueError, match='binary differs'):
        adapter.collect('/source', {}, p)


def test_failed_execution_refused(producer):
    p, _, receipt_path, _ = producer
    receipt = json.loads(receipt_path.read_text())
    receipt['execution']['exit_code'] = 1
    put(receipt_path, receipt)
    with pytest.raises(ValueError, match='successful exact execution'):
        adapter.collect('/source', {}, p)


def test_changed_source_refused(producer, monkeypatch):
    p, _, _, _ = producer
    monkeypatch.setattr(Tree, 'sha256', lambda *args: 'f'*64)
    with pytest.raises(ValueError, match='compiled source bytes differ'):
        adapter.collect('/source', {}, p)


def test_dirty_original_refused(producer):
    p, _, _, _ = producer
    put(Path(p['leg'])/'leg.txt', 'commit='+'a'*40+'\ndirty_tracked=1\n')
    with pytest.raises(ValueError, match='source witness'):
        adapter.collect('/source', {}, p)


def wanted(found):
    g = found['facts']['groups']['_mojolearn_dev_training']
    b = dict(target='linux-64', vendor='cuda', arch='sm_89', tier='identical', name='_mojolearn_training',
             toolchain=g['toolchain_locked'], flags=dict(numeric_mode='identical', target_column='nvidia',
             gpu_archs='sm_89', linux_cpu='x86-64-v3', compile_jobs='32', package_byte_lm='1'))
    return dict(kind='compile', missing_inputs=[], coverage=[common.BASE_COVERAGE,common.binding_key(b)],
                inputs=dict(binding=b, unit=dict(kind='compile', vendor='nvidia', release_modes=['identical'],
                compile_variants=[], build_flags={}, compiler_sha256='c'*64)))


def test_exact_worker_count_and_semantic_refusals(producer, monkeypatch):
    p, _, _, _ = producer
    found = adapter.collect('/source', {}, p)
    w = wanted(found)
    monkeypatch.setattr(common, 'require_closure', lambda *args: ['a.mojo'])
    rid, unit = next(iter(found['units'].items()))
    adapter.check_unit(rid, w, found['facts'], unit)
    assert found['configuration']['dependency_context']['selected_binding_compile_jobs'][unit['binding_key']] == '32'
    for key, value in [('numeric_mode', 'fast'), ('gpu_archs', 'sm_80'), ('compile_jobs', '2')]:
        bad = copy.deepcopy(w);bad['inputs']['binding']['flags'][key] = value
        with pytest.raises(ValueError, match='compiled flag differs'):
            adapter.check_unit(rid, bad, found['facts'], unit)
    bad = copy.deepcopy(found['facts'])
    bad['groups'][unit['group']]['environment']['MOJOLEARN_MOJO_BUILD_FLAGS'] = '--optimization-level 0'
    with pytest.raises(ValueError, match='semantic build flags differ'):
        adapter.check_unit(rid, w, bad, unit)
    bad = copy.deepcopy(found['facts'])
    bad['groups'][unit['group']]['environment']['MOJOLEARN_CUDA_CODE_FORMAT'] = 'ptx'
    with pytest.raises(ValueError, match='code format differs'):
        adapter.check_unit(rid, w, bad, unit)


def test_changed_generated_entry_refused(producer, monkeypatch):
    p, _, _, _ = producer
    original = Tree.text
    monkeypatch.setattr(Tree, 'text', lambda self,path: original(self,path) + ('# changed\n' if path.startswith('bindings/entry/') else ''))
    with pytest.raises(ValueError, match='not the tested canonical wrapper'):
        adapter.collect('/source', {}, p)


def test_recorded_root_hash_refused(producer):
    p, _, _, _ = producer
    d = json.loads(Path(p['result']).read_text())
    d['plan']['groups'][0]['fields']['root_sha256'] = '0'*64
    d['plan']['groups'][0]['key'] = dev.digest(d['plan']['groups'][0]['fields'])
    put(Path(p['result']), d)
    with pytest.raises(ValueError, match='generated development root differs'):
        adapter.collect('/source', {}, p)


def test_wrong_compiler_refused(producer, monkeypatch):
    p, _, _, _ = producer
    found = adapter.collect('/source', {}, p)
    w = wanted(found)
    w['inputs']['unit']['compiler_sha256'] = 'f'*64
    rid, unit = next(iter(found['units'].items()))
    with pytest.raises(ValueError, match='compiler identity differs'):
        adapter.check_unit(rid, w, found['facts'], unit)


def test_native_sm80_does_not_qualify_ptx(producer):
    p, _, _, _ = producer
    found = adapter.collect('/source', {}, p)
    w = wanted(found)
    binding = w['inputs']['binding']
    binding['arch'] = binding['flags']['gpu_archs'] = 'sm_80'
    key = common.binding_key(binding)
    w['coverage'] = [common.BASE_COVERAGE, key]
    unit = dict(group='_mojolearn_dev_training', binding_key=key)
    found['facts']['groups'][unit['group']]['environment']['MOJOLEARN_GPU_ARCHS'] = 'sm_80'
    with pytest.raises(ValueError, match='native does not qualify PTX'):
        adapter.check_unit('compile-nvidia/'+key, w, found['facts'], unit)


@pytest.mark.parametrize('workers', ['0', '032', '-1', '٣٢', 32])
def test_noncanonical_worker_count_refused(producer, workers):
    p, _, _, _ = producer
    d = json.loads(Path(p['result']).read_text())
    row = d['plan']['groups'][0]
    row['fields']['environment']['MOJOLEARN_COMPILE_JOBS'] = workers
    row['key'] = dev.digest(row['fields'])
    put(Path(p['result']), d)
    with pytest.raises(ValueError, match='compiler worker count'):
        adapter.collect('/source', {}, p)


@pytest.mark.parametrize('argument', ['--gpu-arch=sm_80', '--emit=ptx'])
def test_injected_gpu_emission_command_is_not_an_environment_equivalence(producer, argument):
    p, _, receipt_path, _ = producer
    receipt = json.loads(receipt_path.read_text())
    receipt['command'].append(argument)
    receipt['execution']['command'] = list(receipt['command'])
    put(receipt_path, receipt)
    with pytest.raises(ValueError, match='successful exact execution'):
        adapter.collect('/source', {}, p)
