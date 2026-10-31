"""Retained producer admissions on synthetic receipts; nothing is built, installed or run."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import pytest

import development_build as devb
import release_adapter_evidence as adapters
import release_coverage_source_evidence as coverage
import release_ledger_admission as admission
import release_retained_compile_evidence as compiled
from release_requirements import RequirementsLedger, digest, producer_admission_available, requirements

VERSION = '0.9.1'
ADAPTER_COVERAGE = ['torch-inference', 'torch-training', 'max-train', 'max-train-resume']


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


CATALOG = {
    'schema': 'fixture',
    'source_inputs': {'public_runtime': ['python/mojolearn/*.py']},
    'source_exclude': [],
    'requirements': [
        {'id': 'adapters-nvidia', 'kind': 'adapters', 'stage': 'pre',
         'scope': 'Installed candidate whole-model adapters on nvidia', 'coverage': ADAPTER_COVERAGE,
         'source_inputs': {'adapter_harness': ['tools/adapter_*.py']},
         'configuration': ['release_modes', 'artifacts:nvidia', 'build_flags']},
        {'id': 'coverage-source', 'kind': 'source', 'stage': 'pre',
         'scope': 'Changed registrations reconciled; no missing required lanes or fault arms',
         'coverage': ['coverage mapping and exclusions reviewed'],
         'source_inputs': {'checker': ['tools/verify_coverage_audit.py', 'tools/identity_lanes/*']},
         'configuration': ['release_modes']},
    ]}

CHECKER = '''
def native_status_by_entry(rows):
    return {r["entry"]: (r["fixture_status"], []) for r in rows}
def triage(row):
    return row["fixture_status"], [], ""
def sabotage_of(row):
    return row["fixture_arm"], ""
def py_triage(row, by_entry):
    return row["fixture_status"], [], ""
'''


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    put(root / 'tools/release_requirements.json', CATALOG)
    put(root / 'tools/adapter_wheel_smoke.py', 'harness = 1\n')
    put(root / 'python/mojolearn/__init__.py', 'api = 1\n')
    put(root / 'tools/verify_coverage_audit.py', CHECKER)
    put(root / 'tools/mojo_registrations.py', '')
    put(root / 'tools/verify_reference_evidence.py', '')
    put(root / 'tools/identity_lanes/knn.py', 'lane = 1\n')
    put(root / 'bindings/native.mojo', 'fn native(): pass\n')
    put(root / 'bindings/build_native.sh', 'mojo build native\n')
    put(root / 'bindings/merged_build.sh', '# comment\nmojo build merged\n')
    put(root / 'bindings/build_defines.sh', 'defines\n')
    put(root / 'pixi.lock', 'lock\n')
    put(root / 'bench/results/release_verification/2026-10-20_pypi_090/release-provenance.json',
        {'schema': 'mojolearn.release-provenance.v1', 'version': '0.9.0', 'source_commit': '1' * 40})
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture source')
    return root


def ledger_for(tmp_path, wanted, configuration=None):
    ledger = RequirementsLedger(tmp_path / 'ledger.json')
    revision = next(iter(wanted.values()))['source_revision']
    ledger.sync(dict(schema='mojolearn.release-requirements.v1', version=VERSION, source_revision=revision,
                     requirements=wanted, source_root='/fixture', configuration=configuration or {}))
    return ledger


def manifest_rows(root, revision, ids, configuration):
    catalog = dict(CATALOG, requirements=[s for s in CATALOG['requirements'] if s['id'] in ids])
    return requirements(root, revision, VERSION, catalog=catalog, configuration=configuration,
                        expand_dependencies=False)['requirements']


# ---------------------------------------------------------------- adapters-<vendor>
def adapter_smoke(directory, vendor='nvidia', resume=True, device='cuda:0'):
    phases = {}
    results = {
        'torch-compile/adapter-results.json': ['torch_compile:mlp', 'output_word_negative_control'],
        'torch-training/training-results.json': ['mlp: ownership, finite outputs, repeatability',
                                                 'mlp: resumed 2+2 equals uninterrupted 4 (all five outputs)'],
        'max-training/max-training-results.json': ['mlp: compile, execute, repeat, input ownership, Torch exact']
        + (['mlp: resume exact'] if resume else [])}
    for relative, checks in results.items():
        put(directory / relative, dict(complete=True, checks=checks, failures=[]))
    for phase in ('torch-compile', 'torch-training', 'max-training'):
        put(directory / (phase + '.log'), 'argv: fixture\n')
        phases[phase] = dict(exit=0, seconds=1.0, log=f'/root/gate/adapter-smoke/{phase}.log', timed_out=False)
    put(directory / 'adapter-smoke.json', dict(status='PASSED', device=device, problems=[], phases=phases,
        frameworks=dict(torch='ran', max='ran'), installed={'mojolearn-core': VERSION, 'torch': '2.9.0'}))
    return phases


@pytest.fixture
def gate(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    directory = tmp_path / 'gate-nvidia/stamp-release-light-v1'
    wheels = {f'mojolearn_core-{VERSION}-py3-none-manylinux.whl': 'a' * 64}
    identity = dict(source='candidate', vendor='nvidia', version=VERSION, source_commit=revision,
                    release_modes=['identical'], wheels=wheels)
    phases = adapter_smoke(directory / 'adapter-smoke')
    put(directory / 'gate-smoke.json', dict(
        vendor='nvidia', version=VERSION, input_mode='candidate', candidate=identity, adapters_enabled=True,
        adapters_finished=1.0, adapters=dict(enabled=True, ok=True, status='PASSED', rc=0,
            phases={p: 0 for p in phases}, phase_records=phases,
            report='/root/gate/adapter-smoke/adapter-smoke.json')))
    configuration = {'release_modes': ['identical'], 'build_flags': {}, 'artifacts:nvidia': wheels}
    wanted = manifest_rows(repo, revision, {'adapters-nvidia'}, configuration)
    return dict(directory=directory, revision=revision, wanted=wanted, configuration=configuration)


def test_retained_adapter_smoke_is_admitted_and_reused(tmp_path, repo, gate):
    ledger = ledger_for(tmp_path, gate['wanted'], gate['configuration'])
    assert producer_admission_available(gate['wanted']['adapters-nvidia'])
    assert ledger.plan()['adapters-nvidia']['action'] == 'run'
    review = admission.prepare('gate-adapters', repo, ledger.path, tmp_path / 'review',
                               dict(gate=gate['directory']))
    assert review['admitted'] == ['adapters-nvidia'] and review['comparison']['adapters-nvidia']['reusable']
    receipt = json.loads((tmp_path / 'review/receipts.json').read_text())['adapters-nvidia']
    assert {row['id']: row['expected'] for row in receipt['coverage']} == {
        'torch-inference': 2, 'torch-training': 2, 'max-train': 1, 'max-train-resume': 1}
    result = admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    assert result['admitted'] == ['adapters-nvidia'] and result['actions'] == {'adapters-nvidia': 'reuse'}
    attempt = ledger.read()['attempts'][-1]
    assert attempt['run_id'] is None and attempt['origin'] == admission.ORIGIN
    assert attempt['source_revision'] == gate['revision']
    assert ledger.gate(stage='pre') == (True, [])
    # Admitting the same reviewed report again appends nothing.
    again = admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    assert again['admitted'] == [] and again['already_admitted'] == ['adapters-nvidia']


def test_missing_max_resume_record_is_refused(tmp_path, repo, gate):
    adapter_smoke(gate['directory'] / 'adapter-smoke', resume=False)
    ledger = ledger_for(tmp_path, gate['wanted'], gate['configuration'])
    with pytest.raises(ValueError, match='gate adapter record differs|no actual check record'):
        admission.prepare('gate-adapters', repo, ledger.path, tmp_path / 'review', dict(gate=gate['directory']))
    assert not ledger.read()['attempts']


def test_failed_adapter_phase_is_refused(tmp_path, repo, gate):
    smoke = json.loads((gate['directory'] / 'adapter-smoke/adapter-smoke.json').read_text())
    smoke['phases']['max-training']['exit'] = 1
    put(gate['directory'] / 'adapter-smoke/adapter-smoke.json', smoke)
    ledger = ledger_for(tmp_path, gate['wanted'], gate['configuration'])
    with pytest.raises(ValueError, match='adapter smoke did not pass'):
        admission.prepare('gate-adapters', repo, ledger.path, tmp_path / 'review', dict(gate=gate['directory']))


def test_other_wheels_are_not_reused_and_tampering_blocks_reuse(tmp_path, repo, gate):
    ledger = ledger_for(tmp_path, gate['wanted'], gate['configuration'])
    admission.prepare('gate-adapters', repo, ledger.path, tmp_path / 'review', dict(gate=gate['directory']))
    admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    # New candidate bytes are a changed input: the old smoke is not relabelled.
    changed = dict(gate['configuration'], **{'artifacts:nvidia': {'other.whl': 'b' * 64}})
    rows = manifest_rows(repo, gate['revision'], {'adapters-nvidia'}, changed)
    row = ledger.plan(dict(requirements=rows, configuration=changed))['adapters-nvidia']
    assert row['status'] != 'passed' and any('artifacts:nvidia' in r for r in row['reasons'])
    # Retained checker bytes changed after admission: never a pass.
    put(gate['directory'] / 'adapter-smoke/max-training/max-training-results.json',
        dict(complete=True, checks=['mlp: resume exact', 'x'], failures=[]))
    admission._VERIFIED.clear()
    row = ledger.plan()['adapters-nvidia']
    assert row['status'] != 'passed' and row['action'] in ('restore-evidence', 'complete-provenance')


# ---------------------------------------------------------------- coverage-source
def audit(path, revision, rows, scope='all', base=None):
    native = [dict(entry=f'e{i}', src='bindings/native.mojo', fixture_status=s, fixture_arm=a)
              for i, (s, a) in enumerate(rows)]
    python = [dict(public='KNN', op='fit', fixture_status='covered (via native)')]
    return put(path, dict(head=revision, scope=scope, base=base, native=native, python=python))


def test_coverage_audit_is_classified_by_the_tested_checker(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    configuration = {'release_modes': ['identical']}
    wanted = manifest_rows(repo, revision, {'coverage-source'}, configuration)
    ledger = ledger_for(tmp_path, wanted, configuration)
    assert producer_admission_available(wanted['coverage-source'])
    path = audit(tmp_path / 'audit.json', revision, [('covered', 'armed'), ('n/a', 'n/a')])
    review = admission.prepare('coverage-source', repo, ledger.path, tmp_path / 'review', dict(audit=path))
    assert review['admitted'] == ['coverage-source']
    receipt = json.loads((tmp_path / 'review/receipts.json').read_text())['coverage-source']
    assert receipt['coverage'][0]['expected'] == 3 and receipt['producer']['report_type'] == 'source'
    assert admission.admit(ledger.path, repo, tmp_path / 'review/report.json')['actions'] == {'coverage-source': 'reuse'}


def test_coverage_gaps_owed_arms_and_symbolic_bases_are_refused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    configuration = {'release_modes': ['identical']}
    ledger = ledger_for(tmp_path, manifest_rows(repo, revision, {'coverage-source'}, configuration), configuration)
    path = audit(tmp_path / 'audit.json', revision, [('covered', 'owed'), ('GAP', 'armed')])
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('coverage-source', repo, ledger.path, tmp_path / 'gaps', dict(audit=path))
    review = json.loads((tmp_path / 'gaps/review.json').read_text())
    assert '2 audited rows' in review['refused']['coverage-source']
    path = audit(tmp_path / 'changed.json', revision, [('covered', 'armed')], scope='changed', base='origin/main')
    with pytest.raises(ValueError):
        admission.prepare('coverage-source', repo, ledger.path, tmp_path / 'symbolic', dict(audit=path))
    assert not ledger.read()['attempts']


def test_changed_audit_needs_the_last_published_release_base(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    assert coverage.published_base(repo, revision, VERSION) == ('0.9.0', '1' * 40)
    with pytest.raises(ValueError, match='no published release'):
        coverage.published_base(repo, revision, '0.8.0')


# ---------------------------------------------------------------- compile (Mac development build)
def closure_digest(root, rels):
    value = hashlib.sha256()
    for rel in sorted(rels, key=lambda name: PurePosixPath(name).parts):
        value.update(rel.encode() + b'\0' + sha(root / rel).encode() + b'\n')
    return value.hexdigest()


MAC_CPU = {'dependency_context': {'cpu_target': 'osx-arm64'}}
TOOLCHAIN = dict(platform='osx-arm64', packages=['mojo-1.0.0-release'], pixi_lock_sha256='c' * 64)
APPLE = dict(xcode='Xcode 26.6', sdk='26.5')


@pytest.fixture
def development(tmp_path, repo, monkeypatch):
    revision = git(repo, 'rev-parse', 'HEAD')
    build = tmp_path / 'mac-build'
    cache = tmp_path / 'dev-cache'
    scripts = {p: hashlib.sha256(devb.recipe_text((repo / p).read_text()).encode()).hexdigest()
               for p in ('bindings/merged_build.sh', 'bindings/build_defines.sh')}
    stem = '_mojolearn_dev_native_host'
    fields = dict(schema=compiled.DEV_KEY_SCHEMA, repo=str(build), cache=str(cache), mode='host',
                  platform='darwin', stem=stem, flags_class='host', members=['_mojolearn_native_host'],
                  source=dict(scope='closure', files={'bindings/native.mojo': sha(repo / 'bindings/native.mojo')},
                              unresolved=[]),
                  builders=dict(schema='mojolearn.development-recipe.v1', scripts=scripts, mode='host'),
                  root_sha256='d' * 64,
                  toolchain=dict(available=True, compiler_sha256='e' * 64, locked=TOOLCHAIN, apple=APPLE),
                  environment=dict(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_TARGET_COLUMN='cpu',
                                   MOJOLEARN_COMPILE_JOBS='1', MOJOLEARN_PACKAGE_BYTE_LM='1',
                                   MODULAR_HOME=str(build / '.pixi/envs/default/share/max'),
                                   MOJOLEARN_RUNPOD_KEY_FILE='/secret'))
    key = devb.digest(fields)
    binary = put(build / 'python/mojolearn/host/.development' / key / (stem + '.so'), 'compiled bytes\n')
    put(cache / 'artifacts' / key / 'receipt.json', dict(key=key, fields=fields, exit_code=0, sha256=sha(binary)))
    result = put(tmp_path / 'build-evidence/result.json', dict(
        schema=compiled.DEV_SCHEMA, state='activated', missing=[], activated=['host'], source_commit=revision,
        results=[dict(stem=stem, state='built', exit_code=0)],
        plan=dict(platform='darwin', layout='development', repo=str(build), cache=str(cache),
                  groups=[dict(mode='host', stem=stem, key=key, members=fields['members'], fields=fields)])))

    def identity(flags=None, closure=None):
        rels = ['bindings/build_native.sh', 'bindings/native.mojo', 'pixi.lock']
        return dict(schema='mojolearn.binding-identity.v2', target='osx-arm64', vendor='host', arch='', tier='host',
                    name='_mojolearn_native_host', script='build_native.sh',
                    closure=dict(scope='closure', digest=closure or closure_digest(repo, rels)),
                    generators=None, toolchain=TOOLCHAIN, host_toolchain=APPLE,
                    flags=flags or dict(numeric_mode='identical', target_column='cpu', gpu_archs=None,
                                        compile_jobs='1', package_byte_lm='1'),
                    builders={'packaging/macos/build_release_wheel.sh': 'f' * 64},
                    builder_rule='builder-view.v1'), rels

    def row(spec, **changes):
        binding, rels = identity(**changes)
        key = compiled.binding_key(binding)
        coverage = [compiled.BASE_COVERAGE, key]
        vendor = spec.split('-')[1]
        inputs = dict(unit=dict(schema='mojolearn.release-dependencies.v1', kind='compile', vendor=vendor,
                                release_modes=['identical'], build_flags={}, compile_variants=[]),
                      binding=binding, configuration={},
                      contract=dict(id=spec + '/' + key, scope='Integrated identical build for ' + vendor,
                                    coverage=coverage, kind='compile', stage='pre'))
        return spec + '/' + key, dict(source_revision=revision, inputs=inputs, fingerprint=digest(inputs),
            scope=inputs['contract']['scope'], coverage=coverage, missing_inputs=[], kind='compile', stage='pre',
            provenance_sources=sorted(rels + list(binding['builders'])), required=True, parent_requirement=spec)

    originals = dict([row('compile-apple'), row('compile-cpu')])
    monkeypatch.setattr(admission, 'reconstruct', lambda *a, **k: copy.deepcopy(originals))
    return dict(result=result, revision=revision, originals=originals, row=row, binary=binary)


def test_development_host_build_admits_apple_and_cpu_host_bindings(tmp_path, repo, development):
    wanted = development['originals']
    assert all(producer_admission_available(w) for w in wanted.values())
    ledger = ledger_for(tmp_path, copy.deepcopy(wanted), MAC_CPU)
    review = admission.prepare('mac-development-build', repo, ledger.path, tmp_path / 'review',
                               dict(result=development['result']))
    assert review['admitted'] == sorted(wanted) and not review['refused']
    assert review['reconstruction_configuration']['build_flags'] == {}
    result = admission.admit(ledger.path, repo, tmp_path / 'review/report.json')
    assert set(result['actions'].values()) == {'reuse'}
    plan = ledger.plan()
    assert all(plan[name]['status'] == 'passed' for name in wanted)


def test_development_build_never_qualifies_a_linux_cpu_column(tmp_path, repo, development):
    linux = {'dependency_context': {'cpu_target': 'linux-64'}}
    ledger = ledger_for(tmp_path, copy.deepcopy(development['originals']), linux)
    review = admission.prepare('mac-development-build', repo, ledger.path, tmp_path / 'review',
                               dict(result=development['result']))
    assert review['admitted'] == [name for name in sorted(development['originals']) if name.startswith('compile-apple/')]


def test_development_build_refuses_flag_and_closure_mismatches(tmp_path, repo, development, monkeypatch):
    bad = dict([development['row']('compile-apple', flags=dict(numeric_mode='identical', target_column='cpu',
        gpu_archs=None, compile_jobs='2', package_byte_lm='1'))])
    bad.update([development['row']('compile-cpu', closure='0' * 64)])
    monkeypatch.setattr(admission, 'reconstruct', lambda *a, **k: copy.deepcopy(bad))
    ledger = ledger_for(tmp_path, copy.deepcopy(bad), MAC_CPU)
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('mac-development-build', repo, ledger.path, tmp_path / 'review',
                          dict(result=development['result']))
    refused = json.loads((tmp_path / 'review/review.json').read_text())['refused']
    assert any('MOJOLEARN_COMPILE_JOBS' in reason for reason in refused.values())
    assert any('closure digest' in reason for reason in refused.values())


def test_development_build_with_changed_binary_or_source_is_refused(tmp_path, repo, development):
    ledger = ledger_for(tmp_path, copy.deepcopy(development['originals']))
    development['binary'].write_text('other bytes\n')
    with pytest.raises(ValueError, match='binary differs'):
        admission.prepare('mac-development-build', repo, ledger.path, tmp_path / 'a', dict(result=development['result']))
    development['binary'].write_text('compiled bytes\n')
    put(repo / 'bindings/native.mojo', 'fn native(): changed\n')
    git(repo, 'commit', '-qam', 'later edit')
    result = json.loads(development['result'].read_text())
    result['source_commit'] = git(repo, 'rev-parse', 'HEAD')
    put(development['result'], result)
    with pytest.raises(ValueError, match='source bytes differ'):
        admission.prepare('mac-development-build', repo, ledger.path, tmp_path / 'b', dict(result=development['result']))


# ---------------------------------------------------------------- compile (device run, Linux host): unit checks
def test_retained_device_and_linux_host_units_require_exact_identities():
    binding = dict(target='osx-arm64', vendor='metal', arch='apple', tier='identical', name='_mojolearn_gbdt')
    key = compiled.binding_key(binding)
    inputs = dict(unit=dict(kind='compile', vendor='apple', release_modes=['identical'], build_flags={},
                            compile_variants=[]), binding=binding, configuration={})
    wanted = dict(kind='compile', inputs=inputs, coverage=[compiled.BASE_COVERAGE, key], missing_inputs=[])
    group = dict(members=['_mojolearn_gbdt'], receipt={}, stamp={}, index={}, artifact={})
    facts = dict(binding_identities={key: binding}, build_environment={'MOJOLEARN_NUMERIC_MODE': 'identical'},
                 groups={'_mojolearn_native': group}, before_build={}, summary={})
    rows, _ = compiled._device_check('compile-apple/' + key, wanted, facts, dict(binding_key=key))
    assert [r['id'] for r in rows] == wanted['coverage']
    with pytest.raises(ValueError, match='identity differs'):
        compiled._device_check('compile-apple/' + key, wanted,
                               dict(facts, binding_identities={key: dict(binding, name='x')}), dict(binding_key=key))
    with pytest.raises(ValueError, match='build flags'):
        compiled._device_check('compile-apple/' + key, copy.deepcopy(wanted) | {'inputs': dict(inputs,
            unit=dict(inputs['unit'], build_flags={'MOJOLEARN_EXTRA': '1'}))}, facts, dict(binding_key=key))

    import release_host_build as rhb
    host = dict(target='linux-64', vendor='host', arch='', tier='host', name='_mojolearn_core_host')
    key = compiled.binding_key(host)
    unit = dict(kind='compile', vendor='nvidia', release_modes=['identical'], build_flags={},
                compile_variants=[], compiler_sha256='a' * 64)
    wanted = dict(kind='compile', inputs=dict(unit=unit, binding=host, configuration={}),
                  coverage=[compiled.BASE_COVERAGE, key], missing_inputs=[])
    facts = dict(binding_identities={key: host}, compiler_sha256='a' * 64, build_environment=dict(rhb.FLAGS),
                 members=['_mojolearn_core_host'], evidence=[])
    assert compiled.supports_requirement(wanted)
    compiled._host_check('compile-nvidia/' + key, wanted, facts, dict(binding_key=key))
    origin = dict(action='retain', source_commit='b'*40, original_proof_sha256='c'*64)
    retained = dict(facts, group_origins={'host.so':origin}, member_groups={host['name']:'host.so'})
    compiled._host_check('compile-nvidia/' + key, wanted, retained, dict(binding_key=key, compilation_origin=origin))
    with pytest.raises(ValueError, match='compilation origin'):
        compiled._host_check('compile-nvidia/' + key, wanted, retained, dict(binding_key=key))
    with pytest.raises(ValueError, match='compiler executable'):
        compiled._host_check('compile-nvidia/' + key, wanted, dict(facts, compiler_sha256='b' * 64),
                             dict(binding_key=key))


def test_runtime_and_published_rows_with_partial_coverage_have_no_adapter():
    # Full contracts are admitted by release_installed_evidence (test_release_installed_evidence.py).
    for kind, name, coverage_ids in (('runtime', 'runtime-apple', ['constructors']),
                                     ('published', 'published-apple', ['index-install'])):
        wanted = dict(kind=kind, coverage=coverage_ids, missing_inputs=[],
                      inputs=dict(contract=dict(id=name)))
        assert not admission.supports_requirement(wanted)
    assert adapters.supports_requirement(dict(kind='adapters', coverage=ADAPTER_COVERAGE, missing_inputs=[],
                                              inputs=dict(contract=dict(id='adapters-apple'))))


def test_mac_development_glue_interpreter_must_be_absolute(tmp_path, repo, development):
    result = json.loads(development['result'].read_text())
    row = result['plan']['groups'][0]
    row['fields']['environment']['MOJOLEARN_MERGED_PYTHON'] = 'relative/python'
    row['key'] = devb.digest(row['fields'])
    put(development['result'], result)
    ledger = ledger_for(tmp_path, copy.deepcopy(development['originals']), MAC_CPU)
    with pytest.raises(ValueError, match='glue interpreter is not recorded as an absolute path'):
        admission.prepare('mac-development-build', repo, ledger.path, tmp_path/'invalid-glue',
                          dict(result=development['result']))
