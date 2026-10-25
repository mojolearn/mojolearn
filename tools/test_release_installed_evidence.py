"""Installed runtime, multi-GPU and published-install producers on synthetic receipts.

Nothing is installed, compiled or run on a GPU: the producer report functions
(summarize/judge) and the ledger adapters run over fixture directories.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import qualify_native_runtime_installed as q
import release_installed_evidence as installed
import release_ledger_admission as admission
import release_multi_gpu_parallel as multi
import release_published_install as published
import release_runtime_installed as runtime
from release_requirements import RequirementsLedger, producer_admission_available, requirements

VERSION = '0.9.1'
TOOLS = Path(__file__).resolve().parent
PCI_A, PCI_B, PCI_C = '0:3b:00.0', '0:5e:00.0', '0:86:00.0'
CATALOG = {
    'schema': 'fixture', 'source_inputs': {'public_runtime': ['python/mojolearn/*.py']}, 'source_exclude': [],
    'requirements': [
        {'id': 'runtime-nvidia', 'kind': 'runtime', 'stage': 'pre',
         'scope': 'Public operations and transitive callback refusals on nvidia', 'coverage': list(runtime.COVERAGE_IDS),
         'source_inputs': {'tests': ['tools/release_runtime_installed.py']},
         'configuration': ['release_modes', 'build_flags']},
        {'id': 'multi-gpu-nvidia', 'kind': 'multi', 'stage': 'pre', 'scope': 'Two physical GPUs for nvidia',
         'coverage': list(multi.COVERAGE_IDS), 'source_inputs': {'tests': ['tools/*parallel*.py']},
         'configuration': ['release_modes', 'build_flags']},
        {'id': 'published-nvidia', 'kind': 'published', 'stage': 'post', 'scope': 'Installed published wheels on nvidia',
         'coverage': list(published.COVERAGE_IDS),
         'source_inputs': {'published_harness': ['tools/release_published_install.py']},
         'configuration': ['release_modes', 'artifacts:nvidia', 'build_flags']},
    ]}


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, indent=2) + '\n')
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'config', 'user.name', 'Fixture')
    put(root / 'tools/release_requirements.json', CATALOG)
    put(root / 'tools/release_runtime_installed.py', 'producer = 1\n')
    put(root / 'tools/release_multi_gpu_parallel.py', 'producer = 1\n')
    put(root / 'tools/release_published_install.py', 'producer = 1\n')
    put(root / 'python/mojolearn/__init__.py', 'api = 1\n')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture source')
    return root


def wheel(tmp_path, name='mojolearn_core-%s-py3-none-manylinux_2_28_x86_64.whl' % VERSION, body=b'core'):
    return put(tmp_path / 'wheels' / name, body.decode())


def ledger_for(tmp_path, root, revision, ids, configuration):
    catalog = dict(CATALOG, requirements=[s for s in CATALOG['requirements'] if s['id'] in ids])
    rows = requirements(root, revision, VERSION, catalog=catalog, configuration=configuration,
                        expand_dependencies=False)['requirements']
    ledger = RequirementsLedger(tmp_path / 'ledger.json')
    ledger.sync(dict(schema='mojolearn.release-requirements.v1', version=VERSION, source_revision=revision,
                     requirements=rows, source_root=str(root), configuration=configuration))
    return ledger, rows


def configuration_for(artifacts):
    return {'release_modes': ['identical'], 'build_flags': {}, 'artifacts:nvidia': artifacts}


def admit(tmp_path, root, ledger, adapter, report, review='review'):
    result = admission.prepare(adapter, root, ledger.path, tmp_path / review, dict(report=report))
    admitted = admission.admit(ledger.path, root, tmp_path / review / 'report.json')
    return result, admitted


# ---------------------------------------------------------------- runtime-<vendor>
def runtime_dir(tmp_path, revision, core, *, failing=None, skipped=None, vendor='nvidia', drop_file=None):
    out = tmp_path / ('runtime-' + vendor)
    document = json.loads(runtime.COVERAGE_PATH.read_text())
    put(out / 'coverage.json', document)
    mapping = document['coverage']
    selected = [name for names in mapping.values() for name in names]
    cases = []
    for name in selected:
        if name == drop_file:
            continue
        stem = Path(name).stem
        body = ''
        if name == failing:
            body = '<failure message="boom">boom</failure>'
        elif name == skipped:
            body = '<skipped message="no device"/>'
        cases.append(f'<testcase classname="mojolearn.tests.{stem}" name="test_case">{body}</testcase>')
    put(out / 'qualify/tests.xml', '<testsuites><testsuite>' + ''.join(cases) + '</testsuite></testsuites>')
    tests = q.junit_summary(out / 'qualify/tests.xml')
    backend = runtime.BACKEND[vendor]
    scope = q.test_scope_summary(tests, selected, backend)
    status = q.selected_test_status(0 if not (failing or skipped) else 1, tests, scope, True)
    put(out / 'qualify/receipt.json', dict(
        vendor=backend, numeric_mode='identical', source_commit=revision, status=status,
        pytest_exit=0 if status == 'selected-tests-passed' else 1, test_selection_kind='whole-files',
        selected_public_scope={name: q.PUBLIC_TESTS[name] for name in selected}, tests=tests,
        test_scope_summary=scope, wheels=[dict(path=str(core), sha256=sha(core), bytes=4)]))
    put(out / 'qualify/installed-modules.json', dict(vendor=backend, numeric_mode='identical',
                                                     inventory_errors=[], worker_error=None))
    put(out / 'qualify/tests.log', 'fixture\n')
    put(out / 'request.json', dict(vendor=vendor, source_commit=revision, wheels={core.name: sha(core)}))
    report = runtime.summarize(out)
    report['qualifier_exit'] = 0 if report['status'] == 'passed' else 1
    put(out / runtime.REPORT, report)
    return out


def test_coverage_map_assigns_every_registered_public_test_once():
    mapping = runtime.coverage_map(json.loads(runtime.COVERAGE_PATH.read_text()))
    files = [name for names in mapping.values() for name in names]
    assert list(mapping) == list(runtime.COVERAGE_IDS)
    assert sorted(files) == sorted(q.PUBLIC_TESTS) and len(files) == len(set(files))
    assert all(mapping[name] for name in runtime.COVERAGE_IDS)
    for name in files:
        assert (TOOLS.parent / 'python/mojolearn/tests' / name).is_file(), name
    broken = json.loads(runtime.COVERAGE_PATH.read_text())
    broken['coverage']['constructors'] = broken['coverage']['constructors'][1:]
    with pytest.raises(ValueError, match='differs from the public test registry'):
        runtime.coverage_map(broken)


def test_installed_runtime_is_admitted_and_reused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    out = runtime_dir(tmp_path, revision, core)
    ledger, rows = ledger_for(tmp_path, repo, revision, {'runtime-nvidia'}, configuration_for({core.name: sha(core)}))
    assert producer_admission_available(rows['runtime-nvidia'])
    review, result = admit(tmp_path, repo, ledger, 'installed-runtime', out / runtime.REPORT)
    assert review['admitted'] == ['runtime-nvidia']
    assert result['actions'] == {'runtime-nvidia': 'reuse'}
    receipt = json.loads((tmp_path / 'review/receipts.json').read_text())['runtime-nvidia']
    mapping = json.loads(runtime.COVERAGE_PATH.read_text())['coverage']
    assert {row['id']: row['expected'] for row in receipt['coverage']} == {k: len(v) for k, v in mapping.items()}
    assert receipt['producer']['tool'] == 'tools/release_installed_evidence.py'
    assert ledger.gate(stage='pre') == (True, [])


def test_runtime_failure_skip_or_missing_case_is_refused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    configuration = configuration_for({core.name: sha(core)})
    for index, change in enumerate((dict(failing='test_meta_native.py'), dict(skipped='test_cache_get_native.py'),
                                    dict(drop_file='test_native_cv_outputs.py'))):
        out = runtime_dir(tmp_path / str(index), revision, core, **change)
        assert json.loads((out / runtime.REPORT).read_text())['status'] == 'failed'
        ledger, _ = ledger_for(tmp_path / str(index), repo, revision, {'runtime-nvidia'}, configuration)
        with pytest.raises(ValueError, match='did not pass'):
            admission.prepare('installed-runtime', repo, ledger.path, tmp_path / str(index) / 'review',
                              dict(report=out / runtime.REPORT))
        assert not ledger.read()['attempts']


def test_runtime_on_unqualified_wheels_or_tampered_junit_is_refused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    out = runtime_dir(tmp_path, revision, core)
    ledger, _ = ledger_for(tmp_path, repo, revision, {'runtime-nvidia'}, configuration_for({core.name: 'b' * 64}))
    with pytest.raises(ValueError, match='not the ledger'):
        admission.prepare('installed-runtime', repo, ledger.path, tmp_path / 'review', dict(report=out / runtime.REPORT))
    xml = (out / 'qualify/tests.xml').read_text()
    put(out / 'qualify/tests.xml', xml.replace('<testcase ', '<testcase ', 1).replace('</testsuite>', '<testcase classname="mojolearn.tests.test_meta_native" name="extra"/></testsuite>'))
    ledger, _ = ledger_for(tmp_path / 'b', repo, revision, {'runtime-nvidia'}, configuration_for({core.name: sha(core)}))
    with pytest.raises(ValueError, match='differs'):
        admission.prepare('installed-runtime', repo, ledger.path, tmp_path / 'b/review', dict(report=out / runtime.REPORT))


# ---------------------------------------------------------------- multi-gpu-<vendor>
def multi_dir(tmp_path, revision, core, *, stray=False, callback_ok=True, single_gpu=False, unsighted=False):
    out = tmp_path / 'multi'
    inventory = [dict(index=0, uuid='GPU-a', pci_bus_id=PCI_A, name='L40S'),
                 dict(index=1, uuid='GPU-b', pci_bus_id=PCI_B, name='L40S'),
                 dict(index=2, uuid='GPU-c', pci_bus_id=PCI_C, name='L40S')]
    if single_gpu:
        inventory = inventory[:1]
    put(out / 'driver-inventory.json', dict(command=['nvidia-smi'], raw='', parsed=inventory))
    gpu = {0: PCI_A, 1: PCI_B}
    runs, pid = [], 100
    for model in ('regressor', 'classifier'):
        for order in ([0], [0, 1], [1, 0]):
            pids = []
            for d in order:
                pid += 1
                pids.append((pid, gpu[d]))
            folds = {}
            for i in range(5):
                worker, bus = pids[i % len(pids)]
                fixture = hashlib.sha256(f'{model}{i}'.encode()).hexdigest()
                folds[fixture] = dict(fixture=fixture, inventory=dict(pid=worker, devices=[dict(pci_bus_id='0000:' + bus[2:])]))
            runs.append(dict(model=model, devices=order, repeat=0, fold_records=folds))
    controls = [m + ':' + f for m in ('classifier', 'regressor') for f in ('dropped-fold', 'changed-model',
                                                                           'changed-prediction', 'aliased-device')]
    put(out / 'capture/report.json', dict(status='NUMERICS_AND_PLACEMENT_PASS', vendor='cuda', devices=[0, 1],
        runs=runs, controls=controls, physical_execution_trace='OWED', native_fault_controls='OWED',
        distribution=dict(import_matches_distribution=True, editable=False)))
    names = ['parallel-equals-serial', 'assignments', 'recovery-equals-serial', 'native-fold-fault',
             'wrong-device:repeated', 'wrong-device:nonexistent'] + ['callback:' + k for k in multi.CALLBACK_KINDS]
    checks = [dict(name=n, passed=callback_ok or not n.startswith('callback:')) for n in names]
    put(out / 'probe.json', dict(mode='probe', vendor='cuda', numeric_mode='identical', checks=checks, workers=[201, 202],
                                 assignments=[[i, i % 2, 201 + i % 2] for i in range(5)]))
    put(out / 'hidden.json', dict(mode='hidden', checks=[dict(name='hidden-device', passed=True)], workers=[]))
    samples = [dict(t=1.0, rows=[dict(pid=102, pci_bus_id=PCI_A)] + ([] if unsighted else [dict(pid=103, pci_bus_id=PCI_B)])),
               dict(t=1.1, rows=[dict(pid=999, pci_bus_id=PCI_C)] + ([dict(pid=201, pci_bus_id=PCI_C)] if stray else []))]
    put(out / 'driver-processes.jsonl', ''.join(json.dumps(s) + '\n' for s in samples))
    put(out / 'install.log', 'fixture\n')
    put(out / 'request.json', dict(vendor='nvidia', devices=[0, 1], wheels={core.name: sha(core)},
                                   source_commit=revision, capture_exit=0, probe_exit=0, hidden_exit=0))
    if single_gpu:
        with pytest.raises(ValueError, match='fewer than two GPUs'):
            multi.judge(out)
        return out
    put(out / multi.REPORT, multi.judge(out))
    return out


def test_multi_gpu_trace_and_controls_are_admitted(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    out = multi_dir(tmp_path, revision, core)
    report = json.loads((out / multi.REPORT).read_text())
    assert report['status'] == 'passed', report['coverage']
    counts = {row['id']: row['expected'] for row in report['coverage']}
    # 30 fold records + 6 schedules + 5 probe folds + assignments + 2 driver sightings
    assert counts == {'native device placement': 44, 'consumed outputs': 8,
                      'wrong-device and callback negative controls': 17}
    assert report['physical_execution_trace']['sighted'] == {PCI_A: [102], PCI_B: [103]}
    assert report['native_fault_controls']['fold_fault'] is True
    ledger, rows = ledger_for(tmp_path, repo, revision, {'multi-gpu-nvidia'}, configuration_for({core.name: sha(core)}))
    assert producer_admission_available(rows['multi-gpu-nvidia'])
    review, result = admit(tmp_path, repo, ledger, 'multi-gpu', out / multi.REPORT)
    assert result['actions'] == {'multi-gpu-nvidia': 'reuse'}
    assert ledger.gate(stage='pre') == (True, [])


@pytest.mark.parametrize('fault', ['stray', 'callback', 'unsighted'])
def test_multi_gpu_faults_fail_the_report(tmp_path, repo, fault):
    revision = git(repo, 'rev-parse', 'HEAD')
    out = multi_dir(tmp_path, revision, wheel(tmp_path), stray=fault == 'stray', callback_ok=fault != 'callback',
                    unsighted=fault == 'unsighted')
    report = json.loads((out / multi.REPORT).read_text())
    assert report['status'] == 'failed'
    failing = {row['id'] for row in report['coverage'] if row['status'] != 'passed'}
    assert failing == ({'wrong-device and callback negative controls'} if fault == 'callback'
                       else {'native device placement'})


def test_multi_gpu_needs_two_gpus_and_parses_driver_tables(tmp_path, repo):
    multi_dir(tmp_path, git(repo, 'rev-parse', 'HEAD'), wheel(tmp_path), single_gpu=True)
    assert multi.normal_pci('00000000:3B:00.0') == multi.normal_pci('0000:3b:00.0') == '0:3b:00.0'
    assert multi._parse_nvidia_processes('4242, 00000000:3B:00.0\n') == [dict(pid=4242, pci_bus_id=PCI_A)]
    assert multi._parse_nvidia_inventory('0, GPU-a, 00000000:3B:00.0, NVIDIA L40S\n')[0]['pci_bus_id'] == PCI_A
    inventory = [dict(index=0, pci_bus_id=PCI_A), dict(index=1, pci_bus_id=PCI_B)]
    text = 'PID 77 is using 1 DRM device(s):\n1 \n'
    assert multi._parse_amd_processes(text, inventory) == [dict(pid=77, pci_bus_id=PCI_B)]


# ---------------------------------------------------------------- published-<vendor>
def published_dir(tmp_path, revision, core, *, candidate_sha=None, version=VERSION, job_exit=0):
    out = tmp_path / 'published'
    downloaded = put(out / 'wheels' / core.name, core.read_text())
    digest = sha(downloaded)
    site = '/venv/lib/python3.12/site-packages'
    put(out / 'wheel-manifest.json', dict(
        schema='mojolearn-pypi-wheel-manifest/1', index_url=published.PYPI_INDEX, version=version, vendor='nvidia',
        all_digests_match=True,
        files=[dict(distribution='mojolearn-core', file=core.name, path=str(downloaded), sha256=digest,
                    pypi_sha256=digest, pypi_digest_match=True)],
        loaded={'mojolearn-core': dict(version=version, location=site,
                                       direct_url=dict(url='file:///w/' + core.name,
                                                       archive_info=dict(hashes=dict(sha256=digest)))),
                'import mojolearn': dict(module_file=site + '/mojolearn/__init__.py')}))
    put(out / 'installed-bytes.json', dict(site=site, compared=12, differ=[], missing=[]))
    from release_gate_profiles import CORRECTNESS_JOBS, COMMAND_TIMEOUT_SECONDS
    put(out / 'correctness/results.json', dict(
        status='PASSED', scope='expanded', source_commit=revision, installed_from=dict(expected_version=version),
        installed=dict(vendor='cuda'),
        jobs=[dict(name=name, exit_code=job_exit if name == 'models' else 0, timed_out=False,
                   timeout_seconds=COMMAND_TIMEOUT_SECONDS, elapsed_seconds=1.0) for name in CORRECTNESS_JOBS]))
    smoke = out / 'adapter-smoke'
    results = {'torch-compile/adapter-results.json': ['torch_compile:mlp'],
               'torch-training/training-results.json': ['mlp: resumed 2+2 equals uninterrupted 4'],
               'max-training/max-training-results.json': ['mlp: compile, execute', 'mlp: resume exact']}
    phases = {}
    for relative, checks in results.items():
        put(smoke / relative, dict(complete=True, checks=checks, failures=[]))
    for phase in ('torch-compile', 'torch-training', 'max-training'):
        put(smoke / (phase + '.log'), 'fixture\n')
        phases[phase] = dict(exit=0, seconds=1.0, timed_out=False)
    put(smoke / 'adapter-smoke.json', dict(status='PASSED', device='cuda:0', problems=[], phases=phases,
        frameworks=dict(torch='ran', max='ran'), installed={'mojolearn-core': version}))
    put(out / 'request.json', dict(vendor='nvidia', version=version, install_exit=0, source_commit=revision,
                                   candidates={core.name: candidate_sha or sha(core)}))
    put(out / published.REPORT, published.judge(out))
    return out


def test_published_install_is_admitted_against_the_qualified_artifacts(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    out = published_dir(tmp_path, revision, core)
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'passed', report['checks']
    assert {row['id']: row['expected'] for row in report['coverage']} == {
        'index-install': 4, 'published-artifact-hashes': 3, 'installed-smoke': 11, 'installed-adapters': 5}
    ledger, rows = ledger_for(tmp_path, repo, revision, {'published-nvidia'}, configuration_for({core.name: sha(core)}))
    assert producer_admission_available(rows['published-nvidia'])
    # The box directory is pulled home before admission: evidence is judged where it now lies.
    home = tmp_path / 'home/published'
    shutil.copytree(out, home)
    shutil.rmtree(out)
    review, result = admit(tmp_path, repo, ledger, 'published-install', home / published.REPORT)
    receipt = json.loads((tmp_path / 'review/receipts.json').read_text())['published-nvidia']
    assert all(Path(ref['path']).is_relative_to(tmp_path / 'home') or Path(ref['path']).is_relative_to(tmp_path / 'review')
               for ref in receipt['evidence'])
    assert result['actions'] == {'published-nvidia': 'reuse'}
    assert ledger.gate(stage='post') == (True, [])


def test_published_bytes_other_than_the_candidate_are_refused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    core = wheel(tmp_path)
    out = published_dir(tmp_path, revision, core, candidate_sha='c' * 64)
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'failed'
    # A passing report whose bytes differ from the ledger's qualified artifacts is refused too.
    good = published_dir(tmp_path / 'g', revision, core)
    ledger, _ = ledger_for(tmp_path, repo, revision, {'published-nvidia'}, configuration_for({core.name: 'd' * 64}))
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('published-install', repo, ledger.path, tmp_path / 'review',
                          dict(report=good / published.REPORT))
    review = json.loads((tmp_path / 'review/review.json').read_text())
    assert 'qualified artifacts' in review['refused']['published-nvidia']


def test_published_failed_smoke_job_fails(tmp_path, repo):
    out = published_dir(tmp_path, git(repo, 'rev-parse', 'HEAD'), wheel(tmp_path), job_exit=1)
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'failed'
    assert [row['id'] for row in report['coverage'] if row['status'] != 'passed'] == ['installed-smoke']


def test_real_catalog_rows_have_installed_adapters():
    catalog = json.loads((TOOLS / 'release_requirements.json').read_text())
    families = {'runtime': ('nvidia', 'amd', 'apple', 'cpu'), 'multi': ('nvidia', 'amd'),
                'published': ('nvidia', 'amd', 'apple')}
    seen = set()
    for spec in catalog['requirements']:
        if spec.get('kind') in families:
            wanted = dict(kind=spec['kind'], coverage=spec['coverage'], inputs=dict(contract=dict(id=spec['id'])))
            assert installed.supports_requirement(wanted), spec['id']
            assert admission.supports_requirement(wanted), spec['id']
            seen.add(spec['id'])
    assert len(seen) == 9
    optional = dict(kind='runtime', required=False, coverage=list(runtime.COVERAGE_IDS),
                    inputs=dict(contract=dict(id='runtime-apple/public-regressions')))
    assert not installed.supports_requirement(optional)
