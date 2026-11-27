"""Installed runtime producers on synthetic receipts (the published-install family left the ledger on 2026-11-27).

Nothing is installed, compiled or run on a GPU: the producer report functions
(summarize/judge) and the ledger adapters run over fixture directories.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import qualify_native_runtime_installed as q
import release_installed_evidence as installed
import release_ledger_admission as admission
import release_runtime_installed as runtime
from release_requirements import RequirementsLedger, producer_admission_available, requirements

VERSION = '0.9.1'
TOOLS = Path(__file__).resolve().parent
CATALOG = {
    'schema': 'fixture', 'source_inputs': {'public_runtime': ['python/mojolearn/*.py']}, 'source_exclude': [],
    'requirements': [
        {'id': 'runtime-nvidia', 'kind': 'runtime', 'stage': 'pre',
         'scope': 'Public operations and transitive callback refusals on nvidia', 'coverage': list(runtime.COVERAGE_IDS),
         'source_inputs': {'tests': ['tools/release_runtime_installed.py']},
         'configuration': ['release_modes', 'build_flags']},
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


def test_no_published_family_remains_in_the_ledger():
    # Andrew 2026-11-27: the per-vendor published rows left with the published-<vendor> steps; the one published
    # check's report is kept by the release evidence set, not admitted into the ledger
    assert 'published' not in installed.FAMILIES and 'published-install' not in admission.ADAPTERS
    catalog = json.loads((TOOLS / 'release_requirements.json').read_text())
    assert not any(spec.get('kind') == 'published' for spec in catalog['requirements'])


def test_real_catalog_rows_have_installed_adapters():
    catalog = json.loads((TOOLS / 'release_requirements.json').read_text())
    families = {'runtime': ('nvidia', 'amd', 'apple', 'cpu')}
    seen = set()
    for spec in catalog['requirements']:
        if spec.get('kind') in families:
            wanted = dict(kind=spec['kind'], coverage=spec['coverage'], inputs=dict(contract=dict(id=spec['id'])))
            assert installed.supports_requirement(wanted), spec['id']
            assert admission.supports_requirement(wanted), spec['id']
            seen.add(spec['id'])
    assert len(seen) == 4
    optional = dict(kind='runtime', required=False, coverage=list(runtime.COVERAGE_IDS),
                    inputs=dict(contract=dict(id='runtime-apple/public-regressions')))
    assert not installed.supports_requirement(optional)


def test_linux_split_wheels_match_candidate_distribution_names():
    import release_installed_evidence as E
    configuration = {"artifacts:nvidia": {"mojolearn-core": "a" * 64, "mojolearn-nvidia": "b" * 64}}
    wheels = {"mojolearn_core-0.8.63-py3-none-manylinux_2_35_x86_64.whl": "a" * 64,
              "mojolearn_nvidia-0.8.63-py3-none-manylinux_2_35_x86_64.whl": "b" * 64}
    assert E._qualified(configuration, wheels)
    assert not E._qualified(configuration, dict(wheels, **{
        "mojolearn_core-0.8.63-py3-none-manylinux_2_35_x86_64.whl": "c" * 64}))
