"""Installed runtime and published-install producers on synthetic receipts.

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
import release_published_install as published
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


# ---------------------------------------------------------------- published-<vendor>
from test_release_published_bytes import nvidia_set  # noqa: E402  (real wheels with RECORD, version 0.8.66)

PUB_VERSION = '0.8.66'


def published_dir(tmp_path, revision, wheels, *, candidate_sha=None, index_url=None, gate=True):
    """A retained published-<vendor> bytes check as `check` leaves it: the request, the resolved files and the
    gate receipt copied by content (no install, no device)."""
    out = tmp_path / 'published'
    resolved, digests = [], {}
    for w in wheels:
        dist = published.wheel_tags(w.name)['distribution']
        downloaded = out / 'wheels' / dist / w.name
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(w, downloaded)
        digests[dist] = digest = sha(downloaded)
        resolved.append(dict(distribution=dist, requested=dist + '==' + PUB_VERSION, candidate=w.name, pip_exit=0,
                             files=[w.name], file=w.name, sha256=digest, index_sha256=digest))
    evidence = []
    if gate:
        receipt = put(tmp_path / 'gate' / 'gate-smoke.json', dict(verdict='PASS', candidate=dict(wheels=digests)))
        copy = put(out / 'gate-evidence' / (sha(receipt)[:16] + '-gate-smoke.json'), receipt.read_text())
        evidence.append(dict(source=str(receipt), path=copy.relative_to(out).as_posix(), sha256=sha(copy)))
    candidates = {w.name: sha(w) for w in wheels}
    candidates.update(candidate_sha or {})
    put(out / 'request.json', dict(
        mode='bytes', vendor='nvidia', version=PUB_VERSION, index_url=index_url or published.PYPI_INDEX,
        source_commit=revision, candidates=candidates,
        gate=dict(verdict='gate-nvidia PASS: 11 installed correctness records', evidence=evidence),
        resolved=resolved))
    put(out / published.REPORT, published.judge(out, index_url or published.PYPI_INDEX))
    return out


def published_ledger(tmp_path, repo, revision, artifacts):
    configuration = configuration_for(artifacts)
    catalog = dict(CATALOG, requirements=[s for s in CATALOG['requirements'] if s['id'] == 'published-nvidia'])
    rows = requirements(repo, revision, PUB_VERSION, catalog=catalog, configuration=configuration,
                        expand_dependencies=False)['requirements']
    ledger = RequirementsLedger(tmp_path / 'ledger.json')
    ledger.sync(dict(schema='mojolearn.release-requirements.v1', version=PUB_VERSION, source_revision=revision,
                     requirements=rows, source_root=str(repo), configuration=configuration))
    return ledger, rows


def test_published_install_is_admitted_against_the_qualified_artifacts(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    wheels = nvidia_set(tmp_path / 'cand')
    out = published_dir(tmp_path, revision, wheels)
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'passed', report['checks']
    assert {row['id']: row['expected'] for row in report['coverage']} == {
        # mojolearn-core and mojolearn-nvidia (the NVIDIA wheel carries its PTX set, 2026-11-05)
        'index-resolution': 6, 'published-artifact-hashes': 2, 'wheel-integrity': 2, 'gate-evidence': 4}
    ledger, rows = published_ledger(tmp_path, repo, revision, {w.name: sha(w) for w in wheels})
    assert producer_admission_available(rows['published-nvidia'])
    # The directory is moved before admission: evidence is judged where it now lies.
    home = tmp_path / 'home/published'
    shutil.copytree(out, home)
    shutil.rmtree(out)
    review, result = admit(tmp_path, repo, ledger, 'published-install', home / published.REPORT)
    receipt = json.loads((tmp_path / 'review/receipts.json').read_text())['published-nvidia']
    assert all(Path(ref['path']).is_relative_to(tmp_path / 'home') or Path(ref['path']).is_relative_to(tmp_path / 'review')
               for ref in receipt['evidence'])
    assert any('gate-evidence' in ref['path'] for ref in receipt['evidence'])
    assert result['actions'] == {'published-nvidia': 'reuse'}
    assert ledger.gate(stage='post') == (True, [])


def test_published_bytes_other_than_the_candidate_are_refused(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    wheels = nvidia_set(tmp_path / 'cand')
    out = published_dir(tmp_path, revision, wheels, candidate_sha={wheels[1].name: 'c' * 64})
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'failed'
    assert report['mismatch'] == [dict(file=wheels[1].name, published_sha256=sha(wheels[1]), candidate=wheels[1].name,
                                       candidate_sha256='c' * 64)]
    # A passing report whose bytes differ from the ledger's qualified artifacts is refused too.
    good = published_dir(tmp_path / 'g', revision, wheels)
    ledger, _ = published_ledger(tmp_path, repo, revision, {w.name: 'd' * 64 for w in wheels})
    with pytest.raises(ValueError, match='no requirement could be admitted'):
        admission.prepare('published-install', repo, ledger.path, tmp_path / 'review',
                          dict(report=good / published.REPORT))
    review = json.loads((tmp_path / 'review/review.json').read_text())
    assert 'qualified artifacts' in review['refused']['published-nvidia']


def test_published_without_gate_evidence_fails(tmp_path, repo):
    out = published_dir(tmp_path, git(repo, 'rev-parse', 'HEAD'), nvidia_set(tmp_path / 'cand'), gate=False)
    report = json.loads((out / published.REPORT).read_text())
    assert report['status'] == 'failed'
    assert [row['id'] for row in report['coverage'] if row['status'] != 'passed'] == ['gate-evidence']


def test_a_report_made_against_another_index_is_refused_by_the_ledger(tmp_path, repo):
    revision = git(repo, 'rev-parse', 'HEAD')
    wheels = nvidia_set(tmp_path / 'cand')
    out = published_dir(tmp_path, revision, wheels, index_url='file:///fake/simple')
    assert json.loads((out / published.REPORT).read_text())['status'] == 'passed'   # passed for its own index
    ledger, _ = published_ledger(tmp_path, repo, revision, {w.name: sha(w) for w in wheels})
    with pytest.raises(ValueError, match='differs from its raw producer files'):
        admission.prepare('published-install', repo, ledger.path, tmp_path / 'review', dict(report=out / published.REPORT))


def test_real_catalog_rows_have_installed_adapters():
    catalog = json.loads((TOOLS / 'release_requirements.json').read_text())
    families = {'runtime': ('nvidia', 'amd', 'apple', 'cpu'), 'published': ('nvidia', 'amd', 'apple')}
    seen = set()
    for spec in catalog['requirements']:
        if spec.get('kind') in families:
            wanted = dict(kind=spec['kind'], coverage=spec['coverage'], inputs=dict(contract=dict(id=spec['id'])))
            assert installed.supports_requirement(wanted), spec['id']
            assert admission.supports_requirement(wanted), spec['id']
            seen.add(spec['id'])
    assert len(seen) == 7
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
