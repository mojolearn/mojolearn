# SPDX-License-Identifier: Apache-2.0
"""Verifier negative controls. No product imports, compilation or GPU execution."""
import argparse
import json
import os
from pathlib import Path
import sys

import pytest

import verify_neural_provider as verify


def test_pinned_candidate_rejects_missing_native_files_and_changed_bytes(tmp_path):
    package = tmp_path / 'mojolearn'
    package.mkdir()
    (package / '__init__.py').write_text('# inert verifier fixture\n')
    with pytest.raises(ValueError, match='no native binaries'):
        verify.pin_candidate(package, [])
    binary = package / '_mojolearn_fixture.so'
    binary.write_bytes(b'inert binary bytes, never loaded')
    pinned = verify.pin_candidate(package, [])
    observed = {'mojolearn._mojolearn_fixture': str(binary)}
    rows, failures = verify.validate_loaded(observed, pinned)
    assert not failures and rows[0]['verified']
    binary.write_bytes(b'changed bytes')
    _, failures = verify.validate_loaded(observed, pinned)
    assert any('changed bytes' in failure for failure in failures)


def test_loaded_paths_outside_supplied_candidate_and_no_binary_are_rejected(tmp_path):
    binary = tmp_path / 'outside.so'
    binary.write_bytes(b'never loaded')
    _, failures = verify.validate_loaded({'mojolearn.fake': str(binary)}, {})
    assert any('unpinned' in failure for failure in failures)
    code = tmp_path / 'only_python.py'
    code.write_text('# no native execution')
    _, failures = verify.validate_loaded({'mojolearn': str(code)}, {str(code): verify.digest(code)})
    assert 'no loaded native product binary was observed' in failures


@pytest.mark.parametrize('element', ['skipped', 'failure', 'error'])
def test_pytest_refusals_and_failures_cannot_be_reported_as_passed(tmp_path, element):
    xml = tmp_path / 'report.xml'
    xml.write_text(f'<testsuites><testsuite><testcase name="a"><{element} message="why"/></testcase></testsuite></testsuites>')
    result = verify.junit_result(xml)
    assert result['status'] == 'failed' and result['failed'] == 1
    assert result['cases'][0]['outcome'] == element


def test_empty_collection_is_failure(tmp_path):
    xml = tmp_path / 'report.xml'
    xml.write_text('<testsuite tests="0"/>')
    with pytest.raises(ValueError, match='no test cases'):
        verify.junit_result(xml)


def test_incomplete_build_records_preserve_success_and_failure_groups(tmp_path):
    record = tmp_path / 'build.json'
    record.write_text(json.dumps({'schema': 'mojolearn.development-build.v1',
        'results': [{'key': 'a', 'state': 'compiled', 'exit_code': 0, 'sha256': 'a' * 64},
                    {'key': 'b', 'state': 'failed', 'exit_code': 1}]}))
    out = tmp_path / 'out'
    out.mkdir()
    summaries = verify.retain_build_records([record], out)
    assert [g['exit_code'] for g in summaries[0]['groups']] == [0, 1]
    assert Path(summaries[0]['retained']).read_bytes() == record.read_bytes()


def test_process_exception_and_timeout_are_recorded_not_raised(tmp_path):
    missing = verify.execute([str(tmp_path / 'absent')], tmp_path / 'missing.log', tmp_path, os.environ.copy(), 2)
    assert missing['status'] == 'failed' and 'exception' in missing
    timeout = verify.execute([sys.executable, '-c', 'import time; time.sleep(10)'],
                             tmp_path / 'timeout.log', tmp_path, os.environ.copy(), 0.05)
    assert timeout['status'] == 'failed' and timeout['timed_out']
    assert 'Verifier timeout' in (tmp_path / 'timeout.log').read_text()


def test_source_only_runs_remaining_checks_and_never_claims_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'source_provenance', lambda root: {'commit': 'test'})
    monkeypatch.setattr(verify, 'SOURCE_AUDITS', ('first.py', 'second.py'))
    monkeypatch.setattr(verify, 'SOURCE_SUITES', ())
    called = []
    def execute(command, log, cwd, env, timeout):
        called.append(command[-1])
        return {'status': 'failed' if len(called) == 1 else 'passed', 'exit_code': 1 if len(called) == 1 else 0}
    monkeypatch.setattr(verify, 'execute', execute)
    args = argparse.Namespace(output=tmp_path / 'out', mode='source-only', timeout=1)
    assert verify.verify(args) == 1
    report = json.loads((args.output / 'result.json').read_text())
    assert len(called) == 2 and report['failed_checks'] == ['first.py']
    assert not report['selected_runtime_checks_passed']
    assert any('ALL candidate' in row for row in report['pending'])


def test_source_success_still_reports_runtime_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'source_provenance', lambda root: {'commit': 'test'})
    monkeypatch.setattr(verify, 'SOURCE_AUDITS', ('only.py',))
    monkeypatch.setattr(verify, 'SOURCE_SUITES', ())
    monkeypatch.setattr(verify, 'execute', lambda *args: {'status': 'passed', 'exit_code': 0})
    args = argparse.Namespace(output=tmp_path / 'out', mode='source-only', timeout=1)
    assert verify.verify(args) == 0
    report = json.loads((args.output / 'result.json').read_text())
    assert report['status'] == 'source_passed_runtime_pending'
    assert not report['selected_runtime_checks_passed']


def test_runtime_failure_does_not_cancel_other_independent_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'source_provenance', lambda root: {'commit': 'test'})
    monkeypatch.setattr(verify, 'SOURCE_AUDITS', ())
    monkeypatch.setattr(verify, 'SOURCE_SUITES', ())
    monkeypatch.setattr(verify, 'pin_candidate', lambda *args: {})
    monkeypatch.setattr(verify, 'retain_build_records', lambda *args: [])
    calls = []
    def execute(command, log, cwd, env, timeout):
        config = json.loads(Path(command[-1]).read_text())
        calls.append(config['path'])
        assert env['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] == ('1' if config['kind'] == 'pytest' else '0')
        # Simulate a missing API in the first process; no product module is
        # imported and no native/GPU work occurs in this verifier control test.
        failed = len(calls) == 1
        details = {'status': 'failed' if failed else 'passed', 'loaded_files': [],
                   'exception': 'missing candidate API' if failed else None}
        Path(config['result']).write_text(json.dumps(details))
        return {'status': details['status'], 'exit_code': int(failed)}
    monkeypatch.setattr(verify, 'execute', execute)
    args = argparse.Namespace(output=tmp_path / 'out', mode='runtime', timeout=1,
        package=tmp_path, binary_root=[], build_record=[], python=Path(sys.executable))
    assert verify.verify(args) == 1
    report = json.loads((args.output / 'result.json').read_text())
    assert len(calls) == len(verify.RUNTIME_SUITES) + len(verify.RUNTIME_PROBES) == 9
    assert len(report['failed_checks']) == 1
    assert not report['selected_runtime_checks_passed']
