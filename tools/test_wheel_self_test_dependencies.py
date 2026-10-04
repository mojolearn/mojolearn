"""The canonical verifier gate must request the wheel's verification dependencies."""
import json
import subprocess
import sys

import wheel_self_test


def test_gate_installs_verify_extra_for_exact_local_artifact(tmp_path, monkeypatch):
    wheel = tmp_path / 'mojolearn-0.8.37-py3-none-any.whl'
    wheel.write_bytes(b'local candidate')
    output = tmp_path / 'receipt'
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        report = json.dumps({'passed': True, 'clean': {'state': 'IDENTICAL'},
                             'perturbed': {'state': 'DIVERGENT'}})
        return subprocess.CompletedProcess(command, 0, stdout=report, stderr='')

    monkeypatch.setattr(wheel_self_test.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['wheel_self_test.py', str(wheel),
                                    '--out', str(output), '--python', '/test/python'])
    assert wheel_self_test.main() == 0
    install = next(command for command in calls if 'pip' in command)
    assert install[-1] == str(wheel.resolve()) + '[verify]'
    assert wheel_self_test.passed(output / 'results.json', wheel)
