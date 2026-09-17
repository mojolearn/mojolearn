# SPDX-License-Identifier: Apache-2.0
import json
import pytest
import check_python_gates as gates


def test_unscoped_gate_run_refuses():
    with pytest.raises(SystemExit):
        gates.main([])


def test_one_gate_plan_does_not_stage_or_launch(monkeypatch, capsys):
    monkeypatch.setattr(gates.identity_iterate, 'run_job', lambda *a: pytest.fail('plan launched'))
    name = gates.discover()[0]
    assert gates.main(['--gate', name, '--backend', 'cuda', '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['gates'] == [name] and plan['backend'] == 'cuda'
    assert plan['timeout'] == 60 and plan['budget'] == 300


def test_failed_gate_stops_round_and_records_pending(monkeypatch, tmp_path):
    names = [n for n in gates.discover() if n != 'test_linalg_identity'][:2]
    monkeypatch.setattr(gates.identity_iterate, 'cpu_package', lambda *a: (tmp_path, tmp_path))
    calls = []
    def fail(cmd, env, **kwargs):
        calls.append(cmd)
        assert '--deadline' in cmd
        return 124
    monkeypatch.setattr(gates.identity_iterate, 'run_job', fail)
    assert gates.main(['--gate', names[0], '--gate', names[1], '--out', str(tmp_path)]) == 124
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert len(calls) == 1 and not report['complete']
    assert report['pending'] == [names[1]]
    assert report['failed']['gate'] == names[0]


@pytest.mark.parametrize('value', ['nan', 'inf', '0'])
def test_invalid_budget_refuses(value):
    with pytest.raises(SystemExit):
        gates.main(['--all', '--budget', value, '--plan'])
