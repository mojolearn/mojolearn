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


@pytest.mark.parametrize('backend', ['metal', 'cuda', 'hip'])
def test_gpu_all_excludes_declared_cpu_gates(capsys, backend):
    assert gates.main(['--all', '--backend', backend, '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    cpu = {n for n in gates.discover() if gates.gate_backends(n) == {'cpu'}}
    assert cpu and cpu <= set(plan['excluded'])
    assert not cpu.intersection(plan['gates'])
    assert set(plan['gates']) | set(plan['excluded']) == set(gates.discover())


def test_cpu_all_retains_host_training_gates(capsys):
    gates.main(['--all', '--backend', 'cpu', '--plan'])
    plan = json.loads(capsys.readouterr().out)
    cpu_training = {n for n in gates.discover() if n.startswith('test_cpu_training')}
    assert cpu_training and cpu_training <= set(plan['gates'])
    assert all(gates.gate_backends(n) == {'cpu'} for n in cpu_training)


def test_explicit_wrong_backend_refuses_before_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(gates.identity_iterate, 'run_job', lambda *a, **k: pytest.fail('inapplicable gate launched'))
    with pytest.raises(SystemExit):
        gates.main(['--gate', 'test_cpu_training_transformer', '--backend', 'cuda', '--out', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_metadata_is_literal_and_unmarked_gate_remains_conservative(monkeypatch, tmp_path):
    monkeypatch.setattr(gates, 'ROOT', tmp_path)
    root = tmp_path / 'python/mojolearn/tests'
    root.mkdir(parents=True)
    p = root / 'test_example.py'
    p.write_text('raise RuntimeError("must not import")\n')
    assert gates.gate_backends('test_example') == gates.BACKENDS
    for declaration in ['GATE_BACKENDS = make_scope()', 'GATE_BACKENDS = ()',
                        'GATE_BACKENDS = ("typo",)', 'GATE_BACKENDS = "cpu"',
                        'GATE_BACKENDS = ("cpu",)\nGATE_BACKENDS = ("metal",)']:
        p.write_text(declaration)
        with pytest.raises(ValueError):
            gates.gate_backends('test_example')
