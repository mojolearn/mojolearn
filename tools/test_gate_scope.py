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


@pytest.fixture
def scoped_module_gates(monkeypatch, tmp_path):
    """Executable modules, not pytest functions; discovery must not import them."""
    root = tmp_path / 'python/mojolearn/tests'
    root.mkdir(parents=True)
    (root / 'test_host_gate.py').write_text(
        'GATE_BACKENDS = ("cpu",)\nraise RuntimeError("plan must not import")\n')
    (root / 'test_device_gate.py').write_text(
        'GATE_BACKENDS = ("metal", "cuda", "hip")\nraise RuntimeError("plan must not import")\n')
    (root / 'test_unscoped_gate.py').write_text('raise RuntimeError("plan must not import")\n')
    monkeypatch.setattr(gates, 'ROOT', tmp_path)
    return {'test_host_gate', 'test_device_gate', 'test_unscoped_gate'}


@pytest.mark.parametrize('backend', ['metal', 'cuda', 'hip'])
def test_gpu_all_excludes_declared_cpu_gates(capsys, backend, scoped_module_gates):
    assert gates.main(['--all', '--backend', backend, '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan['excluded']) == {'test_host_gate'}
    assert set(plan['gates']) == {'test_device_gate', 'test_unscoped_gate'}
    assert set(plan['gates']) | set(plan['excluded']) == scoped_module_gates


def test_cpu_all_retains_declared_cpu_module_gates(capsys, scoped_module_gates):
    assert gates.main(['--all', '--backend', 'cpu', '--plan']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert set(plan['gates']) == {'test_host_gate', 'test_unscoped_gate'}
    assert set(plan['excluded']) == {'test_device_gate'}


def test_cpu_training_checks_are_retained_in_pytest_inventory():
    # These files define test_* functions as well as a __main__ convenience
    # runner. Since dd999d458, the exhaustive inventory assigns them to pytest
    # so the module-gate pass does not execute the same checks twice.
    suites = gates.inventory(gates.ROOT / 'python/mojolearn/tests')
    cpu_training = {p.stem for p in suites['pytest'] if p.stem.startswith('test_cpu_training')}
    source_files = {p.stem for p in (gates.ROOT / 'python/mojolearn/tests').glob('test_cpu_training*.py')}
    assert source_files and cpu_training == source_files
    assert not cpu_training.intersection(gates.discover())
    assert all(gates.gate_backends(name) == {'cpu'} for name in cpu_training)


def test_explicit_wrong_backend_refuses_before_launch(monkeypatch, tmp_path, scoped_module_gates, capsys):
    monkeypatch.setattr(gates.identity_iterate, 'run_job', lambda *a, **k: pytest.fail('inapplicable gate launched'))
    with pytest.raises(SystemExit):
        gates.main(['--gate', 'test_host_gate', '--backend', 'cuda', '--out', str(tmp_path / 'out')])
    assert 'selected gates do not apply to cuda' in capsys.readouterr().err
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
