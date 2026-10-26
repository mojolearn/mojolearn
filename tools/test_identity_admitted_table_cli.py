"""Current local payloads compare with admitted values, retaining actual raw records."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

PKG = Path(__file__).resolve().parents[1] / 'python/mojolearn'


@pytest.fixture
def cli(monkeypatch, tmp_path):
    name = 'identity_admitted_table_test_package'
    package = ModuleType(name)
    package.__path__ = [str(PKG)]
    monkeypatch.setitem(sys.modules, name, package)
    verify = ModuleType(name + '._verify')
    for value, label in enumerate(('VERIFIED', 'MISMATCH', 'USAGE', 'REFUSED_FAST', 'CANNOT_RUN', 'NO_REFERENCE')):
        setattr(verify, 'EXIT_' + label, value)
    monkeypatch.setitem(sys.modules, verify.__name__, verify)
    surface = ModuleType(name + '.host_surface')
    surface.GBDT_CTR_MODEL_LANES = ()
    surface.comparable_lanes = lambda lanes, vendor: list(lanes)
    surface.covered_lanes = lambda: ['changed', 'new']
    surface.PUBLIC_INAPPLICABLE_PREFIXES = ('par-',)
    surface.lane_exposure = lambda lanes, vendor: {lane: {'status': 'covered'} for lane in lanes}
    monkeypatch.setitem(sys.modules, surface.__name__, surface)
    backend = ModuleType(name + '._backend')
    backend.numeric_mode = lambda: 'identical'
    backend.vendor = lambda: 'cuda'
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    def load(leaf):
        spec = importlib.util.spec_from_file_location(name + '.' + leaf, PKG / (leaf + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        setattr(package, leaf, module)
        return module

    ref, engine, module = load('_verify_reference'), load('_verify_all'), load('_identity')
    monkeypatch.setattr(engine, 'host_surface', lambda: surface)
    harness = SimpleNamespace(LANES={'changed': None, 'new': None}, FIXTURES=['base', 'ties'], LANE_REVISIONS={})
    table = dict(format=ref.FORMAT, records=[dict(dir='current', file='nvidia.json', vendor='gpu', commit='1'*40),
        dict(dir='retained', file='amd.json', vendor='old-gpu', commit='2'*40)],
        fixtures={f: {'X': f, 'y_clf': 'c', 'y_reg': 'r'} for f in harness.FIXTURES},
        heldout={f: {'X': 'held-' + f} for f in harness.FIXTURES}, cells={})
    for lane in harness.LANES:
        for fixture in harness.FIXTURES:
            table['cells'][lane + '/' + fixture] = {
                p: dict(ref='b'*16 if p == 'train' else 'n/a:function', cols={'nvidia': 0, 'amd': 1})
                for p in ('train', 'infer', 'model')}
    table_file = tmp_path / 'table.json'
    monkeypatch.setattr(ref, 'table_path', lambda: str(table_file))
    monkeypatch.setattr(engine, 'load_harness', lambda path: harness)
    monkeypatch.setattr(module, 'harness_path', lambda: ('installed-harness.py', 'wheel copy'))
    monkeypatch.setattr(module, 'commit_witness', lambda: ('3'*40, 'wheel COMMIT witness'))
    monkeypatch.setattr(module, 'columns', lambda: pytest.fail('historical columns must not decide current payloads'))
    args = SimpleNamespace(check=False, lanes='changed', fixtures='base', repeats=1,
                           vendor='named-worker', keep=str(tmp_path / 'local.json'), json=True)
    state = SimpleNamespace(calls=[], rc=0, mutate=lambda record: None, local=None)

    def run(path, argv, env, capture):
        state.calls.append((path, argv, env, capture))
        assert '--diff' not in argv and capture is False
        lanes = argv[argv.index('--lanes')+1].split(',')
        fixtures = argv[argv.index('--fixtures')+1].split(',') if '--fixtures' in argv else harness.FIXTURES
        repeats = int(argv[argv.index('--repeats')+1])
        record = dict(complete=True, repeats=repeats, commit='3'*40, vendor=args.vendor,
                      fixtures=copy.deepcopy(table['fixtures']), heldout=copy.deepcopy(table['heldout']), cells={})
        for lane in lanes:
            for fixture in fixtures:
                record['cells'][lane + '/' + fixture] = dict(verdict='STABLE', hashes=['b'*16]*repeats,
                    infer=['n/a:function']*repeats, infer_verdict='N/A', model=['n/a:function']*repeats, model_verdict='N/A')
        state.mutate(record)
        state.local = copy.deepcopy(record)
        Path(argv[argv.index('--json')+1]).write_text(json.dumps(record))
        return state.rc, ''
    monkeypatch.setattr(module, '_run', run)

    def invoke(capsys):
        table_file.write_text(json.dumps(table))
        rc = module.cmd_identity(args)
        output = capsys.readouterr().out
        report = json.loads(output[output.index('{'):])
        return rc, report
    return SimpleNamespace(module=module, engine=engine, ref=ref, args=args, table=table,
        harness=harness, state=state, invoke=invoke, backend=backend, table_file=table_file)


def test_current_payload_matches_admitted_reference_and_retains_original_origins(cli, capsys):
    rc, report = cli.invoke(capsys)
    assert rc == 0 and report['verdict'] == 'IDENTICAL'
    assert len(cli.state.calls) == 1 and report['diff_exit'] is None and report['comparison_exit'] == 0
    assert {c['recorded_at_commit'] for c in report['reference_evidence']['columns']} == {'1'*40, '2'*40}
    assert json.loads(Path(cli.args.keep).read_text()) == cli.state.local
    assert 'x4' not in report['detail']


def test_flags_keep_actual_raw_column_and_new_table_lane(cli, capsys):
    cli.args.lanes, cli.args.fixtures, cli.args.repeats = 'new', 'ties', 2
    rc, report = cli.invoke(capsys)
    assert rc == 0 and report['lanes'] == ['new'] and report['fixtures'] == ['ties']
    argv = cli.state.calls[0][1]
    assert argv[argv.index('--vendor')+1] == 'named-worker'
    assert argv[argv.index('--repeats')+1] == '2'
    assert report['local_json'] == cli.args.keep
    assert cli.state.local['cells']['new/ties']['hashes'] == ['b'*16]*2


@pytest.mark.parametrize('damage', ['missing-reference', 'conflict', 'stale', 'unknown-lane', 'unknown-fixture', 'zero-repeats'])
def test_bad_selection_or_reference_refuses_before_execution(cli, capsys, damage):
    if damage == 'missing-reference': del cli.table['cells']['changed/base']
    if damage == 'conflict': cli.table['cells']['changed/base']['train']['conflict'] = True
    if damage == 'stale': cli.harness.LANE_REVISIONS = {'changed': 'new-fixture'}
    if damage == 'unknown-lane': cli.args.lanes = 'unknown'
    if damage == 'unknown-fixture': cli.args.fixtures = 'unknown'
    if damage == 'zero-repeats': cli.args.repeats = 0
    rc, report = cli.invoke(capsys)
    assert rc in (2, 5) and not cli.state.calls


@pytest.mark.parametrize('damage', ['incomplete', 'missing-cell', 'missing-part', 'wrong-input', 'divergent', 'unstable', 'collector-failure'])
def test_missing_or_failed_execution_never_passes(cli, capsys, damage):
    def change(record):
        cell = record['cells']['changed/base']
        if damage == 'incomplete': record['complete'] = False
        if damage == 'missing-cell': record['cells'].clear()
        if damage == 'missing-part': cell.pop('model')
        if damage == 'wrong-input': record['fixtures']['base']['X'] = 'other'
        if damage == 'divergent': cell['hashes'] = ['a'*16]
        if damage == 'unstable': cell['verdict'] = 'MOVED'
    cli.state.mutate = change
    if damage == 'collector-failure': cli.state.rc = 1
    rc, report = cli.invoke(capsys)
    assert rc in (1, 4) and report['verdict'] != 'IDENTICAL'
    assert len(cli.state.calls) == 1


def test_check_keeps_historical_file_inspection(cli, monkeypatch):
    cli.args.check = True
    calls = []
    monkeypatch.setattr(cli.module, 'cmd_check', lambda args: calls.append(args) or 0)
    assert cli.module.cmd_identity(cli.args) == 0
    assert calls == [cli.args] and not cli.state.calls
