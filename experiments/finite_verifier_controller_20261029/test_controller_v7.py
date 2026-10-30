"""Exercise actual retained controller AST without launching native producers."""
import ast
import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from controller_local_failures import LocalObservationFailure


SOURCE = Path(__file__).with_name('run_frozen_chunks_v7.py')
TREE = ast.parse(SOURCE.read_text())


def execute_nodes(nodes, namespace):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)


def controller_loop(tmp_path, action, *, keep=True):
    saved, calls = {}, []
    def run(chunk):
        calls.append(chunk['lane'])
        action(chunk)
    scope = dict(B=tmp_path, a=SimpleNamespace(vendor='apple', phase='routine-all', max_units=None,
        lease_deadline_epoch=None, unit_budget_seconds=None, retention_reserve_seconds=None,
        keep_going_local_failures=keep), selected=[{'lane': lane, 'vendor': 'apple', 'phase': 'routine-all'}
                                                 for lane in ['broken', 'healthy']],
        excluded={}, failed_local=[], run=run, save=lambda p,v: saved.__setitem__(Path(p).name,v),
        lease_budget=lambda *a: None, pending_count=lambda p: 1, time=time, json=json,
        LocalObservationFailure=LocalObservationFailure)
    loop = next(node for node in TREE.body if isinstance(node, ast.Try))
    try:
        execute_nodes([loop], scope)
    except BaseException as error:
        return error, calls, saved
    return None, calls, saved


def test_known_local_failure_continues_other_unit_and_finishes_nonzero(tmp_path):
    def action(chunk):
        if chunk['lane'] == 'broken':
            raise LocalObservationFailure({'lane': 'broken', 'state': 'failed-pending-repair'})
    error, calls, saved = controller_loop(tmp_path, action)
    assert isinstance(error, SystemExit) and error.code == 1
    assert calls == ['broken', 'healthy']
    final = saved['apple-routine-all-controller.json']
    assert final['status'] == 'partial-with-local-failures'
    assert len(final['failed_local']) == 1 and final['offsite_pending'] == 1


@pytest.mark.parametrize('error_type', [RuntimeError, ValueError])
def test_infrastructure_or_provenance_error_is_never_swallowed(tmp_path, error_type):
    def action(chunk):raise error_type('retention or provenance failure')
    error, calls, saved = controller_loop(tmp_path, action)
    assert type(error) is error_type and calls == ['broken']
    assert saved['controller-state.json']['status'] == 'stopped'
    assert 'apple-routine-all-controller.json' not in saved


def test_opt_in_required_for_continuation(tmp_path):
    def action(chunk):raise LocalObservationFailure({'lane': 'broken'})
    error, calls, _ = controller_loop(tmp_path, action, keep=False)
    assert isinstance(error, LocalObservationFailure) and calls == ['broken']


def test_existing_failed_unit_is_read_without_reexecuting_producer(tmp_path):
    out = tmp_path / 'runs/apple/routine-all/broken'
    out.mkdir(parents=True)
    for name, value in [('controller-summary.json', {'local_complete': False}),
                        ('verify.json', {'original': 'report'}), ('verifier-run.json', {'original': 'receipt'})]:
        (out / name).write_text(json.dumps(value))
    def retained(*args):
        assert args[-2:] == ({'original': 'report'}, {'original': 'receipt'})
        raise LocalObservationFailure({'lane': 'broken'})
    scope = dict(B=tmp_path, a=SimpleNamespace(keep_going_local_failures=True), json=json, datetime=datetime,
        save=lambda *a: None, preserve_local_failure=retained,
        call=lambda *a, **kw: pytest.fail('original producer must never rerun'))
    execute_nodes([next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'run')], scope)
    with pytest.raises(LocalObservationFailure):
        scope['run']({'vendor': 'apple', 'phase': 'routine-all', 'lane': 'broken'})


def test_retention_failure_prevents_local_failure_continuation(tmp_path):
    (tmp_path / 'verify.json').write_text('{}')
    def fail(*args):raise RuntimeError('retention failed')
    import hashlib
    scope = dict(hashlib=hashlib, validated_failure=lambda *a: {'lane': 'broken'},
        retain_partial=fail, LocalObservationFailure=LocalObservationFailure,
        save=lambda *a: pytest.fail('must not record continuation before retention succeeds'))
    execute_nodes([next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'preserve_local_failure')], scope)
    with pytest.raises(RuntimeError, match='retention failed'):
        scope['preserve_local_failure']({'vendor': 'apple', 'lane': 'broken'}, tmp_path,
                                        tmp_path / 'producer.log', {}, {'exit_code': 4})
