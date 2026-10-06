"""One selected full opponent retry within the original batch's absolute deadline.

No compilation, installation, preparation, smoke or verification work. A new
receipt directory preserves the failed attempt and never creates a new budget.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from run import CAPS, SOURCE, atomic, stop_tree

RACE = 'algos/lars/istella/rows=full'
ARM = 'sklearn-cpu'
MAX_SECONDS = 300
RECIPES = {
    ('algos/lars/istella/rows=full', 'sklearn-cpu'): 'reg-istella',
    ('algos/lars-stable/istella/rows=full', 'sklearn-cpu'): 'reg-istella',
    ('algos/lars-direct/istella/rows=full', 'sklearn-cpu'): 'reg-istella',
    ('algos/dart/istella/rows=full', 'xgboost-cpu'): 'cls-istella',
    ('algos/dart-reg/istella/rows=full', 'xgboost-cpu'): 'reg-istella',
}


def coverage(board_path, race_id, arm):
    if not board_path.exists():
        return dict(status='FAILED_MISSING_RECEIPT')
    board = json.loads(board_path.read_text())
    race = board.get('races', {}).get(race_id, {})
    cells = race.get('cells', [])
    selected = [cell for cell in cells if cell.get('arm') == arm]
    details = dict(race_status=race.get('status'), cells=cells)
    if race.get('status') != 'done' or len(selected) != 1 or len(cells) != 1:
        return dict(status='FAILED_INCOMPLETE_COVERAGE', **details)
    cell = selected[0]
    times = cell.get('times_ms', [])
    warmup = cell.get('warmup_ms')
    if (cell.get('status') != 'ok' or cell.get('rounds') != 1 or len(times) != 1
            or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                       for v in [warmup, *times])):
        return dict(status='FAILED_INCOMPLETE_COVERAGE', **details)
    quality = cell.get('quality', {})
    if (not quality or quality.get('finite') is False or quality.get('error')
            or any(isinstance(value, float) and not math.isfinite(value) for value in quality.values())):
        return dict(status='FAILED_QUALITY', **details)
    if any('/'+lane+'/' in race_id for lane in ('lars-stable', 'lars-direct')) and quality.get('r2', -math.inf) < 0:
        return dict(status='FAILED_QUALITY', **details)
    receipts = cell.get('state_receipts', [])
    if (len(receipts) != 1 or receipts[0].get('output_status') != 'ok'
            or len(receipts[0].get('output_sha256') or '') != 64):
        return dict(status='FAILED_MISSING_OUTPUT_HASH', **details)
    return dict(status='MEASURED', **details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True,
                        help='Existing full input root, or its rows-full directory')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--budget-status', type=Path, required=True,
                        help='Original attempt-01/status.json; deadline is inherited unchanged')
    parser.add_argument('--race', default=RACE)
    parser.add_argument('--arm', default=ARM)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--failed-result', type=Path, required=True,
                        help='Original failed result.json; successful cells cannot be replayed')
    args = parser.parse_args()
    if ((args.race, args.arm) not in RECIPES
            or not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error('Select a LARS/DART opponent repair and a positive finite maximum duration')
    failed_raw = args.failed_result.read_bytes()
    failed = json.loads(failed_raw)
    stabilization = args.race == 'algos/lars-stable/istella/rows=full'
    direct = args.race == 'algos/lars-direct/istella/rows=full'
    original_race = RACE if stabilization or direct else args.race
    if (failed.get('race') != original_race or failed.get('arm') != args.arm
            or not (str(failed.get('status', '')).startswith('FAILED')
                    or failed.get('status') in ('BUDGET_LIMIT', 'NOT_RUN_BUDGET_EXHAUSTED'))):
        parser.error('Original receipt must name this failed opponent cell; never replay a measured cell')
    raw_budget = args.budget_status.read_bytes()
    original = json.loads(raw_budget)
    started = float(original['started_at'])
    deadline = float(original['deadline_at'])
    if (not math.isfinite(started) or not math.isfinite(deadline)
            or not 0 < deadline-started <= 7200.01):
        raise SystemExit('Original budget timestamps are invalid')
    base = args.evidence
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / 'status.json'
    if status_path.exists() or (base / 'result.json').exists():
        raise SystemExit('Preserve previous retry evidence; use a new directory')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if actual != args.expected_sha:
        raise SystemExit('Retry source differs from the requested frozen commit')
    state = dict(status='WAITING_FOR_MACHINE_LOCK', source_sha=actual, owner_pid=os.getpid(),
                 total=1, completed=0, failed=0, budget_limited=0, cells=[],
                 active_race=args.race, worker_pid=None, allocation_seconds=args.max_seconds,
                 retry_of=str(args.failed_result), retry_of_status=failed['status'],
                 retry_of_sha256=hashlib.sha256(failed_raw).hexdigest(),
                 started_at=started, deadline_at=deadline, retry_requested_at=time.time(),
                 inherited_budget_status=str(args.budget_status),
                 inherited_budget_snapshot_sha256=hashlib.sha256(raw_budget).hexdigest(),
                 budget_seconds=original.get('budget_seconds', deadline-started),
                 evidence=str(base))

    def save():
        state['heartbeat_at'] = time.time()
        atomic(status_path, state)

    def finish(result):
        atomic(base / 'result.json', result)
        measured = result['status'] == 'MEASURED'
        limited = result['status'] in ('BUDGET_LIMIT', 'NOT_RUN_BUDGET_EXHAUSTED')
        state.update(status='COMPLETE' if measured else
                     ('BUDGET_EXHAUSTED' if limited else 'FAILED'),
                     completed=int(measured), budget_limited=int(limited),
                     failed=int(not measured and not limited),
                     cells=[{k: result.get(k) for k in ('race', 'arm', 'status', 'returncode', 'elapsed_seconds')}],
                     active_race=None, worker_pid=None, finished_at=time.time())
        save()
        return 0 if measured else 1

    save()
    atomic(base / 'original-failed-result.json', failed)
    atomic(base / 'original-budget-status.json', original)
    result = dict(race=args.race, arm=args.arm, allocation_seconds=args.max_seconds,
                  retry_of=str(args.failed_result), retry_of_status=failed['status'],
                  retry_of_sha256=hashlib.sha256(failed_raw).hexdigest(),
                  inherited_deadline_at=deadline, source_sha=actual, returncode=None)
    if stabilization:
        result['recipe_change'] = dict(original_race=original_race,
                                      eps_before=2.0**-52, eps_after=2.0**-26,
                                      scope='Separate opt-in recipe; historical LARS unchanged')
    if direct:
        result['recipe_change'] = dict(original_race=original_race,
                                      eps_before=2.0**-52, eps_after=2.0**-52,
                                      precompute_before='auto', precompute_after=False,
                                      scope='Separate sklearn-only direct-X recipe; other arms unsupported')
    # Waiting for the active batch also consumes the inherited total budget.
    # There is no new two-hour clock when this process acquires the machine.
    with (args.runtime / 'gpu.lock').open('a') as lock:
        while True:
            if time.time() >= deadline:
                return finish(dict(result, status='NOT_RUN_BUDGET_EXHAUSTED', elapsed_seconds=0))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save()
                time.sleep(min(2, max(0.01, deadline-time.time())))
        inherited = json.loads(args.budget_status.read_text())
        if float(inherited['deadline_at']) != deadline or float(inherited['started_at']) != started:
            return finish(dict(result, status='FAILED_BUDGET_CHANGED', elapsed_seconds=0))
        block = RECIPES[(args.race, args.arm)]
        data_root = args.data.parent if (args.data / (block+'.json')).exists() else args.data
        recipe = data_root / 'rows-full' / (block+'.json')
        try:
            meta = json.loads(recipe.read_text())
            fit = meta['arrays']['X']['shape']
            query = meta['arrays']['Xq']['shape']
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return finish(dict(result, status='FAILED_FULL_INPUT_METADATA',
                               error=repr(exc), elapsed_seconds=0))
        if (meta.get('full_dataset_coverage') is not True or meta.get('smoke_max_rows')
                or meta.get('fit_rows_available') != fit[0]
                or meta.get('fit_rows') != [0, fit[0]]
                or meta.get('eval_rows_available') != query[0]):
            return finish(dict(result, status='FAILED_FULL_INPUT_METADATA', elapsed_seconds=0))
        for name in ('tmp', 'cache', 'board'):
            (base / name).mkdir(exist_ok=True)
        env = {key: value for key, value in os.environ.items()
               if key not in CAPS and not key.startswith('MOJOLEARN_')}
        env.update(TMPDIR=str(base / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
                   DYLD_LIBRARY_PATH=str(args.runtime / 'runtime-libs'),
                   GBM_BENCH_DATA='/Users/ec2-user/datasets/gbm-bench',
                   REPAIR_RACE_ID=args.race, REPAIR_ARM=args.arm)
        python = str(args.runtime / 'venv/bin/python')
        atomic(base / 'resources.json', dict(cpu_count=os.cpu_count(),
               thread_caps={key: env.get(key) for key in CAPS}, source_sha=actual, harness_sha=actual,
               hardware=subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip(),
               policy='Shared machine lock; serial unrestricted CPU; worker pool readback retained',
               dataset_recipe=str(recipe), dataset_recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
               train_shape=fit, eval_shape=query))
        allowance = min(args.max_seconds, deadline-time.time())
        if allowance <= 0:
            return finish(dict(result, status='NOT_RUN_BUDGET_EXHAUSTED', elapsed_seconds=0))
        driver = ('lars_direct_board.py' if direct else
                  ('lars_stable_board.py' if stabilization else 'selected_board.py'))
        command = [python, '-u', str(Path(__file__).with_name(driver)),
                   '--vendor', 'apple', '--modes', 'identical', '--families', 'algos',
                   '--rows', 'full', '--rounds', '1', '--python-env', python, '--skip-install',
                   '--no-smoke-gate', '--opponents-only', '--skip-failed',
                   '--round-seconds', str(max(1, int(allowance))), '--cache', str(base / 'cache'),
                   '--algos-data', str(data_root), '--data-root', '/Users/ec2-user/datasets/gbm-bench',
                   '--opponent-store', str(base / 'opponent-store.jsonl'), '--out', str(base / 'board')]
        launched = time.time()
        local_deadline = min(deadline, launched + args.max_seconds)
        result.update(command=command, effective_allowance_seconds=local_deadline-launched,
                      cell_started_at=launched, cell_deadline_at=local_deadline)
        killed = []
        with (base / 'retry.log').open('x') as log:
            try:
                process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lock.fileno(),))
            except OSError as exc:
                return finish(dict(result, status='FAILED_LAUNCH', error=repr(exc),
                                   elapsed_seconds=time.time()-launched))
            state.update(status='RUNNING', phase='selected-opponent-retry', worker_pid=process.pid,
                         cell_started_at=launched, cell_deadline_at=local_deadline)
            save()
            while process.poll() is None:
                if time.time() >= local_deadline:
                    killed = stop_tree(process)
                    break
                save()
                time.sleep(min(2, max(0.01, local_deadline-time.time())))
        result.update(returncode=process.returncode, elapsed_seconds=time.time()-launched,
                      killed_owned_pids=killed,
                      status='BUDGET_LIMIT' if killed else ('EXIT_OK' if process.returncode == 0 else 'FAILED'))
        try:
            parsed = coverage(base / 'board/board.json', args.race, args.arm)
        except (ValueError, TypeError, OSError) as exc:
            parsed = dict(status='FAILED_INVALID_RECEIPT', receipt_error=repr(exc))
        result.update({key: value for key, value in parsed.items() if key != 'status'})
        if result['status'] == 'EXIT_OK':
            result['status'] = parsed['status']
        return finish(result)


if __name__ == '__main__':
    raise SystemExit(main())
