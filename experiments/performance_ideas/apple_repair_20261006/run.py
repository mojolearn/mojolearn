"""Serial M3 opponent repairs with a persistent, two-hour total deadline.

This is measurement orchestration only. Existing libraries are reused; no
compilation, installation, smoke or numerical verification is launched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[3]
PLAN = [
    ('algos/qn-reg/taxi/rows=full', 'sklearn-cpu', 300),
    ('algos/qn-reg/istella/rows=full', 'sklearn-cpu', 600),
    ('classical2/gmm/taxi/rows=full', 'sklearn-cpu', 2400),
    ('algos/lars/istella/rows=full', 'sklearn-cpu', 120),
    ('algos/dart/istella/rows=full', 'xgboost-cpu', 1800),
    ('algos/dart-reg/istella/rows=full', 'xgboost-cpu', 1800),
]
CAPS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def stop_tree(process):
    """Kill only this cell's descendants, including workers with their own session."""
    parents = {}
    for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True).splitlines():
        pid, parent = map(int, line.split())
        parents[pid] = parent
    owned = {process.pid}
    while True:
        expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
        if expanded == owned:
            break
        owned = expanded
    # Stop creation before killing; do not signal a shared shell/process group.
    for pid in owned:
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
    for pid in sorted(owned - {process.pid}) + [process.pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)
    return sorted(owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    args = parser.parse_args()
    base = args.evidence
    base.mkdir(parents=True, exist_ok=True)
    status_path = base / 'status.json'
    if status_path.exists():
        raise SystemExit('Preserve the previous attempt; this controller never resets an existing budget.')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert actual == args.expected_sha
    state = dict(status='WAITING_FOR_MACHINE_LOCK', source_sha=actual, owner_pid=os.getpid(),
                 total=len(PLAN), completed=0, failed=0, budget_limited=0,
                 budget_seconds=7200, cells=[], evidence=str(base))

    def save():
        state['heartbeat_at'] = time.time()
        atomic(status_path, state)

    def launch(command, label, allowance, env):
        seconds = min(allowance, state['deadline_at'] - time.time())
        if seconds <= 0:
            return dict(returncode=None, status='NOT_RUN_BUDGET_EXHAUSTED')
        state.update(status='RUNNING', phase=label)
        save()
        started = time.time()
        deadline = min(started + seconds, state['deadline_at'])
        with (base / (label + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
            state['worker_pid'] = process.pid
            save()
            killed = []
            while process.poll() is None:
                if time.time() >= deadline:
                    killed = stop_tree(process)
                    break
                save()
                time.sleep(min(2, max(0.05, deadline-time.time())))
        state['worker_pid'] = None
        save()
        return dict(returncode=process.returncode, elapsed_seconds=time.time()-started,
                    status='BUDGET_LIMIT' if killed else ('EXIT_OK' if process.returncode == 0 else 'FAILED'),
                    killed_owned_pids=killed, command=command)

    save()
    with (args.runtime / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state.update(started_at=time.time(), deadline_at=time.time()+7200)
        env = {k:v for k,v in os.environ.items() if k not in CAPS and not k.startswith('MOJOLEARN_')}
        for name in ['tmp', 'cache', 'data', 'cells']:
            (base/name).mkdir(exist_ok=True)
        env.update(TMPDIR=str(base/'tmp'), PYTHONDONTWRITEBYTECODE='1',
                   DYLD_LIBRARY_PATH=str(args.runtime/'runtime-libs'),
                   GBM_BENCH_DATA='/Users/ec2-user/datasets/gbm-bench')
        python = str(args.runtime/'venv/bin/python')
        atomic(base/'resources.json', dict(cpu_count=os.cpu_count(), thread_caps={k:env.get(k) for k in CAPS},
               hardware=subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip(),
               harness_sha=actual, source_sha=actual, policy='Serial full-machine uncapped CPU; worker resource readback retained.'))
        prep = launch([python, str(Path(__file__).with_name('prepare.py')),
                       '--input', str(args.data), '--output', str(base/'data/rows-full')],
                      'prepare', 180, env)
        atomic(base/'prepare-result.json', prep)
        if prep['returncode'] != 0:
            state.update(status='BLOCKED_MEASUREMENT_FAILURE', error='Full-input preparation failed', prepare=prep)
            save()
            return 1
        for index, (race, arm, allowance) in enumerate(PLAN):
            label = 'cell-%02d' % (index+1)
            out = base/'cells'/label
            out.mkdir(exist_ok=True)
            state['active_race'] = race
            cell_env = dict(env, REPAIR_RACE_ID=race, REPAIR_ARM=arm)
            command = [python, '-u', str(Path(__file__).with_name('selected_board.py')),
                       '--vendor','apple','--modes','identical','--families','algos,classical2',
                       '--rows','full','--rounds','1','--python-env',python,'--skip-install',
                       '--no-smoke-gate','--opponents-only','--skip-failed','--round-seconds',str(allowance),
                       '--cache',str(base/'cache'),'--algos-data',str(base/'data'),
                       '--more-data',str(base/'data'),'--data-root','/Users/ec2-user/datasets/gbm-bench',
                       '--opponent-store',str(out/'opponent-store.jsonl'),'--out',str(out/'board')]
            result = dict(race=race, arm=arm, allocation_seconds=allowance,
                          **launch(command,label,allowance,cell_env))
            board_path = out/'board/board.json'
            if board_path.exists():
                board = json.loads(board_path.read_text())
                record = board.get('races',{}).get(race,{})
                result['race_status'] = record.get('status')
                result['cells'] = record.get('cells',[])
                if result['status']=='EXIT_OK' and record.get('status')=='done':
                    result['status']='MEASURED'
                elif result['status']=='EXIT_OK':
                    result['status']='FAILED_INCOMPLETE_COVERAGE'
            elif result['status']=='EXIT_OK':
                result['status']='FAILED_MISSING_RECEIPT'
            atomic(out/'result.json',result)
            state['cells'].append({k:result.get(k) for k in ['race','arm','status','returncode','elapsed_seconds']})
            state['completed'] += result['status']=='MEASURED'
            state['budget_limited'] += result['status'] in ('BUDGET_LIMIT','NOT_RUN_BUDGET_EXHAUSTED')
            state['failed'] += result['status'].startswith('FAILED')
            save()
        state.update(status='COMPLETE' if state['completed']==len(PLAN) else 'COMPLETE_WITH_UNRESOLVED_CELLS',
                     finished_at=time.time(), active_race=None, worker_pid=None)
        save()
        return int(state['completed'] != len(PLAN))


if __name__=='__main__':
    raise SystemExit(main())
