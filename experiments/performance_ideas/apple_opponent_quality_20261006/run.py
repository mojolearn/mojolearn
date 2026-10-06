"""Measure missing full Apple opponents serially within one two-hour allowance.

Reuse accepted inputs and installed libraries. No compilation, installation,
smoke checks, estimator preflight, or changes to estimator settings.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
REPAIR = SOURCE / 'experiments/performance_ideas/apple_repair_20261006'
sys.path.insert(0, str(REPAIR))
# Reuse the existing repair controller's process-tree stop and receipt helpers.
import importlib.util
spec = importlib.util.spec_from_file_location('prior_repair_controller', REPAIR / 'run.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sys.modules['run'] = prior
from retry import coverage

CAPS = tuple(set(prior.CAPS) | {'OMP_THREAD_LIMIT', 'NUMBA_NUM_THREADS',
             'NUMEXPR_MAX_THREADS', 'GOTO_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'})
atomic, stop_tree = prior.atomic, prior.stop_tree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--full-big', type=Path, required=True)
    p.add_argument('--full-reg', type=Path, required=True)
    p.add_argument('--expected-sha', required=True)
    args = p.parse_args()
    base = args.evidence
    base.mkdir(parents=True, exist_ok=True)
    if (base / 'status.json').exists():
        raise SystemExit('Preserve previous attempts and their immutable budgets')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    if sha != args.expected_sha:
        raise SystemExit('Source differs from requested frozen commit')
    plan_path = Path(__file__).with_name('plan.json')
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    state = dict(status='WAITING_FOR_MACHINE_LOCK', source_sha=sha, owner_pid=os.getpid(),
                 budget_seconds=7200, total=len(plan['cells']), completed=0, failed=0,
                 budget_limited=0, cells=[], evidence=str(base),
                 plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
                 excluded_known_unsupported=plan['blocked_roster'],
                 original_two_hour_campaign_unchanged=True)

    def save():
        state['heartbeat_at'] = time.time()
        atomic(base / 'status.json', state)

    atomic(base / 'frozen-plan.json', plan)
    save()
    with (args.runtime / 'gpu.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save()
                time.sleep(2)
        started = time.time()
        state.update(started_at=started, deadline_at=started+7200, status='PREPARING_SYMLINKS')
        save()
        for directory in ('tmp', 'cache', 'cells', 'inputs/ctd/rows-full', 'inputs/reg/rows-full'):
            (base / directory).mkdir(parents=True, exist_ok=True)
        env = {k:v for k,v in os.environ.items() if k not in CAPS and not k.startswith('MOJOLEARN_')}
        env.update(TMPDIR=str(base / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
                   DYLD_LIBRARY_PATH=str(args.runtime / 'runtime-libs'),
                   GBM_BENCH_DATA='/Users/ec2-user/datasets/gbm-bench')
        python = str(args.runtime / 'venv/bin/python')
        atomic(base / 'resources.json', dict(cpu_count=os.cpu_count(),
               hardware=subprocess.check_output(['sysctl','-n','hw.model'],text=True).strip(),
               source_sha=sha, harness_sha=sha, thread_caps={k:env.get(k) for k in CAPS},
               policy='Exclusive shared machine lock; serial unrestricted CPU/GPU; effective worker pools retained'))
        inputs = {}
        try:
            for block, dataset, root, destination in (
                    ('big', 'taxi', args.full_big, base / 'inputs/ctd/rows-full'),
                    ('reg', 'taxi', args.full_reg, base / 'inputs/reg/rows-full'),
                    ('reg', 'istella', args.full_reg, base / 'inputs/reg/rows-full')):
                stem = block+'-'+dataset
                original = root / (stem+'.json')
                raw = original.read_bytes()
                rec = json.loads(raw)
                shape = [5250086, 11] if dataset == 'taxi' else [2043304, 220]
                if (rec['arrays']['X']['shape'] != shape
                        or rec['arrays']['Xq']['shape'] != [500000, shape[1]]
                        or rec.get('fit_rows_available') != shape[0]
                        or rec.get('fit_rows') != [0, shape[0]] or rec.get('smoke_max_rows')):
                    raise ValueError('Existing recipe does not attest expected full workload: '+stem)
                for suffix in ('.npz', '.json'):
                    source = (root / (stem+suffix)).resolve(strict=True)
                    target = destination / (stem+suffix)
                    if target.exists() or target.is_symlink():
                        raise ValueError('Refuse to replace an input alias: '+str(target))
                    target.symlink_to(source)
                inputs[stem] = dict(source_npz=str((root / (stem+'.npz')).resolve()),
                     source_recipe=str(original), recipe_sha256=hashlib.sha256(raw).hexdigest(),
                     arrays=rec['arrays'], data_unchanged=True)
            atomic(base / 'input-receipts.json', inputs)
        except Exception as exc:
            state.update(status='BLOCKED_FULL_INPUT_METADATA', error=repr(exc), finished_at=time.time())
            save()
            return 1
        for index, cell in enumerate(plan['cells'], 1):
            label = 'cell-%02d' % index
            out = base / 'cells' / label
            out.mkdir()
            race, arm = cell['race'], cell['arm']
            allowance = min(cell['seconds'], state['deadline_at']-time.time())
            result = dict(race=race, arm=arm, source_sha=sha, allocation_seconds=cell['seconds'],
                          effective_allowance_seconds=max(0, allowance), returncode=None,
                          expected_fit_shape=cell['fit_shape'], expected_eval_shape=cell['eval_shape'],
                          input_receipt=inputs[cell['block']+'-'+cell['dataset']],
                          status='NOT_RUN_BUDGET_EXHAUSTED')
            if allowance > 0:
                cell_env = dict(env, REPAIR_RACE_ID=race, REPAIR_ARM=arm)
                command = [python, '-u', str(REPAIR / 'selected_board.py'),
                           '--vendor','apple','--modes','identical','--families',cell['family'],
                           '--rows','full','--rounds','1','--python-env',python,
                           '--skip-install','--no-smoke-gate','--opponents-only','--skip-failed',
                           '--round-seconds',str(max(1,int(allowance))), '--cache',str(base/'cache'),
                           '--ctd-data',str(base/'inputs/ctd'),'--more-data',str(base/'inputs/reg'),
                           '--algos-data',str(base/'inputs/reg'), '--data-root','/Users/ec2-user/datasets/gbm-bench',
                           '--opponent-store',str(out/'opponent-store.jsonl'),'--out',str(out/'board')]
                cell_start = time.time()
                deadline = min(state['deadline_at'], cell_start+allowance)
                killed = []
                state.update(status='RUNNING', active_race=race, active_arm=arm, phase=label)
                save()
                try:
                    with (out / 'run.log').open('x') as log:
                        process = subprocess.Popen(command, cwd=SOURCE, env=cell_env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                        state['worker_pid'] = process.pid
                        while process.poll() is None:
                            if time.time() >= deadline:
                                killed = stop_tree(process)
                                break
                            save()
                            time.sleep(min(2,max(0.01,deadline-time.time())))
                    result.update(command=command, returncode=process.returncode,
                                  elapsed_seconds=time.time()-cell_start, killed_owned_pids=killed,
                                  status='BUDGET_LIMIT' if killed else 'FAILED')
                    parsed = coverage(out/'board/board.json', race, arm)
                    result.update({k:v for k,v in parsed.items() if k != 'status'})
                    if not killed and process.returncode == 0:
                        result['status'] = parsed['status']
                except Exception as exc:
                    result.update(status='FAILED_CONTROLLER', error=repr(exc),
                                  elapsed_seconds=time.time()-cell_start)
            atomic(out/'result.json', result)
            state['cells'].append({k:result.get(k) for k in ('race','arm','status','returncode','elapsed_seconds')})
            state['completed'] += result['status']=='MEASURED'
            state['budget_limited'] += result['status'] in ('BUDGET_LIMIT','NOT_RUN_BUDGET_EXHAUSTED')
            state['failed'] += result['status'].startswith('FAILED')
            state['worker_pid'] = None
            save()
        state.update(status='COMPLETE' if state['completed']==len(plan['cells']) else 'COMPLETE_WITH_UNRESOLVED_CELLS',
                     finished_at=time.time(), active_race=None, active_arm=None, worker_pid=None)
        save()
        return int(state['completed'] != len(plan['cells']))


if __name__ == '__main__':
    raise SystemExit(main())
