#!/usr/bin/env python3
"""Run frozen full-workload A/B cells serially, with IDENTICAL before FAST.

This controller never compiles, runs a verification fixture, or substitutes a
component driver. Configured blocked cells remain barriers. Each command must
write its declared result JSON, including model hashes captured by the workload.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def hashes(value):
    """Find explicitly named model/output hashes, never artifact-file hashes."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('model_sha256', 'model_hash', 'model_state_sha256',
                       'output_sha256', 'output_digest', 'model_digest') and child:
                found.append({'field': key, 'value': child})
            else:
                found.extend(hashes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(hashes(child))
    return found


def run(config, root):
    source = config['source_sha']
    repo = Path(config['repo'])
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != source:
        raise ValueError('Frozen checkout changed: ' + actual)
    results_path = root / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    if any(result['source_sha'] != source for result in results.values()):
        raise ValueError('Cannot reuse another freeze in this result directory')
    jobs = config['jobs']
    if len({job['key'] for job in jobs}) != len(jobs):
        raise ValueError('Duplicate cell key')
    ordered = sorted(jobs, key=lambda job: 0 if job['mode'] == 'identical' else 1)
    def status(phase, **extra):
        write(root / 'status.json', dict(phase=phase, pid=os.getpid(), updated=time.time(),
              source_sha=source, total=len(jobs), completed=sum(r['status'] == 'MEASURED_FULL' for r in results.values()),
              compilation='not run; reuse accepted artifacts', **extra))
    for job in ordered:
        key = job['key']
        if key in results:
            if results[key]['status'] != 'MEASURED_FULL':
                status('BLOCKED_MEASUREMENT_FAILURE', current=key, receipt=results[key]['receipt'])
                return 2
            continue
        if job.get('blocked'):
            status('BLOCKED', current=key, mode=job['mode'], reason=job['blocked'],
                   pending=[{'key': j['key'], 'mode': j['mode'], 'blocked': j.get('blocked')} for j in ordered if j['key'] not in results])
            return 2
        required = ('dataset_sha256', 'dimensions', 'estimator_settings', 'timed_boundary',
                    'intrinsic_caps', 'full_dataset_coverage', 'artifact_provenance')
        for field in required:
            if field not in job:
                raise ValueError(key + ': missing workload evidence ' + field)
        if job['full_dataset_coverage'] is not True:
            raise ValueError(key + ': full dataset coverage remains pending')
        if job['mode'] not in ('identical', 'fast') or set(job['arms']) != {'A', 'B'}:
            raise ValueError(key + ': invalid mode or A/B arms')
        cell = root / key
        cell.mkdir(parents=True, exist_ok=True)
        receipt = dict(source_sha=source, key=key, mode=job['mode'], workload=job,
                       started=time.time(), runs=[], receipt=str(cell / 'receipt.json'))
        failed = False
        for phase in ('warmup', 'scored'):
            for arm in ('A', 'B'):
                spec = job['arms'][arm]
                output = cell / (phase + '-' + arm + '.json')
                log = cell / (phase + '-' + arm + '.log')
                substitutions = {'output': str(output), 'phase': phase, 'arm': arm}
                argv = [part.format_map(substitutions) for part in spec['argv']]
                env = dict(os.environ, **config.get('environment', {}), **spec.get('environment', {}))
                env.update(MOJOLEARN_NUMERIC_MODE=job['mode'], MOJOLEARN_VENDOR=config['vendor'])
                # Dedicated Apple uses unrestricted pools; Linux uses the actual
                # worker allocation configured by the full-workload driver.
                if config['vendor'] == 'apple':
                    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MOJOLEARN_BENCH_THREADS'):
                        env.pop(name, None)
                start = time.time()
                with log.open('x') as stream:
                    proc = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    status('MEASURING', current=key, mode=job['mode'], arm=arm, sample=phase,
                           worker_pid=proc.pid, log=str(log))
                    try:
                        rc = proc.wait(timeout=job.get('timeout_seconds', 86400))
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        rc = 124
                record = dict(phase=phase, arm=arm, excluded=phase == 'warmup', returncode=rc,
                              process_wall_seconds=time.time() - start, log=str(log), output=str(output))
                # Process wall time is provenance only. The declared workload
                # result owns operation, fit and inference timing boundaries.
                if rc == 0 and output.exists():
                    data = json.loads(output.read_text())
                    record.update(result=data, result_sha256=digest(output), model_hashes=hashes(data))
                    if phase == 'scored' and not record['model_hashes']:
                        record['error'] = 'Scored result has no model/output hash'
                        failed = True
                else:
                    record['error'] = 'Workload failed or did not write result JSON'
                    failed = True
                receipt['runs'].append(record)
                write(cell / 'receipt.json', receipt)
                if failed:
                    break
            if failed:
                break
        receipt.update(finished=time.time(), status='MEASUREMENT_FAILED' if failed else 'MEASURED_FULL')
        write(cell / 'receipt.json', receipt)
        results[key] = receipt
        write(results_path, results)
        if failed:
            status('BLOCKED_MEASUREMENT_FAILURE', current=key, receipt=receipt['receipt'])
            return 2
    status('COMPLETE' if jobs else 'BLOCKED_NO_FULL_WORKLOAD_RECIPES')
    return 0 if jobs else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(json.loads(args.config.read_text()), args.output)
        except Exception as exc:
            write(args.output / 'controller-error.json', dict(error=repr(exc), time=time.time()))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
