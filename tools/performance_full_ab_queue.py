#!/usr/bin/env python3
"""Run frozen full-workload A/B cells serially, with IDENTICAL before FAST.

This controller never compiles, runs a verification fixture, or substitutes a
component driver. Configured blocked cells remain barriers. Each command must
write its declared result JSON, with output hashes and explicit model-state status.

Result contract: schema=mojolearn.full-ab-result/1, status=PASS, source_sha,
dataset_sha256, mode, vendor, arm, phase, dimensions, full_dataset_coverage=true,
timed_boundary matching the job, timings.full_operation_seconds>0, output_sha256,
loaded_artifacts={absolute_path: sha256}, and model_state={status: CAPTURED,
sha256, scope} or {status: UNAVAILABLE, reason}. Applicable fit/inference/cold/
repeated durations belong in timings separately. Model unavailability is retained
and never presented as model identity evidence. Each arm artifact_provenance list
requires path, sha256, numerical_source_sha, compiler, target and defines.
"""
import argparse
import fcntl
import hashlib
import json
import math
import re
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


def check_freeze(repo, source):
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
                                     '--untracked-files=all'], text=True).strip()
    if actual != source or dirty:
        raise ValueError('Frozen checkout changed or is dirty')


def sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def artifact_evidence(job):
    evidence = job['artifact_provenance']
    if not isinstance(evidence, dict) or set(evidence) != {'A', 'B'}:
        raise ValueError('artifact_provenance requires A and B file lists')
    checked = {}
    for arm, artifacts in evidence.items():
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError('Every arm requires concrete retained artifacts')
        checked[arm] = {}
        for entry in artifacts:
            path = Path(entry['path'])
            expected = entry['sha256']
            if (not path.is_absolute() or not path.is_file() or not sha256_value(expected)
                    or digest(path) != expected):
                raise ValueError('Missing or changed retained artifact: ' + str(path))
            if str(path) in checked[arm]:
                raise ValueError('Duplicate retained artifact: ' + str(path))
            if not entry.get('numerical_source_sha') or not entry.get('compiler') or not entry.get('target') or 'defines' not in entry:
                raise ValueError('Artifact requires numerical source, compiler, target and defines')
            checked[arm][str(path)] = expected
    return checked


def embedded_failures(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in ('error', 'errors', 'failure', 'failures', 'failed') and child:
                return True
            if key.lower() in ('status', 'outcome', 'quality_status') and isinstance(child, str):
                if any(term in child.upper() for term in ('FAIL', 'ERROR', 'BLOCKED', 'PENDING', 'TIMEOUT')):
                    return True
            if embedded_failures(child):
                return True
    elif isinstance(value, list):
        return any(embedded_failures(child) for child in value)
    return False


def validate_result(data, job, config, arm, phase, artifacts):
    """Explicit adapter contract; arbitrary JSON and process duration never qualify.

    Workload adapters must emit the full_operation_seconds measurement from their
    declared boundary, plus separate fit/inference/cold/repeated measurements
    where applicable. A model-state hash may be explicitly unavailable; an output
    hash never silently claims to hash model state. These are recorded evidence,
    not independent proof that an adapter implemented its boundary correctly.
    """
    if not isinstance(data, dict) or data.get('schema') != 'mojolearn.full-ab-result/1':
        raise ValueError('Missing full-operation result schema')
    if data.get('status') != 'PASS' or embedded_failures(data):
        raise ValueError('Result reports failure or incomplete work')
    expected = dict(source_sha=config['source_sha'], dataset_sha256=job['dataset_sha256'],
                    mode=job['mode'], vendor=config['vendor'], arm=arm, phase=phase)
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError('Result source/dataset/mode/vendor/arm/phase provenance differs')
    if data.get('dimensions') != job['dimensions'] or data.get('full_dataset_coverage') is not True:
        raise ValueError('Actual full dataset dimensions/coverage differ from declared recipe')
    if data.get('timed_boundary') != job['timed_boundary']:
        raise ValueError('Result boundary differs from declared full operation')
    timings = data.get('timings', {})
    whole = timings.get('full_operation_seconds')
    if isinstance(whole, bool) or not isinstance(whole, (int, float)) or not math.isfinite(whole) or whole <= 0:
        raise ValueError('Missing positive finite full operation duration')
    for name, duration in timings.items():
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid timing: ' + name)
    if not sha256_value(data.get('output_sha256')):
        raise ValueError('Missing complete output SHA256')
    model = data.get('model_state', {})
    if model.get('status') == 'CAPTURED':
        if not sha256_value(model.get('sha256')) or not model.get('scope'):
            raise ValueError('Captured model state requires SHA256 and scope')
    elif model.get('status') == 'UNAVAILABLE':
        if not model.get('reason'):
            raise ValueError('Unavailable model state requires a reason')
    else:
        raise ValueError('Model state must explicitly be CAPTURED or UNAVAILABLE')
    if data.get('loaded_artifacts') != artifacts[arm]:
        raise ValueError('Workload did not attest expected loaded artifact hashes')
    if job.get('master_selection'):
        from six_lane_evidence import validate_master_result
        data['master_qualification'] = validate_master_result(data, job, config, arm, phase)
    return dict(output_sha256=data['output_sha256'], model_state=model,
                timings=timings, model_identity_available=model['status'] == 'CAPTURED')


def run(config, root, retry_failed=False):
    source = config['source_sha']
    repo = Path(config['repo'])
    check_freeze(repo, source)
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
            if results[key]['status'] != 'MEASURED_FULL' and not retry_failed:
                status('BLOCKED_MEASUREMENT_FAILURE', current=key, receipt=results[key]['receipt'])
                return 2
            if results[key]['status'] == 'MEASURED_FULL':
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
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in ('.', '..'):
            raise ValueError('Unsafe cell key')
        if not sha256_value(job['dataset_sha256']):
            raise ValueError('Dataset SHA256 required')
        if job.get('neural_selection'):
            from neural_identical_integration import resolve_job
            job = resolve_job(job, config, repo)
        check_freeze(repo, source)
        artifacts = artifact_evidence(job)
        attempts = root / key / 'attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        previous = sorted(attempts.iterdir())
        if previous and key not in results and not retry_failed:
            raise ValueError('Interrupted attempt exists; use explicit --retry-failed')
        cell = attempts / ('attempt-%04d' % (len(previous) + 1))
        cell.mkdir()
        receipt = dict(source_sha=source, key=key, mode=job['mode'], workload=job,
                       started=time.time(), runs=[], receipt=str(cell / 'receipt.json'),
                       previous_receipt=results.get(key, {}).get('receipt'),
                       warmup_scope='separate_process', steady_state_claim=False)
        failed = False
        for phase in ('warmup', 'scored'):
            for arm in ('A', 'B'):
                spec = job['arms'][arm]
                output = cell / (phase + '-' + arm + '.json')
                log = cell / (phase + '-' + arm + '.log')
                substitutions = {'output': str(output), 'phase': phase, 'arm': arm}
                argv = [part.format_map(substitutions) for part in spec['argv']]
                inherited = dict(os.environ)
                if job.get('neural_selection') or job.get('master_selection'):
                    # Clean experimental state before applying the explicitly
                    # recorded worker setup and frozen A/B controls. Bindings
                    # read many switches once, at import or first use.
                    inherited = {name: value for name, value in inherited.items()
                                 if not name.startswith('MOJOLEARN_')}
                env = dict(inherited, **config.get('environment', {}), **spec.get('environment', {}))
                env.update(MOJOLEARN_NUMERIC_MODE=job['mode'], MOJOLEARN_VENDOR=config['vendor'])
                # Dedicated Apple uses unrestricted pools; Linux uses the actual
                # worker allocation configured by the full-workload driver.
                if config['vendor'] == 'apple':
                    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MOJOLEARN_BENCH_THREADS'):
                        env.pop(name, None)
                check_freeze(repo, source)
                artifact_evidence(job)
                start = time.time()
                with log.open('x') as stream:
                    proc = subprocess.Popen(argv, cwd=repo, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    status('MEASURING', current=key, mode=job['mode'], arm=arm, sample=phase,
                           worker_pid=proc.pid, log=str(log))
                    try:
                        deadline = time.monotonic() + job.get('timeout_seconds', 86400)
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(argv, job.get('timeout_seconds', 86400))
                            try:
                                rc = proc.wait(timeout=min(15, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                status('MEASURING', current=key, mode=job['mode'], arm=arm,
                                       sample=phase, worker_pid=proc.pid, log=str(log))
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
                    try:
                        check_freeze(repo, source)
                        artifact_evidence(job)
                        data = json.loads(output.read_text())
                        record.update(result_sha256=digest(output), result=data)
                        record.update(validate_result(data, job, config, arm, phase, artifacts))
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        record['error'] = str(exc)
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
        scored = [r for r in receipt['runs'] if r['phase'] == 'scored']
        if job.get('neural_selection') and not failed and len(scored) == 2:
            contracts = job['neural_source_selection']['A']['contracts']
            if all(contract == 'S' for contract in contracts.values()):
                same_outputs = scored[0]['output_sha256'] == scored[1]['output_sha256']
                captured = all(r.get('model_identity_available') for r in scored)
                same_models = (not captured or
                               scored[0]['model_state']['sha256'] == scored[1]['model_state']['sha256'])
                receipt['neural_same_arithmetic_ab'] = 'PASS' if same_outputs and same_models else 'FAIL'
                if not same_outputs or not same_models:
                    receipt['error'] = 'An S neural arm changed output or captured model-state bits'
                    failed = True
            else:
                receipt['neural_same_arithmetic_ab'] = 'NOT_REQUIRED_VERSIONED_ARITHMETIC'
        receipt.update(finished=time.time(), status='MEASUREMENT_FAILED' if failed else 'MEASURED_FULL',
                       model_identity_available=len(scored) == 2 and all(r.get('model_identity_available') for r in scored))
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
    parser.add_argument('--retry-failed', action='store_true',
                        help='Retry failed/interrupted cells in fresh attempt directories; retain prior evidence')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(json.loads(args.config.read_text()), args.output, retry_failed=args.retry_failed)
        except Exception as exc:
            write(args.output / 'controller-error.json', dict(error=repr(exc), time=time.time()))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
