#!/usr/bin/env python3
"""Collect frozen full PCA evidence and render it through the board tool.

Only JSON/log files are fetched. Every attempt has its own immutable identity;
failed attempts survive later successful freezes. This never runs a workload.
The collector index/boards live outside the repository; an owner may publish
those rows into the main board without replacing incomparable historical cells.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from fast_quality_rule import judge
from performance_full_ab_queue import validate_result
import performance_measurement_board as board_tool


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def normalized(receipt, path, run):
    """Normalize recorded scored execution; never reinterpret FAST as identity."""
    workload = receipt['workload']
    idea = workload['key'].split('-', 1)[0]
    source = receipt['source_sha']
    attempt_id = run['name'] + '/' + str(path.parent.relative_to(Path(run['local'])))
    cell = dict(id=idea, vendor='apple', mode='fast', route='metal',
                case=workload['key'] + '/' + path.parent.name, scope='full_workload',
                measurement_id=attempt_id, source_sha=source,
                machine=run['machine'], evidence=str(path),
                remote_evidence=receipt.get('receipt'), status='PENDING_COMPLETE_PAIR',
                dataset_sha256=workload.get('dataset_sha256'), dimensions=workload.get('dimensions'),
                estimator_settings=workload.get('estimator_settings'),
                timed_boundary=workload.get('timed_boundary'),
                artifact_provenance=workload.get('artifact_provenance'),
                retained_prerequisites={arm: spec.get('retained_prerequisites', [])
                                        for arm, spec in workload.get('arms', {}).items()},
                warmup_scope=receipt.get('warmup_scope'), steady_state_claim=False,
                model_hash_policy='FAST hashes are reported separately; equality is not required',
                promotion=False)
    records = receipt.get('runs', [])
    cell['retained_runs'] = [{k: r.get(k) for k in ('phase', 'arm', 'returncode', 'error', 'log', 'output')}
                             for r in records]
    scored = {r['arm']: r for r in records if r.get('phase') == 'scored'}
    cell['scored_arm_timings'] = {arm: r.get('result', {}).get('timings') for arm, r in scored.items()}
    cell['model_states'] = {arm: r.get('result', {}).get('model_state') for arm, r in scored.items()}
    cell['output_hashes'] = {arm: r.get('result', {}).get('output_sha256') for arm, r in scored.items()}
    cell['repeated_output_hashes'] = {arm: r.get('result', {}).get('repeated_output_sha256') for arm, r in scored.items()}
    if receipt.get('status') == 'MEASUREMENT_FAILED' or any(r.get('error') or r.get('returncode') != 0 for r in records):
        cell['status'] = 'MEASUREMENT_FAILED'
        return cell
    if receipt.get('status') != 'MEASURED_FULL':
        return cell
    try:
        if source != run['source_sha']:
            raise ValueError('Receipt differs from configured source freeze')
        if workload.get('mode') != 'fast' or idea not in ('F01', 'F11'):
            raise ValueError('Collector supports only declared FAST PCA workloads')
        counts = [(r.get('phase'), r.get('arm')) for r in records]
        if sorted(counts) != sorted((phase, arm) for phase in ('warmup', 'scored') for arm in ('A', 'B')):
            raise ValueError('Require exactly one excluded warmup and scored run per arm')
        artifacts = {arm: {a['path']: a['sha256'] for a in values}
                     for arm, values in workload['artifact_provenance'].items()}
        for r in records:
            validate_result(r['result'], workload, dict(source_sha=source, vendor='apple'),
                            r['arm'], r['phase'], artifacts)
            if r.get('excluded') != (r['phase'] == 'warmup'):
                raise ValueError('Warmup exclusion differs from declared protocol')
        a, b = (scored[arm]['result'] for arm in ('A', 'B'))
        if a.get('idea') != idea or b.get('idea') != idea:
            raise ValueError('Result candidate differs from receipt')
        errors = []
        for data in (a, b):
            quality = data['quality']
            if quality.get('fitted_state_finite') is not True or quality.get('output_finite') is not True:
                raise ValueError('Scored fitted state or output is not finite')
            squared, norm = quality['reconstruction_squared_error'], quality['query_squared_norm']
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (squared, norm)) or squared < 0 or norm <= 0:
                raise ValueError('Invalid reconstruction sums')
            errors.append(math.sqrt(squared / norm))
        if b.get('candidate_reached') is not True:
            raise ValueError('Candidate mechanism was not reached')
        # F01's retained caller contract uses relative L2 reconstruction error.
        # Preserve F11's incomplete auxiliary quality coverage explicitly.
        quality = judge(errors[0], errors[1], rtol=1e-3, atol=1e-6)
        cell['quality'] = dict(reconstruction=quality, finite=True, candidate_reached=True,
                               scope='full query relative L2 reconstruction; other oracle metrics pending')
        hashes = [data['model_state'].get('sha256') for data in (a, b)]
        cell['model_hashes_equal'] = hashes[0] == hashes[1] if all(hashes) else None
        cell['output_hashes_equal'] = a['output_sha256'] == b['output_sha256']
        cell['status'] = 'MEASURED' if quality['ok'] else 'QUALITY_FAILED'
        cell.update(warmups=1, scored_samples=1, returncode=0,
                    baseline_ms=a['timings']['full_operation_seconds'] * 1000,
                    candidate_ms=b['timings']['full_operation_seconds'] * 1000,
                    baseline_machine=run['machine'], candidate_machine=run['machine'],
                    baseline_source_sha=source, candidate_source_sha=source,
                    artifact_hashes=dict(baseline=artifacts['A'], candidate=artifacts['B']))
        cell['pending_coverage'] = ['all affected estimator workloads and interacting configurations',
                                    'full-data singular/noise oracle quality']
        if idea == 'F11':
            cell['variant'] = 'compensated-pca'
            cell['pending_coverage'].append('downstream LLE full-workload mapping and quality')
        else:
            cell['variant'] = 'default'
            cell['equivalent_candidate_mapping'] = dict(id='F02', variant='pca',
                reason='Same corrected job001/job011 pair and PCA workload; reuse evidence, no additional execution')
    except (KeyError, TypeError, ValueError) as exc:
        cell.update(status='RECEIPT_REJECTED', error=str(exc))
    return cell


def collect(config):
    state = Path(config['state'])
    state.mkdir(parents=True, exist_ok=True)
    prior_path = state / 'index.json'
    prior = read(prior_path) if prior_path.exists() else {}
    cells = {c['measurement_id']: c for c in prior.get('cells', [])}
    outcomes = []
    for run in config['runs']:
        destination = Path(run['local'])
        destination.mkdir(parents=True, exist_ok=True)
        outcome = dict(name=run['name'], source_sha=run['source_sha'])
        if run.get('remote'):
            command = ['rsync', '-rlt', '--safe-links', '--no-links', '--include=*/',
                       '--include=*.json', '--include=*.log', '--exclude=*']
            if config.get('rsync_ssh'):
                command += ['-e', config['rsync_ssh']]
            command += [run['remote'].rstrip('/') + '/', str(destination) + '/']
            try:
                with (state / (run['name'] + '-fetch.log')).open('a') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=config.get('fetch_timeout_seconds', 45))
                outcome['fetch_returncode'] = result.returncode
                if result.returncode:
                    outcomes.append(outcome)
                    continue
            except (OSError, subprocess.TimeoutExpired) as exc:
                outcome['error'] = str(exc)
                outcomes.append(outcome)
                continue
        for path in sorted(destination.glob('*/attempts/*/receipt.json')):
            try:
                cell = normalized(read(path), path, run)
                key = cell['measurement_id']
                # Archive every changed normalization without erasing the
                # original downloaded receipt or previous terminal failures.
                digest = hashlib.sha256(json.dumps(cell, sort_keys=True).encode()).hexdigest()
                atomic(state / 'history' / (hashlib.sha256(key.encode()).hexdigest()[:16] + '-' + digest + '.json'), cell)
                old = cells.get(key)
                if (old and old['status'] in ('MEASUREMENT_FAILED', 'QUALITY_FAILED', 'RECEIPT_REJECTED')
                        and old['status'] != cell['status']):
                    retained = dict(old, measurement_id=key + '/prior-' + hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:16])
                    cells[retained['measurement_id']] = retained
                cells[key] = cell
            except (OSError, ValueError, KeyError, TypeError) as exc:
                outcome.setdefault('errors', []).append(dict(path=str(path), error=str(exc)))
        outcomes.append(outcome)
    index = dict(cells=sorted(cells.values(), key=lambda c: c['measurement_id']),
                 machines=sorted(set(prior.get('machines', [])) | {run['machine'] for run in config['runs']}),
                 notes=['Full PCA scored execution only; original failed attempts retained.',
                        'FAST state/output hashes may differ; this is not an IDENTICAL comparison.',
                        'No opponent ratios or default promotions. F11 downstream LLE and auxiliary oracle quality remain pending.'])
    atomic(prior_path, index)
    # Use the repository board tool for every generated board update.
    board = board_tool.build(read(config['inventory']), index)
    board_tool.write(board, state / 'boards')
    atomic(state / 'status.json', dict(status='COLLECTED', updated=time.time(),
                                     cells=len(index['cells']), runs=outcomes,
                                     index=str(prior_path), boards=str(state / 'boards')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    state = Path(config['state']); state.mkdir(parents=True, exist_ok=True)
    with (state / 'collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (state / 'STOP').exists():
            try:
                collect(read(args.config))
            except Exception as exc:
                atomic(state / 'error.json', dict(updated=time.time(), error=repr(exc)))
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(config.get('interval_seconds', 60))


if __name__ == '__main__':
    main()
