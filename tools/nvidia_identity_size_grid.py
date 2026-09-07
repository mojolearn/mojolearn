#!/usr/bin/env python3
"""Plan, or explicitly run, bounded root-only NVIDIA IDENTICAL size comparisons.

No dependency installation, compilation, rental or package import. Default is
one planned GEMV-small/incumbent-FAST job. --all explicitly selects the full
prepared grid; unsupported cells never launch. All other public features still
need adapters. Run only from the root/main thread on prepared Linux NVIDIA.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

LANES = ('gemv', 'nt', 'gram', 'knn', 'umap', 'gbdt')
SIZES = ('small', 'medium', 'large')
EXTERNAL = ('fast', 'deterministic')
GIB = 2 ** 30
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def workload(lane, size):
    i = SIZES.index(size)
    options = []
    state = 'PLANNED'
    if lane == 'gemv':
        n = (512, 2048, 8192)[i]
        shape, elements = {'dimension': n}, n * n + n
        options = ['--dim', str(n)]
    elif lane == 'nt':
        n = (1024, 16384, 262144)[i]
        shape, elements = {'rows': n, 'inner': 64, 'output': 64}, n * 64 + 4096
        options = ['--rows', str(n)]
    elif lane == 'gram':
        n = (4096, 65536, 1048576)[i]
        shape, elements = {'observations': n, 'features': 32}, 2 * n * 32
        options = ['--gram-rows', str(n)]
    elif lane == 'knn':
        n, q = ((4096, 64), (65536, 128), (262144, 256))[i]
        shape, elements = {'index': n, 'queries': q, 'features': 32, 'k': 10}, (n + q) * 32
        options = ['--index', str(n), '--queries', str(q), '--k', '10', '--knn-external', 'cuml']
    elif lane == 'umap':
        n = (256, 1024, None)[i]
        shape, elements = {'rows': n, 'features': 16, 'epochs': 50}, (n or 0) ** 2
        options = ['--umap-rows', str(n), '--umap-epochs', '50'] if n else []
        if n is None:
            state = 'ADAPTER_REQUIRED'
    else:
        shape, elements = {'rows': 1024 if i == 0 else None, 'features': 8, 'iterations': 16}, 1024 * 8
        if i:
            state = 'ADAPTER_REQUIRED'
    # Conservative preallocation policy, not a measured peak or library promise.
    # Parent + two resident workers + FP64 oracle + output archives are included.
    host = 2 * GIB + elements * 4 * 16
    gpu = 2 * GIB + elements * 4 * 8
    if lane == 'knn':
        # No proven workspace bound for the incumbent/native selection paths.
        # Reserve 256 bytes per query/index pair; deliberately blocks large kNN.
        host += shape['index'] * shape['queries'] * 256
        gpu += shape['index'] * shape['queries'] * 256
    return shape, options, state, {'host_bytes': host, 'gpu_bytes': gpu,
        'policy': 'Conservative reservation, not measured peak; kNN workspace not certified'}


def plan(args):
    lanes = LANES if args.all else (args.lanes or ('gemv',))
    sizes = SIZES if args.all else (args.sizes or ('small',))
    modes = EXTERNAL if args.all else (args.external_modes or ('fast',))
    jobs = []
    for lane in lanes:
        for size in sizes:
            shape, options, base, estimate = workload(lane, size)
            for external in modes:
                status = base
                if external == 'deterministic' and lane in ('knn', 'umap'):
                    status = 'DETERMINISTIC_SUPPORT_UNVERIFIED'
                elif external == 'deterministic' and lane == 'gbdt':
                    status = 'DOCUMENTED_UNSUPPORTED'
                elif status == 'PLANNED' and estimate['host_bytes'] > 12 * GIB:
                    status = 'MEMORY_ADMISSION_REFUSED'
                name = '-'.join((lane, size, 'external', external))
                output = args.out / name / 'comparison'
                command = [sys.executable, str(ROOT / 'tools/nvidia_public_compare.py'),
                    '--lane', lane, '--out', str(output), '--ours-mode', 'identical-only',
                    '--external-mode', external, '--rounds', '7', *options]
                jobs.append(dict(name=name, lane=lane, size=size, external_mode=external,
                    ours_mode='identical-only', status=status, adapter_status=base,
                    shape=shape, memory_estimate=estimate, command=command,
                    incumbent=('CatBoost GPU' if lane == 'gbdt' else 'cuML' if lane in ('knn', 'umap') else 'PyTorch CUDA/cuBLAS FP32')))
    return dict(schema='mojolearn.nvidia-identity-size-grid.v1', status='PLAN_ONLY', jobs=jobs,
        scope='Selected existing adapters only; all remaining public features require adapters. No cross-vendor or large-size identity certification.',
        timing_boundary='Host-array public API, transfers and returned outputs included; seven rounds plus warmup; rotated arms',
        limits=dict(cpu_cores=2, threads=2, rss_gib=12, gpu_fraction=.85,
                    job_seconds=args.job_seconds, total_seconds=args.total_seconds, reserve_seconds=60),
        sources={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(),
                    ROOT / 'tools/nvidia_public_compare.py', ROOT / 'tools/nvidia_serial_guard.py')})


def admit_result(path, job):
    require(path.stat().st_size <= 8 * 1024 * 1024, 'Oversized comparison metadata')
    record = json.loads(path.read_text())
    require(record.get('status') == 'PASSED' and record.get('ours_mode') == 'identical-only'
            and record.get('active_arms') == ['identical', 'external'], 'Wrong or failed comparison arms')
    require(record.get('external_mode') == job['external_mode']
            and record.get('args', {}).get('lane') == job['lane'], 'Wrong external mode/lane')
    meta = record['metadata']
    require(set(meta) == {'identical', 'external'} and meta['identical'].get('mode') == 'identical',
            'Missing IDENTICAL native mode witness or unexpected our FAST arm')
    for arm in meta.values():
        require(arm.get('cuda') and arm.get('nvidia_smi'), 'Missing NVIDIA runtime witness')
    rows = record['records']
    require(len(rows) == 16 and {(r['arm'], r['round']) for r in rows} ==
            {(a, n) for a in ('identical', 'external') for n in range(8)}, 'Wrong sample inventory')
    require(all(r.get('warmup') is (r['round'] == 0) and type(r.get('ms')) in (int, float)
                and math.isfinite(r['ms']) and r['ms'] > 0 for r in rows), 'Invalid timing sample')
    require(record.get('accuracy') and all(r.get('passed') is True for r in record['accuracy']), 'Accuracy refused')
    require(set(record['summary']) == {'identical', 'external'} and not any(path.parent.glob('fast-*.npz')),
            'Unexpected our FAST evidence')
    require(len({tuple(r['hashes']) for r in rows if r['arm'] == 'identical'}) == 1,
            'IDENTICAL repeated hashes differ')
    if job['external_mode'] == 'deterministic':
        flags = meta['external'].get('external_runtime_flags', {})
        require(flags.get('deterministic_algorithms') is True
                and flags.get('deterministic_warn_only') is False
                and flags.get('cublas_workspace_config') == ':4096:8'
                and flags.get('cuda_matmul_allow_tf32') is False
                and flags.get('cudnn_allow_tf32') is False
                and flags.get('cudnn_deterministic') is True
                and flags.get('cudnn_benchmark') is False, 'Missing strict deterministic incumbent flags')
    return {'results_sha256': sha(path), 'summary': record['summary'], 'scope': record.get('scope')}


def run(args, record):
    require(sys.platform == 'linux' and Path('/proc').is_dir(), 'Execution requires prepared Linux NVIDIA')
    # Import only the stdlib-only supervisor, never a package/model/dependency.
    import nvidia_serial_guard as guard
    deadline = time.monotonic() + args.total_seconds
    active = None
    def interrupted(signum, frame):
        raise InterruptedError('Grid controller interrupted: ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    record['status'] = 'RUNNING'
    write(args.out / 'grid.json', record)
    try:
        for job in record['jobs']:
            if job['status'] != 'PLANNED':
                continue
            remaining = int(deadline - time.monotonic()) - 60
            if remaining < args.job_seconds + 20:
                job['status'] = 'DEADLINE_NOT_LAUNCHED'
                record['status'] = 'INCOMPLETE'
                break
            used, total = guard.gpu_memory()
            _, available = guard.memory(-1)
            estimate = job['memory_estimate']
            if used > 512 or estimate['gpu_bytes'] > total * 1024**2 * .85 - used * 1024**2 or estimate['host_bytes'] > available - 2 * GIB:
                job['status'] = 'MEMORY_ADMISSION_REFUSED'
                job['preflight'] = dict(gpu_used_mib=used, gpu_total_mib=total, host_available_bytes=available)
                continue
            require(all(sha(ROOT / name) == digest for name, digest in record['sources'].items()),
                    'Runner/adapter/guard source changed since plan')
            directory = args.out / job['name']
            directory.mkdir()
            cmd = [sys.executable, str(ROOT / 'tools/nvidia_serial_guard.py'), '--seconds', str(args.job_seconds),
                   '--rss-gib', '12', '--', *job['command']]
            write(directory / 'command.json', cmd)
            (directory / 'command.sh').write_text(shlex.join(cmd) + '\n')
            job['status'] = 'RUNNING'
            write(args.out / 'grid.json', record)
            env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_VENDOR='cuda', PYTHONNOUSERSITE='1')
            with (directory / 'guard.log').open('w') as log:
                active = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                code = active.wait(timeout=args.job_seconds + 20)
                active = None
            (directory / 'exit_code').write_text(str(code) + '\n')
            job['returncode'] = code
            require(code == 0, 'Guard/job failed: ' + job['name'])
            job['admission'] = admit_result(directory / 'comparison/results.json', job)
            job['status'] = 'PASSED'
            write(args.out / 'grid.json', record)
        else:
            record['status'] = 'SELECTED_RUNS_COMPLETE' if all(j['status'] == 'PASSED' for j in record['jobs']) else 'INCOMPLETE_GRID'
    except BaseException as exc:
        if 'job' in locals() and job['status'] == 'RUNNING':
            job['status'] = 'FAILED'
        record['status'] = 'FAILED'
        record['reason'] = repr(exc)
        raise
    finally:
        if active is not None and active.poll() is None:
            # Never SIGKILL the supervisor: it owns a separate worker process
            # group and must perform TERM/KILL cleanup before another job runs.
            active.send_signal(signal.SIGTERM)
            try:
                active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                record['cleanup'] = 'UNVERIFIED_SUPERVISOR_STILL_RUNNING; no subsequent work permitted'
            else:
                record['cleanup'] = 'Supervisor exited after termination request'
        if 'job' in locals() and job.get('status') == 'FAILED':
            directory = args.out / job['name']
            if directory.is_dir() and not (directory / 'exit_code').exists():
                (directory / 'exit_code').write_text(str(active.returncode if active is not None and active.returncode is not None else 124) + '\n')
        write(args.out / 'grid.json', record)
        (args.out / 'exit_code').write_text('0\n' if record['status'] == 'SELECTED_RUNS_COMPLETE' else '1\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New output directory, must not exist')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plan', action='store_true', help='Default: write plan JSON only')
    action.add_argument('--run', action='store_true', help='Root-only explicit prepared NVIDIA execution')
    parser.add_argument('--lanes', nargs='+', choices=LANES)
    parser.add_argument('--sizes', nargs='+', choices=SIZES)
    parser.add_argument('--external-modes', nargs='+', choices=EXTERNAL)
    parser.add_argument('--all', action='store_true', help='Explicitly select all prepared lanes/sizes/external modes')
    parser.add_argument('--ours-mode', choices=['identical-only'], default='identical-only')
    parser.add_argument('--job-seconds', type=int, default=180)
    parser.add_argument('--total-seconds', type=int, default=2400)
    args = parser.parse_args()
    require(1 <= args.job_seconds <= 180 and args.job_seconds + 80 <= args.total_seconds <= 2400, 'Invalid bounded deadline')
    require(not args.all or not any((args.lanes, args.sizes, args.external_modes)), '--all cannot be combined with subsets')
    for values in (args.lanes, args.sizes, args.external_modes):
        require(values is None or len(values) == len(set(values)), 'Duplicate selection')
    args.out = args.out.absolute()
    args.out.mkdir(parents=True, exist_ok=False)
    record = plan(args)
    write(args.out / 'grid.json', record)
    if args.run:
        run(args, record)
    print(json.dumps({'status': record['status'], 'output': str(args.out), 'jobs': len(record['jobs'])}))
    return 0 if not args.run or record['status'] == 'SELECTED_RUNS_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
