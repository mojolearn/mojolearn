#!/usr/bin/env python3
"""Bounded, checkpointed full-property captures for the 23 ordinary holds.

This tool collects evidence; it never promotes a reference or removes a hold.
Use --source for development captures, or --wheel with a Python environment
where that exact wheel is already installed. All wheel package files are checked
before execution. GPU runs additionally record/replay six kernel saved models.
Each invocation has a wall-clock budget and preserves its own stage receipt.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LANES = (
    'mamba3', 'transformer', 'transformer-window', 'samba',
    'samba-untied-dropout-accum', 'gbdt-query-rmse', 'gmm-sample',
    'gmm-random-init-sample', 'gp-normalize-y', 'gp-sample-y',
    'gp-sample-y-normalize', 'gp-optimize', 'gp-optimize-restarts',
    'gpc', 'gpc-multiclass', 'ivf-extend', 'svc-poly',
    'kernel-ridge-poly', 'kernel-ridge-sigmoid', 'kernel-ridge-laplacian',
    'nystroem-poly', 'nystroem-sigmoid', 'nystroem-laplacian',
)
KERNELS = LANES[-6:]
PROPERTIES = ('--batch-grad', '--batch-scale', '--ragged', '--step-full')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, doc):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def verify_wheel(wheel, package):
    """Refuse partial/source/mismatched installs before generating evidence."""
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('mojolearn/') or name.endswith('/'):
                continue
            rel = Path(name).relative_to('mojolearn')
            if '..' in rel.parts:
                raise ValueError('unsafe wheel path')
            installed = package / rel
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            if not installed.is_file() or sha(installed) != expected:
                raise ValueError(f'installed wheel mismatch: {name}')
            checked += 1
    if not checked:
        raise ValueError('wheel has no mojolearn package files')
    return dict(wheel_sha256=sha(wheel), package_files_checked=checked,
                package_path=str(package))


def run_stage(command, log, timeout, env, cwd):
    """Bound the whole process group, including spawned numerical workers."""
    with log.open('wb') as out:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def column_command(python, source, lane, backend, vendor, output, resume):
    harness = [str(ROOT / 'tools/identity_break.py')] if source else ['-m', 'mojolearn._identity_break']
    cmd = [python, *harness, '--lanes', lane, '--repeats', '2',
           '--require-backend', backend, '--vendor', vendor,
           '--fail-on-refused', *PROPERTIES, '--json', str(output)]
    if resume and output.exists():
        cmd.append('--resume')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--source', action='store_true')
    scope.add_argument('--wheel', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'metal', 'cuda', 'hip'), required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, required=True)
    parser.add_argument('--stage-seconds', type=int, default=600)
    parser.add_argument('--lanes', default=','.join(LANES))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.budget_seconds, args.stage_seconds) <= 0:
        parser.error('budgets must be positive')
    lanes = args.lanes.split(',')
    if not lanes or len(set(lanes)) != len(lanes) or set(lanes) - set(LANES):
        parser.error('lanes must be unique members of the 23-hold set')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('output is not empty; use --resume to preserve checkpoints')
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / f'attempt-{time.time_ns()}'
    attempt.mkdir()
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(MOJOLEARN_NUMERIC_MODE='identical', PYTHONNOUSERSITE='1',
               MOJOLEARN_CPU_THREADS='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               BLIS_NUM_THREADS='1', OMP_THREAD_LIMIT='1')
    if args.source:
        env['PYTHONPATH'] = str(ROOT / 'python')
    commit_file = ROOT / 'commit.txt'
    commit = (commit_file.read_text().strip() if commit_file.exists() else
              subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('missing exact source commit witness')
    env['MOJOLEARN_COMMIT'] = commit
    env['MOJOLEARN_GATE_COMMIT'] = commit
    started = time.monotonic()
    receipt = dict(status='INCOMPLETE', source_capture=args.source,
                   backend=args.backend, vendor=args.vendor, lanes=lanes, source_commit=commit,
                   repeats=2, properties=list(PROPERTIES), stages=[],
                   harness_sha256=sha(ROOT / 'tools/identity_break.py'),
                   gate_sha256=sha(ROOT / 'tools/classical_host_gate.py'),
                   promotion='NOT_PERFORMED')
    if args.wheel:
        probe = subprocess.check_output([args.python, '-c',
            'import importlib.util; print(importlib.util.find_spec("mojolearn").origin)'],
            cwd=attempt, env=env, text=True, timeout=30)
        package = Path(probe.strip()).resolve().parent
        receipt.update(verify_wheel(args.wheel.resolve(), package))
        if sha(package / '_identity_break.py') != receipt['harness_sha256']:
            raise ValueError('installed harness differs from saved-model capture tooling')
    context = {k: v for k, v in receipt.items() if k not in ('status', 'stages')}
    context_path = output / 'context.json'
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError('resume context changed; use a new output directory')
    atomic_json(context_path, context)
    report = attempt / 'receipt.json'
    atomic_json(report, receipt)

    def stage(name, command):
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            receipt['stages'].append(dict(name=name, status='BUDGET_EXHAUSTED'))
            atomic_json(report, receipt)
            return False
        code = run_stage(command, attempt / (name + '.log'),
                         min(args.stage_seconds, remaining), env, attempt)
        receipt['stages'].append(dict(name=name, command=command, exit_code=code))
        atomic_json(report, receipt)
        return code == 0

    passed = True
    for lane in lanes:
        column = output / (lane + '.json')
        ok = stage('column-' + lane, column_command(args.python, args.source,
                   lane, args.backend, args.vendor, column, args.resume))
        passed = ok and passed
        if lane in KERNELS and args.backend != 'cpu' and ok:
            # Never overwrite a partial or earlier GPU model recording.
            models = attempt / ('saved-' + lane)
            gate = [args.python, str(ROOT / 'tools/classical_host_gate.py'),
                    '--package-root', str(ROOT / 'python') if args.source else '']
            recorded = stage('record-' + lane, [*gate, 'record', str(models), '--lanes', lane])
            passed = recorded and passed
            if recorded:
                replayed = stage('replay-' + lane, [*gate, 'check', str(models),
                    '--gpu-column', str(column), '--report', str(attempt / (lane + '-cpu-replay.json'))])
                passed = replayed and passed
    receipt['status'] = 'CAPTURED_UNQUALIFIED' if passed else 'INCOMPLETE'
    receipt['seconds'] = time.monotonic() - started
    atomic_json(report, receipt)
    print(report)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
