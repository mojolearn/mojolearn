#!/usr/bin/env python3
"""Compile and validate the native autograd branch; retain complete logs.

Run inside the repository's activated Mojo environment. This is a test
controller, never a product execution path. It does not publish or measure
performance, and a local GPU pass is not cross-vendor identity evidence.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'gpu', 'both'), default='cpu')
    parser.add_argument('--only', choices=('all', 'gradients', 'mlp', 'example'), default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mojo', default='mojo')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mojo = shutil.which(args.mojo)
    if not mojo:
        parser.error('mojo must be on PATH (run through pixi run)')
    report = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
              'scope': 'Correctness tests only; no performance or cross-vendor qualification.',
              'stages': [], 'comparisons': [], 'status': 'running'}
    report_path = out / 'summary.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    def stage(name, command):
        log = out / (name + '.log')
        start = time.monotonic()
        result = {'name': name, 'command': command, 'log': str(log)}
        report['stages'].append(result)
        save()
        with log.open('w') as stream:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
                result['exit_code'] = completed.returncode
            except subprocess.TimeoutExpired:
                result['exit_code'] = 124
                result['timeout'] = True
        result['elapsed_seconds'] = round(time.monotonic() - start, 3)
        save()
        print(json.dumps({'stage': name, 'exit_code': result['exit_code'], 'log': str(log)}), flush=True)
        if result['exit_code'] != 0:
            raise RuntimeError(f'{name} failed; see {log}')
        return log

    def native(name, source, defines=()):
        binary = out / name
        command = ['bash', str(ROOT / 'tools/with_build_lock.sh'), mojo, 'build', '-j', '2',
                   '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1', '-I', str(ROOT)]
        for define in defines:
            command += ['-D', define]
        command += [str(ROOT / source), '-o', str(binary)]
        stage(name + '-build', command)
        return stage(name + '-run', ['bash', str(ROOT / 'tools/with_build_lock.sh'), str(binary)])

    def mlp_values(path):
        values, epochs = {}, {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] in ('PARAM', 'LOSS'):
                key = tuple(fields[:3])
                if key in values:
                    raise RuntimeError(f'duplicate MLP output {key}')
                values[key] = float(fields[3])
            elif fields and fields[0] == 'CASE':
                epochs[int(fields[1])] = int(fields[3])
        if len(epochs) != 48 or 'PASS mlp_cases 48' not in path.read_text():
            raise RuntimeError('incomplete MLP case coverage')
        return values, epochs

    try:
        backends = ['cpu', 'gpu'] if args.backend == 'both' else [args.backend]
        gradient_logs = {}
        for backend in backends:
            if args.only in ('all', 'gradients'):
                path = native(f'{backend}-gradients', f'autograd/checks/{backend}.mojo')
                text = path.read_text()
                if 'PASS finite-difference cases=35' not in text or 'PASS lifecycle refusals=10 repeated_graphs=30' not in text:
                    raise RuntimeError('incomplete gradient/lifecycle coverage')
                gradient_logs[backend] = path
            if args.only in ('all', 'mlp'):
                source = f'autograd/checks/mlp_{backend}.mojo'
                baseline = native(f'{backend}-mlp-existing', source)
                candidate = native(f'{backend}-mlp-autograd', source, ['MOJOLEARN_AUTOGRAD_MLP=1'])
                a, ea = mlp_values(baseline)
                b, eb = mlp_values(candidate)
                if a.keys() != b.keys() or ea != eb:
                    raise RuntimeError('MLP training coverage or stopping behavior differs')
                failures = []
                max_abs = 0.0
                for key in a:
                    delta = abs(a[key] - b[key])
                    max_abs = max(max_abs, delta)
                    if not math.isfinite(delta) or delta > 1e-5 + 1e-5 * abs(a[key]):
                        failures.append({'key': key, 'existing': a[key], 'autograd': b[key]})
                result = {'backend': backend, 'kind': 'MLP numerical regression', 'cases': len(ea),
                          'values_compared': len(a), 'max_absolute_difference': max_abs, 'failures': failures}
                report['comparisons'].append(result)
                save()
                if failures:
                    raise RuntimeError(f'{backend} MLP regression failed ({len(failures)} values)')
        if len(gradient_logs) == 2:
            def digests(path):
                return {line.split()[1]: line.split()[-1] for line in path.read_text().splitlines()
                        if line.startswith('PASS ') and ' digest ' in line}
            a, b = digests(gradient_logs['cpu']), digests(gradient_logs['gpu'])
            mismatches = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
            report['comparisons'].append({'kind': 'local CPU/GPU gradient fixture bit comparison',
                                          'results': len(a), 'mismatches': mismatches})
            if mismatches:
                raise RuntimeError(f'local CPU/GPU fixture bits differ: {mismatches}')
        if args.only in ('all', 'example'):
            native('custom-regression', 'autograd/examples/custom_regression.mojo')
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
    save()
    print(json.dumps({'status': report['status'], 'summary': str(report_path)}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
