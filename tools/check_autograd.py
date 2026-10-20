#!/usr/bin/env python3
"""Compile and validate the native autograd branch; retain complete logs.

Run inside the repository's activated Mojo environment. This is a test
controller, never a product execution path. It does not publish or measure
performance, and a local GPU pass is not cross-vendor identity evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
# The 48 mlp_cases digests (case, epochs, digest), recorded on host, Metal, NVIDIA and AMD.
MLP_DIGESTS = ROOT / 'autograd/checks/mlp_cases_digests.tsv'
<<<<<<< HEAD
# The generated higher-order suite names its expected counts on its first line
# (autograd/checks/generate_higher_oracles.py): '# HIGHER_EXPECT cases=N refusals=K fused=F'.
HIGHER_SUITE = ROOT / 'autograd/checks/higher.mojo'


def higher_expect():
    first = HIGHER_SUITE.read_text().splitlines()[0].split()
    if first[:2] != ['#', 'HIGHER_EXPECT']:
        raise RuntimeError(f'{HIGHER_SUITE} has no HIGHER_EXPECT line (regenerate it)')
    return {k: int(v) for k, v in (field.split('=') for field in first[2:])}
=======
# The rank-N lanes' checks (docs/plans/AUTOGRAD_RANKN.md section 6): each
# prints `RANKN <lane> <case> digest <d>` lines and PASS, and raises on a
# failed comparison. The orchestrator diffs the RANKN lines of CPU, Metal,
# NVIDIA and AMD; here the local CPU and GPU lines must agree.
RANKN_LANES = ('core', 'fold', 'bmm', 'layout', 'conv')
>>>>>>> origin/lane/ag-rankn-python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'gpu', 'both'), default='cpu')
<<<<<<< HEAD
    parser.add_argument('--only', choices=('all', 'gradients', 'mlp', 'infer', 'example', 'higher'), default='all')
=======
    parser.add_argument('--only', choices=('all', 'gradients', 'mlp', 'infer', 'example', 'rankn'), default='all')
>>>>>>> origin/lane/ag-rankn-python
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

    def mlp_digests(path):
        found = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] == 'DIGEST':
                found[int(fields[1])] = fields[2]
        return found

    def stored_mlp_rows():
        rows = [line.split('\t') for line in MLP_DIGESTS.read_text().splitlines()
                if line and not line.startswith('#') and not line.startswith('case')]
        if len(rows) != 48:
            raise RuntimeError(f'{MLP_DIGESTS} holds {len(rows)} cases, expected 48')
        return rows

    def stored_mlp_digests():
        return {int(case): digest for case, _, digest in stored_mlp_rows()}

    def stored_mlp_epochs():
        return {int(case): int(epochs) for case, epochs, _ in stored_mlp_rows()}

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
                # The hand-assembled backward (MOJOLEARN_AUTOGRAD_MLP_OFF) was
                # deleted 2026-10-19; the tape is compared with the stored
                # digests it and the tape both produced on every column.
                source = f'autograd/checks/mlp_{backend}.mojo'
                candidate = native(f'{backend}-mlp-autograd', source)
                _, epochs = mlp_values(candidate)
                got = mlp_digests(candidate)
                want = stored_mlp_digests()
                failures = [{'case': case, 'stored': want.get(case), 'autograd': got.get(case)}
                            for case in sorted(want.keys() | got.keys()) if want.get(case) != got.get(case)]
                failures += [{'case': case, 'stored_epochs': want_epochs, 'autograd_epochs': epochs.get(case)}
                             for case, want_epochs in sorted(stored_mlp_epochs().items()) if epochs.get(case) != want_epochs]
                result = {'backend': backend, 'kind': 'MLP stored-digest regression', 'cases': len(epochs),
                          'digests_compared': len(want), 'reference': str(MLP_DIGESTS.relative_to(ROOT)),
                          'failures': failures}
                report['comparisons'].append(result)
                save()
                if failures:
                    raise RuntimeError(f'{backend} MLP digests differ from the stored reference ({len(failures)})')
        infer_logs = {}
        for backend in backends:
            if args.only in ('all', 'infer'):
                path = native(f'{backend}-infer', f'autograd/checks/infer_{backend}.mojo')
                if 'PASS infer models=7 refusals=4' not in path.read_text():
                    raise RuntimeError('incomplete inference-mode coverage')
                infer_logs[backend] = path
        if len(infer_logs) == 2:
            def infer_digests(path):
                return {line.split()[1]: line.split()[-1] for line in path.read_text().splitlines()
                        if line.startswith('PASS ') and ' digest ' in line}
            a, b = infer_digests(infer_logs['cpu']), infer_digests(infer_logs['gpu'])
            mismatches = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
            report['comparisons'].append({'kind': 'local CPU/GPU inference bit comparison',
                                          'results': len(a), 'mismatches': mismatches})
            if mismatches:
                raise RuntimeError(f'local CPU/GPU inference bits differ: {mismatches}')
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
<<<<<<< HEAD
        if args.only == 'higher':
            # Higher-order gradients (create_graph): the generated suite on each
            # backend; HIGHER_DIGEST <case> <digest> lines are kept per backend
            # for the cross-vendor diff (CPU, Metal, NVIDIA, AMD: 0 differing).
            expect = higher_expect()
            higher = {}
            for backend in backends:
                path = native(f'{backend}-higher', f'autograd/checks/higher_{backend}.mojo')
                text = path.read_text()
                for need in (f"PASS higher-order cases={expect['cases']}", f"PASS higher refusals={expect['refusals']}",
                             'PASS higher lifecycle repeated_graphs=30 release=1'):
                    if need not in text:
                        raise RuntimeError(f'incomplete higher-order coverage: no "{need}" in {path}')
                found = {}
                matches = None
                for line in text.splitlines():
                    fields = line.split()
                    if len(fields) == 3 and fields[0] == 'HIGHER_DIGEST':
                        if fields[1] in found:
                            raise RuntimeError(f'duplicate HIGHER_DIGEST {fields[1]}')
                        found[fields[1]] = fields[2]
                    elif len(fields) == 2 and fields[0] == 'FIRST_ORDER_MATCH':
                        matches = fields[1]
                if len(found) != expect['cases']:
                    raise RuntimeError(f"{backend}: {len(found)} HIGHER_DIGEST lines, expected {expect['cases']}")
                digests = out / f'higher-digests-{backend}.tsv'
                digests.write_text(''.join(f'{case}\t{found[case]}\n' for case in sorted(found)))
                report['comparisons'].append({'backend': backend, 'kind': 'higher-order suite',
                                              'cases': len(found), 'refusals': expect['refusals'],
                                              'fused': expect.get('fused'), 'digests': str(digests),
                                              'first_order_match': matches})
                save()
                higher[backend] = found
            if len(higher) == 2:
                a, b = higher['cpu'], higher['gpu']
                mismatches = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
                report['comparisons'].append({'kind': 'local CPU/GPU HIGHER_DIGEST comparison',
                                              'results': len(a), 'mismatches': mismatches})
                if mismatches:
                    raise RuntimeError(f'local CPU/GPU higher-order bits differ: {mismatches}')
=======
        rankn_logs = {}
        for backend in backends:
            if args.only in ('all', 'rankn'):
                for lane in RANKN_LANES:
                    path = native(f'{backend}-rankn-{lane}', f'autograd/checks/rankn_{lane}_{backend}.mojo')
                    text = path.read_text()
                    lines = [line for line in text.splitlines() if line.startswith(f'RANKN {lane} ')]
                    if not lines or 'PASS' not in text or 'FAIL' in text:
                        raise RuntimeError(f'rankn {lane} ({backend}): no RANKN lines, no PASS, or a FAIL')
                    rankn_logs[(backend, lane)] = sorted(lines)
        for lane in RANKN_LANES:
            if ('cpu', lane) in rankn_logs and ('gpu', lane) in rankn_logs:
                a, b = rankn_logs[('cpu', lane)], rankn_logs[('gpu', lane)]
                mismatches = sorted(set(a) ^ set(b))
                report['comparisons'].append({'kind': f'local CPU/GPU rankn {lane} digest comparison',
                                              'results': len(a), 'mismatches': mismatches})
                if mismatches:
                    raise RuntimeError(f'local CPU/GPU rankn {lane} digests differ: {mismatches[:4]}')
>>>>>>> origin/lane/ag-rankn-python
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
