#!/usr/bin/env python3
"""CPU inference gate for the byte LM (DEVIATION 2613).

Loads parameters from the retained three-vendor capture and requires the CPU
forward, through the public `LanguageModelInference` surface, to reproduce
the recorded loss BYTES:

  heldout-initial  8 batches  parameters full128/initial/initial_p.f32
  heldout-final    8 batches  parameters full128/step000128/post_p.f32
  training steps   N steps    parameters stepNNNNNN/initial_p.f32, ids.i32, loss.f32

The capture is Apple's. Its comparison.json records that CUDA and HIP matched
it on every one of these bytes, so a match here is a match against all three.
Every file is checked against the SHA-256 its own manifest recorded before it
is used; a reference that does not verify stops the gate (exit 2).

It also records SHA-256 of the logits of fixed probes, so two CPUs can be
compared at full [batch, length, vocab] resolution and not only through the
loss.

Exit 0: every byte equal. 1: any mismatch. 2: the gate could not run.
--expect-mismatch inverts the verdict for the DEVIATION 2612 sabotage build:
exit 0 only if at least one loss differs.
"""
import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / 'bench/results/resume/2026-09-07-root-byte-lm-three-vendor/apple'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(path, want, what):
    got = sha256(path)
    if got != want:
        raise SystemExit(f'gate: {what} {path} sha256 {got} != recorded {want}')
    return Path(path).read_bytes()


def host_info():
    info = dict(machine=platform.machine(), system=platform.system(),
                release=platform.release(), python=platform.python_version())
    try:
        if platform.system() == 'Darwin':
            info['cpu'] = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                         capture_output=True, text=True, timeout=10).stdout.strip()
        else:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.lower().startswith(('model name', 'cpu model')):
                    info['cpu'] = line.split(':', 1)[1].strip()
                    break
            flags = [l for l in Path('/proc/cpuinfo').read_text().splitlines()
                     if l.startswith(('flags', 'Features'))]
            if flags:
                have = set(flags[0].split(':', 1)[1].split())
                info['isa'] = sorted(have & {'avx2', 'fma', 'avx512f', 'asimd', 'sve', 'sve2'})
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def git_commit():
    env = os.environ.get('MOJOLEARN_GATE_COMMIT')
    if env:
        return env
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_steps(spec, available):
    if spec == 'all':
        return available
    if spec.startswith('every:'):
        stride = int(spec.split(':', 1)[1])
        return [s for s in available if s == 1 or s % stride == 0]
    return [int(s) for s in spec.split(',') if s]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument('--steps', default='all', help="'all', 'every:N' or a comma list")
    parser.add_argument('--report', type=Path, help='new exclusive JSON report')
    parser.add_argument('--expect-mismatch', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / 'python'))
    # The package imports under its default tier through the CPU-only path of
    # DEVIATION 2615, which is part of what this gate exercises. The binding
    # under test is copied to python/mojolearn/host/ by the leg before this
    # runs, because that path is how `_backend` recognizes a CPU-only install.
    try:
        from mojolearn._buffer import frombytes
        from mojolearn._byte_lm_config import ByteLanguageModelConfig
        from mojolearn._byte_lm_host import LanguageModelInference, binary_path
    except Exception as exc:  # the import itself is part of what is gated
        print(f'gate: import failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2

    shape = ByteLanguageModelConfig()
    run = args.capture / 'full128'
    comparison = json.loads((args.capture.parent / 'comparison.json').read_text())
    if comparison.get('identity_admitted') is not True:
        print('gate: capture comparison.json does not admit identity', file=sys.stderr)
        return 2

    def params_from(path, want):
        raw = verified(path, want, 'parameters')
        return frombytes(raw, '<f4', (shape.n_total,))

    models = {}

    def model_for(path, want):
        key = str(path)
        if key not in models:
            models[key] = LanguageModelInference(params_from(path, want), shape=shape)
        return models[key]

    rows = []

    def check(label, model, ids_path, ids_sha, loss_path, loss_sha):
        ids = frombytes(verified(ids_path, ids_sha, 'ids'), '<i4', (shape.batch, shape.length + 1))
        want = struct.unpack('<I', verified(loss_path, loss_sha, 'loss'))[0]
        started = time.perf_counter()
        got = model.loss_bits(ids, threaded=False)
        seconds = time.perf_counter() - started
        started = time.perf_counter()
        got_thr = model.loss_bits(ids, threaded=True)
        seconds_thr = time.perf_counter() - started
        rows.append(dict(case=label, want=f'{want:08x}', got=f'{got:08x}', got_threaded=f'{got_thr:08x}',
                         equal_reference=got == want, equal_threaded=got_thr == want,
                         equal=got == want and got_thr == want,
                         seconds=round(seconds, 6), seconds_threaded=round(seconds_thr, 6)))

    try:
        for phase in ('heldout-initial', 'heldout-final'):
            evaluation = json.loads((run / phase / 'evaluation.json').read_text())
            before = evaluation['state_before']['arrays']['parameters']
            source = run / 'initial/initial_p.f32' if phase == 'heldout-initial' else run / 'step000128/post_p.f32'
            model = model_for(source, before)
            for i, batch in enumerate(evaluation['batches']):
                stem = run / phase / f'batch{i:02d}'
                check(f'{phase}/batch{i:02d}', model, f'{stem}.ids.i32', batch['ids_sha256'],
                      f'{stem}.loss.f32', batch['loss_sha256'])

        available = sorted(int(p.name[4:]) for p in run.glob('step[0-9]*') if p.is_dir())
        for step in parse_steps(args.steps, available):
            directory = run / f'step{step:06d}'
            arrays = json.loads((directory / 'capture.json').read_text())['arrays']
            model = LanguageModelInference(
                params_from(directory / 'initial_p.f32', arrays['initial_p']['sha256']), shape=shape)
            check(f'step{step:06d}', model, directory / 'ids.i32', arrays['ids']['sha256'],
                  directory / 'loss.f32', arrays['loss']['sha256'])
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    evaluation = json.loads((run / 'heldout-final' / 'evaluation.json').read_text())
    final = model_for(run / 'step000128/post_p.f32', evaluation['state_before']['arrays']['parameters'])
    from mojolearn._bufcheck import le_bytes
    width = shape.length + 1
    flat_ids = struct.unpack(f'<{shape.batch * width}i',
                             (run / 'heldout-final/batch00.ids.i32').read_bytes())
    inputs = [v for r in range(shape.batch) for v in flat_ids[r * width: r * width + shape.length]]
    rows_in = frombytes(struct.pack(f'<{len(inputs)}i', *inputs), '<i4', (shape.batch, shape.length))
    first = frombytes(struct.pack('<i', inputs[0]), '<i4', (1, 1))
    first_row = frombytes(struct.pack(f'<{shape.length}i', *inputs[:shape.length]), '<i4', (1, shape.length))
    probes = {}
    for tag, flag in (('', False), ('_threaded', True)):
        probes['final_heldout00_full' + tag] = hashlib.sha256(
            le_bytes(final.logits(rows_in, threaded=flag), 'f')).hexdigest()
        probes['final_heldout00_row0' + tag] = hashlib.sha256(
            le_bytes(final.logits(first_row, threaded=flag), 'f')).hexdigest()
        probes['final_heldout00_row0_len1' + tag] = hashlib.sha256(
            le_bytes(final.logits(first, threaded=flag), 'f')).hexdigest()
        probes['final_heldout00_next_bytes' + tag] = final.next_bytes(rows_in, threaded=flag)
    # DEVIATION 2616: the threaded path must reproduce the reference logits
    # byte for byte at batch 2 (row tasks) and batch 1 (head tasks).
    probe_paths_equal = all(probes[k] == probes[k + '_threaded'] for k in
                            ('final_heldout00_full', 'final_heldout00_row0',
                             'final_heldout00_row0_len1', 'final_heldout00_next_bytes'))

    mismatches = [r for r in rows if not r['equal']]
    ref_mismatches = [r for r in rows if not r['equal_reference']]
    thr_mismatches = [r for r in rows if not r['equal_threaded']]
    if args.expect_mismatch:
        verdict = len(ref_mismatches) > 0 and len(thr_mismatches) > 0
    else:
        verdict = len(mismatches) == 0 and probe_paths_equal
    report = dict(
        schema='mojolearn.byte-lm-host-gate.v1',
        deviation=2613,
        commit=git_commit(),
        host=host_info(),
        binary=dict(path=binary_path(), sha256=sha256(binary_path())),
        capture=str(args.capture.relative_to(ROOT)) if args.capture.is_relative_to(ROOT) else str(args.capture),
        capture_comparison_sha256=sha256(args.capture.parent / 'comparison.json'),
        expect_mismatch=args.expect_mismatch,
        compared=len(rows), equal=len(rows) - len(mismatches), mismatched=len(mismatches),
        mismatched_reference=len(ref_mismatches), mismatched_threaded=len(thr_mismatches),
        probe_paths_equal=probe_paths_equal,
        first_mismatches=mismatches[:5], probes=probes,
        # Wall clock of each loss call through the public surface (one
        # [2, 32] forward plus the loss). Operational, not a matched
        # benchmark, and not compared with anything outside this library.
        timing=dict(loss_calls=len(rows),
                    total_seconds=round(sum(r['seconds'] for r in rows), 6),
                    total_seconds_threaded=round(sum(r['seconds_threaded'] for r in rows), 6),
                    max_seconds=max((r['seconds'] for r in rows), default=0.0),
                    min_seconds=min((r['seconds'] for r in rows), default=0.0)),
        verdict='PASS' if verdict else 'FAIL', rows=rows)
    text = json.dumps(report, indent=1, sort_keys=True) + '\n'
    if args.report:
        with open(args.report, 'x') as stream:
            stream.write(text)
    print(f"gate: {report['verdict']}: {report['equal']}/{report['compared']} loss bytes equal"
          f"{' (sabotage build, a mismatch was required)' if args.expect_mismatch else ''}")
    print(f"gate: host {report['host'].get('cpu', '?')} {report['host']['machine']}")
    for name, value in probes.items():
        print(f'gate: probe {name} {value}')
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
