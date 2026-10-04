#!/usr/bin/env python3
"""Run ONLY named repaired quality gates against a frozen wave revision.

Append-only: every gate/arm gets a fresh `supplements/<gate>--<arm>/` folder
(refused if it already exists) with its full log, rc and a receipt bound to the
wave identity, the frozen binary inventory and the exact gate script bytes.
Original quality receipts and failure logs are never touched. The command and
environment match identical_wave_runner.py's quality phase; the gate script is
taken from the pinned harness directory when it carries a reviewed repair
(e.g. tools/idn_all_checks.py at e98f53361), else from the frozen source tree.
No timing, no opponents, no wheel installs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from identical_wave_runner import run, verify_products, write

SUPPORTED_KINDS = ('python',)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(vendor, arch, source, arm):
    """Same environment construction as identical_wave_runner.main()."""
    backend = {'nvidia': 'cuda', 'amd': 'hip'}[vendor]
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_', 'MOJO_COMPILE_'))}
    env = dict(clean, MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_COMPILE_JOBS='1', PYTHONUNBUFFERED='1',
               MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_TARGET_COLUMN=vendor, MOJOLEARN_VENDOR=backend,
               MOJOLEARN_SKIP_BUILD_GATE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PATH'] = '/root/.pixi/bin:/opt/rocm/bin:' + env.get('PATH', '')
    env['PYTHONPATH'] = str(source / 'python')
    env['MOJOLEARN_MOJO_BUILD_FLAGS'] = '-D MOJOLEARN_IDN_ALL_OFF=1' if arm == 'off' else ''
    if arm == 'off':
        env['MOJOLEARN_IDN_ALL_OFF'] = '1'
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--harness', required=True, type=Path, help='Directory with pinned repaired gate scripts at their repo-relative paths')
    p.add_argument('--harness-commit', required=True)
    p.add_argument('--vendor', required=True, choices=('nvidia', 'amd'))
    p.add_argument('--gpu-arch', required=True)
    p.add_argument('--python', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--cell', action='append', required=True, help='gate_id:arm, repeatable')
    a = p.parse_args()
    wave = a.wave.resolve()
    plan = json.loads(a.plan.read_text())
    identity = json.loads((wave / 'wave.json').read_text())
    prepared = json.loads((wave / 'prepare.json').read_text())
    if prepared.get('status') != 'PASS' or prepared.get('identity') != identity:
        p.error('wave prepare receipt missing, failed or for another identity')
    if identity['plan_sha256'] != digest(a.plan) or identity['vendor'] != a.vendor or identity['gpu_arch'] != a.gpu_arch:
        p.error('plan/vendor/arch differ from the wave identity')
    gates = {g['id']: g for g in plan['quality_gates']}
    cells = []
    for cell in a.cell:
        gate_id, _, arm = cell.partition(':')
        if gate_id not in gates or arm not in ('on', 'off'):
            p.error('unknown cell ' + cell)
        gate = gates[gate_id]
        if arm not in gate.get('arms', ['on', 'off']):
            p.error('gate excludes arm: ' + cell)
        if gate.get('kind') not in SUPPORTED_KINDS or not gate.get('path'):
            p.error('only python gates with a path are supported: ' + cell)
        cells.append((gate, arm))
    for arm in sorted({arm for _, arm in cells}):
        source = wave / arm / 'source'
        head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if head != identity['sha'] or subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=source).returncode:
            p.error('frozen source changed: ' + arm)
        verify_products(source, json.loads((wave / arm / 'build-products.json').read_text()))
    failed = 0
    for gate, arm in cells:
        source = wave / arm / 'source'
        # Earlier attempts stay untouched; the reconciler reports every attempt.
        base = wave / 'supplements' / (gate['id'] + '--' + arm)
        base.parent.mkdir(exist_ok=True)
        folder, n = base, 1
        while folder.exists():
            n += 1
            folder = base.with_name(base.name + '--attempt' + str(n))
        folder.mkdir()  # exist_ok=False: one attempt per folder, never overwritten
        # Use the harness copy only when it IS a reviewed repair (bytes differ from
        # the frozen source); unchanged gates run from the frozen tree beside
        # their own helper modules (e.g. bench_board_probe).
        pinned = a.harness / gate['path']
        frozen = source / gate['path']
        script = pinned if pinned.is_file() and digest(pinned) != digest(frozen) else frozen
        report = folder / (gate['id'] + '.json')
        argv = [str(a.python), str(script)] + [v.replace('{report}', str(report)).replace('{full_data}', str(a.data.parent / 'rows-full')) for v in gate.get('args', [])]
        env = environment(a.vendor, a.gpu_arch, source, arm)
        env['MOJOLEARN_IDN_GATE_ARTIFACTS'] = str(folder / 'artifacts')
        receipt = {'schema': 1, 'kind': 'quality-supplement', 'gate': gate['id'], 'arm': arm, 'attempt': n, 'identity': identity,
                   'wave': str(wave), 'prepare_sha256': digest(wave / 'prepare.json'),
                   'build_products_sha256': digest(wave / arm / 'build-products.json'),
                   'plan_gate': gate, 'harness_commit': a.harness_commit,
                   'script': {'path': str(script), 'sha256': digest(script), 'from': 'harness' if script == pinned else 'frozen-source',
                              'frozen_source_sha256': digest(source / gate['path'])},
                   'argv': argv, 'status': 'RUNNING', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'opponents_executed': 0, 'timed': False}
        write(folder / 'receipt.json', receipt)
        log = folder / (gate['id'] + '.log')
        rc = run(argv, source, env, log, gate.get('timeout', 3600))
        lines = log.read_text(errors='replace').splitlines()
        receipt.update(rc=rc, status='PASS' if rc == 0 else 'FAIL', log_sha256=digest(log), last_line=(lines[-1][:400] if lines else ''),
                       finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        if report.is_file():
            receipt['report_sha256'] = digest(report)
        elif (report / 'result.json').is_file():  # gates whose {report} argument is an output directory
            receipt['report_sha256'] = digest(report / 'result.json')
        write(folder / 'receipt.json', receipt)
        print('SUPPLEMENT', gate['id'], arm, receipt['status'], 'rc', rc, flush=True)
        failed |= bool(rc)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
