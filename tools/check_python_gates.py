#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run explicitly selected installed-package gates with a shared time budget."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import identity_iterate

ROOT = Path(__file__).resolve().parents[1]


def discover():
    return sorted(p.stem for p in (ROOT / 'python/mojolearn/tests').glob('test_*.py')
                  if 'import pytest' not in p.read_text() and 'unittest.TestCase' not in p.read_text())


def main(argv=None):
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument('--gate', action='append', default=[])
    scope.add_argument('--all', action='store_true', help='explicit broad qualification')
    ap.add_argument('--release', action='store_true', help='explicit release qualification for broad Metal gates')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--backend', choices=('cpu', 'metal', 'cuda', 'hip'), default='cpu')
    ap.add_argument('--host-dir', type=Path)
    ap.add_argument('--budget', type=float)
    ap.add_argument('--timeout', type=float, default=60)
    ap.add_argument('--wait-timeout', type=float, default=60)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    args.budget = args.budget if args.budget is not None else (60 if args.backend == 'metal' else 300)
    if any(not math.isfinite(v) or v <= 0 for v in (args.budget, args.timeout, args.wait_timeout)):
        ap.error('budget and timeouts must be finite and positive')
    available = discover()
    if args.list:
        print('\n'.join(available))
        return 0
    gates = available if args.all else list(dict.fromkeys(args.gate))
    if not gates or set(gates) - set(available):
        ap.error('select known --gate NAME entries or explicitly --all; use --list')
    print(json.dumps(dict(gates=gates, backend=args.backend, budget=args.budget,
                          timeout=args.timeout, wait_timeout=args.wait_timeout)), flush=True)
    if args.plan:
        return 0
    if args.backend == 'metal' and len(gates) > 1 and not args.release:
        ap.error('multiple Metal gates require --release; diagnose one --gate at a time')
    out = args.out or Path(tempfile.mkdtemp(prefix='mojolearn-gates-'))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        ap.error('summary already exists; use a fresh output directory')
    print(f'Gate logs and summary: {out}', flush=True)
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_LINALG_GATE_FULL='1')
    env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + env.get('PYTHONPATH', '')
    if args.backend == 'cpu':
        try:
            package, host = identity_iterate.cpu_package(out, args.host_dir or env.get('MOJOLEARN_HOST_DIR') or ROOT / 'python/mojolearn/host')
        except ValueError as exc:
            ap.error(str(exc))
        env['PYTHONPATH'] = str(package) + os.pathsep + env['PYTHONPATH']
        env['MOJOLEARN_HOST_DIR'] = str(host)
    deadline = started + args.budget
    completed = []
    def report(pending, failed=None):
        tmp = out / 'summary.tmp'
        tmp.write_text(json.dumps(dict(complete=not pending and failed is None,
            completed=completed, pending=pending, failed=failed, backend=args.backend,
            elapsed_seconds=time.monotonic()-started), indent=2) + '\n')
        tmp.replace(out / 'summary.json')
    def run(command, mode, label):
        if time.monotonic() >= deadline:
            return 124
        cmd = [sys.executable, str(ROOT / 'tools/mac_slot.py'), '--deadline', str(deadline),
               '--timeout', str(args.timeout), '--wait-timeout', str(args.wait_timeout), mode, *command]
        # run_job forwards termination to the scheduler for process-group cleanup.
        # Keep the scheduler's stdout in a file without changing its process tree.
        import subprocess
        with (out / (label + '.log')).open('w') as log:
            return identity_iterate.run_job(cmd, env, stdout=log, stderr=subprocess.STDOUT)
    report(gates)
    if 'test_linalg_identity' in gates and not env.get('MOJOLEARN_GEMM_CARD') and env.get('MOJOLEARN_GATES_SKIP_CARD') != '1':
        card = out.resolve() / 'oracle.card'
        code = run(['bash', str(ROOT / 'tools/gemm_card.sh'), 'oracle', str(card)], 'run', 'oracle')
        if code:
            report(gates, dict(gate='oracle-card', exit_code=code))
            return code
        env['MOJOLEARN_GEMM_CARD'] = str(card)
    body = ('import sys,runpy,mojolearn as ml\n'
            'actual=ml.vendor()\n'
            'if actual != sys.argv[1]: raise SystemExit(f"requested {sys.argv[1]}, loaded {actual}")\n'
            'sys.argv=[sys.argv[2]]\n'
            'runpy.run_module(sys.argv[0], run_name="__main__")')
    for i, gate in enumerate(gates):
        code = run([sys.executable, '-c', body, args.backend, 'mojolearn.tests.' + gate],
                   'run' if args.backend == 'cpu' else args.backend, gate)
        if code:
            report(gates[i+1:], dict(gate=gate, exit_code=code))
            return code
        completed.append(gate)
        report(gates[i+1:])
    return 0


if __name__ == '__main__':
    sys.exit(main())
