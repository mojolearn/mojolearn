#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quality capture / compare for MOJOLEARN_EIGH_FAST_TRIDIAG (lane w4-eigh).

  dump OUT.json        run every case once under the installed x_decomp
                       binding (FAST, Apple) and record its metrics
  compare A.json B.json  arm A (main) vs arm B (the define)

Metrics (float64 numpy oracle, outside the solve): relative residual
||A V - V diag(w)||_F / ||A||_F, max eigenvalue error / max |lambda| against
numpy.linalg.eigvalsh(float64), orthogonality ||V^T V - I||_F / sqrt(n),
and w ascending.

Tolerances, fixed before any B result exists:
  * routed cases (n >= 512: board:4096, indefinite:1024, board:1000):
    eigenvalue error <= 3.5e-7 (the brief's target: the repaired eigh level),
    residual <= max(1.1 A, A + 5e-8) (no worse than main; main's Jacobi is
    5.4e-5 on the board), orthogonality <= 2e-4 (the original absolute bound
    of tools/apple_fast_eigh_quality.py), ascending.
  * refusal cases (repeated:1024, gram:1024: repeated eigenvalues, the new
    route must hand them to main's Jacobi) and every case below n = 512: each
    metric <= max(1.1 A, A + 5e-8) and <= 2e-4 (A's own bound).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

ROUTED = ('board:4096', 'indefinite:1024', 'board:1000')
OTHER = ('repeated:1024', 'gram:1024', 'board:257', 'indefinite:128', 'repeated:31')
METRICS = ('relative_residual', 'max_eigenvalue_error', 'orthogonality_error')
EIG_MAX = 3.5e-7
ORTH_MAX = 2e-4


def matrix(kind, n):
    import numpy as np
    # seed 7: board:4096 is then the board's own input (tools/bench_board_algos.py sym_system)
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n)).astype(np.float32)
    sym = (m + m.T) * np.float32(0.5)
    if kind == 'board':
        a = sym.copy()
        a[np.diag_indices(n)] += np.float32(2 * np.sqrt(n))
    elif kind == 'indefinite':
        a = sym
    elif kind == 'repeated':
        a = np.diag((np.arange(n) % 5 - 2).astype(np.float32))
    elif kind == 'gram':
        h = max(1, n // 2)
        a = np.asarray(m[:, :h] @ m[:, :h].T, np.float32)
    else:
        raise ValueError(kind)
    return np.ascontiguousarray(a, dtype=np.float32)


def dump(out):
    import numpy as np
    from mojolearn import linalg
    if os.environ.get('MOJOLEARN_VENDOR') != 'apple' or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast':
        raise SystemExit('needs MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast')
    so = ROOT / 'python/mojolearn/_mojolearn_x_decomp.so'
    res = {'binding_sha256': hashlib.sha256(so.read_bytes()).hexdigest(), 'cases': {}}
    for key in ROUTED + OTHER:
        kind, n = key.split(':')
        n = int(n)
        a = matrix(kind, n)
        w, v = linalg.eigh(a, UPLO='L')
        w = np.asarray(w.tolist() if hasattr(w, 'tolist') else w, dtype=np.float64).reshape(-1)
        v = np.asarray(v.tolist() if hasattr(v, 'tolist') else v, dtype=np.float64).reshape(n, n)
        a64 = np.tril(a.astype(np.float64))
        a64 = a64 + np.tril(a64, -1).T
        ref = np.linalg.eigvalsh(a64)
        scale = max(np.linalg.norm(a64), np.finfo(float).tiny)
        m = {
            'relative_residual': float(np.linalg.norm(a64 @ v - v * w[None, :]) / scale),
            'max_eigenvalue_error': float(np.max(np.abs(np.sort(w) - ref)) / max(np.max(np.abs(ref)), np.finfo(float).tiny)),
            'orthogonality_error': float(np.linalg.norm(v.T @ v - np.eye(n)) / math.sqrt(n)),
            'ascending': bool(np.all(np.diff(w) >= 0)),
        }
        res['cases'][key] = m
        print('EIGH-W4-CAPTURE ' + json.dumps(dict(case=key, **m)), flush=True)
    Path(out).write_text(json.dumps(res, indent=2, sort_keys=True) + '\n')


def compare(pa, pb):
    A = json.loads(Path(pa).read_text())['cases']
    B = json.loads(Path(pb).read_text())['cases']
    assert set(A) == set(B) == set(ROUTED + OTHER), 'case set mismatch'
    ok = True
    for key in ROUTED + OTHER:
        a, b = A[key], B[key]
        fails = []
        if not b['ascending']:
            fails.append('ascending')
        for name in METRICS:
            x, ref = b[name], a[name]
            if not (isinstance(x, float) and math.isfinite(x) and x >= 0):
                fails.append(name + '=nonfinite')
                continue
            if key in ROUTED:
                if name == 'max_eigenvalue_error':
                    good = x <= EIG_MAX
                elif name == 'relative_residual':
                    good = x <= max(1.1 * ref, ref + 5e-8)
                else:
                    good = x <= ORTH_MAX
            else:
                good = x <= max(1.1 * ref, ref + 5e-8) and x <= ORTH_MAX
            if not good:
                fails.append(f'{name} {x:.3e} vs A {ref:.3e}')
        ok &= not fails
        print('EIGH-W4-CASE ' + json.dumps(dict(case=key, status='FAIL' if fails else 'OK', fails=fails,
                                                A={k: a[k] for k in METRICS}, B={k: b[k] for k in METRICS})),
              flush=True)
    print('EIGH-W4-AB status=' + ('PASS' if ok else 'FAIL') + ' cases=%d' % len(A), flush=True)
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('dump', 'compare'))
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()
    if args.mode == 'dump':
        assert len(args.paths) == 1
        dump(args.paths[0])
    else:
        assert len(args.paths) == 2
        compare(*args.paths)


if __name__ == '__main__':
    main()
