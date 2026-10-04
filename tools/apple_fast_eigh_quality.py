#!/usr/bin/env python3
"""Quality-only M3 probe: each input solved once; no scored opponent timings.

Run under an already selected FAST x_decomp binding. --output records metrics;
--compare compares a second arm to the first, allowing 10% or 5e-8 absolute
noise. This is an A/B nonregression check, NOT opponent-quality admission.
The 4096 board row remains held unless its stricter recorded opponent errors
are met. Use --sizes 4096 --kinds board for that input after small probes pass.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
from mojolearn import linalg


def matrices(n, kinds):
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n)).astype(np.float32)
    sym = (m + m.T) * np.float32(0.5)
    for kind in kinds:
        if kind == 'board':
            a = sym.copy()
            a[np.diag_indices(n)] += np.float32(2 * np.sqrt(n))
        elif kind == 'indefinite':
            a = sym.copy()
        elif kind == 'repeated':
            # Closed spectrum, including repeated eigenvalues and zero.
            a = np.diag((np.arange(n) % 5 - 2).astype(np.float32))
        elif kind == 'gram':
            a = np.asarray(m[:, :max(1, n // 2)] @ m[:, :max(1, n // 2)].T, np.float32)
        else:
            raise ValueError(kind)
        yield kind, a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='31,128,257')
    ap.add_argument('--kinds', default='board,indefinite,repeated,gram')
    ap.add_argument('--output', required=True)
    ap.add_argument('--compare')
    args = ap.parse_args()
    if os.environ.get('MOJOLEARN_VENDOR') != 'apple' or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast':
        raise SystemExit('quality probe requires MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast on M3')
    previous = json.loads(Path(args.compare).read_text()) if args.compare else None
    results = {}
    passed = True
    for n in map(int, args.sizes.split(',')):
        for kind, a in matrices(n, args.kinds.split(',')):
            # Oracle is quality verification outside the GPU call, not a timed arm.
            a64 = a.astype(np.float64)
            ref = np.linalg.eigvalsh(a64)
            w, v = linalg.eigh(a)
            w = np.asarray(w.tolist(), dtype=np.float64)
            v = np.asarray(v.tolist(), dtype=np.float64)
            scale = max(np.linalg.norm(a64), np.finfo(float).tiny)
            metrics = {
                'relative_residual': float(np.linalg.norm(a64 @ v - v * w) / scale),
                'max_eigenvalue_error': float(np.max(np.abs(w - ref)) / max(np.max(np.abs(ref)), np.finfo(float).tiny)),
                'orthogonality_error': float(np.linalg.norm(v.T @ v - np.eye(n)) / np.sqrt(n)),
            }
            key = f'{kind}:{n}'
            ok = all(np.isfinite(x) and x <= 2e-4 for x in metrics.values()) and bool(np.all(np.diff(w) >= 0))
            if previous:
                ok = ok and all(value <= max(previous[key][name] * 1.1, previous[key][name] + 5e-8)
                                for name, value in metrics.items())
            results[key] = metrics
            passed &= ok
            print('EIGH-QUALITY ' + json.dumps(dict(case=key, status='OK' if ok else 'FAIL', **metrics)), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
