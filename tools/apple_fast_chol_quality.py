#!/usr/bin/env python3
"""M3 quality-only Cholesky A/B: residuals, solves, and failed-minor info.

Each matrix is factored once. No timed opponents. --compare checks the second
arm against the first with 10% or 5e-8 absolute noise, never bit identity.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import numpy as np
from mojolearn import Cholesky


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', default='65,257,2051')
    ap.add_argument('--output', required=True)
    ap.add_argument('--compare')
    args = ap.parse_args()
    if os.environ.get('MOJOLEARN_VENDOR') != 'apple' or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast':
        raise SystemExit('requires MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast on M3')
    previous = json.loads(Path(args.compare).read_text()) if args.compare else None
    result = {}
    passed = True
    for n in map(int, args.sizes.split(',')):
        rng = np.random.default_rng(7)
        m = rng.standard_normal((n, n)).astype(np.float32)
        a = (m + m.T) * np.float32(0.5)
        a[np.diag_indices(n)] += np.float32(2 * np.sqrt(n))
        factor = Cholesky(jitter=0.0).fit(a)
        L = np.asarray(factor.L_.tolist(), dtype=np.float64)
        a64 = a.astype(np.float64)
        rhs = rng.standard_normal((n, 3)).astype(np.float32)
        x = np.asarray(factor.solve(rhs).tolist(), dtype=np.float64)
        q = dict(relative_residual=float(np.linalg.norm(L @ L.T - a64) / np.linalg.norm(a64)),
                 solve_residual=float(np.linalg.norm(a64 @ x - rhs) / np.linalg.norm(rhs)))
        ok = factor.info_ == 0 and np.all(np.isfinite(L)) and np.all(np.diag(L) > 0)
        ok = ok and bool(np.all(L[np.triu_indices(n, 1)] == 0))
        ok = ok and all(np.isfinite(value) and value < 1e-4 for value in q.values())
        if previous:
            ok = ok and all(value <= max(previous[str(n)][name] * 1.1, previous[str(n)][name] + 5e-8)
                            for name, value in q.items())
        result[str(n)] = q
        passed &= ok
        print('CHOL-QUALITY ' + json.dumps(dict(n=n, status='OK' if ok else 'FAIL', **q)), flush=True)
    # The public API reports a failed leading minor rather than raising at fit.
    # Fail after the first panel too: deferred FAST execution must recover info.
    for n, bad in ((65, 0), (513, 300)):
        a = np.eye(n, dtype=np.float32)
        a[bad, bad] = -1
        factor = Cholesky(jitter=0.0).fit(a)
        ok = factor.info_ == bad + 1
        try:
            factor.solve(np.ones(n, dtype=np.float32))
        except Exception as exc:
            # Mojo's Python bridge exposes this documented refusal as a
            # generic Exception, not ValueError (gap26-chol-quality).
            # Accept only the failed-factorization message and expected minor;
            # unrelated bridge/runtime errors must still fail the probe.
            expected = ('cholesky_solve_host: refusing to solve against a '
                        f'FAILED factorization (info={bad + 1})')
            if not str(exc).startswith(expected):
                raise
        else:
            ok = False
        passed &= ok
        print('CHOL-FAILURE ' + json.dumps(dict(n=n, expected_info=bad + 1, info=factor.info_,
                                               status='OK' if ok else 'FAIL')), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
