#!/usr/bin/env python3
"""M3-only LU staging quality/call+read worker; no repeats or warmups.

quality OUT: capture existing LU-MMA hard/odd/singular fixtures.
compare A B: require exact LU/pivots/solutions and unchanged info/residuals.
timing OUT: one board8192 lu_factor and one solve(A,B), first reads included.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
FIXTURE = 'lu-mma-dbuf-v1'


def capture(action, out):
    assert os.environ.get('MOJOLEARN_VENDOR') == 'apple'
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    import numpy as np
    out.mkdir(parents=True, exist_ok=False)
    if action == 'quality':
        from lu_fast_mma_quality import dump
        dump(str(out / 'outputs.npz'))
        old = json.loads((out / 'outputs.npz.json').read_text())
        result = dict(fixture=FIXTURE, binding_sha256=old['binding_sha256'], rows=old['rows'])
    else:
        import mojolearn as ml
        from mojolearn import _backend
        binding = _backend.binding('_mojolearn_x_decomp', 'fast')
        assert int(binding.x_decomp_numeric_mode()) == 0
        rng = np.random.default_rng(7)
        n = 8192
        a = rng.standard_normal((n, n)).astype('float32')
        a[np.arange(n), np.arange(n)] += np.float32(2 * np.sqrt(n))
        b = rng.standard_normal((n, 64)).astype('float32')
        factor = getattr(ml, 'lu_factor', None) or ml.linalg.lu_factor
        solve = getattr(ml, 'solve', None) or ml.linalg.solve
        rows = []
        for name, call in [('lu-factor', lambda: factor(a)), ('lu-solve', lambda: solve(a, b))]:
            start = time.perf_counter_ns()
            output = call()
            returned = time.perf_counter_ns()
            arrays = [np.asarray(v) for v in output] if name == 'lu-factor' else [np.asarray(output)]
            checksums = [float(np.sum(v, dtype='float64')) for v in arrays]
            consumed = time.perf_counter_ns()
            assert all(np.isfinite(v).all() for v in arrays)
            row = dict(algo=name, dataset='synthetic', call_ms=(returned-start)/1e6,
                       first_read_ms=(consumed-returned)/1e6,
                       call_read_ms=(consumed-start)/1e6, checksums=checksums,
                       repetitions=1)
            print('LU-DBUF-TIME ' + json.dumps(row), flush=True)
            rows.append(row)
            del output, arrays
        so = ROOT / 'python/mojolearn/_mojolearn_x_decomp.so'
        result = dict(fixture=FIXTURE, binding_sha256=hashlib.sha256(so.read_bytes()).hexdigest(), rows=rows)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


def compare(a, b):
    import numpy as np
    ma = json.loads((a / 'metrics.json').read_text())
    mb = json.loads((b / 'metrics.json').read_text())
    assert ma['fixture'] == mb['fixture'] == FIXTURE
    A = {r['fixture']: r for r in ma['rows']}
    B = {r['fixture']: r for r in mb['rows']}
    expected = {'boost65', 'boost257', 'boost300', 'boost1000', 'board8192', 'plain1000', 'plain2051', 'zero700'}
    assert set(A) == set(B) == expected
    failures = []
    with np.load(a / 'outputs.npz') as za, np.load(b / 'outputs.npz') as zb:
        assert set(za.files) == set(zb.files) == {name + suffix for name in expected for suffix in ('_lu', '_piv', '_x')}
        for name in sorted(A):
            x, y = A[name], B[name]
            exact = all(za[name+s].dtype == zb[name+s].dtype and
                        za[name+s].shape == zb[name+s].shape and
                        za[name+s].tobytes() == zb[name+s].tobytes()
                        for s in ('_lu', '_piv', '_x'))
            good = exact and x['finite'] and y['finite'] and x['info'] == y['info']
            good = good and all(0 <= y[key] <= x[key] for key in ('factor_residual', 'solve_residual'))
            if not good:
                failures.append(name)
            print('LU-DBUF-CASE ' + json.dumps(dict(fixture=name, exact=exact,
                  status='PASS' if good else 'FAIL', A=x, B=y)), flush=True)
    print('LU-DBUF-QUALITY ' + json.dumps(dict(status='FAIL' if failures else 'PASS', failures=failures)))
    return bool(failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('quality', 'timing', 'compare'))
    p.add_argument('paths', nargs='+', type=Path)
    args = p.parse_args()
    if args.action == 'compare':
        assert len(args.paths) == 2
        return compare(*args.paths)
    assert len(args.paths) == 1
    capture(args.action, args.paths[0])
    return 0


if __name__ == '__main__':
    sys.exit(main())
