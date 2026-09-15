#!/usr/bin/env python3
"""Cloud-only operation-level Cholesky: factor and solve bits versus one device."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--devices', default='0,1')
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    os.environ.pop('MOJOLEARN_CHOLESKY_DEVICE_COUNT', None)
    devices = tuple(int(v) for v in args.devices.split(','))
    import numpy as np
    from mojolearn._cholesky_impl import Cholesky
    from mojolearn.parallel_classical import fit_cholesky, solve_cholesky

    checks, refusals = [], []

    def refuse(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, ImportError):
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    for n, nrhs, jitter, broken in ((1, 1, None, False), (33, 3, None, False), (64, 1, 0.0, False),
                                    (130, 5, None, False), (400, 8, 0.0, False), (777, 2, None, False),
                                    (96, 1, 0.0, True)):
        g = np.random.default_rng(n * 13 + nrhs)
        m = g.normal(size=(n, n))
        a = m @ m.T / n + np.eye(n)
        if broken:
            a[n // 2, n // 2] = -1.0
        a = a.astype('<f4')
        a = np.tril(a) + np.tril(a, -1).T
        one = Cholesky(jitter=jitter)
        one.numeric_mode = 'identical'
        one.fit(a)
        many = Cholesky(jitter=jitter)
        many.numeric_mode = 'identical'
        fit_cholesky(many, a, devices=devices)
        assert np.asarray(one.L_).tobytes() == np.asarray(many.L_).tobytes(), n
        assert (one.info_, one.nb_, one.jitter_) == (many.info_, many.nb_, many.jitter_), n
        assert struct.pack('<d', one._logdet) == struct.pack('<d', many._logdet), n
        h = hashlib.sha256(np.asarray(many.L_).tobytes())
        outputs = {}
        if broken:
            assert many.info_ != 0
            refuse('solve_failed_factor', lambda: solve_cholesky(many, np.ones(n, '<f4'), devices=devices))
        else:
            for shape in ((n,), (n, nrhs)):
                b = g.normal(size=shape).astype('<f4')
                x1 = np.asarray(one.solve(b))
                x2 = np.asarray(solve_cholesky(many, b, devices=devices))
                assert x1.shape == x2.shape and x1.tobytes() == x2.tobytes(), (n, shape)
                outputs[str(shape)] = hashlib.sha256(x2.tobytes()).hexdigest()
        checks.append(dict(n=n, nrhs=nrhs, jitter=jitter, info=int(many.info_), nb=int(many.nb_),
                           factor_sha256=h.hexdigest(), solutions=outputs))
        print('PASS', n, nrhs, jitter, 'info', many.info_, flush=True)

    refuse('wrong_type', lambda: fit_cholesky(object(), np.eye(3, dtype='<f4'), devices=devices))
    fast = Cholesky()
    fast.numeric_mode = 'fast'
    refuse('nonidentical', lambda: fit_cholesky(fast, np.eye(3, dtype='<f4'), devices=devices))
    refuse('unfitted_solve', lambda: solve_cholesky(Cholesky(), np.ones(3, '<f4'), devices=devices))
    bad = Cholesky(jitter=0.0)
    bad.numeric_mode = 'identical'
    refuse('nonsymmetric', lambda: fit_cholesky(bad, np.triu(np.ones((4, 4), '<f4')), devices=devices))
    assert not hasattr(bad, 'L_'), 'failed fit published state'

    args.report.write_text(json.dumps(dict(status='PASS', devices=list(devices), checks=checks,
        refusals=refusals, scope='Cholesky factor with trailing-update rows and solve with right-hand-side '
        'columns across devices versus one device; panel order, pivots and root matrix unchanged; no speed claim'),
        indent=2) + '\n')
    print('PASS', len(checks), 'Cholesky configurations and', len(refusals), 'refusals')


if __name__ == '__main__':
    main()
