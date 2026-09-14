#!/usr/bin/env python3
"""Cloud-only multi-GPU bootstrap, permutation test and Monte Carlo versus one device."""
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
    os.environ.pop('MOJOLEARN_RESAMPLE_DEVICE_COUNT', None)
    devices = tuple(int(v) for v in args.devices.split(','))
    import numpy as np
    from mojolearn import resample
    from mojolearn import parallel_classical as pc

    checks, refusals = [], []

    def fingerprint(result):
        h = hashlib.sha256()
        for name in sorted(vars(result)):
            value = getattr(result, name)
            if hasattr(value, 'tobytes'):
                h.update(name.encode() + np.asarray(value).tobytes())
            elif isinstance(value, tuple):
                h.update(name.encode() + b''.join(struct.pack('<d', v) for v in value))
            elif isinstance(value, float):
                h.update(name.encode() + struct.pack('<d', value))
            else:
                h.update(name.encode() + repr(value).encode())
        return h.hexdigest()

    def refuse(name, function):
        try:
            function()
        except (ValueError, TypeError, RuntimeError, ImportError):
            refusals.append(name)
        else:
            raise AssertionError('expected refusal: ' + name)

    g = np.random.default_rng(1234)
    one_col = g.normal(size=97).astype('<f4')
    two_col = g.normal(size=(61, 2)).astype('<f4')
    for data, statistic in ((one_col, 'mean'), (one_col, 'std'), (one_col, 'quantile'), (two_col, 'pearson'),
                            (two_col, 'diff_means'), (one_col, 'trimmed_mean')):
        for n_resamples, method, alternative, r_first in ((2, 'percentile', 'two-sided', 0), (999, 'basic', 'less', 0),
                                                           (4099, 'percentile', 'greater', 17)):
            kw = dict(statistic=statistic, n_resamples=n_resamples, method=method, alternative=alternative,
                      random_state=3, r_first=r_first, q_or_prop=0.2, confidence_level=0.9)
            a = resample.bootstrap(data, numeric_mode='identical', **kw)
            b = pc.bootstrap(data, devices=devices, **kw)
            fa, fb = fingerprint(a), fingerprint(b)
            assert fa == fb, ('bootstrap', statistic, n_resamples)
            checks.append(dict(op='bootstrap', statistic=statistic, n_resamples=n_resamples, method=method,
                               alternative=alternative, r_first=r_first, sha256=fb))
            print('PASS bootstrap', statistic, n_resamples, method, alternative, r_first, flush=True)
    x = g.normal(size=45).astype('<f4')
    y = (g.normal(size=38) + 0.3).astype('<f4')
    for statistic in ('diff_means', 'mean', 'std'):
        for n_resamples, alternative, r_first in ((3, 'two-sided', 0), (2501, 'greater', 9)):
            kw = dict(statistic=statistic, n_resamples=n_resamples, alternative=alternative, random_state=8, r_first=r_first)
            a = resample.permutation_test(x, y, numeric_mode='identical', **kw)
            b = pc.permutation_test(x, y, devices=devices, **kw)
            assert fingerprint(a) == fingerprint(b), ('permutation', statistic, n_resamples)
            checks.append(dict(op='permutation_test', statistic=statistic, n_resamples=n_resamples,
                               alternative=alternative, r_first=r_first, sha256=fingerprint(b)))
            print('PASS permutation_test', statistic, n_resamples, alternative, r_first, flush=True)
    for integrand in ('const', 'sum', 'product'):
        for n_samples, i_first in ((1, 0), (257, 0), (65536, 0), (123457, 11)):
            kw = dict(random_state=4, i_first=i_first)
            a = resample.monte_carlo_integrate(integrand, [-1.0, 0.5], [2.5, 3.0], n_samples, numeric_mode='identical', **kw)
            b = pc.monte_carlo_integrate(integrand, [-1.0, 0.5], [2.5, 3.0], n_samples, devices=devices, **kw)
            assert fingerprint(a) == fingerprint(b), ('monte_carlo', integrand, n_samples)
            checks.append(dict(op='monte_carlo_integrate', integrand=integrand, n_samples=n_samples,
                               i_first=i_first, sha256=fingerprint(b)))
            print('PASS monte_carlo_integrate', integrand, n_samples, i_first, flush=True)

    refuse('bca', lambda: pc.bootstrap(one_col, devices=devices, method='bca'))
    refuse('fast_mode', lambda: pc.bootstrap(one_col, devices=devices, numeric_mode='fast'))
    refuse('negative_r_first', lambda: pc.bootstrap(one_col, devices=devices, n_resamples=10, r_first=-1))
    refuse('bad_integrand', lambda: pc.monte_carlo_integrate('cos', [0, 0], [1, 1], 10, devices=devices))
    refuse('duplicate_devices', lambda: pc.bootstrap(one_col, devices=(0, 0)))

    args.report.write_text(json.dumps(dict(status='PASS', devices=list(devices), checks=checks, refusals=refusals,
        scope='bootstrap/permutation replicate ranges and Monte Carlo sample chunks with global IDs across devices '
              'versus one device; root sort, interval, p-value and fold; no speed claim'), indent=2) + '\n')
    print('PASS', len(checks), 'resample configurations and', len(refusals), 'refusals')


if __name__ == '__main__':
    main()
