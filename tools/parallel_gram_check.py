#!/usr/bin/env python3
"""Cloud-only full-estimator checks for distributed pinned Gram chunks.

Every case fits once without devices and once through `fit_gram_estimator(..., devices=(0, 1))` and must give the
same bytes. Equal bytes prove nothing when the parallel fit never left device 0, so each parallel fit also records
how many ranks received work (the device-plan work counter every plan binding reports through `device_plan_log()`,
read with tools/logical_shard_identity.py `ranks_take`). A case whose work reached fewer than 2 ranks is a FAIL
("spread"), as is a case whose bindings report no ranks. FAIL CLOSED: every case runs, one line each, and the
report's status is PASS only when every case is MATCH and spread; the exit status is 1 otherwise.

The cases run one after another in one process on purpose: the rank record is process-wide, so two concurrent
cases would read each other's ranks, and each case is a few milliseconds on 259 rows.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logical_shard_identity import one_line, ranks_take  # noqa: E402

#: the fewest ranks the two-device fit must spread its work over (the check's point: a real split)
MIN_SPREAD = 2


def case_verdict(mismatch, ranks, rank_errors):
    """(verdict, reason) of one case: DIFFER on any byte mismatch, FAIL when the parallel fit did not spread."""
    if mismatch:
        return "DIFFER", "%s differs from the single-device fit" % mismatch
    why = ("; device_plan_log raised: " + one_line(rank_errors[0])) if rank_errors else ""
    if type(ranks) is not int:
        return "FAIL", "no rank record (no loaded binding reports the device-plan work counter)" + why
    if ranks < MIN_SPREAD:
        return "FAIL", ("spread its work over %d rank(s), not >= %d: the route never reached the Gram dispatcher"
                        % (ranks, MIN_SPREAD)) + why
    return "MATCH", ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import LinearRegression, Ridge, PCA, TruncatedSVD
    from mojolearn.parallel_classical import fit_gram_estimator
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks, hashes = [], []
    for columns in (7, 16, 65, 128):
        X = raw[:259 * columns].astype('<f4').reshape(259, columns) / np.float32(255)
        y = raw[50000:50259].astype('<f4') / np.float32(255)
        weights = np.ones(259, dtype='<f4')
        weights[::7] = np.float32(0.5)
        # Ridge / LinearRegression are kept on purpose (plan N455): their unweighted identical fit takes
        # `linear_gram_fit` (bindings/classical_ridge.mojo), which reads no DevicePlan, so until that door shards the
        # Gram these cases FAIL on spread instead of passing trivially. PCA / TruncatedSVD reach
        # core/gram_multi_gpu.mojo `gram_splitk_parallel`.
        cases = [(LinearRegression, dict(fit_intercept=False), y, {}),
                 (LinearRegression, dict(fit_intercept=True), y, {}),
                 (LinearRegression, dict(fit_intercept=True), y, dict(sample_weight=weights)),
                 (Ridge, dict(alpha=0.25, fit_intercept=True), y, {}),
                 (Ridge, dict(alpha=0.25, fit_intercept=True), y, dict(sample_weight=weights)),
                 (PCA, dict(n_components=3, svd_solver='covariance_eigh', whiten=True), None, {}),
                 (TruncatedSVD, dict(n_components=3), None, {})]
        for cls, params, target, kwargs in cases:
            params['numeric_mode'] = 'identical'
            label = [cls.__name__, columns, dict(params), sorted(kwargs)]
            try:
                verdict, reason, ranks, hexdigest = run_case(cls, params, X, target, kwargs, fit_gram_estimator)
            except Exception as error:
                verdict, reason, ranks, hexdigest = 'FAIL', one_line('%s: %s' % (type(error).__name__, error)), None, None
            checks.append(dict(case=label, verdict=verdict, reason=reason, ranks=ranks))
            hashes.append(hexdigest)
            print(verdict, cls.__name__, columns, params, sorted(kwargs), 'ranks=%s' % ranks, reason, flush=True)
    ok = all(c['verdict'] == 'MATCH' for c in checks)
    args.report.write_text(json.dumps(dict(status='PASS' if ok else 'FAIL', checks=checks, hashes=hashes,
        failed=[c['case'] for c in checks if c['verdict'] != 'MATCH'],
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two GPUs; original 128 Gram chunks; 7/16/65/128 columns; every case must spread over >= %d ranks; '
              'no pooled root-memory or cross-vendor claim' % MIN_SPREAD), indent=2) + '\n')
    return 0 if ok else 1


def run_case(cls, params, X, target, kwargs, fit_gram_estimator):
    """One case: (verdict, reason, ranks, digest hex). The rank record is cleared after the plain fit, so only the
    two-device fit's work is counted."""
    serial = cls(**params).fit(X, target, **kwargs)
    ranks_take(sys.modules)
    parallel = fit_gram_estimator(cls(**params), X, target, devices=(0, 1), **kwargs)
    rank_errors = []
    ranks = ranks_take(sys.modules, rank_errors)
    digest = hashlib.sha256()
    mismatch = None
    for name in ('coef_', '_x_mean', 'components_', 'mean_', 'explained_variance_',
                 'explained_variance_ratio_', 'singular_values_'):
        if hasattr(serial, name):
            a, b = getattr(serial, name), getattr(parallel, name)
            if not (a.shape == b.shape and a.tobytes() == b.tobytes()):
                mismatch = mismatch or name
            digest.update(name.encode() + b.tobytes())
    for name in ('intercept_', '_y_mean', 'noise_variance_'):
        if hasattr(serial, name):
            a, b = struct.pack('<d', getattr(serial, name)), struct.pack('<d', getattr(parallel, name))
            if a != b:
                mismatch = mismatch or name
            digest.update(name.encode() + b)
    method = 'predict' if target is not None else 'transform'
    output = getattr(parallel, method)(X)
    if getattr(serial, method)(X).tobytes() != output.tobytes():
        mismatch = mismatch or method
    digest.update(output.tobytes())
    verdict, reason = case_verdict(mismatch, ranks, rank_errors)
    return verdict, reason, ranks, digest.hexdigest()


if __name__ == '__main__':
    raise SystemExit(main())
