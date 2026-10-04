# SPDX-License-Identifier: Apache-2.0
"""Fixed ordered-cov-v1 saved-output oracle. No fitting or timing here.
Unlike SKIP, raw numeric fields intentionally may change: judge them against
float64 while requiring exact masks/flags. Every B metric must be <= A.
"""
import numpy as np
from scipy.stats import chi2

def json_ready(value):
    """Normalize NumPy scalars before writing or printing evidence."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def error_metrics(value, reference):
    value = np.asarray(value, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if value.shape != reference.shape:
        raise ValueError(f'oracle shape mismatch: {value.shape} != {reference.shape}')
    diff = value - reference
    if not np.isfinite(diff).all():
        raise ValueError('nonfinite saved value or oracle')
    denom = max(float(np.linalg.norm(reference.ravel())), np.finfo(np.float64).tiny)
    return dict(rel_l2=float(np.linalg.norm(diff.ravel()) / denom),
                max_abs=float(np.max(np.abs(diff), initial=0.0)))


def distances(x, location, precision):
    centered = x - location
    return np.einsum('ij,ij->i', centered @ precision, centered)


def analyze(a, b, x, q, lane):
    exact_names = ('raw_support_', 'support_', 'flags')
    exact = {key: (a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                   and a[key].tobytes() == b[key].tobytes()) for key in exact_names}
    # A common conditional oracle is only valid when the saved masks coincide.
    if not exact['support_'] or not exact['raw_support_']:
        return dict(status='HOLD', reason='different reweighting supports', exact=exact)
    for saved in (a, b):
        for key in ('raw_location_', 'raw_covariance_', 'location_', 'covariance_',
                    'precision_', 'dist_', 'distances'):
            if not np.isfinite(saved[key]).all():
                raise ValueError('nonfinite captured field: ' + key)
        if not np.isin(saved['support_'], (0, 1)).all():
            raise ValueError('support contains values other than zero/one')
        if np.asarray(saved['flags']).shape != (len(q),):
            raise ValueError('query flag shape mismatch')
    support = np.asarray(a['support_']).astype(bool)
    if support.shape != (len(x),) or not support.any():
        raise ValueError('invalid saved reweighting support')
    selected = x[support].astype(np.float64)
    location = selected.mean(axis=0)
    centered = selected - location
    d = x.shape[1]
    factor = .975 / chi2.cdf(chi2.ppf(.975, d), d + 2)
    covariance = centered.T @ centered / len(selected) * factor
    eigenvalues, vectors = np.linalg.eigh(covariance)
    cutoff = float(float(np.max(np.abs(eigenvalues))) * d * float(np.finfo(np.float32).eps))
    keep = np.abs(eigenvalues) > cutoff
    precision = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
    train_dist = distances(x.astype(np.float64), location, precision)
    query_dist = distances(q.astype(np.float64), location, precision)
    raw_support = np.asarray(a['raw_support_']).astype(bool)
    if raw_support.shape != (len(x),) or not raw_support.any():
        raise ValueError('invalid raw support')
    raw_selected = x[raw_support].astype(np.float64)
    raw_location = raw_selected.mean(axis=0)
    raw_centered = raw_selected - raw_location
    raw_covariance = raw_centered.T @ raw_centered / len(raw_selected)
    oracle = dict(raw_location_=raw_location, raw_covariance_=raw_covariance,
                  location_=location, covariance_=covariance, precision_=precision,
                  dist_=train_dist, distances=query_dist)
    if lane == 'elliptic-envelope':
        offset = np.percentile(-train_dist, 10.0, method='linear')
        oracle.update(offset_=np.asarray(offset), decision_function=-query_dist-offset)
        oracle_flags = oracle['decision_function'] < 0
    else:
        oracle_flags = query_dist > chi2.ppf(.975, d)
    metrics = {}
    passed = all(exact.values())
    for key, reference in oracle.items():
        ma, mb = error_metrics(a[key], reference), error_metrics(b[key], reference)
        checks = {name: mb[name] <= ma[name] for name in ma}
        metrics[key] = dict(A=ma, B=mb, no_worse=checks)
        passed &= all(checks.values())
    flag_errors = {arm: int(np.count_nonzero(np.asarray(saved['flags'], dtype=bool) != oracle_flags))
                   for arm, saved in (('A', a), ('B', b))}
    metrics['oracle_flag_disagreement_count'] = dict(**flag_errors,
                                                    no_worse=flag_errors['B'] <= flag_errors['A'])
    passed &= flag_errors['B'] <= flag_errors['A']
    # Same-input covariance drift is consistent with split-K atomics, but this
    # diagnostic neither measures a noise range nor relaxes any oracle gate.
    ab = {key: error_metrics(b[key], a[key]) for key in oracle}
    return dict(status='PASS' if passed else 'HOLD', exact=exact, metrics=metrics,
                A_B_differences=ab, oracle_rank=int(keep.sum()), oracle_cutoff=cutoff,
                support_count=int(support.sum()), consistency_factor=float(factor),
                criterion='all exact checks and every B error <= corresponding A error; zero allowance',
                scope='float64 numerical accuracy conditional on common saved support; no opponent refit')

