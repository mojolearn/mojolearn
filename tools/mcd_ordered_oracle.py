# SPDX-License-Identifier: Apache-2.0
"""Fixed ordered-cov-v1 saved-output oracle. No fitting or timing here.
Raw numeric fields may change: each arm is judged against float64 built from
its own saved supports, under the FAST rule (fast_quality_rule.py): every B
error within noise of A, ranks equal; exact masks/flags and strict B <= A are
info only.
"""
import numpy as np
from scipy.stats import chi2
from fast_quality_rule import RULE, judge

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


# FAST rule tolerances (rtol, atol-factor), lower is better: an error vs the
# FP64 oracle may grow by half of itself (fp32 fold order) above an fp32 floor
# (rel_l2: 1e-6; max_abs: 1e-6 * max |oracle|); flag disagreements with the
# oracle by 0.1% of the query rows (at least 1).
REL_TOL = (0.5, 1e-6)
MAXABS_RTOL, MAXABS_FLOOR = 0.5, 1e-6


def oracle_for(saved, x, q, lane):
    """FP64 oracle conditional on one arm's own saved supports."""
    support = np.asarray(saved['support_']).astype(bool)
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
    raw_support = np.asarray(saved['raw_support_']).astype(bool)
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
        flags = oracle['decision_function'] < 0
    else:
        flags = query_dist > chi2.ppf(.975, d)
    sign, logdet = np.linalg.slogdet(raw_covariance)
    info = dict(rank=int(keep.sum()), cutoff=cutoff, support_count=int(support.sum()),
                consistency_factor=float(factor), raw_logdet=float(logdet) if sign > 0 else None)
    return oracle, flags, info


def analyze(a, b, x, q, lane):
    exact_names = ('raw_support_', 'support_', 'flags')
    # FAST rule: equal masks / flags are info, not a gate; each arm is judged
    # against the FP64 oracle built from its own supports.
    exact = {key: (a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                   and a[key].tobytes() == b[key].tobytes()) for key in exact_names}
    for saved in (a, b):
        for key in ('raw_location_', 'raw_covariance_', 'location_', 'covariance_',
                    'precision_', 'dist_', 'distances'):
            if not np.isfinite(saved[key]).all():
                raise ValueError('nonfinite captured field: ' + key)
        if not np.isin(saved['support_'], (0, 1)).all():
            raise ValueError('support contains values other than zero/one')
        if np.asarray(saved['flags']).shape != (len(q),):
            raise ValueError('query flag shape mismatch')
    (oa, fa, ia), (ob, fb, ib) = oracle_for(a, x, q, lane), oracle_for(b, x, q, lane)
    metrics = {}
    passed = ia['rank'] == ib['rank']
    for key in oa:
        ma, mb = error_metrics(a[key], oa[key]), error_metrics(b[key], ob[key])
        scale = max(float(np.max(np.abs(oa[key]))), float(np.max(np.abs(ob[key]))), 1e-300)
        checks = dict(rel_l2=judge(ma['rel_l2'], mb['rel_l2'], *REL_TOL),
                      max_abs=judge(ma['max_abs'], mb['max_abs'], MAXABS_RTOL, MAXABS_FLOOR * scale))
        metrics[key] = dict(A=ma, B=mb, judged=checks)
        passed &= all(c['ok'] for c in checks.values())
    flag_errors = {arm: int(np.count_nonzero(np.asarray(saved['flags'], dtype=bool) != flags))
                   for arm, saved, flags in (('A', a, fa), ('B', b, fb))}
    flag_judge = judge(flag_errors['A'], flag_errors['B'], 0.0, max(1.0, 1e-3 * len(q)))
    metrics['oracle_flag_disagreement_count'] = dict(**flag_errors, judged=flag_judge)
    passed &= flag_judge['ok']
    if ia['raw_logdet'] is not None and ib['raw_logdet'] is not None:
        lj = judge(ia['raw_logdet'], ib['raw_logdet'], 1e-6, 1e-6)
        metrics['raw_objective_logdet'] = dict(judged=lj)
        passed &= lj['ok']
    strict = all(c['strict_le_info'] for m in metrics.values()
                 for c in (m['judged'].values() if 'rel_l2' in m['judged'] else [m['judged']]))
    ab = {key: error_metrics(b[key], a[key]) for key in oa}
    return dict(status='PASS' if passed else 'HOLD', exact_info=exact, metrics=metrics,
                A_B_differences=ab, oracle_A=ia, oracle_B=ib, strict_le_all_info=strict,
                criterion=RULE + '; oracle per arm from its own supports; ranks equal',
                scope='float64 numerical accuracy per arm; no opponent refit '
                      '(the board metric vs the best opponent is judged on the board)')
