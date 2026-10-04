#!/usr/bin/env python3
"""Actual-caller quality gate for MOJOLEARN_MCD_FAST_G1_GRAM (MinCovDet and
EllipticEnvelope). M3 only; no timing, no build, no fallback.

Arm 0 (A): x_decomp built with -D MOJOLEARN_MCD_FAST_G1_GRAM_AUDIT.
Arm 1 (B): A plus -D MOJOLEARN_MCD_FAST_G1_GRAM.
One fresh process per capture. The public estimator resolves x_decomp through
mojolearn._backend.binding; that module's file must be --binding and its
sha256 must be --binding-sha256. The gate assumes no shape window: cases are a
generic spread of features x rows. The AUDIT exports (mcd_g1_gram_on/_count/
_last) decide reach per case: B must select every eligible launch and A none.
A case the route refuses (B eligible == 0) is reported as REFUSED, never PASS.
compare checks both arms against an independent FP64 NumPy reference built
from each arm's own fitted supports: B must be <= A on every error metric
(zero allowance), supports, labels and ranks equal.
"""
import argparse, hashlib, json, math, os, subprocess, zlib
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
if not __debug__: raise RuntimeError('Assertions required')

POLICY = 'mcd-g1-gram-real-fit-zero-regression-v1'
SEED = 2026100419      # estimator random_state, every case
QROWS = 2048
CONTAM = 0.1           # data contamination and EllipticEnvelope contamination
# Generic spread, no shape window: every feature count x every row count.
FEATURES = (8, 33, 64, 100, 300, 700, 1500)
ROWS = (2000, 20000, 200000)
CASES = {'n%d-d%d' % (n, d): (n, d) for d in FEATURES for n in ROWS}
LANES = ('min-cov-det', 'elliptic-envelope')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, data):
    with Path(p).open('x') as f: json.dump(data, f, indent=2, allow_nan=False)


def support_size(case):
    n, d = CASES[case]
    return int(math.ceil(0.5 * (n + d + 1)))


def phases(case):
    """MinCovDet._fast_mcd_native's plan: the covariance K (rows per candidate
    support) of each search phase, in launch order, as (name, K, nc, r)."""
    n, d = CASES[case]
    h = support_size(case)
    if n <= 500:
        return [('A', h, 30, n), ('C', h, 10, n)]
    n_sub = n // 300
    n_ss = n // n_sub
    h_sub = int(math.ceil(n_ss * (h / float(n))))
    n_trials = max(10, 500 // n_sub)
    n_m = min(1500, n)
    h_m = int(math.ceil(n_m * (h / float(n))))
    out = [('A', h_sub, n_sub * n_trials, n_ss), ('B', h_m, n_sub * min(10, n_trials), n_m)]
    if n >= 1500:
        out.append(('C', h, 10 if n > 1500 else 1, n))
    return out


def make(case):
    import numpy as np
    n, d = CASES[case]
    rng = np.random.default_rng(zlib.crc32(('mcd-g1|' + case).encode()))
    L = rng.normal(size=(d, d)) / np.sqrt(d)
    chol = np.linalg.cholesky(L @ L.T + .5 * np.eye(d))
    mu = rng.normal(size=d)
    shift = rng.normal(size=d)
    shift *= 4.0 / np.linalg.norm(shift) * np.sqrt(d) ** .5

    def draw(rows):
        z = rng.normal(size=(rows, d)) @ chol.T + mu
        bad = rng.random(rows) < CONTAM
        z[bad] = 3.0 * (z[bad] - mu) + mu + shift
        return z.astype(np.float32)
    return draw(n), draw(QROWS)


def capture(a):
    import numpy as np
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert source == a.source and len(source) == 40
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    binary = Path(a.binding).resolve(strict=True); assert sha(binary) == a.binding_sha256
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists() and not Path(str(output) + '.json').exists()
    write(str(output) + '.started.json', dict(source=source, case=a.case, lane=a.lane, arm=a.arm, status='RUNNING'))
    import mojolearn as ml
    from mojolearn import _backend
    binding = _backend.binding('_mojolearn_x_decomp', 'fast')
    assert Path(binding.__file__).resolve(strict=True) == binary, binding.__file__
    assert int(binding.x_decomp_numeric_mode()) == 0 and str(binding.x_decomp_vendor()) == 'metal'
    assert int(binding.mcd_g1_gram_on()) == a.arm
    resolved = []
    original = _backend.binding

    def pinned(name, mode=None):
        module = original(name, mode)
        if name == '_mojolearn_x_decomp':
            resolved.append(str(Path(module.__file__).resolve(strict=True)))
        return module
    _backend.binding = pinned

    def snap():
        return dict(counts=[int(binding.mcd_g1_gram_count(i)) for i in range(4)],
                    last=[int(binding.mcd_g1_gram_last(i)) for i in range(7)])
    n, d = CASES[a.case]
    h = support_size(a.case)
    x, q = make(a.case)
    s0 = snap()
    assert s0['counts'] == [0] * 4 and s0['last'] == [0] * 7, 'not a fresh process'
    cls = ml.MinCovDet if a.lane == 'min-cov-det' else ml.EllipticEnvelope
    kw = dict(random_state=SEED, numeric_mode='fast')
    if a.lane == 'elliptic-envelope':
        kw['contamination'] = CONTAM
    model = cls(**kw)
    arrays = dict(x=x, q=q, h=np.array(h))
    fit_error = None
    try:
        model.fit(x)
    except Exception as e:  # recorded, never retried; compare requires both arms identical
        fit_error = type(e).__name__ + ': ' + str(e)
    s1 = snap()
    audit = dict(before=s0, fit=s1)
    if fit_error is None:
        for name in ('raw_location_', 'raw_covariance_', 'raw_support_', 'location_', 'covariance_',
                     'precision_', 'support_', 'dist_'):
            arrays[name] = np.asarray(getattr(model, name))
        arrays['query_dist'] = np.asarray(model.mahalanobis(q), dtype=np.float32)
        if a.lane == 'elliptic-envelope':
            arrays['offset'] = np.array(float(model.offset_))
            arrays['query_decision'] = np.asarray(model.decision_function(q))
            arrays['query_labels'] = np.asarray(model.predict(q)).astype(np.int32)
        s2 = snap()
        audit['query'] = s2
        assert s2 == s1, 'G1 audit moved outside fit'
        for k, v in arrays.items():
            assert np.isfinite(v).all(), k
        assert int(np.count_nonzero(arrays['raw_support_'])) == h
    assert resolved and set(resolved) == {str(binary)}, resolved
    # reach / refusal from the audit deltas (s0 is all zero); refusal is an outcome
    c, last = s1['counts'], s1['last']
    plan = phases(a.case)
    assert c[0] >= c[1] and c[2] <= c[1]
    assert c[2] == (c[1] if a.arm else 0), 'arm selection mismatch'
    if c[1]:
        reach = 'REACHED'
        # the last eligible launch is a d x d self-Gram over one planned phase K
        assert last[0] == d and last[2] == d and last[1] in [p[1] for p in plan], (last, plan)
        assert last[4] >= d * last[1] and last[5] >= d * d and c[3] >= last[3] > 0
    else:
        reach = 'REFUSED' if c[0] else 'NO_BATCHED_ROUTE'
        assert c[3] == 0 and last == [0] * 7
    if fit_error is not None:
        reach = 'FIT_ERROR'
    with output.open('xb') as f: np.savez(f, **arrays)
    write(str(output) + '.json', dict(
        policy=POLICY, source=source, case=a.case, lane=a.lane, arm=a.arm, n=n, d=d, h=h, binding=str(binary), binding_sha256=a.binding_sha256,
        resolved_bindings=sorted(set(resolved)), capture_sha256=sha(output),
        input_sha256=hashlib.sha256(x.tobytes() + q.tobytes()).hexdigest(),
        planned_phases=[list(p) for p in plan], reach=reach,
        audit=audit, fit_error=fit_error, scored_timings=0, promotion_authorized=False))
    print('MCDG1-CAPTURE ' + json.dumps(dict(case=a.case, lane=a.lane, arm=a.arm, reach=reach, counts=c, last=last,
          fit_error=fit_error, output=str(output))), flush=True)
    return 0


def reference(arm, lane, d):
    """Independent FP64 NumPy: every fitted quantity recomputed from the arm's
    own supports (raw: MCD raw estimate; final: reweighted, Pison factor)."""
    import numpy as np
    from scipy.stats import chi2
    x = arm['x'].astype(np.float64); q = arm['q'].astype(np.float64)
    raw = arm['raw_support_'].astype(bool); fin = arm['support_'].astype(bool)
    rl = x[raw].mean(axis=0); rc = (x[raw] - rl).T @ (x[raw] - rl) / raw.sum()
    loc = x[fin].mean(axis=0); cov = (x[fin] - loc).T @ (x[fin] - loc) / fin.sum()
    alpha = .975
    cov *= alpha / chi2.cdf(chi2.ppf(alpha, d), d + 2)
    w, V = np.linalg.eigh(cov)
    cut = np.max(np.abs(w)) * d * np.finfo(np.float32).eps
    inv = np.where(np.abs(w) > cut, 1.0 / np.where(w == 0, 1, w), 0.0)
    prec = (V * inv) @ V.T

    def mahal(z):
        zc = z - loc
        return np.einsum('ij,jk,ik->i', zc, prec, zc)
    ref = dict(raw_location_=rl, raw_covariance_=rc, location_=loc, covariance_=cov, precision_=prec,
               dist_=mahal(x), query_dist=mahal(q))
    sign, logdet = np.linalg.slogdet(rc)
    ref['raw_objective'] = logdet if sign > 0 else None
    if lane == 'elliptic-envelope':
        offset = np.percentile(-ref['dist_'], 100.0 * CONTAM)
        ref['offset'] = offset
        ref['query_decision'] = -ref['query_dist'] - offset
        ref['query_labels'] = np.where(ref['query_decision'] < 0, -1, 1)
    else:
        ref['query_labels'] = np.where(ref['query_dist'] > chi2.ppf(alpha, d), -1, 1)
    return ref


def compare(a):
    import numpy as np
    A, B = [np.load(p, allow_pickle=False) for p in (a.a, a.b)]
    ma, mb = [json.loads(Path(str(p) + '.json').read_text()) for p in (a.a, a.b)]
    for key in ('policy', 'source', 'case', 'lane', 'input_sha256', 'n', 'd', 'h'):
        assert ma[key] == mb[key], key
    assert ma['policy'] == POLICY and ma['arm'] == 0 and mb['arm'] == 1
    assert ma['capture_sha256'] == sha(a.a) and mb['capture_sha256'] == sha(a.b)
    assert all(np.array_equal(A[k], B[k]) for k in ('x', 'q', 'h'))
    common = dict(policy=POLICY, source=ma['source'], case=ma['case'], lane=ma['lane'],
                  binding_A=ma['binding_sha256'], binding_B=mb['binding_sha256'],
                  audit_A=ma['audit'], audit_B=mb['audit'], reach_A=ma['reach'], reach_B=mb['reach'],
                  degradation_allowance=0, scored_timings=0, promotion_authorized=False)
    if ma['fit_error'] or mb['fit_error']:
        same = ma['fit_error'] == mb['fit_error']
        status = 'FIT_ERROR_BOTH' if same else 'HOLD'
        write(a.output, dict(common, status=status, fit_error_A=ma['fit_error'], fit_error_B=mb['fit_error']))
        print('MCDG1-COMPARE ' + json.dumps(dict(case=ma['case'], lane=ma['lane'], status=status)), flush=True)
        return 1
    lane, d = ma['lane'], ma['d']
    refs = [reference(arm, lane, d) for arm in (A, B)]
    metrics = {}

    def le(name, av, bv):
        metrics[name] = dict(A=float(av), B=float(bv), ok=bool(bv <= av))
    for key in ('raw_location_', 'raw_covariance_', 'location_', 'covariance_', 'precision_', 'dist_',
                'query_dist') + (('query_decision',) if lane == 'elliptic-envelope' else ()):
        errs = []
        for arm, ref in zip((A, B), refs):
            diff = arm[key].astype(np.float64) - ref[key]
            errs.append((np.linalg.norm(diff) / max(np.linalg.norm(ref[key]), 1e-300), np.max(np.abs(diff))))
        le(key + 'rel' if key.endswith('_') else key + '_rel', errs[0][0], errs[1][0])
        le(key + 'maxabs' if key.endswith('_') else key + '_maxabs', errs[0][1], errs[1][1])
    if refs[0]['raw_objective'] is not None and refs[1]['raw_objective'] is not None:
        le('raw_objective_logdet', refs[0]['raw_objective'], refs[1]['raw_objective'])
    if lane == 'elliptic-envelope':
        le('offset_abs', abs(float(A['offset']) - refs[0]['offset']), abs(float(B['offset']) - refs[1]['offset']))
        labels = [arm['query_labels'] for arm in (A, B)]
    else:
        from scipy.stats import chi2
        thr = chi2.ppf(.975, d)
        labels = [np.where(arm['query_dist'].astype(np.float64) > thr, -1, 1) for arm in (A, B)]
    le('label_disagreements_vs_ref', np.count_nonzero(labels[0] != refs[0]['query_labels']),
       np.count_nonzero(labels[1] != refs[1]['query_labels']))
    equal = {}

    def jaccard(key):
        av, bv = A[key].astype(bool), B[key].astype(bool)
        union = int(np.count_nonzero(av | bv))
        return float(np.count_nonzero(av & bv) / union) if union else 1.0
    for key in ('raw_support_', 'support_'):
        equal[key + 'jaccard'] = jaccard(key)
        equal[key + 'equal'] = bool(np.array_equal(A[key].astype(bool), B[key].astype(bool)))
    equal['labels_equal'] = bool(np.array_equal(labels[0], labels[1]))
    for key in ('raw_covariance_', 'covariance_'):
        ra, rb = [int(np.linalg.matrix_rank(arm[key].astype(np.float64))) for arm in (A, B)]
        equal[key + 'rank'] = dict(A=ra, B=rb, ref=int(np.linalg.matrix_rank(refs[0][key])))
        equal[key + 'rank_equal'] = ra == rb
    good = (all(m['ok'] for m in metrics.values())
            and all(v for k, v in equal.items() if k.endswith('_equal')))
    # quality is still scored for a refused case, but refusal is never a PASS
    status = ('PASS' if mb['reach'] == 'REACHED' else 'REFUSED') if good else 'HOLD'
    write(a.output, dict(common, status=status, metrics=metrics, equality=equal))
    print('MCDG1-COMPARE ' + json.dumps(dict(case=ma['case'], lane=lane, status=status, reach_B=mb['reach'],
          failed=[k for k, m in metrics.items() if not m['ok']]
          + [k for k, v in equal.items() if k.endswith('_equal') and not v])), flush=True)
    return 0 if status == 'PASS' else (2 if status == 'REFUSED' else 1)


def main():
    assert subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() == 'Apple M3 Ultra'
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='mode', required=True)
    c = s.add_parser('capture')
    for key in ('source', 'binding', 'binding-sha256', 'output'): c.add_argument('--' + key, required=True)
    c.add_argument('--case', choices=CASES, required=True)
    c.add_argument('--lane', choices=LANES, required=True)
    c.add_argument('--arm', type=int, choices=(0, 1), required=True)
    c = s.add_parser('compare')
    for key in ('a', 'b', 'output'): c.add_argument('--' + key, required=True)
    a = p.parse_args(); return capture(a) if a.mode == 'capture' else compare(a)


if __name__ == '__main__': raise SystemExit(main())
