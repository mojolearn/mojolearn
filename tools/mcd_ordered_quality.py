#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M3-only unscored quality worker; no clock, benchmarking or opponent fit.
Worker: dump SOURCE ARM CASE OUT_DIRECTORY. Compare: compare CASE DIRECTORY.
Cases: direct, mcd-istella, ee-istella, mcd-taxi, ee-taxi, mcd-synthetic.
B repeats are reproducibility checks within this unscored fixture, never scored
A/B repetitions. Receipts use a new fixture, not the old permissive MCD gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import chi2
from mcd_ordered_oracle import analyze, error_metrics, json_ready
from mcd_compat_quality import load_inputs

FIXTURE = 'mcd-ordered-cov-v1'
SHAPES = [(1023, 11), (1024, 65), (1025, 11), (3000, 220), (5001, 65), (100000, 11)]
CASES = ['direct', 'mcd-istella', 'ee-istella', 'mcd-taxi', 'ee-taxi', 'mcd-synthetic']


def fixture(case):
    if case == 'mcd-synthetic':
        rng = np.random.default_rng(192)
        x = rng.normal(size=(1100, 17)).astype(np.float32)
        x[:, -1] = 0
        x[:, -2] = x[:, 0]
        q = x[:173].copy()
        return x, q, 'min-cov-det'
    lane = 'elliptic-envelope' if case.startswith('ee-') else 'min-cov-det'
    dataset = case.split('-', 1)[1]
    x, q = load_inputs(SimpleNamespace(data=str(Path.home()/'board-0834/cache/algos-data/rows-full'),
                      dataset=dataset, lane=lane, rows=3000 if dataset=='istella' else None))
    return x, q, lane


def inputs(index):
    n, d = SHAPES[index]
    rng = np.random.default_rng(812 + index)
    x = rng.normal(size=(3, n, d)) * np.geomspace(.125, 8., d)
    x[:, :, -1] = 0
    # Near cancellation in off-diagonal products, plus deterministic tails.
    x[:, 1::2, 0] *= -1
    return np.ascontiguousarray(x, dtype=np.float32)


def reach(b):
    return {name: int(b.x_decomp_mcd_cov_reach(i)) for i, name in
            enumerate(('enabled', 'raw', 'final', 'raw_split', 'final_split'))}


def perturb_pool(b):
    # Exercise differently sized resident allocations between repeated B calls.
    ids = [b.x_decomp_dev_alloc(n) for n in (133, 8193, 65537)]
    for ident in ids:
        b.x_decomp_dev_free(ident)


def direct_dump(b, arm, out):
    expected = int(arm == 'B')
    input_hashes = {}
    for repeat in range(2 if expected else 1):
        saved = {}
        order = range(len(SHAPES)) if repeat == 0 else reversed(range(len(SHAPES)))
        for i in order:
            x = inputs(i)
            n, d = SHAPES[i]
            input_hashes[str(i)] = hashlib.sha256(x.tobytes()).hexdigest()
            for gi, flags in enumerate(((1,0,1), (0,1,0), (0,0,0))):
                gates = np.array(flags, dtype=np.int32)
                result = np.empty((3,d,d), dtype=np.float32)
                r = b.x_decomp_mcd_cov_probe(x.ctypes.data, gates.ctypes.data, result.ctypes.data, [3,n,d])
                assert int(r) == expected
                assert np.array_equal(result[gates==0], np.full_like(result[gates==0], -123.5))
                saved[f'batch_{i}_{gi}'] = result.copy()
            # Actual final resident API with same input and pool reuse.
            xid = b.x_decomp_dev_alloc(n*d)
            yid = b.x_decomp_dev_alloc(d*d)
            try:
                xx = np.ascontiguousarray(x[0])
                b.x_decomp_dev_upload(xid, xx.ctypes.data, xx.size)
                if expected:
                    b.x_decomp_dev_mcd_cov(xid, yid, [n,d])
                else:
                    b.x_decomp_dev_gemm(xid, xid, yid, [d,n,d,1,0])
                result = np.empty((d,d), dtype=np.float32)
                b.x_decomp_dev_download(yid, result.ctypes.data, result.size)
                saved[f'resident_{i}'] = result.copy()
            finally:
                b.x_decomp_dev_free(xid)
                b.x_decomp_dev_free(yid)
            perturb_pool(b)
        np.savez_compressed(out/f'{arm}{repeat}.npz', **saved)
    rr = reach(b)
    if expected:
        assert rr['raw'] > 0 and rr['raw_split'] > 0 and rr['final'] > 0 and rr['final_split'] > 0, rr
    else:
        assert not any(rr.values()), rr
    return dict(counters=rr, input_sha256=input_hashes)


def model_dump(b, arm, case, out):
    import mojolearn as ml
    x, q, lane = fixture(case)
    cls = ml.MinCovDet if lane == 'min-cov-det' else ml.EllipticEnvelope
    reaches = []
    for repeat in range(2 if arm == 'B' else 1):
        b.x_decomp_mcd_cov_reach(-1)
        model = cls(random_state=7, numeric_mode='fast').fit(x)
        distances = np.asarray(model.mahalanobis(q), dtype=np.float64)
        saved = {key: np.asarray(getattr(model,key)).copy() for key in (
            'raw_location_','raw_covariance_','raw_support_', 'location_', 'covariance_',
            'precision_','support_','dist_')}
        saved['distances'] = distances
        if lane == 'elliptic-envelope':
            saved['offset_'] = np.asarray(model.offset_, dtype=np.float64)
            saved['decision_function'] = np.asarray(model.decision_function(q), dtype=np.float64)
            saved['flags'] = np.asarray(model.predict(q)) < 0
        else:
            saved['flags'] = distances > chi2.ppf(.975,x.shape[1])
        rr = reach(b)
        if arm == 'B':
            assert rr['raw'] > 0 and rr['final'] > 0 and rr['final_split'] > 0, rr
            if case != 'mcd-synthetic':
                assert rr['raw_split'] > 0, rr
        else:
            assert not any(rr.values()), rr
        reaches.append(rr)
        np.savez_compressed(out/f'{arm}{repeat}.npz', **saved)
        perturb_pool(b)
    return dict(repeats=reaches, input_sha256=hashlib.sha256(x.tobytes()+q.tobytes()).hexdigest(),
                shape=list(x.shape), lane=lane)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return dict(z)


def compare(case, out):
    a, b, repeat = [load(out/name) for name in ('A0.npz','B0.npz','B1.npz')]
    assert a.keys() == b.keys() == repeat.keys()
    stable = {key: b[key].shape == repeat[key].shape and b[key].dtype == repeat[key].dtype
              and b[key].tobytes() == repeat[key].tobytes() for key in b}
    if case == 'direct':
        metrics = {}
        for i, (n,d) in enumerate(SHAPES):
            x = inputs(i).astype(np.float64)
            refs = np.matmul(x.transpose(0,2,1),x)
            for gi, gates in enumerate(((1,0,1),(0,1,0),(0,0,0))):
                key = f'batch_{i}_{gi}'
                truth = refs.copy()
                truth[np.array(gates)==0] = -123.5
                metrics[key] = {arm: error_metrics(arr[key],truth) for arm,arr in (('A',a),('B',b))}
            key=f'resident_{i}'
            metrics[key] = {arm: error_metrics(arr[key],refs[0]) for arm,arr in (('A',a),('B',b))}
        for m in metrics.values():
            m['no_worse'] = {k:m['B'][k]<=m['A'][k] for k in m['A']}
        result=dict(status='PASS' if all(all(m['no_worse'].values()) for m in metrics.values()) else 'HOLD',metrics=metrics)
    else:
        x,q,lane=fixture(case)
        result=analyze(a,b,x,q,lane)
    result['repeat_identical']=stable
    if not all(stable.values()):
        result['status']='HOLD'
    # Require source/hash/input provenance agreement across capture workers.
    ma,mb=[json.loads((out/(arm+'.json')).read_text()) for arm in ('A','B')]
    assert ma['source']==mb['source'] and ma['case']==mb['case']==case
    assert ma['reach']['input_sha256']==mb['reach']['input_sha256']
    result['captures']={'A':ma,'B':mb}
    result['fixture']=FIXTURE
    result['timing']=False
    result=json_ready(result)
    with (out/'report.json').open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    for key,m in result.get('metrics',{}).items():
        print('MCD-ORDERED-METRIC '+json.dumps(dict(field=key,**m),sort_keys=True))
    print('MCD-ORDERED-QUALITY '+json.dumps(dict(case=case,status=result['status'],
          repeat_identical=all(stable.values()),report=str(out/'report.json')),sort_keys=True))
    return 0 if result['status']=='PASS' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('dump','compare'))
    p.add_argument('args',nargs='+')
    args=p.parse_args()
    if args.action=='compare':
        case,path=args.args
        assert case in CASES
        return compare(case,Path(path))
    source,arm,case,path=args.args
    assert case in CASES and arm in ('A','B')
    from mojolearn import _backend
    b=_backend.binding('_mojolearn_x_decomp','fast')
    assert int(b.x_decomp_numeric_mode())==0 and str(b.x_decomp_vendor())=='metal'
    assert int(b.x_decomp_mcd_cov_reach(0))==int(arm=='B')
    out=Path(path)
    b.x_decomp_mcd_cov_reach(-1)
    rr=direct_dump(b,arm,out) if case=='direct' else model_dump(b,arm,case,out)
    metadata=dict(source=source,arm=arm,case=case,fixture=FIXTURE,reach=rr,
                  binding_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
                  execution_policy='unscored quality; B repeat for reproducibility; no clock')
    with (out/(arm+'.json')).open('x') as f:
        json.dump(json_ready(metadata),f,sort_keys=True,allow_nan=False)
    print('MCD-ORDERED-CAPTURE '+json.dumps(json_ready(metadata),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
