#!/usr/bin/env python3
"""Untimed RF/ET candidate state witnesses; fixtures, never runtime arithmetic.

RF drain-frequency candidates must match baseline model/prediction bits.
ET u16 changes the model: compare SAME arm across vendors/host, report baseline
RMSE/R2 differences separately. No accuracy threshold or default promotion is
implied. Full timing fixtures remain canonical forest_speed_arm RF taxi/istella
and ET REGRESSION istellareg/year, not the classification ET board rows.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

PROFILES = {'rf-k4': [], 'rf-k1': ['MOJOLEARN_IDN_RF_DEVICE_LOOP_K1=1'],
            'rf-k2': ['MOJOLEARN_IDN_RF_DEVICE_LOOP_K2=1'],
            'rf-k8': ['MOJOLEARN_IDN_RF_DEVICE_LOOP_K8=1'],
            'et-float': [], 'et-u16': ['MOJOLEARN_IDN_ET_BINNED_U16=1']}
SOURCE = 'a006da73d78634683cfc77a12dcb99666c55748c'


def arrays_hash(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        if a.dtype.hasobject: raise ValueError('object state: ' + key)
        for part in (json.dumps([key, a.dtype.str, list(a.shape)]).encode(), a.tobytes()):
            digest.update(len(part).to_bytes(8, 'little')); digest.update(part)
    return digest.hexdigest()


def fixtures(kind):
    dims = (15, 16, 17) if kind == 'rf' else (15, 16, 17, 63, 64, 65)
    for i, d in enumerate(dims):
        n = (1023, 1024, 1025)[i % 3]
        rng = np.random.default_rng(700 + d)
        # Discrete values exercise duplicate rows, tied splits, and exact borders.
        X = rng.integers(-8, 9, size=(n + 67, d)).astype(np.float32) / np.float32(8)
        X[1:9] = X[0]; X[:, -1] = 0
        y = ((X[:, 0] * 2 + X[:, 1] / 2 + X[:, 2] * X[:, 2])
             if kind == 'et' else ((X[:, 0] + X[:, 1]) > 0).astype(np.int64))
        for capped in (False, True):
            # ET cap arm also crosses the actual 2*k>=d binning eligibility.
            k = max(1, (d - 1) // 2) if capped and kind == 'et' else d
            yield dict(name=f'{kind}-n{n}-d{d}-cap{int(capped)}', X=X[:n].copy(),
                       y=y[:n].copy(), Xq=X[n:].copy(), yq=y[n:].copy(),
                       max_features=k, max_leaves=31 if capped else -1,
                       u16_eligible=(kind == 'et' and 2*k >= d))


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    import mojolearn as ml
    if ml.numeric_mode().lower() != 'identical': raise ValueError('IDENTICAL required')
    if ml.vendor().lower() != args.vendor: raise ValueError('unexpected imported vendor')
    report = dict(source_sha=SOURCE, profile=args.profile, declared_defines=PROFILES[args.profile],
                  package_file=ml.__file__, vendor=ml.vendor(), numeric_mode=ml.numeric_mode(),
                  status='RUNNING', cases=[], timing_vote=False)
    def save():
        tmp=out/'status.tmp'; tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(out/'status.json')
    save(); kind = args.profile.split('-')[0]
    for case in fixtures(kind):
        dest = out/case['name']; dest.mkdir(); rec=dict(name=case['name'], u16_eligible=case['u16_eligible'])
        try:
            inputs={k:case[k] for k in ('X','y','Xq','yq')}; np.savez(dest/'inputs.npz',**inputs)
            rec['input_hash']=arrays_hash(inputs)
            cls=ml.RandomForestClassifier if kind=='rf' else ml.ExtraTreesRegressor
            params=dict(n_estimators=3,max_depth=8,max_leaves=case['max_leaves'],
                        max_features=case['max_features'],random_state=7,device='gpu',
                        bootstrap=kind=='rf')
            if kind=='rf': params['n_bins']=128
            rec['params']=params; model=cls(**params); model.fit(case['X'],case['y'])
            model.save(dest/'model.npz')
            with np.load(dest/'model.npz',allow_pickle=False) as z:
                # Device metadata differs for some exporters. Keep entire archive;
                # bind every other member, including structure, leaves and classes.
                state={k:z[k].copy() for k in z.files if k!='device'}
            required={'offsets','colid','quesval','left_child','leaves','meta'}
            if not required <= state.keys(): raise ValueError('incomplete forest export')
            pred={'predict':np.asarray(model.predict(case['Xq']))}
            if kind=='rf': pred['proba']=np.asarray(model.predict_proba(case['Xq']))
            if any(not np.isfinite(v).all() for v in pred.values()): raise ValueError('nonfinite prediction')
            np.savez(dest/'predictions.npz',**pred)
            rec.update(status='CAPTURED', model_state_hash=arrays_hash(state), prediction_hash=arrays_hash(pred),
                       model_file_sha256=hashlib.sha256((dest/'model.npz').read_bytes()).hexdigest(),
                       state_members={k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in state.items()})
            if kind=='et':
                diff=pred['predict'].astype(np.float64)-case['yq']; mse=float(np.mean(diff*diff))
                denom=float(np.sum((case['yq']-np.mean(case['yq']))**2))
                rec['quality']={'rmse':float(np.sqrt(mse)), 'r2':1-float(np.sum(diff*diff))/denom}
        except Exception as exc: rec.update(status='FAILED',error=repr(exc))
        report['cases'].append(rec);save()
    report['bindings']={n:{'file':str(Path(m.__file__).resolve()),'sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
                        for n,m in list(sys.modules.items()) if n.startswith('mojolearn') and str(getattr(m,'__file__','')).endswith('.so')}
    report['status']='CAPTURE_COMPLETE_REQUIRES_COMPARISON' if all(c['status']=='CAPTURED' for c in report['cases']) else 'FAILED'
    save(); return 0 if report['status'].startswith('CAPTURE_COMPLETE') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--vendor',required=True);p.add_argument('--out',required=True)
    raise SystemExit(run(p.parse_args()))
