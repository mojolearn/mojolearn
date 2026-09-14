#!/usr/bin/env python3
"""Cloud-only classifier/regressor contracts through distributed boosting."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', action='store_true', required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import GradientBoostingClassifier, GradientBoostingRegressor
    from mojolearn.parallel_ensemble import fit_boosting
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    X = raw[:513*73].astype('<f4').reshape(513,73)
    X[:,:33] %= np.float32(2)
    X[:,33:50] %= np.float32(8)
    weights = np.ones(513,dtype='<f4')
    weights[::7] = 0
    checks = []
    for cls in (GradientBoostingClassifier, GradientBoostingRegressor):
        y = raw[1000000:1000513].astype('<f4') / np.float32(255)
        if cls is GradientBoostingClassifier:
            y = ['left' if v < .4 else 'right' for v in y]
        for pointwise in (False, True):
            opts = dict(n_estimators=3,max_depth=3,border_count=31,random_state=7,
                        use_pointwise_searcher=pointwise,bootstrap_type='No',numeric_mode='identical')
            kwargs = dict(sample_weight=weights,eval_set=(X[:17],y[:17]))
            one = cls(**opts).fit(X,y,**kwargs)
            many = fit_boosting(cls(**opts),X,y,devices=(0,1),**kwargs)
            assert one.model_ == many.model_
            for name in ('loss_curve_','test_loss_curve_'):
                assert getattr(one,name).tobytes() == getattr(many,name).tobytes()
            digest = hashlib.sha256(many.model_.encode())
            if cls is GradientBoostingClassifier:
                assert one.classes_ == many.classes_
                assert list(one.predict(X)) == list(many.predict(X))
                for method in ('decision_function','predict_proba'):
                    a,b = getattr(one,method)(X),getattr(many,method)(X)
                    assert a.tobytes() == b.tobytes()
                    digest.update(b.tobytes())
            else:
                assert one.predict(X).tobytes() == many.predict(X).tobytes()
                digest.update(many.predict(X).tobytes())
            before = many.model_
            try:
                fit_boosting(many,X,y[:-1],devices=(0,1))
            except RuntimeError:
                pass
            else:
                raise AssertionError('invalid target accepted')
            assert many.model_ == before
            checks.append(dict(estimator=cls.__name__,pointwise=pointwise,sha256=digest.hexdigest()))
            print('PASS adapter',cls.__name__,pointwise,flush=True)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest()),indent=2)+'\n')


if __name__ == '__main__':
    main()
