#!/usr/bin/env python3
"""Cloud-only two-level categorical feature-partition identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action='store_true', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import ExperimentalTwoLevelFeatureFreq
    from mojolearn.parallel_ensemble import fit_feature_freq
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []
    for n in (65, 513):
        X = raw[:n*73].astype('<f4').reshape(n, 73)
        X[:, :33] %= np.float32(2)
        X[:, 33:50] %= np.float32(8)
        y = raw[1000000:1000000+n].astype('<f4') / np.float32(255)
        for border in (7,):
            for reverse in (False, True):
                weights = np.ones(n, dtype='<f4')
                weights[::7] = 0
                weights[::11] = 0.5
                X %= np.float32(8)
                options = dict(sources=(0, 34) if not reverse else (34, 0), random_state=7)
                one = ExperimentalTwoLevelFeatureFreq(**options).fit(X,y,sample_weight=weights)
                many = fit_feature_freq(ExperimentalTwoLevelFeatureFreq(**options),X,y,
                                        sample_weight=weights,devices=(0,1))
                assert one.model_ == many.model_
                assert one.loss_curve_ is None and many.loss_curve_ is None
                assert one.predict(X).tobytes() == many.predict(X).tobytes()
                before = many.model_
                try:
                    fit_feature_freq(many,X,y[:-1],devices=(0,1))
                except ValueError:
                    pass
                else:
                    raise AssertionError('invalid target accepted')
                assert many.model_ == before
                checks.append(dict(rows=n,border=border,reverse=reverse,
                    sha256=hashlib.sha256(many.model_.encode()+many.predict(X).tobytes()).hexdigest()))
                print('PASS two-level FeatureFreq',n,border,reverse,flush=True)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; original categorical generation and both levels with greedy feature-group histograms; no pooled state claim'),indent=2)+'\n')


if __name__ == '__main__':
    main()
