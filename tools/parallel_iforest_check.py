#!/usr/bin/env python3
"""Cloud-only IsolationForest tree partition and ordered score equality."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', required=True, action='store_true')
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import IsolationForest
    from mojolearn.parallel_ensemble import fit_isolation_forest, score_isolation_forest
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    X = raw[:257 * 7].astype('<f4').reshape(257, 7) / np.float32(255)
    X[:, -1] = 1
    query = X[::9].copy()
    checks = []
    for trees in (1, 5):
        for bootstrap in (False, True):
            for contamination in ('auto', 0.2):
                params = dict(n_estimators=trees, max_samples=33, max_depth=5,
                              max_features=0.6 if bootstrap else 1.0, bootstrap=bootstrap,
                              contamination=contamination, random_state=42, numeric_mode='identical')
                one = IsolationForest(**params).fit(X)
                many = fit_isolation_forest(IsolationForest(**params), X, devices=(0, 1))
                digest = hashlib.sha256()
                for name in ('offset_', 'max_samples_', 'n_features_in_'):
                    a, b = struct.pack('<d', getattr(one, name)), struct.pack('<d', getattr(many, name))
                    assert a == b, (params, name)
                    digest.update(a)
                for method in ('score_samples', 'decision_function', 'predict'):
                    a = getattr(one, method)(query)
                    b = score_isolation_forest(many, query, devices=(0, 1), method=method)
                    assert a.shape == b.shape and a.tobytes() == b.tobytes(), (params, method)
                    digest.update(b.tobytes())
                before = many._x.tobytes(), many.offset_
                invalid = X.copy()
                invalid[-1, -1] = np.nan
                try:
                    fit_isolation_forest(many, invalid, devices=(0, 1))
                except Exception:
                    pass
                else:
                    raise AssertionError('NaN input accepted')
                assert before == (many._x.tobytes(), many.offset_)
                checks.append(dict(params=params, sha256=digest.hexdigest()))
                print('PASS', trees, bootstrap, contamination, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; whole-tree builds and original scoring; full data replicated and assembled model on root'), indent=2) + '\n')


if __name__ == '__main__':
    main()
