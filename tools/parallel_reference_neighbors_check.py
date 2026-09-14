#!/usr/bin/env python3
"""Cloud-only reference-sharded KNN byte identity and partition gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', required=True, action='store_true')
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []

    def equal(a, b, digest):
        if isinstance(a, (tuple, list)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y, digest)
        else:
            assert a.shape == b.shape and a.dtype == b.dtype
            assert a.tobytes() == b.tobytes(), (a.tolist(), b.tolist())
            digest.update(a.tobytes())

    for d in (3, 129):
        X = raw[:41*d].astype('<f4').reshape(41, d) / np.float32(255)
        X[17] = X[3]  # identical references live in different shards
        X[38] = X[3]
        Q = X[:11].copy()
        Q[-1] = np.float32(1)
        for metric in ('euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'cosine'):
            model = NearestNeighbors(metric=metric, numeric_mode='identical').fit(X)
            with ReferenceShardedNeighbors(model, devices=(0, 1), reference_rows_per_shard=7,
                                            query_rows_per_shard=5) as pool:
                for k in (1, 13, 41):
                    digest = hashlib.sha256()
                    equal(model.kneighbors(Q, n_neighbors=k), pool.kneighbors(Q, n_neighbors=k), digest)
                    assert max(s['reference_end']-s['reference_start'] for s in pool.last_shards_) == 7
                    assert {s['device'] for s in pool.last_shards_} == {0, 1}
                    checks.append(dict(estimator=type(model).__name__, features=d, metric=metric,
                                       k=k, sha256=digest.hexdigest()))
                    print('PASS reference KNN', d, metric, k, flush=True)
        for cls in (KNeighborsClassifier, KNeighborsRegressor):
            for weights in ('uniform', 'distance'):
                for multi in (False, True):
                    y = raw[20000:20041].astype('<i4') % 3
                    if multi:
                        y = np.column_stack((y, y % 2))
                    if cls is KNeighborsRegressor:
                        y = y.astype('<f4') / np.float32(3)
                    model = cls(n_neighbors=13, weights=weights, numeric_mode='identical').fit(X, y)
                    methods = ('predict', 'predict_proba') if cls is KNeighborsClassifier else ('predict',)
                    with ReferenceShardedNeighbors(model, devices=(0, 1), reference_rows_per_shard=7,
                                                    query_rows_per_shard=5) as pool:
                        for method in methods:
                            digest = hashlib.sha256()
                            equal(getattr(model, method)(Q), getattr(pool, method)(Q), digest)
                            checks.append(dict(estimator=cls.__name__, features=d, weights=weights,
                                multioutput=multi, method=method, sha256=digest.hexdigest()))
                            print('PASS reference vote', cls.__name__, d, weights, multi, method, flush=True)
                        before = pool.last_shards_.copy()
                        try:
                            pool.kneighbors(Q, n_neighbors=42)
                        except ValueError:
                            pass
                        else:
                            raise AssertionError('invalid k accepted')
                        assert before == pool.last_shards_
        # Same logical reference/query partitions replay sequentially on one GPU.
        model = NearestNeighbors(n_neighbors=13, numeric_mode='identical').fit(X)
        with ReferenceShardedNeighbors(model, devices=(0,), reference_rows_per_shard=7,
                                      query_rows_per_shard=5) as replay:
            digest = hashlib.sha256()
            equal(model.kneighbors(Q), replay.kneighbors(Q), digest)
            checks.append(dict(estimator='one-device-replay', features=d, sha256=digest.hexdigest()))
            print('PASS reference one-device replay', d, flush=True)
    # Overflow produces NaN distances: native membership uses radix bits,
    # while its subsequent host insertion sort compares floats. Gate both.
    X = np.full((41, 3), np.float32(1e20), dtype='<f4')
    model = NearestNeighbors(n_neighbors=13, numeric_mode='identical').fit(X)
    with ReferenceShardedNeighbors(model, devices=(0, 1), reference_rows_per_shard=7) as pool:
        digest = hashlib.sha256()
        equal(model.kneighbors(X[:1]), pool.kneighbors(X[:1]), digest)
        checks.append(dict(estimator='overflow-distance-order', sha256=digest.hexdigest()))
        print('PASS reference overflow-distance order', flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; reference shards capped at seven rows; original bit-key merge and vote halves; no beyond-80GiB measurement'), indent=2)+'\n')


if __name__ == '__main__':
    main()
