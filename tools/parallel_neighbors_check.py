#!/usr/bin/env python3
"""Cloud-only bitwise query-sharding gate; never execute locally."""
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
    from mojolearn import (NearestNeighbors, RadiusNeighbors,
                           KNeighborsClassifier, KNeighborsRegressor, KernelDensity)
    from mojolearn.parallel_neighbors import ParallelQueries
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
            assert a.tobytes() == b.tobytes(), (a.shape, a.tolist(), b.tolist())
            digest.update(a.tobytes())

    def check(model, Q, method, **kwargs):
        expected = getattr(model, method)(Q, **kwargs)
        original = {k: getattr(model, k) for k in
                    ('used_query_tile_', 'n_candidate_distances_') if hasattr(model, k)}
        digest = hashlib.sha256()
        with ParallelQueries(model, devices=(0, 1), rows_per_shard=7) as driver:
            equal(expected, driver.query(Q, method=method, **kwargs), digest)
            assert {s['device'] for s in driver.last_shards_} == {0, 1}
            # A second wave uses the same workers and retains deterministic output.
            equal(expected, driver.query(Q, method=method, **kwargs), digest)
            diagnostics = driver.last_shards_.copy()
            try:
                driver.query(np.zeros((3, model.n_features_in_ + 1), dtype='<f4'), method=method)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid feature count accepted')
            assert driver.last_shards_ == diagnostics
            if method == 'kneighbors':
                try:
                    driver.query(Q, method=method, n_neighbors=model.n_samples_fit_ + 1)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('worker accepted too many neighbors')
                assert driver.last_shards_ == diagnostics
                equal(expected, driver.query(Q, method=method, **kwargs), digest)
        assert original == {k: getattr(model, k) for k in original}
        case = dict(estimator=type(model).__name__, method=method,
                    metric=model.metric, features=model.n_features_in_, kwargs=kwargs,
                    weights=getattr(model, 'weights', None),
                    kernel=getattr(model, 'kernel', None),
                    algorithm=getattr(model, 'algorithm', None),
                    multioutput=getattr(model, 'outputs_2d_', None),
                    sha256=digest.hexdigest())
        checks.append(case)
        print('PASS', case, flush=True)

    for d in (3, 129):
        X = raw[:41*d].astype('<f4').reshape(41, d) / np.float32(255)
        X[3] = X[2]
        Q = X[:17].copy()
        Q[-1] = np.float32(100)  # empty compact-kernel/radius row
        for metric in ('euclidean', 'manhattan', 'chebyshev', 'cosine'):
            model = NearestNeighbors(n_neighbors=5, metric=metric, numeric_mode='identical').fit(X)
            check(model, Q, 'kneighbors')
            check(model, Q, 'kneighbors', n_neighbors=1, return_distance=False)
        if d == 3:
            for metric in ('euclidean', 'manhattan', 'chebyshev'):
                model = NearestNeighbors(n_neighbors=5, algorithm='rbc', metric=metric,
                                         numeric_mode='identical').fit(X)
                check(model, Q, 'kneighbors')
                radius = RadiusNeighbors(radius=0.2, metric=metric, numeric_mode='identical').fit(X)
                check(radius, Q, 'radius_neighbors', sort_results=True)
                check(radius, Q, 'radius_neighbors', return_distance=False)
                check(radius, None, 'radius_neighbors')
        for cls in (KNeighborsClassifier, KNeighborsRegressor):
            for weights in ('uniform', 'distance'):
                for multi in (False, True):
                    y = raw[20000:20041].astype('<i4') % 3
                    if multi:
                        y = np.column_stack((y, y % 2))
                    if cls is KNeighborsRegressor:
                        y = y.astype('<f4') / np.float32(3)
                    model = cls(n_neighbors=5, weights=weights, numeric_mode='identical').fit(X, y)
                    check(model, Q, 'predict')
                    if cls is KNeighborsClassifier:
                        check(model, Q, 'predict_proba')
        if d == 3:
            for kernel in ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'):
                for weighted in (False, True):
                    weights = np.arange(41, dtype='<f4') if weighted else None
                    model = KernelDensity(kernel=kernel, bandwidth=0.5,
                                          numeric_mode='identical').fit(X, sample_weight=weights)
                    check(model, Q, 'score_samples')
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; whole query rows, full replicated reference index; no pooled-index or cross-vendor claim'), indent=2) + '\n')


if __name__ == '__main__':
    main()
