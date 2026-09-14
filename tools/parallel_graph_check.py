#!/usr/bin/env python3
"""Cloud-only complete graph-estimator outputs with native row partitions."""
import argparse
import hashlib
import json
import os
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
    from mojolearn import AgglomerativeClustering, SpectralClustering, UMAP
    from mojolearn.parallel_graph import fit_graph, transform_umap
    corpus = args.corpus.read_bytes()
    raw = np.frombuffer(corpus, dtype=np.uint8)
    checks = []

    def compare(a, b, names):
        digest = hashlib.sha256()
        for name in names:
            x, y = getattr(a, name), getattr(b, name)
            if hasattr(x, 'tobytes'):
                assert x.shape == y.shape and x.dtype == y.dtype, name
                x, y = x.tobytes(), y.tobytes()
                digest.update(name.encode()+y)
            assert x == y, name
        return digest.hexdigest()

    for n in (17, 65):
        X = raw[:n*7].astype('<f4').reshape(n, 7) / np.float32(255)
        X[5] = X[4]
        for clusters in (1, 3, n):
            one = AgglomerativeClustering(n_clusters=clusters).fit(X)
            many = fit_graph(AgglomerativeClustering(n_clusters=clusters), X, devices=(0, 1))
            digest = compare(one, many, ('labels_', 'children_', 'n_clusters_', 'n_leaves_',
                'n_boruvka_rounds_', 'n_connected_components_', 'n_features_in_'))
            before = many.children_.tobytes()
            try:
                fit_graph(many, X[:1], devices=(0, 1))
            except Exception:
                pass
            else:
                raise AssertionError('one-row hierarchy accepted')
            assert many.children_.tobytes() == before
            checks.append(dict(estimator='AgglomerativeClustering', rows=n, clusters=clusters, sha256=digest))
            print('PASS hierarchy', n, clusters, flush=True)
    for n in (33, 65):
        X = raw[10000:10000+n*7].astype('<f4').reshape(n, 7) / np.float32(255)
        X[5] = X[4]
        for components in (2, 3):
            params = dict(n_clusters=2, n_components=components, n_neighbors=7,
                          n_init=1, random_state=7, eigen_tol=1e-3)
            one = SpectralClustering(**params).fit(X)
            many = fit_graph(SpectralClustering(**params), X, devices=(0, 1))
            digest = compare(one, many, ('labels_', 'embedding_', 'n_components_', 'n_features_in_', 'input_copied_'))
            checks.append(dict(estimator='SpectralClustering', rows=n, components=components, sha256=digest))
            print('PASS spectral', n, components, flush=True)
        for components, negatives in ((2, 0), (3, 3)):
            params = dict(n_neighbors=7, n_components=components, n_epochs=3,
                          random_state=7, negative_sample_rate=negatives, numeric_mode='identical')
            one = UMAP(**params).fit(X)
            many = fit_graph(UMAP(**params), X, devices=(0, 1))
            digest = compare(one, many, ('embedding_', '_transform_training', '_transform_embedding',
                '_transform_config', '_transform_mode', 'n_features_in_', 'input_copied_'))
            Q = X[:11].copy()
            a, b = one.transform(Q), transform_umap(many, Q, devices=(0, 1))
            assert a.shape == b.shape and a.tobytes() == b.tobytes()
            before = many.embedding_.tobytes()
            try:
                fit_graph(many, X[:3], devices=(0, 1))
            except Exception:
                pass
            else:
                raise AssertionError('too-small UMAP accepted')
            assert many.embedding_.tobytes() == before
            checks.append(dict(estimator='UMAP', rows=n, components=components, negatives=negatives,
                               sha256=digest, transform_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
            print('PASS UMAP fit/transform', n, components, negatives, flush=True)
    # No neighbor search here. More than two assignment tiles make the
    # precomputed-affinity path reach the existing multi-GPU KMeans driver.
    n = 257
    A = raw[30000:30000+n*n].astype('<f4').reshape(n, n) / np.float32(255) + np.float32(1)
    A = (A+A.T) * np.float32(0.5)
    np.fill_diagonal(A, 0)
    params = dict(n_clusters=3, n_components=2, n_init=1, random_state=7,
                  eigen_tol=1e-2, affinity='precomputed')
    one = SpectralClustering(**params).fit(A)
    many = fit_graph(SpectralClustering(**params), A, devices=(0, 1))
    digest = compare(one, many, ('labels_', 'embedding_', 'n_components_'))
    checks.append(dict(estimator='SpectralClustering-precomputed', rows=n, sha256=digest))
    print('PASS spectral precomputed', n, flush=True)
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; native distance/neighbor rows and original root graph/eigensolver/optimizer order; full root state remains'), indent=2)+'\n')


if __name__ == '__main__':
    main()
