#!/usr/bin/env python3
"""RunPod-only cooperative KMeans identity gate using R2 corpus bytes."""
import argparse
import hashlib
import struct
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
    from mojolearn import KMeans
    from mojolearn.parallel_classical import fit_kmeans
    source = np.frombuffer(args.corpus.read_bytes()[:65536], dtype=np.uint8)
    checks = []
    hashes = []
    for features in (7, 8, 32):
        X = source[:257 * features].astype('<f4').reshape(257, features) / np.float32(255)
        for init in ('array', 'random', 'k-means++'):
            params = dict(n_clusters=4, init=init, n_init=2, max_iter=4, random_state=213,
                          numeric_mode='identical')
            if init == 'array':
                params['init_centroids'] = X[:4].copy()
            serial = KMeans(**params).fit(X)
            parallel = fit_kmeans(KMeans(**params), X, devices=(0, 1))
            for key in ('cluster_centers_', 'labels_'):
                assert getattr(serial, key).tobytes() == getattr(parallel, key).tobytes(), (features, init, key)
            for key in ('inertia_', 'n_iter_', 'sum_scale_', 'weight_scale_'):
                assert getattr(serial, key) == getattr(parallel, key), (features, init, key)
            checks.append([features, init])
            hashes.append(hashlib.sha256(parallel.cluster_centers_.tobytes() + parallel.labels_.tobytes()
                + struct.pack('<didd', parallel.inertia_, parallel.n_iter_, parallel.sum_scale_, parallel.weight_scale_)).hexdigest())
    args.report.write_text(json.dumps(dict(status='PASS', checks=checks, hashes=hashes,
        scope='Two devices; whole-row assignment tiles, serial update order'), indent=2) + '\n')
    print(args.report.read_text())


if __name__ == '__main__':
    main()
