# SPDX-License-Identifier: Apache-2.0
"""Independent IPCA covariance/eigensystem references on original inputs.

The product stacks centered observations and previous SVD rows. This oracle
updates a covariance matrix, diagonalizes it with NumPy in Float64, and reads
mean/population variance directly from all observations. The repeated batch
retains the specified rank-two approximation, not an exact full-PCA stand-in.
Independent scenarios and file writes use the host-sized worker pool. No
Mojolearn code or duplicated Python implementation is imported or executed.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
X = np.array([[1, 2, 1], [3, 1, 2], [1, 1, 4], [2, 3, 1], [1, 2, 2], [4, 1, 1]], dtype='f4')


def prepare(name):
    batches = [X[:3]] if name == 'first3' else ([X[:3], X[3:]] if name == 'all6' else [X, X])
    scatter = np.zeros((3, 3), dtype='f8')
    seen = 0
    mean = np.zeros(3, dtype='f8')
    observations = []
    for batch in batches:  # Dependent incremental observation history.
        batch = batch.astype('f8')
        observations.append(batch)
        centered = batch - batch.mean(axis=0)
        correction = np.outer(mean - batch.mean(axis=0), mean - batch.mean(axis=0))
        scatter += centered.T @ centered + correction * (seen * len(batch) / (seen + len(batch)))
        seen += len(batch)
        all_values = np.concatenate(observations)
        mean = all_values.mean(axis=0)
        values, vectors = np.linalg.eigh(scatter)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 0)
        vectors = vectors[:, order].T
        # Only the retained covariance participates in the next update.
        scatter = (vectors[:2].T * values[:2]) @ vectors[:2]
    population_variance = all_values.var(axis=0)
    outputs = {'mean': mean, 'population-variance': population_variance,
               'singular': np.sqrt(values), 'vectors': vectors,
               'explained': values / (seen - 1),
               'ratio': values / (population_variance.sum() * seen),
               'noise': np.array([values[2] / (seen - 1)], dtype='f8')}
    return [(name + '-' + suffix + '.f64', value) for suffix, value in outputs.items()]


def write(item):
    name, values = item
    raw = values.astype('<f4' if name.endswith('.f32') else '<f8').tobytes()
    (ROOT / name).write_bytes(raw)
    return {'file': name, 'shape': list(values.shape), 'sha256': hashlib.sha256(raw).hexdigest()}


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(prepare, ['first3', 'all6', 'repeated12']))
        records = list(pool.map(write, [('source.f32', X)] + [item for group in groups for item in group]))
    metadata = {'source': 'test_ipca_native_boundary.py::_data/_model and partial-fit running count',
                'shape': [6, 3], 'components': 2, 'batch_size': 3,
                'numpy': np.__version__, 'workers': os.cpu_count(),
                'oracle': 'Float64 covariance update/eigh, with explicit rank-two retention; direct full-history mean/population variance. No product execution.',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'files': records, 'execution': 'Reference preparation only; no compile/runtime/identity/performance result.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'files': len(records), 'numpy': np.__version__, 'metadata': str(ROOT/'metadata.json')}))
