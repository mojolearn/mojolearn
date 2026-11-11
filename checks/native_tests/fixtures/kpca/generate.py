# SPDX-License-Identifier: Apache-2.0
"""Original KPCA inputs and independent NumPy kernel/centering/eigen references.

Each kernel profile and artifact write runs in the host-sized worker pool.
No Mojolearn, legacy implementation, fitted native output or identity run is
used. Two exact one-feature examples also use analytic nonzero eigenvalues.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def data(n, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, 6)).astype('f4')
    x[:5] *= 6.0
    return x


def kernel(x, y, kind, gamma):
    if kind == 'linear':
        return x @ y.T
    if kind == 'poly':
        return (gamma * (x @ y.T) + 1.0) ** 3
    if kind == 'cosine':
        return (x / np.linalg.norm(x, axis=1)[:, None]) @ (y / np.linalg.norm(y, axis=1)[:, None]).T
    return np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))


def prepare(profile):
    small = profile in ('positive2', 'signed2', 'zero2')
    x = np.array([[1.0], [2.0]] if profile == 'positive2' else ([[0.0], [0.0]] if profile == 'zero2' else [[2.0], [-1.0]]), dtype='f4') if small else data(80)
    q = np.array([[0.0], [3.0]], dtype='f4') if small else data(20, 3)
    kind = 'linear' if small else ('rbf' if profile == 'rbf02' else profile)
    gamma = 0.2 if profile == 'rbf02' else 1.0 / x.shape[1]
    k = kernel(x.astype('f8'), x.astype('f8'), kind, gamma)
    kh = kernel(q.astype('f8'), x.astype('f8'), kind, gamma)
    columns = k.mean(axis=0)
    total = k.mean()
    centered = k - columns[None, :] - k.mean(axis=1)[:, None] + total
    query_centered = kh - columns[None, :] - kh.mean(axis=1)[:, None] + total
    count = 2 if small else 3
    if small:
        values = np.array([0.5 if profile == 'positive2' else (0.0 if profile == 'zero2' else 4.5), 0.0], dtype='f8')
        axis = x[:, 0].astype('f8') - x[:, 0].mean(dtype='f8')
        vectors = np.eye(2) if profile == 'zero2' else np.column_stack([axis / np.linalg.norm(axis), np.ones(2) / np.sqrt(2.0)])
    else:
        values, vectors = np.linalg.eigh(centered)
        order = np.argsort(values)[::-1][:count]
        values = np.maximum(values[order], 0)
        vectors = vectors[:, order]
    alpha = np.zeros_like(vectors)
    positive = values > 0
    alpha[:, positive] = vectors[:, positive] / np.sqrt(values[positive])
    fit_scores = vectors * np.sqrt(values)[None, :]
    query_scores = query_centered @ alpha
    outputs = {'source.f32': x, 'query.f32': q, 'kernel.f64': k, 'query-kernel.f64': kh,
               'columns.f64': columns, 'total.f64': np.array([total]), 'centered.f64': centered,
               'query-centered.f64': query_centered, 'values.f64': values, 'vectors.f64': vectors,
               'fit-scores.f64': fit_scores, 'query-scores.f64': query_scores}
    return [(profile + '-' + suffix, value) for suffix, value in outputs.items()]


def write(item):
    name, values = item
    raw = values.astype('<f4' if name.endswith('.f32') else '<f8').tobytes()
    (ROOT / name).write_bytes(raw)
    return {'file': name, 'shape': list(values.shape), 'sha256': hashlib.sha256(raw).hexdigest()}


if __name__ == '__main__':
    profiles = ['rbf', 'linear', 'poly', 'cosine', 'rbf02', 'positive2', 'signed2', 'zero2']
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(prepare, profiles))
        records = list(pool.map(write, [item for group in groups for item in group]))
    metadata = {'source': 'test_x_neighbors_sanity.py::_data/test_kernel_pca/test_kernel_pca_options',
                'train_shape': [80, 6], 'query_shape': [20, 6], 'seeds': [0, 3],
                'first_five_rows_scale': 6, 'components': 3, 'degree': 3, 'coef0': 1.0,
                'default_gamma': 1.0/6, 'precomputed_gamma': 0.2,
                'small_analytic': {'positive2': {'global_mean': 2.25, 'eigenvalues': [0.5, 0.0]},
                                   'signed2': {'global_mean': 0.25, 'eigenvalues': [4.5, 0.0]},
                                   'zero2': {'global_mean': 0.0, 'eigenvalues': [0.0, 0.0]}},
                'numpy': np.__version__, 'workers': os.cpu_count(),
                'oracle': 'Independent Float64 NumPy kernel definitions, global mean centering and eigh; analytic tiny eigenvalues. No product/legacy code execution.',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'files': records, 'execution': 'Preparation only; not compilation/runtime/identity/performance evidence.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'files': len(records), 'profiles': len(profiles), 'numpy': np.__version__, 'metadata': str(ROOT/'metadata.json')}))
