# SPDX-License-Identifier: Apache-2.0
"""Original LOF inputs and independent Float64 mathematical references.

The oracle uses NumPy whole-array Euclidean norms, stable full sorts and
vectorized reachability-density definitions. It does not import Mojolearn,
its old Python implementation or any product kernel. Independent k variants
and file writes use a host-sized worker pool. This prepares fixtures only.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def data(n=60, d=5, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype('f4')
    x[::9] *= 6.0
    return x


def neighbors(query, reference, k, exclude_self=False):
    distance = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    if exclude_self:
        np.fill_diagonal(distance, np.inf)
    indices = np.argsort(distance, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(distance, indices, axis=1), indices


def density(distances, indices, reference_distances):
    reachability = np.maximum(distances, reference_distances[indices, -1])
    return 1.0 / (reachability.mean(axis=1) + 1e-10)


def prepare(k):
    x = data().astype('f8')
    distances, indices = neighbors(x, x, k, True)
    lrd = density(distances, indices, distances)
    scores = -(lrd[indices] / lrd[:, None]).mean(axis=1)
    offsets = np.r_[-1.5, np.percentile(scores, [10, 25, 50]).astype('f4').astype('f8')]
    records = [(f'fit-k{k}-distances.f64', distances), (f'fit-k{k}-density.f64', lrd),
               (f'fit-k{k}-scores.f64', scores), (f'fit-k{k}-offsets.f64', offsets)]
    if k == 6:
        query = data(25, seed=3).astype('f8')
        query_distances, query_indices = neighbors(query, x, k)
        query_lrd = density(query_distances, query_indices, distances)
        query_scores = -(lrd[query_indices] / query_lrd[:, None]).mean(axis=1)
        records += [('query-distances.f64', query_distances), ('query-density.f64', query_lrd),
                    ('query-scores.f64', query_scores),
                    ('query-offsets.f64', np.array([-1.5, np.float32(np.percentile(scores, 20))], dtype='f8'))]
    return records


def write(item):
    name, value = item
    raw = value.astype('<f4' if name.endswith('.f32') else '<f8').tobytes()
    (ROOT / name).write_bytes(raw)
    return {'file': name, 'shape': list(value.shape), 'sha256': hashlib.sha256(raw).hexdigest()}


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        groups = list(pool.map(prepare, [1, 7, 59, 6]))
        records = list(pool.map(write, [('fit.f32', data()), ('query.f32', data(25, seed=3))]
                               + [item for group in groups for item in group]))
    metadata = {'source': 'test_neighbors_loc_native.py::_data and original LOF parameterizations',
                'numpy': np.__version__, 'workers': os.cpu_count(),
                'source_dimensions': {'fit': [60, 5], 'query': [25, 5]},
                'source_seeds': {'fit': 0, 'query': 3}, 'k': [1, 7, 59, 6],
                'k_admission': 'Public k=200 is clamped to 59 by the retained Python-boundary check.',
                'oracle': 'Float64 Euclidean norm/full sort and mathematical reachability-density ratios; independent NumPy percentiles. No product code execution.',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'files': records, 'execution': 'Fixture preparation only; no compile, runtime test, timing or identity result.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'files': len(records), 'numpy': np.__version__, 'metadata': str(ROOT/'metadata.json')}))
