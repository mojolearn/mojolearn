# SPDX-License-Identifier: Apache-2.0
"""Independent density references for the original KDE Python workloads.

Each kernel/metric/weight profile is an independent host-worker unit. This
uses NumPy Float64 distances, stable log sums, and mathematical radial-volume
normalization. The cosine integral uses Gauss-Legendre quadrature, not the
product recurrence. No Mojolearn or legacy implementation is imported.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KERNELS = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
METRICS = ['euclidean', 'manhattan', 'chebyshev', 'sqeuclidean']
TRAIN = np.random.default_rng(3).standard_normal((37, 5)).astype('f4')
QUERY = np.random.default_rng(9).standard_normal((11, 5)).astype('f4')
# Public Float64 sample_weight is admitted to Float32 storage before RUN.
WEIGHTS = np.linspace(.25, 3., 37).astype('f4')


def norm(kernel, dimension):
    volume = math.pi ** (dimension / 2) / math.gamma(dimension / 2 + 1)
    surface = dimension * volume
    if kernel == 'gaussian':
        return (2 * math.pi) ** (dimension / 2)
    if kernel == 'tophat':
        return volume
    if kernel == 'epanechnikov':
        return 2 * volume / (dimension + 2)
    if kernel == 'exponential':
        return surface * math.gamma(dimension)
    if kernel == 'linear':
        return volume / (dimension + 1)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    radii = (nodes + 1) / 2
    return surface * np.dot(weights / 2, radii ** (dimension - 1) * np.cos(math.pi * radii / 2))


def reference(case):
    kernel, metric, weighted = case
    delta = QUERY.astype('f8')[:, None, :] - TRAIN.astype('f8')[None, :, :]
    if metric == 'manhattan':
        distance = np.abs(delta).sum(axis=2)
    elif metric == 'chebyshev':
        distance = np.abs(delta).max(axis=2)
    else:
        distance = (delta * delta).sum(axis=2)
        if metric == 'euclidean':
            distance = np.sqrt(distance)
    radius = distance / .75
    if kernel == 'gaussian':
        log_kernel = -.5 * radius * radius
    elif kernel == 'exponential':
        log_kernel = -radius
    else:
        log_kernel = np.full(radius.shape, -np.inf)
        inside = radius < 1
        if kernel == 'tophat':
            log_kernel[inside] = 0
        elif kernel == 'epanechnikov':
            log_kernel[inside] = np.log1p(-(radius[inside] ** 2))
        elif kernel == 'linear':
            log_kernel[inside] = np.log1p(-radius[inside])
        else:
            log_kernel[inside] = np.log(np.cos(math.pi * radius[inside] / 2))
    weights = WEIGHTS.astype('f8') if weighted else np.ones(37)
    scores = np.logaddexp.reduce(log_kernel + np.log(weights)[None, :], axis=1)
    scores -= math.log(weights.sum()) + 5 * math.log(.75) + math.log(norm(kernel, 5))
    name = kernel + '-' + metric + ('-weighted' if weighted else '-uniform')
    raw = scores.astype('<f8').tobytes()
    (ROOT / (name + '.f64')).write_bytes(raw)
    return {'name': name, 'kernel': kernel, 'metric': metric, 'weighted': weighted,
            'file': name + '.f64', 'sha256': hashlib.sha256(raw).hexdigest(),
            'finite_scores': int(np.isfinite(scores).sum()), 'negative_infinity_scores': int(np.isneginf(scores).sum())}


if __name__ == '__main__':
    cases = [(kernel, metric, weighted) for kernel in KERNELS for metric in METRICS for weighted in (False, True)]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        records = list(pool.map(reference, cases))
    inputs = []
    for name, values in [('train', TRAIN), ('query', QUERY), ('weights', WEIGHTS)]:
        raw = values.astype('<f4').tobytes()
        (ROOT / (name + '.f32')).write_bytes(raw)
        inputs.append({'file': name + '.f32', 'shape': list(values.shape), 'sha256': hashlib.sha256(raw).hexdigest()})
    metadata = {'origin': 'test_neighbors_kde_native.py::test_scores_equal_former_path',
                'train_shape': [37, 5], 'query_shape': [11, 5], 'seeds': [3, 9], 'bandwidth': .75,
                'weight_admission': 'Float64 linspace(.25,3,37) converted to Float32, matching public RUN storage',
                'profiles': records, 'inputs': inputs, 'workers': os.cpu_count(), 'numpy': np.__version__,
                'bound': 'Existing kde_check: abs(actual-reference) <= 2e-4 * max(1,abs(reference)); mathematical -inf requires actual <= -1e38.',
                'bound_origin': 'kde/checks/kde_check.mojo::_close/check_kde_oracle_vs_float64; not chosen from native test results',
                'oracle': 'Independent Float64 mathematical distances/radial kernels/stable log sums; cosine normalization by128-point quadrature, not product recurrence.',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'execution': 'Input/reference preparation only; no product compile/test/timing/identity run.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'profiles': len(records), 'finite_scores': sum(x['finite_scores'] for x in records),
                      'negative_infinity_scores': sum(x['negative_infinity_scores'] for x in records),
                      'metadata': str(ROOT / 'metadata.json')}))
