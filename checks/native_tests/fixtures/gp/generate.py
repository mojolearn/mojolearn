# SPDX-License-Identifier: Apache-2.0
"""Original GP inputs and independent Float64 mathematical references.

This is test-data preparation, with no Mojolearn import, native execution,
optimizer implementation or identity run. Independent profiles and writes use
host-sized pools. Each original dependent random stream is kept in its original
order. The 48-row classifier's Python random.gauss stream is deliberately not
replaced: its labels are ranked before admitting the source to Float32.

Posterior references use NumPy linear algebra and the declared kernel equations.
Hyperparameter gradients differentiate those equations analytically, without
copying the product's finite differences, factorization or gradient kernels.
The existing external scikit-learn tests remain independent public API checks.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parent
ALPHA = 2.0 ** -20


def hashed_unit(count, salt):
    index = np.arange(count, dtype=np.uint64) + np.uint64(1)
    z = index * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)
    z ^= z >> np.uint64(30)
    z *= np.uint64(0xBF58476D1CE4E5B9)
    z ^= z >> np.uint64(27)
    z *= np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    q = (z >> np.uint64(43)).astype(np.int64)
    return (q - 1048576).astype(np.float32) / np.float32(1048576)


def normalized_data(seed=4, queries=24):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 3)).astype(np.float32)
    y = (50.0 + 12.0 * np.sin(x[:, 0]) + 3.0 * x[:, 1]).astype(np.float32)
    q = rng.normal(size=(queries, 3)).astype(np.float32)
    return x, y, q


def optimizer_data(seed, rows):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3)).astype(np.float32)
    y = (np.sin(1.5 * x[:, 0]) + .5 * x[:, 1] + .1 * rng.normal(size=rows)).astype(np.float32)
    return x, y


def kernel_options(kind='rbf', scales=(1.0,), amplitude=1.0, white=.1,
                   additive=0.0, free_amplitude=True):
    # These are mathematical parameters, not a copy of the native postfix ABI.
    return dict(kind=kind, scales=list(scales), amplitude=amplitude, white=white,
                additive=additive, free_amplitude=free_amplitude)


def profiles():
    result = []

    def add(name, source, target, query, options=None, normalize=False,
            origin='', purpose='', gradient=False):
        result.append(dict(name=name, source=source, target=target, query=query,
                           options=options or kernel_options(), normalize=normalize,
                           origin=origin, purpose=purpose, gradient=gradient))

    x = hashed_unit(32, 0x67700001).reshape(16, 2)
    y = hashed_unit(16, 0x67700002)
    add('surface16', x, y, x.copy(), kernel_options(scales=(.25, .25), white=0),
        origin='test_gp_surface.py::main POSTERIOR',
        purpose='original training recovery bound 2^-14; status/clamp/covariance invariants')
    for normalize in (False, True):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 3)).astype(np.float32)
        y = (np.sin(x[:, 0]) + .1 * rng.normal(size=60) + (5.0 if normalize else 0.0)).astype(np.float32)
        add('cov60norm' if normalize else 'cov60', x, y, x[:20] + np.float32(.25),
            normalize=normalize, origin='test_gp_return_cov.py::test_cov_is_symmetric_and_its_diagonal_is_the_variance',
            purpose='exact symmetry; diagonal sqrt versus native std rtol1e-3/atol1e-4')
    x, y, q = normalized_data()
    add('normalize96', x, y, q, normalize=True,
        origin='test_gp_normalize_y.py::_data',
        purpose='target mean within1e-3; original external sklearn mean/std seams retained')
    add('constant96', x.copy(), np.full(96, 7, dtype=np.float32), q.copy(), normalize=True,
        origin='test_gp_normalize_y.py::test_constant_target_scales_by_one',
        purpose='constant normalization mean7/std1; finite prediction')
    add('sample_shape96', x.copy(), y.copy(), q.copy(),
        origin='test_gp_sample_y.py::arm_shape',
        purpose='original 5 draws/key1 and default1; finite values; public rank/shape retained')
    x, y, q = normalized_data(seed=6, queries=6)
    add('sample_moments96', x, y, q, origin='test_gp_sample_y.py::arm_moments',
        purpose='4000 draws/key3; original6SE+1e-4 mean bound and .08 relative std bound')
    x, y = optimizer_data(4, 40)
    options = [
        ('gradient_rbf40', kernel_options(scales=(.8, 1.2, 2.0), amplitude=1.5, white=.2)),
        ('gradient_matern05_40', kernel_options(kind='matern05', scales=(1.1,))),
        ('gradient_matern15_40', kernel_options(kind='matern15', scales=(1.0, .6, 1.4), amplitude=.7)),
        ('gradient_matern25_40', kernel_options(kind='matern25', scales=(.9,), white=.2,
                                               additive=.3, free_amplitude=False)),
    ]
    for name, option in options:
        add(name, x.copy(), y.copy(), x.copy(), option, gradient=True,
            origin='test_gp_optimizer.py::arm_gradient',
            purpose='native central differences h1e-2/<5e-3; independent analytic gradient bound1e-3')
    x, y = optimizer_data(5, 48)
    add('optimize48', x, y, x.copy(), origin='test_gp_optimizer.py::arm_optimize',
        purpose='unchanged fmin_l_bfgs_b start; positive steps and improved likelihood; no reference optimizer')
    x, y = optimizer_data(6, 48)
    add('restart48', x, y, x.copy(), kernel_options(kind='matern25', scales=(1.0, 1.0, 1.0)),
        origin='test_gp_optimizer.py::arm_restarts',
        purpose='original2 restarts/key7; best saved run/earliest tie; no repeated-fit bits check')
    x = np.random.default_rng(0).standard_normal((96, 4)).astype(np.float32)
    y = (x[:, 0] - .5 * x[:, 2]).astype(np.float32)
    add('parallel64', x[:64].copy(), y[:64].copy(), x[64:].copy(),
        origin='test_parallel_public_drivers.py::_gaussian_process',
        purpose='single actual native graph; retain public multi-device controller')
    return result


def kernel(x, y, options, same=False, derivatives=False):
    scales = np.asarray(options['scales'], dtype=np.float64)
    parts = ((x[:, None, :] - y[None, :, :]) / scales) ** 2
    square = parts.sum(axis=2)
    distance = np.sqrt(square)
    kind = options['kind']
    if kind == 'rbf':
        base = np.exp(-.5 * square)
        slope = base
    elif kind == 'matern05':
        base = np.exp(-distance)
        slope = np.divide(base, distance, out=np.zeros_like(base), where=distance != 0)
    elif kind == 'matern15':
        exponential = np.exp(-np.sqrt(3.0) * distance)
        base = (1 + np.sqrt(3.0) * distance) * exponential
        slope = 3 * exponential
    elif kind == 'matern25':
        exponential = np.exp(-np.sqrt(5.0) * distance)
        base = (1 + np.sqrt(5.0) * distance + 5 * square / 3) * exponential
        slope = (5 / 3) * (1 + np.sqrt(5.0) * distance) * exponential
    else:
        raise ValueError(kind)
    scaled = options['amplitude'] * base
    value = scaled + options['additive']
    white = options['white'] * np.eye(len(x)) if same else np.zeros_like(value)
    value += white
    if not derivatives:
        return value
    grads = []
    names = []
    if options['free_amplitude']:
        grads.append(scaled)
        names.append('log_constant_product')
    if len(scales) == 1:
        grads.append(options['amplitude'] * slope * square)
        names.append('log_length_scale')
    else:
        grads.extend(np.moveaxis(options['amplitude'] * slope[:, :, None] * parts, 2, 0))
        names.extend(f'log_length_scale_{index}' for index in range(len(scales)))
    if options['additive']:
        grads.append(np.full_like(value, options['additive']))
        names.append('log_constant_sum')
    grads.append(white)
    names.append('log_white')
    return value, np.stack(grads), names


def prepare(profile):
    x = profile['source'].astype(np.float64)
    y = profile['target'].astype(np.float64)
    q = profile['query'].astype(np.float64)
    options = profile['options']
    mean = float(y.mean()) if profile['normalize'] else 0.0
    scale = float(y.std()) if profile['normalize'] else 1.0
    if scale == 0:
        scale = 1.0
    target = (y - mean) / scale
    covariance = kernel(x, x, options, same=True) + ALPHA * np.eye(len(x))
    factor = np.linalg.cholesky(covariance)
    dual = np.linalg.solve(covariance, target)
    cross = kernel(q, x, options)
    query_mean = (cross @ dual) * scale + mean
    query_cov = (kernel(q, q, options, same=True) - cross @ np.linalg.solve(covariance, cross.T)) * scale ** 2
    logdet = 2 * np.log(np.diag(factor)).sum()
    quadratic = target @ dual
    lml = -.5 * (quadratic + logdet + len(x) * np.log(2 * np.pi))
    arrays = dict(source=profile['source'], target=profile['target'], query=profile['query'],
                  mean=query_mean, covariance=query_cov,
                  statistics=np.array([mean, scale, lml, logdet, quadratic], dtype=np.float64))
    metadata = {key: value for key, value in profile.items() if key not in ('source', 'target', 'query')}
    metadata.update(rows=len(x), features=x.shape[1], queries=len(q), alpha=ALPHA,
                    covariance_condition=float(np.linalg.cond(covariance)),
                    statistics_order=['target_mean', 'target_std', 'lml', 'logdet', 'y_dot_dual'])
    if profile['gradient']:
        _, dk, order = kernel(x, x, options, same=True, derivatives=True)
        score = np.outer(dual, dual) - np.linalg.solve(covariance, np.eye(len(x)))
        arrays['gradient'] = .5 * np.einsum('ij,pij->p', score, dk)
        metadata['gradient_order'] = order
    if profile['name'] == 'surface16':
        metadata['reference_recovery_max_abs'] = float(np.max(np.abs(query_mean - y)))
    return metadata, [(profile['name'], key, value) for key, value in arrays.items()]


def classifier_inputs():
    rng = random.Random(0)
    x64 = np.asarray([[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(48)], dtype=np.float64)
    order = np.argsort(x64[:, 0] + .5 * x64[:, 1], kind='stable')
    ranks = np.empty(48, dtype=np.int64)
    ranks[order] = np.arange(48)
    x = x64.astype(np.float32)
    records = [('classifier48', 'source', x), ('classifier48', 'query', x[:10].copy()),
               ('classifier48', 'labels2', (ranks >= 24).astype(np.int32)),
               ('classifier48', 'labels3', (ranks * 3 // 48).astype(np.int32))]
    x = np.random.default_rng(75).normal(size=(9, 3)).astype(np.float32)
    records.extend([('partition9', 'source', x), ('partition9', 'query', x[:5].copy()),
                    ('partition9', 'labels2', (np.arange(9) % 2).astype(np.int32)),
                    ('partition9', 'labels3', (np.arange(9) % 3).astype(np.int32))])
    x = np.random.default_rng(0).standard_normal((96, 4)).astype(np.float32)
    y2 = (x[:, 0] + x[:, 1] > 0).astype(np.int32)
    y3 = (np.argsort(np.argsort(x[:64, 3], kind='stable'), kind='stable') * 3 // 64).astype(np.int32)
    records.extend([('classifier64', 'source', x[:64].copy()), ('classifier64', 'query', x[64:].copy()),
                    ('classifier64', 'labels2', y2[:64].copy()), ('classifier64', 'labels3', y3)])
    return records


def write(item):
    name, part, values = item
    dtype = '<i4' if part.startswith('labels') else ('<f4' if part in ('source', 'target', 'query') else '<f8')
    suffix = 'i32' if dtype == '<i4' else ('f32' if dtype == '<f4' else 'f64')
    filename = f'{name}-{part}.{suffix}'
    raw = np.ascontiguousarray(values, dtype=dtype).tobytes()
    (ROOT / filename).write_bytes(raw)
    return dict(file=filename, shape=list(values.shape), dtype=dtype, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        prepared = list(pool.map(prepare, profiles()))
        artifacts = list(pool.map(write, [item for _, group in prepared for item in group] + classifier_inputs()))
    report = dict(status='Independent reference preparation only; native tests not compiled or run',
                  numpy=np.__version__, workers=os.cpu_count(), gpr_profiles=len(prepared),
                  classifier_input_profiles=3, artifacts=artifacts,
                  profiles=[metadata for metadata, _ in prepared],
                  classifier_reference='Original inputs/labels only; no mirrored Newton or OVR implementation',
                  product_imports=False, existing_python_removed=0)
    (ROOT / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('artifacts', 'profiles')}, sort_keys=True))
    print(json.dumps(dict(artifacts=len(artifacts), bytes=sum(item['bytes'] for item in artifacts)), sort_keys=True))
