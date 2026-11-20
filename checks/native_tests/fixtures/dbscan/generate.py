# SPDX-License-Identifier: Apache-2.0
"""Independent DBSCAN graph and nearest-core references, without product fits.

Profiles are independent host-worker units. Connected components depend on
the radius graph and are discovered with an ordinary queue, not the product's
label-propagation/fold implementation. Border rows retain every valid adjacent
core component; expected labels never require arbitrary cluster numbering.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def grid():
    b = [(4. + i, float(j)) for i in range(3) for j in range(3)]
    a = [(-4. + i, float(j)) for i in range(3) for j in range(3)]
    return np.asarray(b + a + [(0., 10.), (20., -20.)], dtype='f4')


def distances(left, right, metric):
    delta = left.astype('f8')[:, None, :] - right.astype('f8')[None, :, :]
    if metric == 'manhattan':
        return np.abs(delta).sum(axis=2)
    return np.sqrt((delta * delta).sum(axis=2))


def profiles():
    x = grid()
    queries = {'train': x.copy(), 'at_eps': np.array([[7., 1.], [-5., 1.]], dtype='f4'),
               'past_eps': np.array([[np.nextafter(np.float32(7), np.float32(8)), 1.]], dtype='f4'),
               'far': np.array([[1000., -1000.], [-500., 700.]], dtype='f4'), 'reverse': x[::-1].copy()}
    result = []
    def add(name, source, eps=1., minimum=4, method='brute', metric='euclidean', weights=None, query=None, origin='test_transductive_predict.py::arm_dbscan'):
        result.append(dict(name=name, source=source, eps=eps, minimum=minimum, method=method,
                           metric=metric, weights=weights, query=query or {}, origin=origin))
    add('grid_brute', x, query=queries)
    add('grid_rbc', x, method='rbc', query={'train': x.copy()})
    add('grid_wide', x, eps=3., query={'tie': np.array([[1., 1.]], dtype='f4')})
    duplicate = np.concatenate([x, x]).astype('f4')
    add('grid_duplicates', duplicate, query={'train': duplicate.copy()})
    add('grid_manhattan', x, metric='manhattan', query={'boundary': np.array([[7., 1.], [6.5, 1.5], [6.75, 1.75]], dtype='f4')})
    add('grid_ones', x, weights=np.ones(20, dtype='f4'), query={'train': x.copy()})
    add('grid_empty', x, eps=.1, minimum=50, method='rbc', query={'first_three': x[:3].copy()})
    add('host512', np.random.default_rng(7).standard_normal((512, 5)).astype('f4'), eps=.9, minimum=5, method='rbc', origin='test_cpu_training_e.py::test_host_fits_run_and_repeat_on_a_cpu_only_install (fit workload only; no repeated-bits campaign)')
    add('parallel400', np.random.default_rng(0).standard_normal((400, 4)).astype('f4'), eps=.9, minimum=5, method='rbc', origin='test_parallel_public_drivers.py::_dbscan (single native mathematical fit; external multi-device seam retained)')
    pair = np.array([[0., 0.], [.5, 0.]] + [[50. + .3 * k, 0.] for k in range(6)], dtype='f4')
    w2 = np.ones(8, dtype='f4'); w2[0] = 2
    w15 = np.ones(8, dtype='f4'); w15[0] = 1.5
    for method in ('brute', 'rbc'):
        origin = 'dbscan/checks/dbscan_check.mojo::check_dbscan_duplicate_equals_weight_two; original8/9 rows, eps1/min3'
        add('weighted2_' + method, pair, minimum=3, method=method, weights=w2, origin=origin)
        add('weighted15_' + method, pair, minimum=3, method=method, weights=w15, origin=origin)
        add('duplicated_' + method, np.concatenate([pair, pair[:1]]), minimum=3, method=method, origin=origin)
    negative = np.array([[0., 0.]] * 3 + [[10. + .25 * k, 0.] for k in range(3)], dtype='f4')
    add('negative_inhibit', negative, eps=.75, minimum=3, weights=np.array([3., -2., 1., 1., 1., 1.], dtype='f4'),
        origin='Additional analytic admitted-negative-weight case: coincident sum2<3 stays noise; far three unit weights sum3 are core')
    add('precomputed_grid', distances(x, x, 'euclidean').astype('f4'), metric='precomputed',
        origin='Additional precomputed form of the original20x2 grid; no prediction feature added')
    return result


def write(name, value, dtype):
    raw = np.asarray(value, dtype=dtype).tobytes()
    (ROOT / name).write_bytes(raw)
    return {'file': name, 'shape': list(np.asarray(value).shape), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def prepare(profile):
    source = profile['source']; rows, features = source.shape
    width = float(np.float32(profile['eps']))
    distance = source.astype('f8') if profile['metric'] == 'precomputed' else distances(source, source, profile['metric'])
    adjacent = distance <= width
    weights = np.ones(rows, dtype='f8') if profile['weights'] is None else profile['weights'].astype('f8')
    degree = adjacent.astype('f8') @ weights
    core = degree >= profile['minimum']
    component = np.full(rows, -1, dtype='i4')
    anchors = []
    for start in np.flatnonzero(core):
        if component[start] >= 0:
            continue
        group = len(anchors); anchors.append(int(start)); component[start] = group
        queue = deque([int(start)])
        while queue:
            current = queue.popleft()
            fresh = np.flatnonzero(adjacent[current] & core & (component < 0))
            component[fresh] = group
            queue.extend(int(value) for value in fresh)
    allowed = np.zeros((rows, len(anchors)), dtype='u1')
    for row in range(rows):
        if core[row]:
            allowed[row, component[row]] = 1
        else:
            groups = component[adjacent[row] & core]
            allowed[row, groups] = 1
    noise = ~allowed.any(axis=1)
    prefix = profile['name'] + '-'
    records = [write(prefix + 'source.f32', source, '<f4'), write(prefix + 'weights.f32', [] if profile['weights'] is None else profile['weights'], '<f4'),
               write(prefix + 'core.u8', core, 'u1'), write(prefix + 'noise.u8', noise, 'u1'),
               write(prefix + 'anchors.i32', anchors, '<i4'), write(prefix + 'allowed.u8', allowed, 'u1'),
               write(prefix + 'degree.f64', degree, '<f8'), write(prefix + 'core-indices.i32', np.flatnonzero(core), '<i4')]
    query_records = []
    core_indices = np.flatnonzero(core)
    for name, query in profile['query'].items():
        expected = np.full(len(query), -1, dtype='i4')
        if len(core_indices):
            d = distances(query, source[core_indices], profile['metric'])
            nearest = np.argmin(d, axis=1)  # Core indices are ascending: ties choose the original index.
            inside = d[np.arange(len(query)), nearest] <= width
            expected[inside] = core_indices[nearest[inside]]
        records += [write(prefix + 'query-' + name + '.f32', query, '<f4'), write(prefix + 'query-' + name + '-core.i32', expected, '<i4')]
        query_records.append({'name': name, 'rows': len(query)})
    exact_radius = int(np.count_nonzero(distance == width))
    nonzero_gaps = np.abs(distance - width); nonzero_gaps = nonzero_gaps[nonzero_gaps != 0]
    return {'name': profile['name'], 'origin': profile['origin'], 'rows': rows, 'features': features,
            'eps': profile['eps'], 'effective_float32_eps': width, 'min_samples': profile['minimum'],
            'method': profile['method'], 'metric': profile['metric'], 'weighted': profile['weights'] is not None,
            'max_iterations': 0, 'budget_mb': 0, 'clusters': len(anchors), 'core_count': int(core.sum()),
            'noise_count': int(noise.sum()), 'ambiguous_border_rows': int(np.count_nonzero(allowed.sum(axis=1) > 1)),
            'exact_radius_pairs': exact_radius, 'minimum_nonzero_radius_gap': float(nonzero_gaps.min()) if len(nonzero_gaps) else None,
            'queries': query_records, 'single_row_queries_from_train': profile['name'] == 'grid_brute', 'files': records}


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        records = list(pool.map(prepare, profiles()))
    metadata = {'profiles': records, 'numpy': np.__version__, 'workers': os.cpu_count(),
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'oracle': 'Independent Float64 radius graph; mathematical weighted neighborhood core criterion; queue-connected core components; all valid border memberships; nearest core with lowest original-index tie.',
                'label_policy': 'Native labels may be renumbered. Core component anchors must map injectively; border assignments must belong to a radius-adjacent core component; noise/core flags exact.',
                'execution': 'Reference preparation only; no Mojolearn import, product fitting, compile/test/timing or same-bits campaign.'}
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'profiles': len(records), 'files': sum(len(r['files']) for r in records),
                      'ambiguous_border_rows': sum(r['ambiguous_border_rows'] for r in records),
                      'metadata': str(ROOT / 'metadata.json')}))
