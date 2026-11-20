"""Independent mathematical Agglomerative references; no product imports.

Profiles run across host cores. Minimax closure pivots and hierarchical merge
decisions depend on their preceding state and remain sequential; pairwise
arithmetic is vectorized NumPy. Ward recomputes cluster centroids from original
observations. Average/complete aggregate original point distances, without
the product's Lance-Williams recurrence or MST implementation. This prepares
oracles only, not a product test, timing or development identity campaign.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def grid():
    b = [(4 + i, j) for i in range(3) for j in range(3)]
    a = [(-4 + i, j) for i in range(3) for j in range(3)]
    return np.asarray(b + a, dtype='<f4')


def profiles():
    x = grid()
    queries = {'train': x.copy(), 'tie': np.array([[1., 1.]], dtype='<f4'),
               'near': np.array([[-1., 1.], [3.5, 0.]], dtype='<f4')}
    rng = np.random.default_rng(7)
    # Preserve the original dependent RNG stream and original concatenation.
    ward = np.ascontiguousarray(np.concatenate([
        rng.normal(center, .4, size=(60, 3)) for center in (0., 4., 8.)]), dtype='<f4')
    six = np.array([[0., 0.], [.2, .7], [.8, .1], [7., 8.], [7.5, 8.2], [8.3, 8.7]], dtype='<f4')
    result = []

    def add(name, source, clusters, linkage='single', route='solver', query=None, origin='additional analytic native route'):
        result.append(dict(name=name, source=source, clusters=clusters,
                           linkage=linkage, route=route, query=query or {}, origin=origin))

    origin = 'test_transductive_predict.py::arm_agglomerative'
    add('grid18', x, 2, query=queries, origin=origin)
    dup = np.concatenate([x, x]).astype('<f4')
    add('duplicate36', dup, 2, query={'train': dup.copy()}, origin=origin)
    add('singletons17', x, 17, query={'train': x.copy()}, origin=origin)
    add('parallel200', np.random.default_rng(0).standard_normal((200, 4)).astype('<f4'), 3,
        origin='test_parallel_public_drivers.py::_graph; multi-device seam retained')
    add('ward180', ward, 3, linkage='ward', route='tree',
        origin='test_x_cluster_twice.py::_CHILD; one numerical fit, no repeated-bits campaign')
    add('grid18_tree', x, 2, route='tree', query=queries,
        origin=origin + '; additional compute_distances=True general-tree route')
    add('average6', six, 2, linkage='average', route='tree')
    add('complete6', six, 2, linkage='complete', route='tree')
    return result


def distances(source):
    x = source.astype(np.float64)
    return np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))


def single_ultrametric(distance):
    # Floyd's minimax closure: independent of choosing/building an MST.
    value = distance.copy()
    for pivot in range(len(value)):
        value = np.minimum(value, np.maximum(value[:, pivot, None], value[None, pivot, :]))
    return value


def other_ultrametric(source, distance, linkage):
    x = source.astype(np.float64)
    n = len(source)
    members = np.zeros((2 * n - 1, n), dtype=bool)
    members[:n] = np.eye(n, dtype=bool)
    active = list(range(n))
    result = np.zeros((n, n), dtype=np.float64)
    smallest_gap = float('inf')
    for step in range(n - 1):
        ids = np.asarray(active)
        membership = members[ids]
        sizes = membership.sum(axis=1).astype(np.float64)
        if linkage == 'ward':
            means = membership.astype(np.float64) @ x / sizes[:, None]
            squared = np.sum((means[:, None, :] - means[None, :, :]) ** 2, axis=2)
            costs = 2 * sizes[:, None] * sizes[None, :] / (sizes[:, None] + sizes[None, :]) * squared
        elif linkage == 'average':
            costs = (membership.astype(np.float64) @ distance @ membership.T.astype(np.float64)) / (sizes[:, None] * sizes[None, :])
        else:
            mask = membership[:, None, :, None] & membership[None, :, None, :]
            costs = np.where(mask, distance[None, None, :, :], -np.inf).max(axis=(2, 3))
        costs[np.tril_indices(len(active))] = np.inf
        at = np.unravel_index(np.argmin(costs), costs.shape)
        chosen = costs[at]
        finite = costs[np.isfinite(costs)]
        larger = finite[finite > chosen]
        if larger.size:
            smallest_gap = min(smallest_gap, float(larger.min() - chosen))
        left, right = active[at[0]], active[at[1]]
        height = np.sqrt(chosen) if linkage == 'ward' else chosen
        li, ri = np.flatnonzero(members[left]), np.flatnonzero(members[right])
        result[np.ix_(li, ri)] = height
        result[np.ix_(ri, li)] = height
        members[n + step] = members[left] | members[right]
        active = [node for node in active if node != left and node != right]
        active.append(n + step)
    return result, smallest_gap


def prepare(profile):
    source = np.ascontiguousarray(profile['source'], dtype='<f4')
    name = profile['name']
    files = []

    def save(suffix, values, dtype):
        value = np.ascontiguousarray(values, dtype=dtype)
        path = ROOT / (name + '-' + suffix)
        data = value.tobytes()
        path.write_bytes(data)
        files.append(dict(file=path.name, shape=list(value.shape), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))

    distance = distances(source)
    if profile['linkage'] == 'single':
        ultrametric = single_ultrametric(distance)
        gap = None
    else:
        ultrametric, gap = other_ultrametric(source, distance, profile['linkage'])
    save('source.f32', source, '<f4')
    save('cophenetic.f64', ultrametric, '<f8')
    query_rows = []
    for batch, queries in profile['query'].items():
        q = np.ascontiguousarray(queries, dtype='<f4')
        squared = np.sum((q.astype(np.float64)[:, None, :] - source.astype(np.float64)[None, :, :]) ** 2, axis=2)
        nearest = squared == squared.min(axis=1, keepdims=True)
        counts = nearest.sum(axis=1)
        assert np.all((counts >= 1) & (counts <= 2)), 'Original workload has at most two tied nearest references'
        candidates = np.full((len(q), 2), -1, dtype='<i4')
        rank = np.cumsum(nearest, axis=1) - 1
        row, column = np.nonzero(nearest)
        candidates[row, rank[row, column]] = column
        save('query-' + batch + '.f32', q, '<f4')
        save('query-' + batch + '-nearest.i32', candidates, '<i4')
        query_rows.append(dict(name=batch, rows=len(q), maximum_ties=int(counts.max())))
    return dict(name=name, origin=profile['origin'], rows=source.shape[0], features=source.shape[1],
                clusters=profile['clusters'], linkage=profile['linkage'], route=profile['route'],
                solver_metric=1, use_knn=False, tree_metric=-1, full_merges=len(source) - 1,
                compute_distances=profile['route'] == 'tree', minimum_distinct_merge_cost_gap=gap,
                single_row_queries=name == 'grid18', queries=query_rows, files=files)


def main():
    workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(prepare, profiles()))
    metadata = dict(profiles=prepared, numpy=np.__version__, workers=workers,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    oracle='Float64 minimax closure for single linkage; Ward direct cluster centroid SSE; average/complete aggregate original point distances. No product/MST/Lance-Williams implementation or output references.',
                    label_policy='Compare cophenetic relations and labels as partitions; tied children orientation/order and label numbering may differ.',
                    execution='Independent reference preparation only; no Mojolearn import/fit/test/compile/timing or repeated-bits campaign.')
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(profiles=len(prepared), files=sum(len(p['files']) for p in prepared), workers=workers, execution=metadata['execution'])))


if __name__ == '__main__':
    main()
