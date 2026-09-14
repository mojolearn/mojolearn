# SPDX-License-Identifier: Apache-2.0
"""Whole-tree data replication: global RNG IDs and canonical prediction order."""
from ._parallel_pool import DevicePool
from ._array import Array
from ._buffer import as_f32_c, empty, addr, addr_ro
from ._bufcheck import memcopy


def fit_forest(estimator, X, y, *, devices=(0,), trees_per_shard=1):
    """Fit one RandomForest or ExtraTrees model across GPUs using global tree IDs.

    Every shard reads the FULL dataset and original forest seed. Tree ranges
    are concatenated in ID order; floating-point predictions use the existing
    complete-forest predictor. GPU count never selects the reduction order.
    Publishes the fitted estimator only after every worker succeeds.
    """
    from .randomforest import RandomForestClassifier, RandomForestRegressor
    from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
    if type(estimator) not in (RandomForestClassifier, RandomForestRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor):
        raise TypeError('fit_forest requires a mojolearn RandomForest or ExtraTrees estimator')
    if type(trees_per_shard) is not int or trees_per_shard < 1:
        raise ValueError('trees_per_shard must be a positive integer')
    params = estimator.get_params()
    if params['numeric_mode'] not in (None, 'identical'):
        raise ValueError('parallel forests require IDENTICAL numeric mode')
    params['numeric_mode'] = 'identical'
    X, _ = as_f32_c(X, ndim=2, name='X')
    # Keep classification label encoding identical on every full-data shard.
    if isinstance(estimator, (RandomForestRegressor, ExtraTreesRegressor)):
        y, _ = as_f32_c(y, ndim=1, name='y')
    else:
        from ._labels import flatten_labels
        y = flatten_labels(y)
    count = params['n_estimators']
    if count < 1:
        raise ValueError('n_estimators must be positive')
    requests = []
    for start in range(0, count, trees_per_shard):
        shard_params = dict(params, n_estimators=min(trees_per_shard, count - start))
        requests.append(('forest_fit', (type(estimator).__name__, shard_params), (X, y, start)))
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    result = type(estimator)(**params)
    result.__dict__.update(parts[0].__dict__)
    offsets = [0]
    for part in parts:
        base = offsets[-1]
        offsets.extend(base + offset for offset in part._offsets.tolist()[1:])
    result._offsets = Array.from_list(offsets, parts[0]._offsets.dtype)
    for name in ('_colid', '_quesval', '_left_child', '_leaves'):
        arrays = [getattr(part, name) for part in parts]
        merged = empty((sum(a.size for a in arrays),), arrays[0].dtype)
        cursor = 0
        for array in arrays:
            memcopy(addr(merged, name=name) + cursor, addr_ro(array, name=name), array.nbytes)
            cursor += array.nbytes
        setattr(result, name, merged)
    # Restore the full constructor/config; worker model metadata is otherwise
    # identical (feature count, class order, criterion and output count).
    result._n_trees = count
    result.n_estimators = count
    result._cfg['n_estimators'] = count
    if hasattr(result, 'depth_cap_bound_'):
        result.depth_cap_bound_ = any(part.depth_cap_bound_ for part in parts)
    result._resident_forest = None
    estimator.__dict__ = result.__dict__.copy()
    return estimator
