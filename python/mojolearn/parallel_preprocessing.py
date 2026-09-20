# SPDX-License-Identifier: Apache-2.0
"""Column partitions retaining each scaler's original row-reduction tree."""
from ._parallel_pool import DevicePool, driver_read_shift
from ._buffer import empty, addr, addr_ro
from ._bufcheck import memcopy


def _statistics(estimator):
    from .preprocessing import MinMaxScaler, StandardScaler
    if type(estimator) is MinMaxScaler:
        return ('data_min_', 'data_max_', 'data_range_', 'scale_', 'min_')
    if type(estimator) is StandardScaler:
        return ('mean_', 'var_', 'scale_')
    raise TypeError('parallel preprocessing requires MinMaxScaler or StandardScaler')


def _ranges(columns, width):
    if type(width) is not int or width < 1:
        raise ValueError('columns_per_shard must be a positive integer')
    return [(start, min(columns, start + width)) for start in range(0, columns, width)]


def fit_scaler(estimator, X, *, devices=(0,), columns_per_shard=16, sample_weight=None):
    """Fit column shards without storing the whole input on any one GPU.

    Rows within a column keep the original chunking and final reduction order.
    The complete host input and each column shard must fit in their respective
    memories. All fitted statistics are published only after every shard passes.
    """
    names = _statistics(estimator)
    if estimator.numeric_mode not in (None, 'identical'):
        raise ValueError('parallel scalers require IDENTICAL numeric mode')
    if sample_weight is not None:
        raise NotImplementedError('scalers do not implement sample_weight')
    data = estimator._input(X)
    n, d = data.shape
    ranges = _ranges(d, columns_per_shard)
    params = estimator.get_params()
    params['numeric_mode'] = 'identical'
    pool = DevicePool(devices)
    try:
        # The READ is shifted by `driver_read_shift` (0 unless the driver
        # sabotage switch is on); the MERGE below still writes at `start`, so
        # every width and every statistic keeps its position.
        parts = pool.map([
            ('scaler_fit', (type(estimator).__name__, params),
             (data[:, start - driver_read_shift(index, start):
                     end - driver_read_shift(index, start)],))
            for index, (start, end) in enumerate(ranges)])
    finally:
        pool.close()
    result = type(estimator)(**params)
    result.__dict__.update(parts[0].__dict__)
    for name in names:
        if getattr(parts[0], name) is None:
            setattr(result, name, None)
            continue
        merged = empty((d,), '<f4')
        for (start, end), part in zip(ranges, parts):
            value = getattr(part, name)
            memcopy(addr(merged, name=name) + start * 4, addr_ro(value, name=name), (end - start) * 4)
        setattr(result, name, merged)
    result.n_features_in_, result.n_samples_seen_ = d, n
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def transform_scaler(estimator, X, *, devices=(0,), columns_per_shard=16, inverse=False):
    """Transform independent column shards and assemble their exact output bytes."""
    import copy
    names = _statistics(estimator)
    if not estimator.__sklearn_is_fitted__():
        raise ValueError('scaler is not fitted')
    if estimator.numeric_mode_ != 'identical':
        raise ValueError('parallel scaler transforms require an IDENTICAL fit')
    if type(inverse) is not bool:
        raise ValueError('inverse must be a bool')
    data = estimator._input(X)
    n, d = data.shape
    if d != estimator.n_features_in_:
        raise ValueError('input feature count differs from fit')
    ranges = _ranges(d, columns_per_shard)
    requests = []
    for index, (start, end) in enumerate(ranges):
        part = copy.copy(estimator)
        part.__dict__ = estimator.__dict__.copy()
        part.n_features_in_ = end - start
        for name in names:
            value = getattr(estimator, name)
            setattr(part, name, None if value is None else value[start:end])
        # Only the DATA columns are shifted; the fitted statistics stay on the
        # true columns, so the shard transforms the wrong rows of the right
        # column block rather than changing any shape.
        shift = driver_read_shift(index, start)
        requests.append(('scaler_transform', part,
                         (data[:, start - shift:end - shift], inverse)))
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    merged = empty((n, d), '<f4')
    destination = addr(merged, name='output')
    for (start, end), part in zip(ranges, parts):
        source = addr_ro(part, name='output shard')
        width = end - start
        for row in range(n):
            memcopy(destination + (row * d + start) * 4, source + row * width * 4, width * 4)
    return merged
