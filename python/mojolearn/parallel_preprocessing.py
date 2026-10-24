# SPDX-License-Identifier: Apache-2.0
"""Fit and apply scalers with their feature columns spread across several GPUs.

Each GPU receives a block of whole columns. The fitted statistics and the
transformed output are the same, bit for bit, as a one-GPU
``fit``/``transform``.

Example
-------
>>> import numpy as np
>>> from mojolearn import StandardScaler
>>> from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
>>> X = np.random.default_rng(0).standard_normal((1000, 32)).astype(np.float32)
>>> scaler = fit_scaler(StandardScaler(), X, devices=(0, 1), columns_per_shard=16)
>>> Z = transform_scaler(scaler, X, devices=(0, 1), columns_per_shard=16)
"""
# Column partitions retaining each scaler's original row-reduction tree.
__all__ = [
    'fit_scaler',
    'transform_scaler',
]

import sys

from ._parallel_pool import DevicePool, driver_read_shift, public_method_result
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
    return [(start, min(columns, start + width)) for start in range(0, columns, width)]  # glue: lists the column shard bounds


def fit_scaler(estimator, X, *, devices=(0,), columns_per_shard=16, sample_weight=None):
    """Fit a ``StandardScaler`` or ``MinMaxScaler`` with column blocks on separate GPUs.

    Parameters
    ----------
    estimator : StandardScaler or MinMaxScaler
        An unfitted scaler using the ``'identical'`` numeric mode (or the default).
    X : array-like of shape (n_samples, n_features)
        Training data, converted to float32.
    devices : tuple of int, default=(0,)
        Distinct GPU indices; column blocks are handed to them in turn.
    columns_per_shard : int, default=16
        Number of feature columns in each block.
    sample_weight : None
        Not supported by the scalers; passing weights raises ``NotImplementedError``.

    Returns
    -------
    estimator
        The same scaler, fitted, with the statistics a one-GPU ``fit`` produces.
    """
    # Fit column shards without storing the whole input on any one GPU.
    #
    # Rows within a column keep the original chunking and final reduction order.
    # The complete host input and each column shard must fit in their respective
    # memories. All fitted statistics are published only after every shard passes.
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
             (data[:, start - driver_read_shift(index, start, devices):
                     end - driver_read_shift(index, start, devices)],))
            for index, (start, end) in enumerate(ranges)])  # glue: dispatches one device job per shard
    finally:
        pool.close()
    result = type(estimator)(**params)
    result.__dict__.update(parts[0].__dict__)
    for name in names:  # glue: walks the fitted statistic names
        if getattr(parts[0], name) is None:
            setattr(result, name, None)
            continue
        merged = empty((d,), '<f4')
        for (start, end), part in zip(ranges, parts):  # glue: copies each shard result block
            value = getattr(part, name)
            memcopy(addr(merged, name=name) + start * 4, addr_ro(value, name=name), (end - start) * 4)
        setattr(result, name, merged)
    result.n_features_in_, result.n_samples_seen_ = d, n
    estimator.__dict__ = result.__dict__.copy()
    return estimator


def transform_scaler(estimator, X, *, devices=(0,), columns_per_shard=16, inverse=False):
    """Apply a fitted scaler, or its inverse, with column blocks on separate GPUs.

    Parameters
    ----------
    estimator : StandardScaler or MinMaxScaler
        A scaler fitted in the ``'identical'`` numeric mode.
    X : array-like of shape (n_samples, n_features)
        Data to transform, with the feature count seen during fit.
    devices : tuple of int, default=(0,)
        Distinct GPU indices.
    columns_per_shard : int, default=16
        Number of feature columns in each block.
    inverse : bool, default=False
        ``True`` applies ``inverse_transform`` instead of ``transform``.

    Returns
    -------
    Array of shape (n_samples, n_features)
        The same values as ``estimator.transform(X)`` (or ``inverse_transform``).
    """
    # Transform independent column shards and assemble their exact output bytes.
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
    for index, (start, end) in enumerate(ranges):  # glue: builds one device job per shard
        part = copy.copy(estimator)
        part.__dict__ = estimator.__dict__.copy()
        part.n_features_in_ = end - start
        for name in names:  # glue: walks the fitted statistic names
            value = getattr(estimator, name)
            setattr(part, name, None if value is None else value[start:end])
        # Only the DATA columns are shifted; the fitted statistics stay on the
        # true columns, so the shard transforms the wrong rows of the right
        # column block rather than changing any shape.
        shift = driver_read_shift(index, start, devices)
        requests.append(('scaler_transform', part,
                         (data[:, start - shift:end - shift], inverse)))
    pool = DevicePool(devices)
    try:
        parts = pool.map(requests)
    finally:
        pool.close()
    # the shards' column blocks joined into one row-major matrix by the
    # device (x_prep `colblock`), not row by row in Python
    from ._expansion_prep import join_column_blocks
    out = join_column_blocks(parts, ranges, n, d)
    # the joined raw columns; the plain method's public conversion for our caller
    return public_method_result(estimator, 'inverse_transform' if inverse else 'transform',
                                out, sys._getframe(1))
