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

from . import preprocessing as _prep
from . import _native_parallel_shard as _shard
from ._native_parallel_shard import _native_parallel_shard

# The scalers' plain owners, for the Python-sharded drivers' native owner
# (bindings/parallel_shard_native.mojo): a column block's fit or transform is
# the plain native one in a worker process.
_shard.register(standard_class=_prep.StandardScaler, standard_call=_prep._STANDARD_CALL,
                standard_fit=_prep._STANDARD_FIT, standard_context=_prep._STANDARD_CONTEXT,
                minmax_class=_prep.MinMaxScaler, minmax_call=_prep._MINMAX_CALL,
                minmax_context=_prep._MINMAX_CONTEXT)

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
    # One native call (bindings/parallel_shard_native.mojo `fit_scaler`):
    # whole columns in `columns_per_shard` blocks, each fitted by the plain
    # native fit in a worker process per device (block i on device
    # i % len(devices)); each statistic is per column, so every block's
    # statistics are placed at its column offset.
    return _native_parallel_shard('fit_scaler', (estimator, X, devices, columns_per_shard, sample_weight))


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
    # One native call (`transform_scaler`): every column block transformed by
    # the plain native transform with its columns' statistics in a worker
    # process, the blocks joined at their column offsets; the plain method's
    # public conversion for our caller.
    return _native_parallel_shard('transform_scaler', (estimator, X, devices, columns_per_shard, inverse))
