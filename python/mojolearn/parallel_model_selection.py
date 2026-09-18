# SPDX-License-Identifier: Apache-2.0
"""Independent cross-validation folds assigned to explicit GPU workers.

Physical multi-GPU qualification is pending; see LANE_STATUS_multigpu-cv.md.
This schedules whole fits. It does not partition one fit across device memory.
"""
import pickle

from . import _backend
from ._array import Array
from ._parallel_pool import DevicePool
from ._gpu_witness import require_distinct_workers
from .model_selection import _clone, _prepare_folds, _take_rows

__all__ = ['cross_val_score']


def cross_val_score(estimator, X, y, *, devices, cv=None, scoring=None,
                    groups=None, error_score='raise'):
    """Fit one fresh estimator per fold on the requested CUDA/HIP devices.

    Fold validation, cloning, scoring and output order match the serial API.
    Workers receive only their fold's data and an unfitted clone. At most one
    fold runs per device; each fit must fit within that device's memory. Folds
    are dispatched in bounded waves, including a final partial wave. A failed
    fit or scoring call raises and closes the pool; no partial scores return.

    Estimators, constructor parameters and custom scorers must be pickleable
    and importable in fresh Python workers. Define custom classes/scorers in
    importable modules. Workers use IDENTICAL mode; explicitly fast estimator
    parameters are refused. A custom pipeline's GPU execution and numerical
    identity remain the pipeline author's responsibility.

    As with serial cross_val_score, only dense data and error_score='raise'
    are supported; fit metadata/weights and automatic refitting are absent.
    Fold indices are host metadata. There is no distributed CPU fit route.
    """
    pool = DevicePool(devices)
    vendor = _backend.vendor()
    if vendor not in ('cuda', 'hip'):
        raise NotImplementedError('parallel cross-validation requires CUDA or HIP GPU workers')
    X, y, folds = _prepare_folds(estimator, X, y, cv, scoring, groups, error_score)
    prototype = _clone(estimator)
    params = prototype.get_params(deep=True)
    if any((name == 'numeric_mode' or name.endswith('__numeric_mode'))
           and value not in (None, 'identical') for name, value in params.items()):
        raise ValueError('parallel cross-validation requires IDENTICAL estimator numeric modes')
    try:
        pickle.dumps((prototype, scoring), protocol=5)
    except Exception as exc:
        raise TypeError('parallel cross-validation requires pickleable estimators and scorers') from exc
    scores = []
    try:
        width = len(pool.devices)
        inventory = pool.map([('device_inventory', None, ()) for _ in pool.devices])
        require_distinct_workers(inventory, vendor, width)
        for start in range(0, len(folds), width):
            requests = []
            for train, test in folds[start:start + width]:
                requests.append(('cross_val_fold', _clone(prototype),
                                 (_take_rows(X, train), _take_rows(y, train),
                                  _take_rows(X, test), _take_rows(y, test), scoring)))
            scores.extend(pool.map(requests))
    finally:
        pool.close()
    return Array.from_list(scores, '<f8')
