# SPDX-License-Identifier: Apache-2.0
"""Independent cross-validation folds assigned to explicit GPU workers.

Physical multi-GPU qualification is pending; see LANE_STATUS_multigpu-cv.md.
This schedules whole fits. It does not partition one fit across device memory.
"""
import os
import pickle

from . import _backend
from ._array import Array
from ._parallel_pool import DevicePool
from ._gpu_witness import require_distinct_processes, require_distinct_workers
from .model_selection import _clone, _prepare_folds, _take_rows

__all__ = ['cross_val_score']


def _require_single_metal_worker(records):
    """Check the one child process, without asserting physical GPU isolation."""
    if len(records) != 1:
        raise RuntimeError('single-device Metal cross-validation requires one worker')
    record = records[0]
    if (record.get('kind') != 'single-metal-worker'
            or record.get('vendor') != 'metal'
            or type(record.get('pid')) is not int or record['pid'] < 1
            or record.get('ppid') != os.getpid()):
        raise RuntimeError('single-device Metal worker identity is invalid')


def cross_val_score(estimator, X, y, *, devices, cv=None, scoring=None,
                    groups=None, error_score='raise'):
    """Fit one fresh estimator per fold on CUDA/HIP devices or Metal device 0.

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
    Fold indices are host metadata.

    THE CPU HOST ROUTE (lane/cpu-routes-gpu-only-four, 2026-09-20). On a
    CPU-only install a "device index" is one WORKER PROCESS: the fold
    partition, the cloning, the wave dispatch, the fold order and the merge
    are the driver's own Python, unchanged, and each fold runs the
    estimator's public `fit` and `score` on that process's host bindings.
    That column checks the DISPATCH against the serial API byte for byte and
    nothing else; it is not device isolation, not placement and not a
    throughput claim, and the two-device CUDA/HIP column remains owed. CPU
    training is public, so the fold fits run on a CPU-only install.
    Metal supports only explicit ``devices=(0,)``: one child executes folds
    sequentially on the available Metal device. Its process identity is
    checked, but no physical isolation or multi-GPU execution is claimed.
    """
    pool = DevicePool(devices)
    vendor = _backend.vendor()
    host = vendor not in ('cuda', 'hip') and _backend._CPU_ONLY is not None
    metal = vendor == 'metal' and pool.devices == (0,)
    if vendor not in ('cuda', 'hip') and not host and not metal:
        raise NotImplementedError(
            'parallel cross-validation requires CUDA or HIP GPU workers, '
            'Metal devices=(0,), or a CPU-only install where one worker is one process')
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
        # ONE WITNESS PER ROUTE, EACH SAYING ONLY WHAT IT CAN. On CUDA/HIP
        # that is one visible physical GPU per worker at local ordinal zero,
        # with no repeated UUID or PCI id. On the host route there is no
        # device to ask about, so it is one distinct worker PROCESS per index,
        # under its own record kind and its own checker; the GPU check is
        # untouched, deliberately.
        if host:
            require_distinct_processes(
                pool.map([('worker_identity', None, ()) for _ in pool.devices]), width)
        elif metal:
            _require_single_metal_worker(pool.map([('metal_worker_identity', None, ())]))
        else:
            inventory = pool.map([('device_inventory', None, ()) for _ in pool.devices])
            require_distinct_workers(inventory, vendor, width)
        for start in range(0, len(folds), width):
            requests = []
            for train, test in folds[start:start + width]:
                requests.append(('cross_val_fold', _clone(prototype),
                                 (_take_rows(X, train), _take_rows(y, train),
                                  _take_rows(X, test), _take_rows(y, test), scoring)))
            results = pool.map(requests)
            if len(results) != len(requests):
                raise ValueError("cross-validation workers returned an incomplete fold batch")
            scores.extend(results)
    finally:
        pool.close()
    return Array.from_list(scores, '<f8')
