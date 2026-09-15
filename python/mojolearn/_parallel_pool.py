# SPDX-License-Identifier: Apache-2.0
"""Ordered persistent subprocess pool with device selection before runtime import."""
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


#: THE CPU REFERENCE ROUTE (lane/cpu-training-par-classical, 2026-09-15).
#: On a CPU-only install (`_backend.vendor() == 'cpu'`) a pool runs ONLY
#: these operations: the ones whose driver splits the work into logical
#: shards in Python (column ranges, series ranges), sends each shard to a
#: worker as its own request and merges the shard results byte for byte in
#: shard order. On CPU each device index is one worker process with no
#: device selection, each shard runs the host binding's plain fit of that
#: shard, and the driver's own split and merge code runs unchanged, so the
#: CPU column checks the sharding logic the GPU column checks. Everything
#: else refuses by name: a COOPERATIVE pool hands the whole fit to one
#: worker and its shards are device row tiles, chunks or ranges inside the
#: GPU binding (`MOJOLEARN_<X>_DEVICE_COUNT`, `*/multi_gpu.mojo`), which a
#: host binding does not restate, and the other non-cooperative operations
#: (neural gradients among them) have no host route declared yet. The
#: operations run only inside `_cpu_reference.reference_training()` (the
#: internal verifier); outside it the worker's fit refuses exactly as a
#: plain CPU fit does.
#:
#: Wave 2 (lane/cpu-training-par-wave2, 2026-09-15) adds the neighbor
#: drivers, which cut query rows (`ParallelQueries`) or reference rows
#: (`ReferenceShardedNeighbors`, merged by composite key in Python, then
#: one vote request on the merged neighbors) in Python.
CPU_OPERATIONS = frozenset((
    'scaler_fit', 'scaler_transform', 'arima_fit', 'holtwinters_fit',
    'neighbor_query', 'neighbor_reference', 'neighbor_vote',
))


def _cpu_refusal(requests, cooperative):
    names = sorted({request[0] for request in requests})
    if cooperative:
        return NotImplementedError(
            'no CPU implementation of the cooperative multi-GPU driver ' + ', '.join(names) + ' yet: '
            'its shards are device row tiles, chunks or ranges inside the GPU binding, '
            'which no host binding restates')
    missing = [name for name in names if name not in CPU_OPERATIONS]
    if missing:
        return NotImplementedError(
            'no CPU implementation of the parallel worker operation ' + ', '.join(missing) + ' yet')
    return None


class DevicePool:
    def __init__(self, devices, *, cooperative=False):
        self.cooperative = cooperative
        self.devices = tuple(devices)
        if (not self.devices or any(type(i) is not int or i < 0 for i in self.devices)
                or len(set(self.devices)) != len(self.devices)):
            raise ValueError('devices must be distinct nonnegative integer indices')
        self._workers = []
        self._threads = None

    def _start(self):
        if self._workers:
            return
        self._threads = ThreadPoolExecutor(max_workers=len(self.devices))
        try:
            groups = [self.devices] if self.cooperative else [(d,) for d in self.devices]
            for group in groups:
                env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='identical')
                from . import _backend
                vendor = _backend.vendor()
                if vendor == 'cuda':
                    names = ('CUDA_VISIBLE_DEVICES',)
                elif vendor == 'hip':
                    # ROCR filters physical devices before HIP enumerates them.
                    # Do not stack an original HIP index on a one-device ROCR
                    # subset (rank 1 would become an invalid local index).
                    names = ('ROCR_VISIBLE_DEVICES',) if 'ROCR_VISIBLE_DEVICES' in env else ('HIP_VISIBLE_DEVICES',)
                    env.pop('HIP_VISIBLE_DEVICES' if names[0] == 'ROCR_VISIBLE_DEVICES' else 'ROCR_VISIBLE_DEVICES', None)
                elif vendor == 'metal' and group == (0,):
                    names = ()
                elif vendor == 'cpu' and not self.cooperative:
                    # A logical worker process per device index; map() has
                    # already admitted only the CPU_OPERATIONS.
                    names = ()
                else:
                    raise ValueError('device selection is unavailable for this vendor/device group')
                for name in names:
                    visible = os.environ.get(name)
                    if visible is not None:
                        ids = visible.split(',')
                        if max(group) >= len(ids) or any(not ids[d] for d in group):
                            raise ValueError('device index outside ' + name)
                        env[name] = ','.join(ids[d] for d in group)
                    else:
                        env[name] = ','.join(str(d) for d in group)
                if self.cooperative:
                    env['MOJOLEARN_KMEANS_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_GBDT_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_GRAM_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_QR_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_OPTIMIZER_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_GLM_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_SOLVER_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_IFOREST_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_FOREST_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_SVM_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_GP_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_GMM_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_RESAMPLE_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_DBSCAN_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_NEIGHBORS_DEVICE_COUNT'] = str(len(group))
                    env['MOJOLEARN_HIERARCHY_DEVICE_COUNT'] = str(len(group))
                self._workers.append(subprocess.Popen(
                    [sys.executable, '-m', 'mojolearn._parallel_worker'], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _call(worker, request):
        pickle.dump(request, worker.stdin, protocol=5)
        worker.stdin.flush()
        ok, value = pickle.load(worker.stdout)
        if not ok:
            raise RuntimeError('GPU worker failed:\n' + value)
        return value

    def map(self, requests):
        requests = list(requests)
        from . import _backend
        if _backend._CPU_ONLY is not None and requests:
            refusal = _cpu_refusal(requests, self.cooperative)
            if refusal is not None:
                raise refusal
            from ._cpu_reference import _active
            if _active.get():
                requests = [('cpu_reference', None, request) for request in requests]
        self._start()
        result = []
        # Waves preserve logical order and never use one worker concurrently.
        for start in range(0, len(requests), len(self._workers)):
            wave = requests[start:start + len(self._workers)]
            futures = [self._threads.submit(self._call, worker, request)
                       for worker, request in zip(self._workers, wave)]
            error = None
            for future in futures:
                try:
                    result.append(future.result())
                except BaseException as exc:
                    error = exc
            if error is not None:
                self.close()
                raise error
        return result

    def close(self):
        for worker in self._workers:
            if worker.poll() is None:
                worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            worker.stdin.close()
            worker.stdout.close()
        self._workers = []
        if self._threads is not None:
            self._threads.shutdown(wait=True, cancel_futures=True)
            self._threads = None
