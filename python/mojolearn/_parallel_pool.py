# SPDX-License-Identifier: Apache-2.0
"""Ordered persistent subprocess pool with device selection before runtime import."""
import os
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


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
        self._start()
        requests = list(requests)
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
