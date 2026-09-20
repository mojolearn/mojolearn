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
#: one vote request on the merged neighbors) in Python, and the forest
#: driver (`parallel_ensemble.fit_forest`), which cuts global tree ID ranges
#: in Python, fits each range through the rf and trees host bindings' shard
#: fits (`rf_*_fit_shard`, `et_*_fit_shard`, the GPU bindings' tree_start
#: offset restated on the host) and concatenates the trees in ID order.
#:
#: Wave 3 (lane/cpu-verifier-par-samba, 2026-09-16) adds the Samba stack's
#: gradient shards. `ParallelNeuralTrainer` sends one `samba_gradient` per
#: logical shard from the same non-cooperative `DevicePool(devices)` that
#: carries par-mlp's `mlp_gradient`, and the shard gradients come back to
#: the driver and are folded by `parallel_training.ordered_sum_gradients`
#: inside the one `samba_update` its one-device cooperative pool carries.
#: Since lane/cpu-training-samba (2026-09-15) the training, mamba and
#: transformer host bindings serve every call SambaStack makes, so both
#: requests run the same host arithmetic the covered `samba` lane does.
#: Random Fourier feature transforms also shard whole rows in Python. Each
#: worker receives identical fitted weights/offsets and uses the kernel_methods
#: host transform; ordered assembly in transform_rbf_sampler is unchanged.
#: This is a logical-shard CPU route, not physical multi-GPU qualification.
#:
#: lane/laneless-public-classes (2026-09-19) adds the disjoint IVF shards.
#: `parallel_ivf.DistributedIVFIndex` cuts a BUILT index's stored rows into
#: contiguous row ranges in Python, maps each range's original ids to local
#: ids, sends one `ivf_store` per shard and one `ivf_search_stored` per
#: query batch, and merges the shard candidates back to global ids in
#: Python by `(distance, original id)`. `ivf_finalize` is the Euclidean root
#: the shards withheld, taken once on the merged row. Every one of those
#: four steps is the driver's own Python, and the shard's search is the ivf
#: family's host binding under its `ivf_flat_partial_search` name, so the
#: CPU column runs the partition and the merge the GPU column runs. It is
#: NOT a device claim: at one device the partition is one shard, which is
#: what `_par_devices`'s docstring says of every `par-*` lane.
#:
#: lane/lm-attention-fallback (2026-09-19) adds `forecast_predict`, the one
#: operation the four `parallel_forecasting` drivers send. IT WAS CHECKED
#: AGAINST THIS DOCSTRING'S BAR, NOT ASSUMED: the partition is
#: `parallel_forecasting._ranges(count, series_per_shard)`, a Python range of
#: SERIES; each shard is a `copy.copy` of the fitted estimator whose
#: series-major state (`_y`/`params_`, or the packed Holt-Winters components)
#: is sliced in Python; the merge is the driver's own `memcopy` of each
#: shard's block at its own offset, in `zip(ranges, parts)` order. NOTHING is
#: split inside a binding: `_parallel_worker.execute` serves the operation as
#: the bare `state.predict(*positional)` with no capability call and no
#: `MOJOLEARN_*_DEVICE_COUNT`, `arima/` and `holtwinters/` name no device
#: count and no `multi_gpu` module at all (their predict and forecast entries
#: take a single default `DeviceContext()`), and the `_run` pool is NOT
#: cooperative, so each worker is handed exactly one device index and could
#: not split across devices even if a kernel wanted to. That last fact holds
#: at EVERY device count, which is why this is `CPU_OPERATIONS` and not
#: `CPU_SINGLE_DEVICE_COOPERATIVE`: `mlp_update` is restricted to one device
#: because its cooperative pool makes the whole device group visible to a
#: single worker and the binding splits inside it above one; no
#: `forecast_predict` worker ever sees more than one device. The shard's own
#: work is served on the host by `_mojolearn_arima_host`'s `arima_predict`
#: and `arima_forecast` (bindings/arima_host_predict.mojo) and by
#: `_mojolearn_tsa_host`'s `holtwinters_predict` and `holtwinters_forecast`
#: (bindings/holtwinters_host_predict.mojo), the same doors the installed
#: forecast inference binding uses, so the CPU column runs the driver's split
#: and merge unchanged against the same host arithmetic the GPU column's
#: shard runs.
#: lane/unlaned-public-algorithms (2026-09-20) adds the GaussianProcess
#: CLASSIFIER's class-level shards, the two operations
#: `parallel_gaussian_process` sends. IT WAS CHECKED AGAINST THIS DOCSTRING'S
#: BAR, NOT ASSUMED: the partition is one-vs-rest CLASSES, built in Python
#: (`columns = [1] if len(classes) == 2 else range(len(classes))`, one
#: `('gpc_class_fit', ...)` request per column with its own 0/1 target vector
#: built by a Python comprehension); the merge is the driver's own Python,
#: `_set_fitted` over the fits in class order for the fit and, for the
#: prediction, the per-class columns normalized or arg-maxed row by row in
#: `predict_gaussian_process_classifier` itself. NOTHING is split inside a
#: binding: the worker calls `GaussianProcessClassifier._fit_binary` and
#: `._latent`, which are the single `gpc_fit` and `gpc_predict` entries the
#: gp family's host binding exports (bindings/gp_host.mojo over
#: gaussian_process/host/gpc_oracle.mojo and gpc_steps.mojo), and the host
#: kernel matrix those stand on says in its own docstring that it is the
#: one-device path with MOJOLEARN_GP_DEVICE_COUNT unset -- the ROW-sharded GP
#: driver (par-gp's `gp_fit`) is deliberately still absent from this set for
#: exactly that reason. The pool is NOT cooperative, so each worker is handed
#: one device index and could not split across devices even if a kernel
#: wanted to. The class shard's own arithmetic is the same host arithmetic the
#: covered `gpc` and `gpc-multiclass` lanes already hash. It is NOT a device
#: claim: at one device the partition is one process per class, which is what
#: `identity_break._par_devices`'s docstring says of every `par-*` lane.
#: lane/cpu-routes-gpu-only-four (2026-09-20) adds the last two operations of
#: the four lanes that had NO CPU route at all, and the two were checked
#: against this docstring's bar separately because they are not alike.
#:
#: `causal_lm_layer` is `models.ParallelCausalLM`'s one operation. THE
#: PARTITION IS LAYERS, cut in the driver's own Python: `layer_devices` is one
#: index per checkpoint layer, `_make_blocks` builds one `_RemoteBlock` per
#: layer and sends it to its owner, and THE MERGE IS `CausalLM._run`'s own
#: sequential chain -- embedding on the first owner, each layer's output
#: handed to the next owner through host memory, the final norm and head on
#: the last owner. Nothing is split inside a binding: the worker builds
#: `_block_classes(route)[kind]`, which on the CPU route is the
#: `neural_inference` block class the covered `hf-causal-lm` lane already
#: hashes, and `_CpuPrimitives`, which is `_mojolearn_neural_host`'s
#: `embedding_forward`/`rms_norm_forward`/`linear_forward` at the same
#: addresses in the same order. The pool is NOT cooperative, so each worker is
#: handed one device index and could not split across devices even if a kernel
#: wanted to, and no `MOJOLEARN_*_DEVICE_COUNT` is set for it. It is INFERENCE
#: only -- `CausalLM.load` plus `forward`; no operation here fits anything --
#: which is why it needs no `require_training` the way `gpc_class_fit` did.
#: `_rpc` addresses a single worker and so never reaches `map()`; it calls
#: `_cpu_refusal` itself so this set still gates it.
#:
#: `cross_val_fold` is `parallel_model_selection.cross_val_score`'s. THE
#: PARTITION IS FOLDS, cut in the driver's own Python by `_prepare_folds` and
#: `_take_rows` (the serial API's own fold code, unchanged), one request per
#: fold with its own cloned estimator, dispatched in waves of
#: `len(pool.devices)`; THE MERGE IS `scores.extend(results)` in fold order.
#: Nothing is split inside a binding: the worker runs
#: `model_selection._fit_score_fold`, the estimator's PUBLIC `fit` and
#: `score`, which on the CPU route is the same host arithmetic the serial
#: `cross-val` lane hashes. IT IS A FIT, so it carries the same hole
#: `gpc_class_fit` opened and the same guard: `_parallel_worker` calls
#: `require_training` before the fold runs, and a CPU-only install outside
#: `reference_training()` refuses in the worker exactly as the plain fit
#: refuses in the parent. The device inventory that admits the WORKERS is not
#: a device claim on this route and does not pretend to be one; see
#: `_gpu_witness.require_distinct_processes`.
CPU_OPERATIONS = frozenset((
    'scaler_fit', 'scaler_transform', 'arima_fit', 'holtwinters_fit',
    'forecast_predict',
    'neighbor_query', 'neighbor_reference', 'neighbor_vote',
    'forest_fit', 'mlp_gradient', 'samba_gradient', 'rbf_sampler_rows',
    'ivf_store', 'ivf_search_stored', 'ivf_finalize',
    'gpc_class_fit', 'gpc_class_predict',
    'causal_lm_layer', 'cross_val_fold', 'worker_identity',
))

#: The cooperative operations the CPU route admits, and only from a
#: ONE-device pool: ParallelNeuralTrainer's updates (`mlp_update`, and
#: `samba_update` since wave 3), which fold the logical shards' gradients in
#: Python (`ordered_sum_gradients`) and apply one optimizer step. Their
#: gradient columns, clip tensors and optimizer ranges are split inside the
#: GPU binding only at MOJOLEARN_OPTIMIZER_DEVICE_COUNT above one
#: (`training/*_multi_gpu.mojo` take the plain path at one), so a one-device
#: update hides no partition a host binding would have to restate; two or
#: more devices refuse by name. `samba_update` also carries the global norm
#: clip (par-samba-clip's max_norm), whose host arithmetic the covered
#: samba-untied-dropout-accum lane already checks.
CPU_SINGLE_DEVICE_COOPERATIVE = frozenset(('mlp_update', 'samba_update'))


#: THE NON-COOPERATIVE DRIVERS' NEGATIVE CONTROL, AND WHY IT IS HERE AND NOT
#: IN A BINDING (lane/par-sabotage-defines, 2026-09-20).
#:
#: A COOPERATIVE driver hands the whole fit to one worker and splits inside the
#: GPU binding, so its arm is a `-D MOJOLEARN_<X>_PARALLEL_SABOTAGE=1` build
#: (`cluster/multi_gpu.mojo`, `solver/multi_gpu.mojo`, and the eleven others).
#: A NON-cooperative driver has no such binding to rebuild: its partition and
#: its merge are the driver's OWN PYTHON -- the column ranges of
#: `parallel_preprocessing`, the global tree-ID ranges of
#: `parallel_ensemble.fit_forest`, the series ranges of `parallel_forecasting`,
#: the query and reference rows of the neighbor drivers, the row ranges of
#: `parallel_ivf`, the class columns of `parallel_gaussian_process`. No define
#: reaches that code, so before this switch existed, the only negative control
#: those lanes had was a HOST ARM, which perturbs the shard's arithmetic and
#: the plain call's arithmetic EQUALLY and therefore says nothing about the
#: partition. This switch perturbs the partition itself.
#:
#: WHAT IT DOES. Every shard after the first reads its slice ONE POSITION
#: EARLY while the merge still writes at the true offset. Widths, shard counts,
#: allocations and every validation are untouched, exactly as the Mojo arms
#: leave them; the answer is simply assembled from the wrong rows. It is the
#: `_sabotage_fold_order` shape of `model_selection.py`: an invariant a checker
#: could test still holds, so the difference has to be caught by the HASH.
#:
#: INERT AT ONE DEVICE OR ONE SHARD, which is the whole point. `index > 0` is
#: false for a single shard, so a `par-*` cell that moves under this switch
#: moved because of the switch and not because of the second device.
#:
#: TWO VARIABLES, as `MOJOLEARN_FOLD_ORDER_SABOTAGE` has needed since
#: 2026-09-17: a switch that quietly returns wrong answers on one env var is a
#: footgun. No build script, no workflow and no gate sets either one.
PAR_DRIVER_SABOTAGE = 'MOJOLEARN_PAR_DRIVER_SABOTAGE'


def par_driver_sabotage():
    """True when both switch variables are set. Read at call time, never
    cached, so a test can turn it on and off inside one process."""
    return (os.environ.get(PAR_DRIVER_SABOTAGE) == '1'
            and os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE') == '1')


def driver_read_shift(index, first=1):
    """How far back shard `index` should READ, in the driver's own units.

    0 always, except under the switch above for a shard past the first, where
    it is 1 and the caller must still MERGE at the unshifted offset. `first`
    is the shard's true start: a shard starting at 0 is never shifted, so the
    shift can never make an index negative.
    """
    if index > 0 and first >= 1 and par_driver_sabotage():
        return 1
    return 0


def _cpu_refusal(requests, cooperative, n_devices=1):
    names = sorted({request[0] for request in requests})
    if cooperative:
        if all(name in CPU_SINGLE_DEVICE_COOPERATIVE for name in names):
            if n_devices == 1:
                return None
            return NotImplementedError(
                'no CPU implementation of the cooperative multi-GPU driver ' + ', '.join(names) +
                ' across ' + str(n_devices) + ' devices yet: its gradient columns, clip tensors and '
                'optimizer ranges are split inside the GPU binding above one device, which no host '
                'binding restates')
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
                elif vendor == 'cpu' and (not self.cooperative or len(group) == 1):
                    # A logical worker process per device index; map() has
                    # already admitted only the CPU_OPERATIONS (and, from a
                    # one-device cooperative pool, CPU_SINGLE_DEVICE_COOPERATIVE).
                    names = ()
                else:
                    raise ValueError('device selection is unavailable for this vendor/device group')
                for name in names:
                    visible = os.environ.get(name)
                    if visible is not None:
                        ids = [token.strip() for token in visible.split(',')]
                        if max(self.devices) >= len(ids) or any(not ids[d] for d in self.devices):
                            raise ValueError('device index outside ' + name)
                        # Validate the entire pool before its first worker:
                        # distinct logical indices can repeat the same visible
                        # token. This catches duplicate masks, not UUID aliases;
                        # physical qualification still needs device witnesses.
                        if len({ids[d] for d in self.devices}) != len(self.devices):
                            raise ValueError('selected devices repeat an identifier in ' + name)
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
            refusal = _cpu_refusal(requests, self.cooperative, len(self.devices))
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
