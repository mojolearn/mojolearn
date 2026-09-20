# SPDX-License-Identifier: Apache-2.0
"""Private local-child RPC. Input is trusted parent process data, never a network."""
import pickle
import sys
import traceback


_forest_snapshot = None
_ivf_snapshot = None


def execute(request):
    global _forest_snapshot, _ivf_snapshot
    operation, state, args = request
    if operation == 'device_inventory':
        from . import _backend
        from ._gpu_witness import visible_gpu_inventory
        return visible_gpu_inventory(_backend.vendor())
    if operation == 'worker_identity':
        from ._gpu_witness import worker_process_inventory
        return worker_process_inventory()
    if operation == 'metal_worker_identity':
        import os
        from . import _backend
        if _backend.vendor() != 'metal':
            raise ValueError('single-device Metal worker requires the Metal backend')
        return dict(kind='single-metal-worker', vendor='metal',
                    pid=os.getpid(), ppid=os.getppid())
    if operation == 'cross_val_fold':
        from . import _backend
        from .model_selection import _fit_score_fold
        if _backend.vendor() not in ('cuda', 'hip', 'metal') and _backend._CPU_ONLY is None:
            raise NotImplementedError('cross_val_fold requires a CUDA, HIP or Metal GPU worker')
        # THE SAME HOLE `gpc_class_fit` OPENED, CLOSED THE SAME WAY
        # (lane/cpu-routes-gpu-only-four, 2026-09-20). A fold IS a fit, and
        # admitting `cross_val_fold` to `_parallel_pool.CPU_OPERATIONS` put a
        # fit behind the driver on an install where the plain fit refuses.
        # `_fit_score_fold` does call the estimator's PUBLIC `fit`, so
        # `_mode._guard_cpu_training` stands behind this for every estimator
        # that carries it -- but that is one decorator on a surface the caller
        # chooses, and `cross_val_score` takes ANY pickleable estimator. The
        # guard is stated here, once, on the operation itself.
        from ._cpu_reference import require_training
        require_training(state)
        return _fit_score_fold(state, *args)
    if operation == 'causal_lm_layer':
        from ._causal_lm_worker import execute as run_layer
        return run_layer(state, args)
    if operation == 'cpu_reference':
        # The pool wraps a request this way only on a CPU-only install and
        # only while its caller is inside reference_training() (the internal
        # verifier), so the shard's host fit runs in that scope here too.
        from . import _backend
        from ._cpu_reference import reference_training
        from ._parallel_pool import CPU_OPERATIONS, CPU_SINGLE_DEVICE_COOPERATIVE
        if _backend._CPU_ONLY is None or args[0] not in CPU_OPERATIONS | CPU_SINGLE_DEVICE_COOPERATIVE:
            raise ValueError('cpu_reference wraps only a CPU route operation on a CPU-only install')
        with reference_training():
            return execute(args)
    if operation == 'forest_prepare':
        from .parallel_ensemble import _admit_forest_predictor
        _admit_forest_predictor(state)
        if _forest_snapshot is not None:
            raise RuntimeError('forest worker already owns a prepared snapshot')
        binding = state._bind()
        capability = getattr(binding, 'forest_pool_available', None)
        if not callable(capability) or capability() != 1:
            raise ImportError('rebuild RF/ET binding for pooled forest inference')
        state._prepare_resident_forest(binding)
        _forest_snapshot = state
        return dict(n_features=int(state.n_features_in_), outputs=int(state._num_outputs),
                    trees=int(state._n_trees), classes=getattr(state, 'classes_', None))
    if operation == 'forest_predict':
        if _forest_snapshot is None:
            raise RuntimeError('forest worker has no prepared snapshot')
        method, X = args
        if method not in ('predict', 'predict_proba'):
            raise ValueError('unsupported pooled forest prediction method')
        function = getattr(_forest_snapshot, method, None)
        if not callable(function):
            raise ValueError('prepared forest does not support ' + method)
        return function(X)
    if operation == 'forest_release':
        model, _forest_snapshot = _forest_snapshot, None
        if model is not None:
            resident = getattr(model, '_resident_forest', None)
            if resident is not None:
                resident._finalizer()
                model._resident_forest = None
        return True
    if operation in ('mlp_update', 'samba_update'):
        import os
        if int(os.environ.get('MOJOLEARN_OPTIMIZER_DEVICE_COUNT', '1')) > 1:
            from ._training_impl import _load
            binding = _load('identical')
            for name in ('optimizer_parallel_available', 'accumulate_parallel_available', 'clip_parallel_available'):
                function = getattr(binding, name, None)
                if not callable(function):
                    raise ImportError('rebuild training binding for pooled neural gradients and updates')
                if function() != 1:
                    raise RuntimeError('training binding refused ' + name)
    if operation in ('mlp_gradient', 'mlp_update'):
        from ._mlp_impl import SmallMLPTrainer, _validate_state
        weights, _, config, schedule = _validate_state(state)
        model = SmallMLPTrainer(*weights, data_schedule=schedule)
        model.load_state_dict(state)
        if operation == 'mlp_gradient':
            return model.loss_and_grads(*args)
        from .parallel_training import ordered_sum_gradients
        gradients = ordered_sum_gradients(args)
        retained = [g.copy() for g in gradients]
        step = model.apply_gradients(gradients)
        return model.state_dict(), retained, step
    if operation in ('samba_gradient', 'samba_update'):
        from ._samba_impl import SambaStack, SambaConfig
        from ._training_impl import Generator
        model = SambaStack(SambaConfig.from_dict(state['config']),
                           generator=Generator(0, 'identical'), numeric_mode='identical')
        model.load_state_dict(state)
        if operation == 'samba_gradient':
            inputs, targets, stream, offset = args
            return model.loss_and_grads(inputs, targets, dropout_stream=stream, token_offset=offset)
        from .parallel_training import ordered_sum_gradients
        gradients = ordered_sum_gradients(args)
        retained = [g.copy() for g in gradients]
        model.optimizer.step(gradients, max_norm=model.max_norm)
        return model.state_dict(), retained, model.optimizer.t
    if operation == 'forest_fit':
        from .randomforest import RandomForestClassifier, RandomForestRegressor
        from .extratrees import ExtraTreesClassifier, ExtraTreesRegressor
        name, params = state
        model = {'RandomForestClassifier': RandomForestClassifier,
                 'RandomForestRegressor': RandomForestRegressor,
                 'ExtraTreesClassifier': ExtraTreesClassifier,
                 'ExtraTreesRegressor': ExtraTreesRegressor}[name](**params)
        X, y, start = args
        model._fit_with_tree_start(X, y, start)
        return model
    if operation == 'kmeans_fit':
        from .cluster import KMeans
        model = KMeans(**state)
        binding = model._bind('_mojolearn')
        if (not callable(getattr(binding, 'kmeans_parallel_available', None))
                or binding.kmeans_parallel_available() != 1):
            raise ImportError('rebuild base binding for cooperative KMeans')
        X, weights = args
        model.fit(X, sample_weight=weights)
        return model
    if operation in ('graph_fit', 'umap_transform'):
        from .parallel_graph import _binding
        native, capability = _binding(state)
        if not callable(getattr(native, capability, None)) or getattr(native, capability)() != 1:
            raise ImportError('rebuild graph binding for native multi-GPU rows')
        if operation == 'umap_transform':
            return state.transform(*args)
        state.fit(*args)
        return state
    if operation == 'dbscan_fit':
        binding = state._bind('_mojolearn_estimators')
        if (not callable(getattr(binding, 'dbscan_parallel_available', None))
                or binding.dbscan_parallel_available() != 1):
            raise ImportError('rebuild estimators binding for parallel DBSCAN neighborhoods')
        X, weights = args
        state.fit(X, sample_weight=weights)
        return state
    if operation in ('gbdt_fit', 'ordered_rmse_fit'):
        model = state
        if callable(getattr(model, '_bind', None)):
            binding = model._bind('_mojolearn_gbdt')
        else:
            from . import _backend
            binding = _backend.binding('_mojolearn_gbdt', 'identical')
        if (not callable(getattr(binding, 'gbdt_parallel_available', None))
                or binding.gbdt_parallel_available() != 1):
            raise ImportError('rebuild GBDT binding for feature-parallel training')
        # the fold searcher of Ordered boosting is the pointwise searcher's
        # fold arm (lane/catboost-parity), so it needs the same capability
        X, y, kwargs = args
        ordered = False
        resolve = getattr(model, '_resolved_boosting_type', None)
        if callable(resolve):
            ordered = resolve(len(X)) == 'Ordered'
        if operation == 'ordered_rmse_fit' or getattr(model, 'use_pointwise_searcher', False) or ordered:
            if not callable(getattr(binding, 'pointwise_parallel_available', None)) or binding.pointwise_parallel_available() != 1:
                raise ImportError('rebuild GBDT binding for parallel pointwise histograms')
        model.fit(X, y, **kwargs)
        return model
    if operation == 'ivf_store':
        state._entry(state._extension(), 'ivf_flat_partial_search')
        _ivf_snapshot = state
        return state.n_rows_
    if operation == 'ivf_search_stored':
        from .parallel_ivf import _partial_search
        if _ivf_snapshot is None:
            raise RuntimeError('IVF shard is not stored in this worker')
        return _partial_search(_ivf_snapshot, args[0])
    if operation == 'ivf_finalize':
        from ._buffer import addr
        if _ivf_snapshot is None:
            raise RuntimeError('IVF shard is not stored in this worker')
        native = _ivf_snapshot._extension()
        native.ivf_finalize_distances(addr(state, name='distances'), state.size, args[0])
        return state
    if operation == 'gpc_class_fit':
        # THE SAME GUARD THE PUBLIC `fit` CARRIES. `_fit_binary` is the class
        # shard's work, and it sits BELOW `GaussianProcessClassifier.fit`'s
        # `_mode._guard_cpu_training` decorator, so calling it here would have
        # been a public CPU TRAINING path on an install where the plain
        # `fit` refuses -- measured on a CPU-only install, 2026-09-20,
        # lane/unlaned-public-algorithms: the plain fit raised
        # "fit/training is reserved for the internal bitwise verifier" and the
        # driver trained anyway. The other admitted fits (`arima_fit`,
        # `holtwinters_fit`, ...) go through their estimator's public `fit`
        # and inherit the guard; this one has to state it.
        from ._cpu_reference import require_training
        from ._gpc_impl import _kernel_arrays
        require_training(state)
        x, y01 = args
        return state._fit_binary(state._extension(), x, y01, *_kernel_arrays(state.kernel))
    if operation == 'gpc_class_predict':
        fit, q, want_proba = args
        mean, _, probability = state._latent(state._extension(), fit, q, want_proba)
        return probability if want_proba else mean
    if operation == 'forecast_predict':
        method, positional = args
        if method != 'predict':
            raise ValueError('invalid forecasting worker operation')
        return state.predict(*positional)
    if operation == 'arima_fit':
        from ._arima_impl import ARIMA
        return ARIMA(**state).fit(*args)
    if operation == 'holtwinters_fit':
        from ._tsa_impl import ExponentialSmoothing
        return ExponentialSmoothing(args[0], ts_num=args[0].shape[0], **state).fit()
    if operation == 'gram_fit':
        binding = state._bind('_mojolearn_estimators')
        if (not callable(getattr(binding, 'gram_parallel_available', None))
                or binding.gram_parallel_available() != 1):
            raise ImportError('rebuild estimators binding for parallel Gram chunks')
        X, y, kwargs = args
        if X.shape[1] > 128 or X.shape[0] < X.shape[1]:
            if not callable(getattr(binding, 'gram_outputs_parallel_available', None)) or binding.gram_outputs_parallel_available() != 1:
                raise ImportError('rebuild estimators binding for Gram output partitions')
        if getattr(state, 'svd_solver', None) == 'full':
            if not callable(getattr(binding, 'qr_parallel_available', None)) or binding.qr_parallel_available() != 1:
                raise ImportError('rebuild estimators binding for parallel QR panels')
        state.fit(X, y, **kwargs)
        return state
    if operation in ('gmm_fit', 'gmm_predict'):
        native = state._extension()
        if (not callable(getattr(native, 'gmm_parallel_available', None))
                or native.gmm_parallel_available() != 1):
            raise ImportError('rebuild mixture binding for row-sharded GaussianMixture E-steps')
        if operation == 'gmm_fit':
            state.fit(*args)
            return state
        method, X = args
        if method not in ('score_samples', 'predict_proba', 'predict'):
            raise ValueError('invalid GaussianMixture prediction operation')
        return getattr(state, method)(X)
    if operation == 'resample':
        from . import resample
        native = resample._extension('identical')
        if (not callable(getattr(native, 'resample_ranges_parallel_available', None))
                or native.resample_ranges_parallel_available() != 1):
            raise ImportError('rebuild resample binding for distributed replicate ranges')
        name, kwargs = args
        if name not in ('bootstrap', 'permutation_test', 'monte_carlo_integrate'):
            raise ValueError('invalid resample operation')
        return getattr(resample, name)(numeric_mode='identical', **kwargs)
    if operation == 'hdbscan_fit':
        from .hdbscan import HDBSCAN
        if type(state) is not HDBSCAN:
            raise TypeError('requires mojolearn.HDBSCAN')
        native = state._extension()
        if (not callable(getattr(native, 'hdbscan_rows_parallel_available', None))
                or native.hdbscan_rows_parallel_available() != 1):
            raise ImportError('rebuild HDBSCAN binding for distributed neighbor and distance rows')
        state.fit(*args)
        return state
    if operation in ('km_fit', 'km_apply'):
        import os
        native = state._extension()
        if (not callable(getattr(native, 'kernel_methods_rows_parallel_available', None))
                or native.kernel_methods_rows_parallel_available() != 1):
            raise ImportError('rebuild kernel methods binding for distributed kernel rows')
        from .parallel_classical import _admit_kernel_method
        _admit_kernel_method(state)
        # Kernel rows follow MOJOLEARN_SVM_DEVICE_COUNT (set by the pool); the
        # KernelRidge factorization and multi-target solve follow the scoped
        # Cholesky switch, which the GP and GaussianMixture workers never see.
        previous = os.environ.get('MOJOLEARN_CHOLESKY_DEVICE_COUNT')
        os.environ['MOJOLEARN_CHOLESKY_DEVICE_COUNT'] = os.environ.get('MOJOLEARN_SVM_DEVICE_COUNT', '1')
        try:
            if operation == 'km_fit':
                state.fit(*args)
                return state
            method, X = args
            if method not in ('predict', 'transform'):
                raise ValueError('invalid kernel method operation')
            return getattr(state, method)(X)
        finally:
            if previous is None:
                os.environ.pop('MOJOLEARN_CHOLESKY_DEVICE_COUNT', None)
            else:
                os.environ['MOJOLEARN_CHOLESKY_DEVICE_COUNT'] = previous
    if operation == 'rbf_sampler_rows':
        from .kernel_methods import RBFSampler
        if type(state) is not RBFSampler:
            raise TypeError('requires mojolearn.RBFSampler')
        return state.transform(args[0])
    if operation in ('cholesky_fit', 'cholesky_solve'):
        import os
        native = state._extension()
        if (not callable(getattr(native, 'cholesky_parallel_available', None))
                or native.cholesky_parallel_available() != 1):
            raise ImportError('rebuild GP binding for operation-level multi-GPU Cholesky')
        # Scoped to this operation: the GP and GaussianMixture workers keep
        # their own root factorization paths.
        previous = os.environ.get('MOJOLEARN_CHOLESKY_DEVICE_COUNT')
        os.environ['MOJOLEARN_CHOLESKY_DEVICE_COUNT'] = os.environ.get('MOJOLEARN_GP_DEVICE_COUNT', '1')
        try:
            if operation == 'cholesky_fit':
                state.fit(*args)
                return state
            return state.solve(*args)
        finally:
            if previous is None:
                os.environ.pop('MOJOLEARN_CHOLESKY_DEVICE_COUNT', None)
            else:
                os.environ['MOJOLEARN_CHOLESKY_DEVICE_COUNT'] = previous
    if operation in ('gp_fit', 'gp_predict'):
        native = state._extension()
        if (not callable(getattr(native, 'gp_parallel_available', None))
                or native.gp_parallel_available() != 1):
            raise ImportError('rebuild GP binding for distributed covariance rows')
        if operation == 'gp_fit':
            state.fit(*args)
            return state
        X, return_std = args
        result = state.predict(X, return_std=return_std)
        return result, (state.clamped_, state.n_clamped_) if return_std else None
    if operation in ('svm_fit', 'svm_predict'):
        from ._svm_impl import _extension
        native = _extension('identical')
        if (not callable(getattr(native, 'svm_parallel_available', None))
                or native.svm_parallel_available() != 1):
            raise ImportError('rebuild SVM binding for distributed kernel rows')
        if operation == 'svm_fit':
            state.fit(*args)
            return state
        X, method = args
        if method not in ('predict', 'decision_function'):
            raise ValueError('invalid SVM prediction operation')
        return getattr(state, method)(X)
    if operation in ('iforest_fit', 'iforest_score'):
        from ._svm_impl import _extension
        native = _extension('identical')
        if (not callable(getattr(native, 'iforest_parallel_available', None))
                or native.iforest_parallel_available() != 1):
            raise ImportError('rebuild SVM binding for parallel IsolationForest')
        if operation == 'iforest_fit':
            state.fit(*args)
            return state
        X, method = args
        if method not in ('score_samples', 'decision_function', 'predict'):
            raise ValueError('invalid IsolationForest score operation')
        return getattr(state, method)(X)
    if operation == 'solver_fit':
        from ._backend import binding
        native = binding('_mojolearn_solver')
        if (not callable(getattr(native, 'solver_parallel_available', None))
                or native.solver_parallel_available() != 1):
            raise ImportError('rebuild solver binding for parallel dot leaves')
        state.fit(*args)
        return state
    if operation == 'glm_fit':
        binding = state._bind('_mojolearn_estimators')
        if (not callable(getattr(binding, 'glm_parallel_available', None))
                or binding.glm_parallel_available() != 1):
            raise ImportError('rebuild estimators binding for parallel GLM gradients')
        state.fit(*args)
        return state
    if operation == 'neighbor_reference':
        from .neighbors import NearestNeighbors
        index, query, k = args
        return NearestNeighbors(**state).fit(index).kneighbors(query, n_neighbors=k)
    if operation == 'neighbor_vote':
        from .parallel_neighbors_reference import _vote
        return _vote(state, *args)
    if operation == 'neighbor_query':
        from .parallel_neighbors import _methods
        X, method, kwargs = args
        if method not in _methods(state):
            raise ValueError('invalid neighbors/density query operation')
        result = getattr(state, method)(X, **kwargs)
        diagnostics = {name: getattr(state, name) for name in
                       ('used_query_tile_', 'n_candidate_distances_')
                       if hasattr(state, name)}
        return result, diagnostics
    if operation == 'scaler_fit':
        from .preprocessing import MinMaxScaler, StandardScaler
        name, params = state
        return {'MinMaxScaler': MinMaxScaler, 'StandardScaler': StandardScaler}[name](**params).fit(*args)
    if operation == 'scaler_transform':
        X, inverse = args
        return state.inverse_transform(X) if inverse else state.transform(X)
    raise ValueError('unknown parallel worker operation: ' + operation)


def main():
    # Native diagnostics may print to stdout. Reserve a duplicate fd for RPC
    # and redirect ordinary stdout (including native printf) to stderr.
    import os
    channel = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    while True:
        try:
            request = pickle.load(sys.stdin.buffer)
        except EOFError:
            break
        response = None
        try:
            try:
                response = (True, execute(request))
            except Exception:
                response = (False, traceback.format_exc())
            pickle.dump(response, channel, protocol=5)
            channel.flush()
        finally:
            # Persistent state belongs only to explicit operation caches.
            # Idle workers must not retain complete neural RPC snapshots.
            request = None
            response = None


if __name__ == '__main__':
    main()
