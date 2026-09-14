# SPDX-License-Identifier: Apache-2.0
"""Private local-child RPC. Input is trusted parent process data, never a network."""
import pickle
import sys
import traceback


def execute(request):
    operation, state, args = request
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
    if operation == 'gbdt_fit':
        model = state
        binding = model._bind('_mojolearn_gbdt')
        if (not callable(getattr(binding, 'gbdt_parallel_available', None))
                or binding.gbdt_parallel_available() != 1):
            raise ImportError('rebuild GBDT binding for feature-parallel training')
        X, y, kwargs = args
        model.fit(X, y, **kwargs)
        return model
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
        state.fit(X, y, **kwargs)
        return state
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
        try:
            response = (True, execute(request))
        except Exception:
            response = (False, traceback.format_exc())
        pickle.dump(response, channel, protocol=5)
        channel.flush()


if __name__ == '__main__':
    main()
