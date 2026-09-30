# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S PUBLIC DOOR.

Owned by the `sequence` expansion lane. `mojolearn/__init__.py` imports this
module and exposes every name in `__all__` lazily as `mojolearn.<name>`; a name
that is already public is refused at import. Register implementations in
`_LAZY_EXPORTS` so optional NumPy imports happen on access. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_sequence": "_mojolearn_x_sequence_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""

__all__ = ["LSTMRegressor", "LSTMClassifier", "GRURegressor", "GRUClassifier", "RMSprop", "Adagrad", "AutoARIMA", "STL", "VAR", "MLPClassifier", "MLPRegressor",
           "RNNRegressor", "RNNClassifier", "Lion", "Adafactor", "LAMB", "Adamax", "NAdam",
           "StepLR", "ExponentialLR", "OneCycleLR", "LayerNorm", "layer_norm_forward",
           "layer_norm_backward", "Theta", "OptimizedTheta", "DynamicTheta", "DynamicOptimizedTheta",
           "AutoTheta", "CrostonClassic", "CrostonOptimized", "CrostonSBA", "ETS", "DampedETS", "GARCH", "ProphetForecaster", "MoEBlock"]

# Keep the public registry importable without NumPy. Implementations load only
# when their public symbol is requested; the core/verifier CLI stays usable.
_LAZY_EXPORTS = {
    'AutoARIMA': ('_x_sequence_autoarima', 'AutoARIMA'),
    'MLPClassifier': ('_x_sequence_mlp', 'MLPClassifier'),
    'MLPRegressor': ('_x_sequence_mlp', 'MLPRegressor'),
    'LAMB': ('_x_sequence_optim', 'LAMB'),
    'Adafactor': ('_x_sequence_optim', 'Adafactor'),
    'Adagrad': ('_x_sequence_optim', 'Adagrad'),
    'Adamax': ('_x_sequence_optim', 'Adamax'),
    'Lion': ('_x_sequence_optim', 'Lion'),
    'NAdam': ('_x_sequence_optim', 'NAdam'),
    'RMSprop': ('_x_sequence_optim', 'RMSprop'),
    'STL': ('_x_sequence_stl', 'STL'),
    'VAR': ('_x_sequence_var', 'VAR'),
    'LayerNorm': ('_x_sequence_norm', 'LayerNorm'),
    'layer_norm_backward': ('_x_sequence_norm', 'layer_norm_backward'),
    'layer_norm_forward': ('_x_sequence_norm', 'layer_norm_forward'),
    'ETS': ('_x_sequence_ets', 'ETS'),
    'DampedETS': ('_x_sequence_ets', 'DampedETS'),
    'GARCH': ('_x_sequence_garch', 'GARCH'),
    'MoEBlock': ('_x_sequence_moe', 'MoEBlock'),
    'ProphetForecaster': ('_x_sequence_prophet', 'ProphetForecaster'),
    'CrostonClassic': ('_x_sequence_croston', 'CrostonClassic'),
    'CrostonOptimized': ('_x_sequence_croston', 'CrostonOptimized'),
    'CrostonSBA': ('_x_sequence_croston', 'CrostonSBA'),
    'AutoTheta': ('_x_sequence_theta', 'AutoTheta'),
    'DynamicOptimizedTheta': ('_x_sequence_theta', 'DynamicOptimizedTheta'),
    'DynamicTheta': ('_x_sequence_theta', 'DynamicTheta'),
    'OptimizedTheta': ('_x_sequence_theta', 'OptimizedTheta'),
    'Theta': ('_x_sequence_theta', 'Theta'),
    'ExponentialLR': ('_x_sequence_sched', 'ExponentialLR'),
    'OneCycleLR': ('_x_sequence_sched', 'OneCycleLR'),
    'StepLR': ('_x_sequence_sched', 'StepLR'),
    'GRUClassifier': ('_x_sequence_rnn', 'GRUClassifier'),
    'GRURegressor': ('_x_sequence_rnn', 'GRURegressor'),
    'LSTMClassifier': ('_x_sequence_rnn', 'LSTMClassifier'),
    'LSTMRegressor': ('_x_sequence_rnn', 'LSTMRegressor'),
    'RNNClassifier': ('_x_sequence_rnn', 'RNNClassifier'),
    'RNNRegressor': ('_x_sequence_rnn', 'RNNRegressor'),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    module, attribute = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module("." + module, __package__), attribute)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        raise ModuleNotFoundError(
            f"mojolearn.{name} requires NumPy. Install it with: "
                'python -m pip install "mojolearn[numpy]"', name="numpy") from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
