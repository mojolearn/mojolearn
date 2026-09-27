# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `sequence` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_sequence": "_mojolearn_x_sequence_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level
"""
from ._x_sequence_autoarima import AutoARIMA
from ._x_sequence_mlp import MLPClassifier, MLPRegressor
from ._x_sequence_optim import LAMB, Adafactor, Adagrad, Adamax, Lion, NAdam, RMSprop
from ._x_sequence_stl import STL
from ._x_sequence_var import VAR
from ._x_sequence_rnn import (GRUClassifier, GRURegressor, LSTMClassifier, LSTMRegressor, RNNClassifier,
                              RNNRegressor)

__all__ = ["LSTMRegressor", "LSTMClassifier", "GRURegressor", "GRUClassifier", "RMSprop", "Adagrad", "AutoARIMA", "STL", "VAR", "MLPClassifier", "MLPRegressor",
           "RNNRegressor", "RNNClassifier", "Lion", "Adafactor", "LAMB", "Adamax", "NAdam"]
