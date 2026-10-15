# SPDX-License-Identifier: Apache-2.0
"""Small neural-network trainers with complete, resumable state.

`SmallMLPTrainer` trains a fixed 8-16-3 ReLU network with cross-entropy and AdamW on the
GPU, with transactional steps, state snapshots and checkpoints. For scikit-learn style
estimators, see `mojolearn.MLPClassifier` and `mojolearn.MLPRegressor`.
"""
from ._mlp_impl import SmallMLPTrainer

__all__ = ['SmallMLPTrainer']


# scikit-learn keeps MLPClassifier and MLPRegressor in this module; they resolve from
# the package at first use (`_sklearn_modules`).
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("neural_network")
__all__ += _SKLEARN_LAYOUT["neural_network"]
