# SPDX-License-Identifier: Apache-2.0
"""Small neural-network trainers with complete, resumable state.

`SmallMLPTrainer` trains a fixed 8-16-3 ReLU network with cross-entropy and AdamW on the
GPU, with transactional steps, state snapshots and checkpoints. For scikit-learn style
estimators, see `mojolearn.MLPClassifier` and `mojolearn.MLPRegressor`.
"""
from ._mlp_impl import SmallMLPTrainer

__all__ = ['SmallMLPTrainer']
