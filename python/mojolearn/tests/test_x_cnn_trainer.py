# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CNNClassifier learns a separable task through its public fit. The Linear
layer's forward and backward against float64 are native
(checks/native_tests/cnn_trainer.mojo, `pixi run
check-native-cnn-trainer-cpu`); the whole trajectory against torch is a
two-stage check (torch and the bindings live in different interpreters on
the dev pod)."""
import numpy as np


def test_cnn_classifier_learns():
    import mojolearn as ml
    rng = np.random.default_rng(1)
    X = rng.standard_normal((128, 1, 6, 6)).astype(np.float32)
    y = (X[:, 0, :3].sum((1, 2)) > X[:, 0, 3:].sum((1, 2))).astype(int)
    m = ml.CNNClassifier(input_shape=(1, 6, 6), conv_channels=(4,), learning_rate=0.1, batch_size=32,
                         max_iter=15, random_state=0).fit(X.reshape(128, -1), y)
    assert m.loss_curve_[-1] < 0.5 * m.loss_curve_[0]
    assert (m.predict(X) == y).mean() > 0.9
