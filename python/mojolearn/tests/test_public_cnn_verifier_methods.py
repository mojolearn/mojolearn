# SPDX-License-Identifier: Apache-2.0
"""Public CNN endpoints that the original verifier lanes did not execute."""
import numpy as np
import pytest

import mojolearn as ml


@pytest.mark.parametrize("cls,shape", [(ml.Conv1d, (2, 7)), (ml.Conv2d, (2, 3, 5))])
def test_convolution_fit_records_features_without_changing_weights(cls, shape):
    rows = (np.arange(4 * np.prod(shape), dtype=np.float32) / np.float32(16)).reshape(4, -1)
    model = cls(2, 3, 3, padding=1, random_state=8, input_shape=shape, numeric_mode="identical")
    original_weight = np.asarray(model.weight_).copy()
    original_bias = np.asarray(model.bias_).copy()
    before = np.asarray(model.transform(rows)).copy()
    assert model.fit(rows) is model
    assert model.n_features_in_ == np.prod(shape)
    np.testing.assert_array_equal(model.weight_, original_weight)
    np.testing.assert_array_equal(model.bias_, original_bias)
    np.testing.assert_array_equal(model.transform(rows), before)


@pytest.mark.parametrize("cls,shape", [(ml.BatchNorm1d, (3,)), (ml.BatchNorm2d, (3, 2, 2))])
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm_fit_updates_one_training_batch_and_restores_mode(cls, shape, training):
    images = (np.arange(9 * np.prod(shape), dtype=np.float32) / np.float32(16)).reshape((9,) + shape)
    rows = images.reshape(9, -1)
    model = cls(3, momentum=0.25, input_shape=shape, numeric_mode="identical")
    model.training = training
    assert model.fit(rows) is model
    axes = (0,) + tuple(range(2, images.ndim))
    expected_mean = np.float64(0.25) * images.astype(np.float64).mean(axis=axes)
    expected_var = np.float64(0.75) + np.float64(0.25) * images.astype(np.float64).var(axis=axes, ddof=1)
    np.testing.assert_allclose(model.running_mean_, expected_mean, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(model.running_var_, expected_var, rtol=2e-6, atol=2e-6)
    assert model.num_batches_tracked_ == 1
    assert model.training is training


@pytest.mark.parametrize("image_input", [False, True])
def test_cnn_predict_gathers_actual_class_words(image_input):
    rows = np.random.default_rng(37).normal(size=(33, 16)).astype(np.float32)
    labels = np.asarray([-11, 23, 71], dtype=np.int64)[np.arange(33) % 3]
    model = ml.CNNClassifier(input_shape=(1, 4, 4), conv_channels=(2,), kernel_size=3, pool_size=2,
                             batch_size=11, max_iter=1, random_state=9, numeric_mode="identical").fit(rows, labels)
    query = rows.reshape(33, 1, 4, 4) if image_input else rows
    probabilities = np.asarray(model.predict_proba(query))
    predicted = np.asarray(model.predict(query))
    assert predicted.shape == (33,)
    assert predicted.dtype == np.asarray(model.classes_).dtype
    np.testing.assert_array_equal(predicted, np.asarray(model.classes_)[probabilities.argmax(axis=1)])
    np.testing.assert_array_equal(predicted[:11], model.predict(query[:11]))
