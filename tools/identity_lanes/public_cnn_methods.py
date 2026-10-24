# SPDX-License-Identifier: Apache-2.0
# The existing x-cnn lanes do not call the layer fit methods or classifier
# predict. Keep their admitted payloads intact; these separate public-operation
# lanes remain OWED until frozen runs/reference admission qualify their outputs.


@lane("public-cnn-layer-fit")
def _(ml, X, yc, yr, Xh=None):
    rows = np.ascontiguousarray(X[:32, :16], dtype=np.float32)
    conv1 = ml.Conv1d(2, 3, 3, padding=1, random_state=5, input_shape=(2, 8), numeric_mode="identical")
    conv2 = ml.Conv2d(1, 3, 3, padding=1, random_state=7, input_shape=(1, 4, 4), numeric_mode="identical")
    if conv1.fit(rows) is not conv1 or conv2.fit(rows) is not conv2:
        raise AssertionError("convolution fit must return its layer")
    bn1 = ml.BatchNorm1d(16, momentum=0.25, numeric_mode="identical")
    bn2 = ml.BatchNorm2d(4, momentum=0.25, input_shape=(4, 2, 2), numeric_mode="identical")
    bn1.eval()
    bn2.eval()
    if bn1.fit(rows) is not bn1 or bn2.fit(rows) is not bn2:
        raise AssertionError("batch normalization fit must return its layer")
    return _fit(dict(conv1=_h(conv1.transform(rows), conv1.weight_, conv1.bias_),
                     conv2=_h(conv2.transform(rows), conv2.weight_, conv2.bias_),
                     bn1=_h(bn1.running_mean_, bn1.running_var_, bn1.transform(rows)),
                     bn2=_h(bn2.running_mean_, bn2.running_var_, bn2.transform(rows)),
                     metadata=_h(np.asarray([conv1.n_features_in_, conv2.n_features_in_,
                                             bn1.num_batches_tracked_, bn2.num_batches_tracked_,
                                             bn1.training, bn2.training], dtype=np.int64))))


@lane("public-cnn-classifier-predict")
def _(ml, X, yc, yr, Xh=None):
    rows = np.ascontiguousarray(X[:33, :16], dtype=np.float32)
    labels = np.asarray([-11, 23, 71], dtype=np.int64)[np.arange(len(rows)) % 3]
    model = ml.CNNClassifier(input_shape=(1, 4, 4), conv_channels=(2,), kernel_size=3, pool_size=2,
                             batch_size=11, max_iter=1, random_state=9, numeric_mode="identical").fit(rows, labels)
    predicted = model.predict(rows)
    probabilities = model.predict_proba(rows)
    # The independent harness oracle checks the class-word gather, including
    # nonconsecutive labels; hashes retain the actual public returned labels.
    np.testing.assert_array_equal(predicted, np.asarray(model.classes_)[np.asarray(probabilities).argmax(axis=1)])
    return _fit(dict(predicted=_h(predicted), probabilities=_h(probabilities)), model,
                lambda e: (e.predict(np.ascontiguousarray(Xh[:33, :16], dtype=np.float32)),))


_batch_decl("n/a:coverage-probe (the lane checks public call paths the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "public-cnn-layer-fit", "public-cnn-classifier-predict")
