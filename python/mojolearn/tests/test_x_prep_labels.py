# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Prep lane sanity: LabelEncoder, LabelBinarizer and MultiLabelBinarizer
against scikit-learn (exact: codes and indicators)."""
import numpy as np
try:
    import pytest
    pytest.importorskip("sklearn")
except ImportError:
    pass
import sklearn.preprocessing as skp
import mojolearn as ml


def test_label_encoder():
    rng = np.random.default_rng(0)
    for y in (rng.integers(-5, 9, 200) * 3, rng.standard_normal(100).astype(np.float32),
              np.array(["b", "a", "c", "a"])):
        m, r = ml.LabelEncoder().fit(y), skp.LabelEncoder().fit(y)
        np.testing.assert_array_equal(np.asarray(m.classes_), r.classes_)
        np.testing.assert_array_equal(np.asarray(m.transform(y)), r.transform(y))
        np.testing.assert_array_equal(np.asarray(m.inverse_transform(np.asarray(m.transform(y)))), y)


if __name__ == "__main__":
    test_label_encoder()
    print("PASS test_x_prep_labels")


def test_label_binarizer():
    rng = np.random.default_rng(1)
    for y in (rng.integers(0, 4, 50) * 2, rng.integers(0, 2, 30), np.array(["x", "y", "x"]), np.array([5, 5])):
        for kw in (dict(), dict(neg_label=-1, pos_label=3)):
            m, r = ml.LabelBinarizer(**kw).fit(y), skp.LabelBinarizer(**kw).fit(y)
            np.testing.assert_array_equal(np.asarray(m.classes_), r.classes_)
            np.testing.assert_array_equal(np.asarray(m.transform(y)), r.transform(y))


def test_multilabel_binarizer():
    ys = [[1, 3], [2], [], [3, 1, 7], [7, 7]]
    m, r = ml.MultiLabelBinarizer().fit(ys), skp.MultiLabelBinarizer().fit(ys)
    np.testing.assert_array_equal(np.asarray(m.classes_), r.classes_)
    np.testing.assert_array_equal(np.asarray(m.transform(ys)), r.transform(ys))
    ys = [["a", "b"], ["c"]]
    np.testing.assert_array_equal(np.asarray(ml.MultiLabelBinarizer().fit_transform(ys)),
                                  skp.MultiLabelBinarizer().fit_transform(ys))


if __name__ == "__main__":
    test_label_binarizer()
    test_multilabel_binarizer()
    print("PASS test_x_prep_labels (binarizers)")
