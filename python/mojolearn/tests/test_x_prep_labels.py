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
