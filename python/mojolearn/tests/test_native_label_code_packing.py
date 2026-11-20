# SPDX-License-Identifier: Apache-2.0
"""Native code scratch preserves public label values, order and refusal seams."""
import numpy as np
import pytest

from mojolearn import Array, NearestCentroid
from mojolearn._labels import decode_labels


@pytest.mark.parametrize("code_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("count", [0, 5])
@pytest.mark.parametrize("classes", [
    [-(2**63), 0, 2**63 - 1],
    [0.5, -2.5, 7.0],
    [False, True, False],
    ["🦊", "a\x00b", "é"],
    [2**70, -2**70, 7],
])
def test_shared_decoder_widths_keep_label_values_and_output_kind(code_dtype, count, classes):
    order = [2, 0, 1, 2, 1][:count]
    words = np.array(order, dtype=code_dtype)
    codes = Array.from_buffer(words)
    result = decode_labels(classes, codes)
    expected = [classes[i] for i in order]
    if type(result) is Array:
        assert result.dtype == ("<f8" if type(classes[0]) is float else "<i8")
        assert result.tolist() == expected
    else:
        assert result == expected
    np.testing.assert_array_equal(words, order)
    del codes, words
    assert (result.tolist() if type(result) is Array else result) == expected


@pytest.mark.parametrize("code_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("classes", [["🦊", "a\x00b"], [2**70, -2**70]])
def test_list_label_pack_keeps_negative_indices_and_range_refusal(code_dtype, classes):
    codes = Array.from_buffer(np.array([-1, -2, 0, 1], dtype=code_dtype))
    assert decode_labels(classes, codes) == [classes[1], classes[0], classes[0], classes[1]]
    for invalid in [-3, 2]:
        bad = Array.from_buffer(np.array([0, invalid], dtype=code_dtype))
        with pytest.raises(IndexError, match="list index out of range"):
            decode_labels(classes, bad)


def _centroid_model():
    source = np.array([[-3, -1], [-3, 1], [-1, -1], [-1, 1],
                       [1, -1], [1, 1], [3, -1], [3, 1]], dtype=np.float32)
    return NearestCentroid(priors="uniform").fit(source, [0, 0, 0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize("classes, expected, kind", [
    ([-(2**63), 2**63 - 1], [-(2**63), -(2**63), 2**63 - 1], "i"),
    ([False, True], [0, 0, 1], "i"),
    ([-3.0, 7.0], [-3.0, -3.0, 7.0], "f"),
    (["🦊", "a\x00b"], ["🦊", "🦊", "a\x00b"], "U"),
    ([2**70, -2**70], [2**70, 2**70, -2**70], "O"),
])
def test_nearest_centroid_native_codes_pack_rebound_class_order(classes, expected, kind):
    model = _centroid_model()
    model.classes_ = classes
    query = np.array([[-2, 0], [0, 0], [2, 0]], dtype=np.float32)
    result = model.predict(query)
    actual = np.asarray(result)
    assert actual.tolist() == expected
    assert actual.dtype.kind == kind
    # Packing returns its own public result; neither the code scratch nor
    # the model's array owners may be needed after predict returns.
    del model, query
    assert np.asarray(result).tolist() == expected


def test_nearest_centroid_rebound_short_string_classes_keep_index_refusal():
    model = _centroid_model()
    model.classes_ = ["only"]
    with pytest.raises(IndexError, match="list index out of range"):
        model.predict(np.array([[2, 0]], dtype=np.float32))
    assert list(model.classes_) == ["only"]
