# SPDX-License-Identifier: Apache-2.0
"""CatBoost-compatible tree inspection using authoritative archive bits."""
import numpy as np
import pytest

from mojolearn.ensemble import GradientBoosting


def model(text):
    fitted = GradientBoosting()
    fitted.model_ = text
    fitted._bind = lambda name: pytest.fail("metadata inspection loaded a GPU binding")
    return fitted


@pytest.mark.parametrize("record", ["tree 0 depth 1 dim 2 weights 0",
                                    "ntree 0 nodes 1 dim 2 weights 0"])
def test_tree_values_recover_bits_and_preserve_leaf_dimension_order(record):
    obj = model("trees 1\n" + record + "\n"
                "leaf 0 0 ignored/80000000\nleaf 0 1 ignored/00000001\n"
                "leaf 0 2 ignored/3f800001\nleaf 0 3 ignored/bf800001\n")
    counts = obj.get_tree_leaf_counts()
    assert counts.dtype == np.uint32
    np.testing.assert_array_equal(counts, [2])
    values = obj.get_leaf_values()
    assert values.dtype == np.float64
    np.testing.assert_array_equal(values.astype(np.float32).view(np.uint32),
                                  [0x80000000, 1, 0x3f800001, 0xbf800001])


def test_tree_offsets_and_depth_zero():
    obj = model("trees 2\ntree 0 depth 0 dim 1 weights 0\nleaf 0 0 0/00000000\n"
                "tree 1 depth 1 dim 1 weights 0\nleaf 1 0 1/3f800000\nleaf 1 1 2/40000000\n")
    np.testing.assert_array_equal(obj.get_tree_leaf_counts(), [1, 2])
    np.testing.assert_array_equal(obj.get_leaf_values(), [0, 1, 2])


@pytest.mark.parametrize("text", ["trees 1\n", "trees 1\ntree 0 depth 1 dim 1 weights 0\n",
                                    "trees 1\ntree 1 depth 0 dim 1 weights 0\n"])
def test_incomplete_metadata_refused(text):
    with pytest.raises(ValueError):
        model(text).get_leaf_values()


def test_unfitted_inspection_refused():
    with pytest.raises(RuntimeError, match="before fit"):
        GradientBoosting().get_tree_leaf_counts()
