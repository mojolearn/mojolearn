"""CatBoost-compatible tree inspection, through the public methods.

The leaf counts and leaf bits of each record kind, the depth-zero offsets
and the refusals of incomplete records are the native check
`gbdt_tree_metadata`. What stays here is the boundary: the returned dtypes
(through `np.asarray`, zero-copy), that inspection loads no GPU binding,
that a refusal arrives as `ValueError`, and the unfitted refusal.
"""
import numpy as np
import pytest

from mojolearn.ensemble import GradientBoosting


def model(text):
    fitted = GradientBoosting()
    fitted.model_ = text
    fitted._bind = lambda name: pytest.fail("metadata inspection loaded a GPU binding")
    return fitted


def test_inspection_returns_uint32_counts_and_float64_values():
    obj = model("trees 1\ntree 0 depth 1 dim 2 weights 0\n"
                "leaf 0 0 ignored/80000000\nleaf 0 1 ignored/00000001\n"
                "leaf 0 2 ignored/3f800001\nleaf 0 3 ignored/bf800001\n")
    counts = obj.get_tree_leaf_counts()
    assert np.asarray(counts).dtype == np.uint32
    np.testing.assert_array_equal(counts, [2])
    values = obj.get_leaf_values()
    assert np.asarray(values).dtype == np.float64
    np.testing.assert_array_equal(np.asarray(values).astype(np.float32).view(np.uint32),
                                  [0x80000000, 1, 0x3f800001, 0xbf800001])


def test_incomplete_metadata_refused():
    with pytest.raises(ValueError):
        model("trees 1\n").get_leaf_values()


def test_unfitted_inspection_refused():
    with pytest.raises(RuntimeError, match="before fit"):
        GradientBoosting().get_tree_leaf_counts()
