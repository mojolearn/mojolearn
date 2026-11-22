# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The tree inspection records behind `GradientBoosting.get_tree_leaf_counts`
and `get_leaf_values`, on the literal model texts of test_gbdt_tree_metadata.py
(moved from its `test_tree_values_recover_bits_and_preserve_leaf_dimension_order`,
`test_tree_offsets_and_depth_zero` and `test_incomplete_metadata_refused`).

The public door (`bindings/trees_public_gbdt_door.mojo` `gbdt_tree_metadata_op`)
reads `model_` and calls `core.trees_gbdt_metadata.gbdt_tree_metadata` on it;
this check calls that entry with the same texts.

  0  `tree 0 depth 1 dim 2` with four leaf words      -> counts [2], the four
     float32 words back exactly from the float64 values
  1  the same as an `ntree 0 nodes 1 dim 2` record      -> the same
  2  two trees, depth 0 and depth 1, dim 1              -> counts [1, 2], values [0, 1, 2]
  3  `trees 1` and no tree record                       -> refused
  4  one depth-1 tree with no leaves                    -> refused
  5  a tree record numbered 1 first                     -> refused

The entry is host code on both columns (it reads text, no device). The
returned dtypes, the unfitted refusal and the no-GPU-binding guarantee stay
in the Python test (boundary).
"""
from std.memory import bitcast

from core.trees_gbdt_metadata import gbdt_tree_metadata
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require

comptime GBDT_TREE_METADATA_CASES = 6


def _leaf_text(record: String) -> String:
    return (String("trees 1\n") + record + "\n"
        + "leaf 0 0 ignored/80000000\nleaf 0 1 ignored/00000001\n"
        + "leaf 0 2 ignored/3f800001\nleaf 0 3 ignored/bf800001\n")


def _check_bits(text: String, label: String) raises:
    var parsed = gbdt_tree_metadata(text)
    require(len(parsed.counts) == 1 and parsed.counts[0] == UInt32(2), label + ": leaf counts are not [2]")
    var words: List[UInt32] = [0x80000000, 0x00000001, 0x3F800001, 0xBF800001]
    require(len(parsed.values) == len(words), label + ": wrong number of leaf values")
    for i in range(len(words)):  # small-loop(four leaf words)
        var narrow = parsed.values[i].cast[DType.float32]()
        require(bitcast[DType.uint32](narrow) == words[i], label + ": leaf " + String(i) + " bits differ")


def _refused(text: String) -> Bool:
    try:
        _ = gbdt_tree_metadata(text)
    except:
        return True
    return False


def run_case(index: Int) raises:
    var label = "gbdt_tree_metadata case " + String(index)
    if index == 0:
        _check_bits(_leaf_text("tree 0 depth 1 dim 2 weights 0"), label)
    elif index == 1:
        _check_bits(_leaf_text("ntree 0 nodes 1 dim 2 weights 0"), label)
    elif index == 2:
        var text = String("trees 2\ntree 0 depth 0 dim 1 weights 0\nleaf 0 0 0/00000000\n"
            + "tree 1 depth 1 dim 1 weights 0\nleaf 1 0 1/3f800000\nleaf 1 1 2/40000000\n")
        var parsed = gbdt_tree_metadata(text)
        require(len(parsed.counts) == 2 and parsed.counts[0] == UInt32(1) and parsed.counts[1] == UInt32(2),
                label + ": leaf counts are not [1, 2]")
        require(len(parsed.values) == 3 and parsed.values[0] == 0.0 and parsed.values[1] == 1.0
                and parsed.values[2] == 2.0, label + ": leaf values are not [0, 1, 2]")
    elif index == 3:
        require(_refused(String("trees 1\n")), label + ": a text with no tree record was accepted")
    elif index == 4:
        require(_refused(String("trees 1\ntree 0 depth 1 dim 1 weights 0\n")),
                label + ": a tree with no leaves was accepted")
    else:
        require(_refused(String("trees 1\ntree 1 depth 0 dim 1 weights 0\n")),
                label + ": an out-of-order tree record was accepted")


def test_cases() raises:
    check_cases("gbdt_tree_metadata", GBDT_TREE_METADATA_CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
