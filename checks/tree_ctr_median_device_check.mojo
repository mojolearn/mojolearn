# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FeatureFreq tree-CTR Median grid ON THE DEVICE against the host
statement, word for word.

    pixi run check-native-gpu tree_ctr_median_device_check

Three FeatureFreq-shaped columns (few distinct values, many, and more
borders than values) and a constant one go through
`launch_tree_median_words` into ONE level-wide words buffer, read home once.
Checked:

  * every column's record equals the record the gather statement gives over
    a host-sorted copy (`median_words_host`), bit for bit;
  * the grid from each record (`tree_ctr_median_from_words`) equals the
    host column's (`tree_ctr_median_borders_host`, the reference's
    `select_borders` Median), border for border;
  * the constant column takes the single 0.5;
  * the columns are not modified by the sort.
"""

from std.testing import TestSuite

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.native_tests.kit import download, upload
from gbdt.checks.tree_ctr_check import median_fixture_column, median_words_host
from gbdt.ctrs.tree_ctr import (
    TreeF32Ptr,
    tree_ctr_median_borders_host,
    tree_ctr_median_from_words,
    tree_ctr_median_words,
)
from gbdt.ctrs.tree_ctr_device import TreeMedianScratch, launch_tree_median_words


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("tree CTR Median device check: " + what)


def check_tree_ctr_median_device() raises:
    var ctx = DeviceContext()
    var n = 1531
    var cards: List[Int] = [7, 211, 3, 1]
    var counts: List[Int] = [15, 15, 31, 15]
    var columns = List[List[Float32]]()
    var bases = List[Int]()
    var total = 0
    for k in range(len(cards)):
        columns.append(median_fixture_column(n, cards[k], 40 + k))
        bases.append(total)
        total += tree_ctr_median_words(counts[k])
    var scratch = TreeMedianScratch(ctx, n)
    var d_words = ctx.enqueue_create_buffer[DType.float32](total)
    var d_columns = List[DeviceBuffer[DType.float32]]()
    for k in range(len(cards)):
        d_columns.append(upload[DType.float32](ctx, columns[k]))
    # every column's launches on the one queue, then ONE readback
    for k in range(len(cards)):
        var d_column = d_columns[k]
        launch_tree_median_words(
            ctx, d_column, n, counts[k], scratch, d_words, bases[k]
        )
    var words = download[DType.float32](ctx, d_words, total)
    for k in range(len(cards)):
        var want_words = median_words_host(columns[k], counts[k])
        for i in range(len(want_words)):
            _need(
                words[bases[k] + i] == want_words[i],
                "column " + String(k) + " word " + String(i),
            )
        var got = tree_ctr_median_from_words(words, bases[k], counts[k])
        var host = tree_ctr_median_borders_host(
            TreeF32Ptr(unsafe_from_address=Int(columns[k].unsafe_ptr())),
            n, counts[k],
        )
        _need(len(got) == len(host), "column " + String(k) + " border count")
        for i in range(len(host)):
            _need(got[i] == host[i], "column " + String(k) + " border " + String(i))
        if cards[k] == 1:
            _need(len(got) == 1 and got[0] == Float32(0.5), "a constant column takes 0.5")
        var d_column = d_columns[k]
        var back = download[DType.float32](ctx, d_column, n)
        for r in range(n):
            _need(back[r] == columns[k][r], "the sort left the column as it was")
    ctx.synchronize()
    _ = scratch^
    _ = len(columns)
    print("  tree CTR Median grid on the device equals the host statement")


def test_tree_ctr_median_device() raises:
    check_tree_ctr_median_device()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
