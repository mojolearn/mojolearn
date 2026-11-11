# SPDX-License-Identifier: Apache-2.0
"""Typed native calls of the ridge multi table
(core/linear_entry_table_multi.mojo; lane calltable-base-helpers,
2026-11-11). No Python: a caller passes buffer addresses and plain words
and keeps the buffers alive until the call returns. These calls do not
touch the GIL."""
from core.linear_entry_table import LinearEntryKey
from core.linear_entry_table_multi import (
    ridge_multi_table_entry, RIDGE_MULTI_ABI, RM_OP_COUNT, RM_COL_SUMS, RM_MEANS_FINISH, RM_CENTER, RM_FIT,
    RM_PREDICT,
)
from core.native_descriptor_call import native_descriptor_call


def ridge_multi_call(key: LinearEntryKey, op: Int, addrs: List[Int], ints: List[Int],
                     floats: List[Float64]) raises -> Int:
    """One ridge multi operation in the estimators image the key names."""
    var entry = ridge_multi_table_entry(key, 0, String(RIDGE_MULTI_ABI), RM_OP_COUNT)
    return native_descriptor_call(entry, op, addrs, ints, floats)


def ridge_multi_available(key: LinearEntryKey) -> Bool:
    """True when the image the key names published the ridge multi table
    (a build with the one-program ridge)."""
    try:
        _ = ridge_multi_table_entry(key, 0, String(RIDGE_MULTI_ABI), RM_OP_COUNT)
        return True
    except:
        return False


def rm_col_sums(key: LinearEntryKey, x: Int, sums: Int, rows: Int, cols: Int) raises:
    var addrs: List[Int] = [x, sums]
    var ints: List[Int] = [rows, cols]
    var floats = List[Float64]()
    _ = ridge_multi_call(key, RM_COL_SUMS, addrs, ints, floats)


def rm_means_finish(key: LinearEntryKey, sums: Int, mean64: Int, mean32: Int, cols: Int, rows: Int,
                    weighted: Bool, total: Float64) raises:
    var addrs: List[Int] = [sums, mean64, mean32]
    var ints: List[Int] = [cols, rows, 1 if weighted else 0]
    var floats: List[Float64] = [total]
    _ = ridge_multi_call(key, RM_MEANS_FINISH, addrs, ints, floats)


def rm_center(key: LinearEntryKey, x: Int, mu: Int, dst: Int, rows: Int, cols: Int) raises:
    var addrs: List[Int] = [x, mu, dst]
    var ints: List[Int] = [rows, cols]
    var floats = List[Float64]()
    _ = ridge_multi_call(key, RM_CENTER, addrs, ints, floats)


def rm_fit(key: LinearEntryKey, x: Int, y: Int, coef: Int, ymean: Int, n: Int, d: Int, m: Int, alpha: Float64,
           center_y: Bool, xmean: Int, icpt: Int) raises:
    """Ridge on every column of Y; with xmean and icpt nonzero also the
    intercepts (the former 7-word params)."""
    var want = xmean != 0 and icpt != 0
    var addrs: List[Int] = [x, y, coef, ymean, xmean if want else 0, icpt if want else 0]
    var ints: List[Int] = [n, d, m, 1 if center_y else 0, 1 if want else 0]
    var floats: List[Float64] = [alpha]
    _ = ridge_multi_call(key, RM_FIT, addrs, ints, floats)


def rm_predict(key: LinearEntryKey, x: Int, coef: Int, icpt: Int, dst: Int, n: Int, d: Int, m: Int) raises:
    var addrs: List[Int] = [x, coef, icpt, dst]
    var ints: List[Int] = [n, d, m]
    var floats = List[Float64]()
    _ = ridge_multi_call(key, RM_PREDICT, addrs, ints, floats)
