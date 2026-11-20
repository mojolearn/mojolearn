# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries behind `bindings/_mojolearn_x_cnn_host.mojo`'s Python
doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of that
shell, so a Python-free caller reaches the same source; no Python here).

`out_f32` is the CPU column's output seam the shell's address-out entries
call; the end-to-end sabotage arm
(x_cnn/checks/sabotage/e2e_host_output_bit.patch) flips every word's low
bit here.
"""
from std.memory import unsafe_memcpy

from core.hostptr import read_i32
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW
from x_cnn.ops import FP, IP


@always_inline
def out_f32(p: FP, n: Int):
    """THE CPU COLUMN'S OUTPUT SEAM for the address-out entries: every
    float32 word such an entry returns is in p[0:n] when this runs. In
    production it does nothing. The end-to-end sabotage arm
    (x_cnn/checks/sabotage/e2e_host_output_bit.patch) flips every word's low
    bit here, and in the shell's `copy_f32` for the entries that still copy
    a result."""
    _ = p
    _ = n


def pool_counts(prm: List[Int32]) -> Tuple[Int, Int]:
    var nc = Int(prm[PP_N]) * Int(prm[PP_C])
    return (nc * Int(prm[PP_H]) * Int(prm[PP_W]), nc * Int(prm[PP_OH]) * Int(prm[PP_OW]))


def check_labels_below(py_: IP, n: Int, k: Int) raises:
    """The softmax entry's label check: every label of the n rows < k."""
    for i in range(n):
        if Int(py_[i]) >= k:
            raise Error("x_cnn softmax: a label is not a class index")


def csr_checked(csr_address: Int, n: Int, F: Int, nnz: Int, mode: Int) raises -> Tuple[List[Int32], List[Int32]]:
    """Validate a CSR block [rowptr | col | row] at `csr_address` against
    [n, F, nnz, mode]; returns (the block, the int32 params)."""
    if n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var csr = read_i32(csr_address, n + 1 + 2 * nnz)
    if Int(csr[0]) != 0 or Int(csr[n]) != nnz:
        raise Error("x_cnn spmm: rowptr must start at 0 and end at nnz")
    for r in range(n):
        if csr[r + 1] < csr[r]:
            raise Error("x_cnn spmm: rowptr must be non-decreasing")
    for e in range(nnz):
        var c = Int(csr[n + 1 + e])
        var rr = Int(csr[n + 1 + nnz + e])
        if c < 0 or c >= n or rr < 0 or rr >= n:
            raise Error("x_cnn spmm: a column or row index is out of range")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    return (csr^, prm^)


# TOMBSTONE: x_cnn_res_argmax (CNNClassifier.predict's device argmax; the classifier predicts on the x_sequence program executor since lane x8) deleted 2026-11-09 by lane/x8b-cnn-resident-predict; code recoverable at 7c9d9d5575.
# Restore: git apply experiments/removed/X8B_XCNN_RES_ARGMAX.patch


# TOMBSTONE: x_cnn_res_gather (gather_rows_f32, the CPU body; unused: no caller since lane x8, only python/mojolearn/tests/test_x_cnn_repeat.py) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/XCNN_RES_GATHER.patch


# TOMBSTONE: the host fit epoch (epoch_rows_fill, adam_rows, widen_f32_into, widen_f32_list_into, sgd_hyper_rows, fit_epoch_r_into) deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
# Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
