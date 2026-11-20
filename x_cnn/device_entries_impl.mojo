# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed device entries behind `bindings/_mojolearn_x_cnn.mojo`'s Python
doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of that
shell, so a Python-free caller reaches the same source; no Python here):
the device label and CSR checks and the CSR validation.

TOMBSTONE: the resident fit epochs (`fit_epoch_r_device`, `fit_epoch_d_device`,
`sgd_base_rows`, `widen_f32_into`, `_tape_words`; CNNClassifier is a captured
program on the executor, sequence/cnn_classifier_program.mojo) deleted
2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
"""
from core.hostptr import read_i32
from x_cnn.ops import IP
from x_cnn.device import cnn_ctx
from core.device_zero import enqueue_fill
from max.gpu import block_dim, block_idx, thread_idx


# The host entries' label and CSR refusals, as one bulk upload, one check
# launch and two flag words read back (the host walks they replace read
# every label, rowptr entry and edge on the CPU). Same predicates, same
# messages in the same priority; every writer of a flag stores the same 1.
comptime _CHK_TPB = 256


def _label_bad_kernel(labels: IP, n: Int32, k: Int32, flag: IP):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if labels.unsafe_load(i) >= k:
            flag.unsafe_store(0, Int32(1))


def _csr_bad_kernel(csr: IP, n: Int32, nnz: Int32, flag: IP):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if csr.unsafe_load(i + 1) < csr.unsafe_load(i):
            flag.unsafe_store(0, Int32(1))
    if i < Int(nnz):
        var c = csr.unsafe_load(Int(n) + 1 + i)
        var r = csr.unsafe_load(Int(n) + 1 + Int(nnz) + i)
        if c < Int32(0) or c >= n or r < Int32(0) or r >= n:
            flag.unsafe_store(1, Int32(1))


def _device_int_check(src: IP, words: Int, n: Int, m: Int, csr: Bool) raises -> Tuple[Int32, Int32]:
    """`words` int32 of the caller's `src` checked on the device: the
    label check (`m` = classes) or the CSR check (`m` = nnz). Returns the
    two flag words."""
    var ctx = cnn_ctx()
    var d = ctx.enqueue_create_buffer[DType.int32](max(words, 1))
    if words > 0:
        ctx.enqueue_copy(dst_buf=d, src_ptr=src)
    var flag = ctx.enqueue_create_buffer[DType.int32](2)
    enqueue_fill(ctx, flag, Int32(0))
    var span = max(n, m) if csr else n
    if span > 0:
        if csr:
            ctx.enqueue_function[_csr_bad_kernel](
                d.unsafe_ptr(), Int32(n), Int32(m), flag.unsafe_ptr(),
                grid_dim=((span + _CHK_TPB - 1) // _CHK_TPB, 1, 1), block_dim=(_CHK_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[_label_bad_kernel](
                d.unsafe_ptr(), Int32(n), Int32(m), flag.unsafe_ptr(),
                grid_dim=((span + _CHK_TPB - 1) // _CHK_TPB, 1, 1), block_dim=(_CHK_TPB, 1, 1),
            )
    var host = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_buf=host, src_buf=flag)
    ctx.synchronize()
    var f0 = host.unsafe_ptr().unsafe_load(0)
    var f1 = host.unsafe_ptr().unsafe_load(1)
    _ = d^
    _ = flag^
    _ = host^
    return (f0, f1)


def csr_checked_device(csr_address: Int, n: Int, F: Int, nnz: Int, mode: Int) raises -> Tuple[List[Int32], List[Int32]]:
    """Validate a CSR block [rowptr | col | row] at `csr_address` against
    [n, F, nnz, mode] (the ordering checks on the device); returns (the
    block, the int32 params)."""
    if n <= 0 or F <= 0 or nnz < 0 or mode < 0 or mode > 3:
        raise Error("x_cnn spmm: positive n and F, nnz >= 0, mode in {0, 1, 2, 3}")
    var csr = read_i32(csr_address, n + 1 + 2 * nnz)
    if Int(csr[0]) != 0 or Int(csr[n]) != nnz:
        raise Error("x_cnn spmm: rowptr must start at 0 and end at nnz")
    var bad = _device_int_check(
        csr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), len(csr), n, nnz, True
    )
    if bad[0] != Int32(0):
        raise Error("x_cnn spmm: rowptr must be non-decreasing")
    if bad[1] != Int32(0):
        raise Error("x_cnn spmm: a column or row index is out of range")
    var prm: List[Int32] = [Int32(n), Int32(F), Int32(nnz), Int32(mode)]
    return (csr^, prm^)
