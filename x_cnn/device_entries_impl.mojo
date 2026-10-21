# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed device entries behind `bindings/_mojolearn_x_cnn.mojo`'s Python
doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of that
shell, so a Python-free caller reaches the same source; no Python here):
the device label and CSR checks, the CSR validation, and the resident fit
epochs (`x_cnn_fit_epoch_r` and `x_cnn_fit_epoch_d`) with their step loops.
"""
from bindings.hostptr import read_i32
from x_cnn.ops import FP, IP, AH_ROW
from x_cnn.device import (
    linear_forward_into,
    softmax_xent_into,
    conv_block_forward_into,
    res_alloc,
    res_free,
    res_upload,
    res_download,
    res_gather_pair,
    opt_many_resident,
)
from x_cnn.device import (
    res_gather_pair_perm, adam_hyper_resident, opt_many_resident_h, softmax_xent_res_loss,
)
from x_cnn.device import cnn_ctx
from autograd.rules_cnn import cnn_backward_autograd
from core.device_zero import enqueue_fill
from std.gpu import block_dim, block_idx, thread_idx


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


@always_inline
def _at(a: Int) -> FP:
    return FP(unsafe_from_address=a)


@always_inline
def _at_i(a: Int) -> IP:
    return IP(unsafe_from_address=a)


def _tape_words() raises -> Int:
    """Two resident words for the tape step: [the unit seed 1.0, the loss's
    recorded value (never read)]."""
    var w = res_alloc(2)
    var one: List[Float32] = [Float32(1), Float32(0)]
    res_upload(w, one.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 2)
    _ = one^
    return w


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


def fit_epoch_r_device(
    n: Int,
    batch: Int,
    adam: Bool,
    nh: Int,
    nb: Int,
    steps: Int,
    flat: Int,
    k: Int,
    bh: List[List[Int]],
    head: List[Int],
    a: List[Int],
    data: List[Int],
    t: Tuple[List[Int], List[Int], List[Int], List[Int]],
    cs: List[List[List[Int32]]],
    ps: List[List[List[Int32]]],
    pools: List[List[Bool]],
    rows: IP,
    hyp: MutPointer[Float64, MutUntrackedOrigin],
    losses: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """`x_cnn_fit_epoch_r`'s epoch: per step the row gathers, each block's
    forward, the head, the softmax, the tape backward and one optimizer
    launch over every parameter."""
    var tw = _tape_words()
    for st in range(steps):  # small-loop(steps: optimizer steps of one epoch): orchestration that launches device work per step
        var s0 = st * batch
        var m = min(batch, n - s0)
        var q = 0 if m == batch else 1
        res_gather_pair(a[0], data[0], data[2], a[1], data[1], 1, rows + s0, m)
        var src = a[0]
        for j in range(nb):
            conv_block_forward_into[True](
                _at(src), _at(bh[j][0]), _at(bh[j][1]), cs[q][j], ps[q][j], pools[q][j], _at(bh[j][4]),
                _at_i(bh[j][5]), bh[j][7], bh[j][8],
            )
            src = bh[j][4]
        linear_forward_into[True](_at(src), _at(head[0]), _at(head[1]), m, flat, k, _at(a[2]))
        var loss = softmax_xent_into[True](_at(a[2]), _at_i(a[1]), m, k, _at(a[3]), _at(a[4]))
        losses[st] = Float64(loss)
        # the head and every block's backward on the tape (autograd/rules_cnn.mojo)
        # TOMBSTONE: MOJOLEARN_AUTOGRAD_CNN_OFF (removed: tape is the only backward) deleted 2026-10-19 by ddd4fa509; code recoverable at ddd4fa509^.
        # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_CNN_OFF.patch (the hand walk arm)
        cnn_backward_autograd(bh, head, a, cs[q], ps[q], pools[q], m, flat, k, tw, tw + 4)
        var h = List[Float32]()
        for e in range(nh):  # small-loop(nh: optimizer scalars of one step, six or nine): builds one launch parameter list, not data
            h.append(Float32(hyp[st * nh + e]))
        if adam:
            opt_many_resident[True](t[0], t[1], t[2], t[3], h)
        else:
            opt_many_resident[False](t[0], t[1], t[2], t[3], h)
    res_free(tw)


def sgd_base_rows(mut base: List[Float32], f5: List[Float64]):
    """SGD's two hyper rows for `x_cnn_fit_epoch_d`: the five caller words
    through float32 and the first-step flag 1, then the same with flag 0."""
    for r in range(2):
        for e in range(5):
            base.append(Float32(f5[e]))
        base.append(Float32(1) if r == 0 else Float32(0))


def widen_f32_into(dst: MutPointer[Float64, MutUntrackedOrigin], src: List[Float32], n: Int):
    """dst[e] = Float64(src[e]) for e < n: the per-step optimizer scalars
    (AH_ROW per step) or one loss per step, already downloaded."""
    for e in range(n):  # small-loop(n: per-step optimizer scalars or per-step losses, not row data)
        dst[e] = Float64(src[e])


def fit_epoch_d_device(
    n: Int,
    batch: Int,
    adam: Bool,
    done: Int,
    nb: Int,
    steps: Int,
    flat: Int,
    k: Int,
    shuffle: Bool,
    key: UInt64,
    bh: List[List[Int]],
    head: List[Int],
    a: List[Int],
    data: List[Int],
    t: Tuple[List[Int], List[Int], List[Int], List[Int]],
    cs: List[List[List[Int32]]],
    ps: List[List[List[Int32]]],
    pools: List[List[Bool]],
    mut base: List[Float32],
    mut l32: List[Float32],
) raises:
    """`x_cnn_fit_epoch_d`'s epoch on the device: the resident hyper rows,
    per step the permuted row gathers, the forward, the resident loss, the
    tape backward and the optimizer launch; the losses come down once into
    `l32`."""
    var hbuf = res_alloc(steps * AH_ROW if adam else 12)
    var lbuf = res_alloc(steps)
    var tw = _tape_words()
    if adam:
        adam_hyper_resident(base, done + 1, steps, hbuf)
    else:
        res_upload(hbuf, base.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 12)
    for st in range(steps):
        var s0 = st * batch
        var m = min(batch, n - s0)
        var q = 0 if m == batch else 1
        res_gather_pair_perm(a[0], data[0], data[2], a[1], data[1], 1, n, s0, m, shuffle, key)
        var src = a[0]
        for j in range(nb):
            conv_block_forward_into[True](
                _at(src), _at(bh[j][0]), _at(bh[j][1]), cs[q][j], ps[q][j], pools[q][j], _at(bh[j][4]),
                _at_i(bh[j][5]), bh[j][7], bh[j][8],
            )
            src = bh[j][4]
        linear_forward_into[True](_at(src), _at(head[0]), _at(head[1]), m, flat, k, _at(a[2]))
        softmax_xent_res_loss(_at(a[2]), _at_i(a[1]), m, k, _at(a[3]), _at(a[4]), lbuf, st)
        # the head and every block's backward on the tape (autograd/rules_cnn.mojo)
        # TOMBSTONE: MOJOLEARN_AUTOGRAD_CNN_OFF (removed: tape is the only backward) deleted 2026-10-19 by ddd4fa509; code recoverable at ddd4fa509^.
        # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_CNN_OFF.patch (the hand walk arm)
        cnn_backward_autograd(bh, head, a, cs[q], ps[q], pools[q], m, flat, k, tw, tw + 4)
        if adam:
            opt_many_resident_h[True](t[0], t[1], t[2], t[3], hbuf + 4 * AH_ROW * st)
        else:
            opt_many_resident_h[False](t[0], t[1], t[2], t[3], hbuf + (0 if done + st == 0 else 24))
    res_download(lbuf, l32.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), steps)
    res_free(hbuf)
    res_free(lbuf)
    res_free(tw)
