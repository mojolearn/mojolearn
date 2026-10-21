# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries behind `bindings/_mojolearn_x_cnn_host.mojo`'s Python
doors (lane thin-trees-metrics, 2026-10-21: moved word for word out of that
shell, so a Python-free caller reaches the same source; no Python here).

`out_f32` is the CPU column's output seam the shell's address-out entries
and the host fit epoch both call; the end-to-end sabotage arm
(x_cnn/checks/sabotage/e2e_host_output_bit.patch) flips every word's low
bit here.
"""
from std.memory import memcpy

from bindings.hostptr import read_i32
from x_cnn.ops import CP_N, CP_C, CP_H, CP_W, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW
from x_cnn.ops import PP_N, PP_C, PP_H, PP_W, PP_OH, PP_OW
from x_cnn.ops import FP, IP, argmax_row_at
from x_cnn.ops import epoch_rows_prm, epoch_rows_at, adam_hyper_base, adam_hyper_at, AH_ROW
from x_cnn.host.ops_host import conv_block_forward_into, conv_block_backward_into
from x_cnn.host.ops_host import linear_forward_into, linear_backward_into
from x_cnn.host.ops_host import softmax_xent_into, sgd_into, adam_into


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


@always_inline
def _at(a: Int) -> FP:
    return FP(unsafe_from_address=a)


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


def argmax_rows_into(src: FP, dst: IP, n: Int, k: Int):
    """`x_cnn_res_argmax` on host memory: `argmax_row_at` for rows 0 .. n-1."""
    var prm = List[Int32](length=1, fill=Int32(k))
    var pp = prm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for i in range(n):
        argmax_row_at(i, src, src, src, src, dst, pp)
    _ = prm^


def gather_rows_f32(
    d: MutPointer[Float32, MutUntrackedOrigin],
    s: MutPointer[Float32, MutUntrackedOrigin],
    rp: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    row: Int,
):
    """dst rows = src rows[rp[r]], `row` 4-byte words each."""
    for r in range(n):
        memcpy(dest=d + r * row, src=s + Int(rp[r]) * row, count=row)


def epoch_rows_fill(dst: IP, n: Int, shuffle: Bool, key: UInt64):
    var prm = epoch_rows_prm(n, 0, shuffle, key)
    var pp = prm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var none = FP(unsafe_from_address=Int(dst))
    for i in range(n):
        epoch_rows_at(i, none, none, none, none, dst, pp)
    _ = prm^


def adam_rows(
    lr: Float64, b1: Float64, b2: Float64, eps: Float64, wd: Float64, dec: Float64, step0: Int, nsteps: Int,
) -> List[Float32]:
    """`adam_hyper_at`'s rows of steps step0 .. step0 + nsteps - 1."""
    var base = adam_hyper_base(lr, b1, b2, eps, wd, dec)
    var out = List[Float32](length=nsteps * AH_ROW + 1, fill=Float32(0))
    var prm: List[Int32] = [Int32(step0)]
    var pb = base.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var po = out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pp = prm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for t in range(nsteps):
        adam_hyper_at(t, pb, po, po, po, pp, pp)
    _ = base^
    _ = prm^
    return out^


def widen_f32_into(dst: MutPointer[Float64, MutUntrackedOrigin], src: List[Float32], n: Int):
    """dst[e] = Float64(src[e]) for e < n."""
    for e in range(n):
        dst[e] = Float64(src[e])


def widen_f32_list_into(mut dst: List[Float64], src: List[Float32], n: Int):
    """dst[e] = Float64(src[e]) for e < n."""
    for e in range(n):
        dst[e] = Float64(src[e])


def sgd_hyper_rows(mut hyp: List[Float64], f5: List[Float64], steps: Int, done: Int):
    """SGD's per-step hyper rows: the five caller words through float32,
    then the first-step flag."""
    for st in range(steps):
        for e in range(5):
            hyp[st * 6 + e] = Float64(Float32(f5[e]))
        hyp[st * 6 + 5] = Float64(1) if done + st == 0 else Float64(0)


def fit_epoch_r_into(
    n: Int,
    batch: Int,
    adam: Bool,
    nh: Int,
    nb: Int,
    steps: Int,
    flat: Int,
    k: Int,
    row: Int,
    bh: List[List[Int]],
    head: List[Int],
    a: List[Int],
    cnt: List[Int],
    hp: List[Int],
    hg: List[Int],
    hb: List[Int],
    cs: List[List[List[Int32]]],
    ps: List[List[List[Int32]]],
    pools: List[List[Bool]],
    rows: MutPointer[Int32, MutUntrackedOrigin],
    hyp: MutPointer[Float64, MutUntrackedOrigin],
    losses: MutPointer[Float64, MutUntrackedOrigin],
    xd: MutPointer[Float32, MutUntrackedOrigin],
    yd: MutPointer[Float32, MutUntrackedOrigin],
    xs: MutPointer[Float32, MutUntrackedOrigin],
    ys: MutPointer[Float32, MutUntrackedOrigin],
    yl: IP,
) raises:
    """The host twin of the GPU binding's `x_cnn_fit_epoch_r` loop: per
    step the two row gathers, each block's forward, the head, the softmax
    (with its label check), the head's backward, each block's backward and
    one optimizer entry per parameter, each followed by its output seam."""
    var noidx = List[Int32](length=1, fill=Int32(0))
    var pnoidx = noidx.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for st in range(steps):
        var s0 = st * batch
        var m = min(batch, n - s0)
        var q = 0 if m == batch else 1
        # the two row gathers (x_cnn_res_gather: the batch, then its labels)
        for r in range(m):
            memcpy(dest=xd + r * row, src=xs + Int(rows[s0 + r]) * row, count=row)
        for r in range(m):
            memcpy(dest=yd + r, src=ys + Int(rows[s0 + r]), count=1)
        var src = a[0]
        for j in range(nb):
            var cprm = cs[q][j].copy()
            var pprm = ps[q][j].copy()
            var pool = pools[q][j]
            var ny = Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])
            var no = pool_counts(pprm)[1] if pool else ny
            var pi = IP(unsafe_from_address=bh[j][5]) if pool else pnoidx
            conv_block_forward_into(
                _at(src), _at(bh[j][0]), _at(bh[j][1]), _at(bh[j][4]), pi, cprm, pprm, pool, True,
                _at(bh[j][7]), _at(bh[j][8]),
            )
            out_f32(_at(bh[j][4]), no)
            src = bh[j][4]
        linear_forward_into(_at(src), _at(head[0]), _at(head[1]), _at(a[2]), m, flat, k)
        out_f32(_at(a[2]), m * k)
        for i in range(m):
            if Int(yl[i]) >= k:
                raise Error("x_cnn softmax: a label is not a class index")
        var loss = softmax_xent_into(_at(a[2]), yl, _at(a[3]), _at(a[4]), m, k)
        out_f32(_at(a[3]), m * k)
        out_f32(_at(a[4]), m * k)
        losses[st] = Float64(loss)
        var glast = bh[nb - 1][6] if nb > 0 else a[5]
        linear_backward_into(_at(src), _at(head[0]), _at(a[3]), _at(glast), _at(head[2]), _at(head[3]), m, flat, k)
        out_f32(_at(glast), m * flat)
        out_f32(_at(head[2]), k * flat)
        out_f32(_at(head[3]), k)
        for jj in range(nb):
            var j = nb - 1 - jj
            var cprm = cs[q][j].copy()
            var pprm = ps[q][j].copy()
            var pool = pools[q][j]
            var N = Int(cprm[CP_N]); var C = Int(cprm[CP_C]); var OC = Int(cprm[CP_OC])
            var ckk = C * Int(cprm[CP_KH]) * Int(cprm[CP_KW])
            var nx = N * C * Int(cprm[CP_H]) * Int(cprm[CP_W])
            var bsrc = bh[j - 1][4] if j > 0 else a[0]
            var dx = bh[j - 1][6] if j > 0 else 0
            var need_dx = dx != 0
            var pi = IP(unsafe_from_address=bh[j][5]) if pool else pnoidx
            var pdb = _at(bh[j][3])
            var pdx = _at(dx) if need_dx else pdb
            conv_block_backward_into(
                _at(bsrc), _at(bh[j][0]), _at(bh[j][1]), _at(bh[j][6]), pi, pdx, _at(bh[j][2]), pdb, cprm,
                pprm, pool, need_dx, True, _at(bh[j][7]), _at(bh[j][8]),
            )
            if need_dx:
                out_f32(pdx, nx)
            out_f32(_at(bh[j][2]), OC * ckk)
            out_f32(pdb, OC)
        var h = List[Float32]()
        for e in range(nh):
            h.append(Float32(hyp[st * nh + e]))
        # one optimizer entry per parameter, in the parameters' order
        for j in range(len(cnt)):
            var nj = cnt[j]
            if adam:
                adam_into(_at(hp[j]), _at(hg[j]), _at(hb[j]), h, nj)
                out_f32(_at(hp[j]), nj)
                out_f32(_at(hb[j]), 2 * nj)
            else:
                sgd_into(_at(hp[j]), _at(hg[j]), _at(hb[j]), h, nj)
                out_f32(_at(hp[j]), nj)
                out_f32(_at(hb[j]), nj)
    _ = noidx^
