# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Data movement and orders the decomp lane's Python did one element at a
time (lane/py-decomp-nbrs, 2026-09-28): gathers and scatters by an int32
index list, the nonzero upper triangle of a square matrix, and the
(value, index) and (x, y, index) orders of Python's stable `sorted`.
Exact copies and compares only: no float arithmetic, so nothing here can
move a bit (except `dsum_sq`, software binary64 in a fixed chunked order
shared with the device form, lane cpu3-python). Host code, compiled into
both x_decomp bindings; the per-element helpers are also the device
kernels' bodies (x_decomp/select_dev.mojo)."""
from std.memory import bitcast
from std.builtin.sort import sort
from std.math import trunc
from std.sys.compile import is_defined

from checks.soft_f64 import SF64_ZERO, sf64_add, sf64_from_f32, sf64_mul

from x_decomp.cells import F32Ptr, I32Ptr


def f32_key(v: Float32) raises -> UInt32:
    """v's monotone uint32 image in Python's float order (-0.0 is +0.0,
    as Python's `==` makes them). A NaN has no place in that order and is
    refused."""
    if v != v:
        raise Error("x_decomp: a NaN has no order (refused)")
    var w = v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    return ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)


def gather(src: F32Ptr, idx: I32Ptr, m: Int, dst: F32Ptr):
    for a in range(m):
        dst.unsafe_store(a, src.unsafe_load(Int(idx.unsafe_load(a))))


def scatter(dst: F32Ptr, idx: I32Ptr, m: Int, src: F32Ptr):
    for a in range(m):
        dst.unsafe_store(Int(idx.unsafe_load(a)), src.unsafe_load(a))


def triu_nonzero(dis: F32Ptr, n: Int, pos: I32Ptr, mir: I32Ptr) -> Int:
    """`[i * n + j for i in range(n) for j in range(i + 1, n) if dis[i * n + j] != 0]`
    into `pos` (and each one's mirror `j * n + i` into `mir`); returns the count.
    `!= 0` is Python's: NaN is kept, -0.0 is dropped."""
    var m = 0
    for i in range(n):
        for j in range(i + 1, n):
            var v = dis.unsafe_load(i * n + j)
            if v != Float32(0):
                pos.unsafe_store(m, Int32(i * n + j))
                mir.unsafe_store(m, Int32(j * n + i))
                m += 1
    return m


def argsort_f32(x: F32Ptr, m: Int, dst: I32Ptr) raises:
    """`sorted(range(m), key=lambda i: x[i])` (stable: ties in index order)."""
    var keys = List[UInt64](capacity=m)
    for i in range(m):
        keys.append((UInt64(f32_key(x.unsafe_load(i))) << 32) | UInt64(i))
    sort(keys)
    for a in range(m):
        dst.unsafe_store(a, Int32(Int(keys[a] & UInt64(0xFFFFFFFF))))


def iso_order(x: F32Ptr, y: F32Ptr, xorder: I32Ptr, m: Int, dst: I32Ptr) raises:
    """`sorted(range(m), key=lambda i: (x[i], y[i]))` given `xorder`, the
    stable order by x alone: each run of equal x is re-sorted by (y, index)."""
    var a = 0
    var keys = List[UInt64]()
    while a < m:
        var xa = x.unsafe_load(Int(xorder.unsafe_load(a)))
        var b = a + 1
        while b < m and x.unsafe_load(Int(xorder.unsafe_load(b))) == xa:
            b += 1
        if b - a == 1:
            dst.unsafe_store(a, xorder.unsafe_load(a))
        else:
            keys.clear()
            for t in range(a, b):
                var i = Int(xorder.unsafe_load(t))
                keys.append((UInt64(f32_key(y.unsafe_load(i))) << 32) | UInt64(i))
            sort(keys)
            for t in range(b - a):
                dst.unsafe_store(a + t, Int32(Int(keys[t] & UInt64(0xFFFFFFFF))))
        a = b
    for t in range(m):
        _ = f32_key(y.unsafe_load(t))          # a NaN y is refused as the full sort would meet it


# ---- lane apple-fast-py2mojo-decomp (2026-10-03): the rest of the decomp
# module's data path that Python ran one element at a time. Exact copies,
# compares and IEEE double sums in Python's order: nothing here moves a bit.

#: -D MOJOLEARN_PY2MOJO_decomp_OFF restores the Python data path (the A/B
#: arm); Python reads it through `x_decomp_py2mojo`.
comptime PY2MOJO_DECOMP = not is_defined["MOJOLEARN_PY2MOJO_decomp_OFF"]()

#: `move` ops. TAKE_ROWS: dst (m x c) row a = src row Int(idx[a * ist + ioff]) + a2
#: (a1 = c). TRANSPOSE: dst (c x r) = src (r x c)^T (a1 = r, a2 = c).
#: PLACE_COLS: src (r x c) into columns a3 .. a3 + c of dst (r x a2) (a1 = c).
#: FILL0: dst[a1 + t * a2] = 0 for t < count (src unused).
comptime MOVE_TAKE_ROWS = 0
comptime MOVE_TRANSPOSE = 1
comptime MOVE_PLACE_COLS = 2
comptime MOVE_FILL0 = 3
#: TAKE_COLS (lane fam2-decomp, 2026-10-04): dst (r x w) column b = src (r x c)
#: column Int(idx[b]) (a1 = w, a2 = c; idx holds w exact floats).
comptime MOVE_TAKE_COLS = 4
#: the highest op (the entries' range check)
comptime MOVE_LAST = MOVE_TAKE_COLS


@always_inline
def move_src(op: Int, t: Int, idx: F32Ptr, a1: Int, a2: Int, ist: Int, ioff: Int) -> Int:
    """The source index of move element t (-1: FILL0 stores a zero)."""
    if op == MOVE_TAKE_ROWS:
        var row = Int(idx.unsafe_load(t // a1 * ist + ioff)) + a2
        return row * a1 + t % a1
    if op == MOVE_TRANSPOSE:
        return (t % a1) * a2 + t // a1
    if op == MOVE_PLACE_COLS:
        return t
    if op == MOVE_TAKE_COLS:
        return (t // a1) * a2 + Int(idx.unsafe_load(t % a1))
    return -1


@always_inline
def move_dst(op: Int, t: Int, a1: Int, a2: Int, a3: Int) -> Int:
    """The destination index of move element t."""
    if op == MOVE_PLACE_COLS:
        return (t // a1) * a2 + a3 + t % a1
    if op == MOVE_FILL0:
        return a1 + t * a2
    return t


def move_host(
    op: Int, src: F32Ptr, idx: F32Ptr, dst: F32Ptr, count: Int, a1: Int, a2: Int, a3: Int, ist: Int, ioff: Int,
    nsrc: Int, ndst: Int,
) raises:
    """`move` on host buffers, every index checked against the buffer lengths."""
    if op < MOVE_TAKE_ROWS or op > MOVE_LAST:
        raise Error("x_decomp: unknown move op")
    if op != MOVE_FILL0 and a1 <= 0 and count > 0:
        raise Error("x_decomp: move needs a positive width")
    for t in range(count):
        if op == MOVE_TAKE_ROWS:
            var v = idx.unsafe_load(t // a1 * ist + ioff)
            if not (v >= Float32(0) and v < Float32(16777216)):
                raise Error("x_decomp: take_rows index out of range")
        if op == MOVE_TAKE_COLS:
            var vc = idx.unsafe_load(t % a1)
            if not (vc >= Float32(0) and vc < Float32(a2)):
                raise Error("x_decomp: take_cols index out of range")
        var s = move_src(op, t, idx, a1, a2, ist, ioff)
        var d = move_dst(op, t, a1, a2, a3)
        if s >= nsrc or d < 0 or d >= ndst:
            raise Error("x_decomp: move index out of range")
        if s >= 0:
            dst.unsafe_store(d, src.unsafe_load(s))
        else:
            dst.unsafe_store(d, Float32(0))


#: values per first-level partial of `dsum_sq` (and per later fold)
comptime DSUM_CHUNK = 256


@always_inline
def dsum_ld(f: F32Ptr, at: Int) -> UInt64:
    """A binary64 word stored as two float32 words, low first."""
    return (UInt64(bitcast[DType.uint32](f.unsafe_load(at + 1))) << UInt64(32)) | UInt64(
        bitcast[DType.uint32](f.unsafe_load(at))
    )


@always_inline
def dsum_st(f: F32Ptr, at: Int, v: UInt64):
    f.unsafe_store(at, bitcast[DType.float32](UInt32(v & UInt64(0xFFFFFFFF))))
    f.unsafe_store(at + 1, bitcast[DType.float32](UInt32(v >> UInt64(32))))


@always_inline
def dsum_sq_chunk(x: F32Ptr, lo: Int, hi: Int) -> UInt64:
    """Software-binary64 sum, ascending, of x[lo:hi]^2 (each square of a
    float32 is exact in binary64)."""
    var acc = SF64_ZERO
    for i in range(lo, hi):
        var v = sf64_from_f32(x.unsafe_load(i))
        acc = sf64_add(acc, sf64_mul(v, v))
    return acc


@always_inline
def dsum_fold_chunk(p: F32Ptr, lo: Int, hi: Int) -> UInt64:
    """Software-binary64 sum, ascending, of the binary64 words p[lo:hi]."""
    var acc = SF64_ZERO
    for i in range(lo, hi):
        acc = sf64_add(acc, dsum_ld(p, 2 * i))
    return acc


def dsum_sq_host(x: F32Ptr, m: Int) -> UInt64:
    """The host column of the device `dsum_sq` (lane cpu3-python): the sum
    of squares in software binary64, DSUM_CHUNK values per partial
    (ascending), then the partials folded DSUM_CHUNK at a time, ascending,
    until one is left. Every vendor and this column add the same pairs."""
    if m <= 0:
        return SF64_ZERO
    var nb = (m + DSUM_CHUNK - 1) // DSUM_CHUNK
    var cur = List[Float32](length=2 * nb, fill=Float32(0))
    var pc = F32Ptr(unsafe_from_address=Int(cur.unsafe_ptr()))
    for b in range(nb):
        dsum_st(pc, 2 * b, dsum_sq_chunk(x, b * DSUM_CHUNK, min((b + 1) * DSUM_CHUNK, m)))
    while nb > 1:
        var nn = (nb + DSUM_CHUNK - 1) // DSUM_CHUNK
        var nxt = List[Float32](length=2 * nn, fill=Float32(0))
        var pn = F32Ptr(unsafe_from_address=Int(nxt.unsafe_ptr()))
        for b in range(nn):
            dsum_st(pn, 2 * b, dsum_fold_chunk(pc, b * DSUM_CHUNK, min((b + 1) * DSUM_CHUNK, nb)))
        cur = nxt^
        pc = F32Ptr(unsafe_from_address=Int(cur.unsafe_ptr()))
        nb = nn
    var r = dsum_ld(pc, 0)
    _ = cur^
    return r


@always_inline
def order_key(v: Float32, neg: Bool) -> UInt32:
    """`f32_key(-v if neg else v)` for a non-NaN v, without the refusal
    (the device form's callers refuse NaN before they launch)."""
    var w = -v if neg else v
    if w == Float32(0):
        w = Float32(0)
    var ub = bitcast[DType.uint32](w)
    return ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == 1 else ub | UInt32(0x80000000)


def order_f(x: F32Ptr, m: Int, dst: F32Ptr) raises:
    """`argsort_f32` with the indices written as exact float32 (m < 2^24)."""
    if m >= 16777216:
        raise Error("x_decomp: order_f exceeds the float32 index bound")
    var o = List[Int32](length=m, fill=Int32(0))
    argsort_f32(x, m, I32Ptr(unsafe_from_address=Int(o.unsafe_ptr())))
    for a in range(m):
        dst.unsafe_store(a, Float32(Int(o[a])))


def select_smallest(x: F32Ptr, m: Int, h: Int, sel: F32Ptr, mask: I32Ptr) raises:
    """`sorted(sorted(range(m), key=lambda i: (x[i], i))[:h])` into sel (as
    exact floats) and the membership of each row into mask (1 / 0)."""
    if m >= 16777216 or h < 0 or h > m:
        raise Error("x_decomp: select_smallest out of range")
    var o = List[Int32](length=m, fill=Int32(0))
    argsort_f32(x, m, I32Ptr(unsafe_from_address=Int(o.unsafe_ptr())))
    for i in range(m):
        mask.unsafe_store(i, Int32(0))
    for a in range(h):
        mask.unsafe_store(Int(o[a]), Int32(1))
    var c = 0
    for i in range(m):
        if mask.unsafe_load(i) != 0:
            sel.unsafe_store(c, Float32(i))
            c += 1


def argmin_all(x: F32Ptr, m: Int, dst: F32Ptr) -> Int:
    """`dmin = min(x); [i for i, v in enumerate(x) if v == dmin]` (as exact
    floats); returns their count. Python's `min` keeps the first of equal
    values, and `==` makes -0.0 and +0.0 one value."""
    if m <= 0:
        return 0
    # lane cpu3-python: a NaN anywhere gives no positions, as the device
    # form's min (a NaN anywhere is the min) compares equal to nothing
    for i in range(m):
        var q = x.unsafe_load(i)
        if q != q:
            return 0
    var dmin = x.unsafe_load(0)
    for i in range(1, m):
        var v = x.unsafe_load(i)
        if v < dmin:
            dmin = v
    var c = 0
    for i in range(m):
        if x.unsafe_load(i) == dmin:
            dst.unsafe_store(c, Float32(i))
            c += 1
    return c


def sign_labels(x: F32Ptr, m: Int, dst: I32Ptr):
    """`[1 if v >= 0 else -1 for v in x]` (NaN is -1)."""
    for i in range(m):
        dst.unsafe_store(i, Int32(1) if x.unsafe_load(i) >= Float32(0) else Int32(-1))


comptime F64Ptr = MutPointer[Float64, MutAnyOrigin]
comptime U32Ptr = MutPointer[UInt32, MutAnyOrigin]


def accuracy(y: F64Ptr, pred: I32Ptr, w: F64Ptr, weighted: Bool, m: Int) raises -> Tuple[Float64, Float64]:
    """accuracy_score's two sums: (matches, m) unweighted; (the matching
    weights added in order, every weight added in order) weighted. A label
    is `int(y)`: truncated, NaN and inf refused as Python's `int` refuses them."""
    var hit = Float64(0)
    var tw = Float64(0)
    var cnt = 0
    for i in range(m):
        var v = y.unsafe_load(i)
        if v - v != Float64(0):
            raise Error("x_decomp: a label that is not a finite number")
        var same = Float64(Int(pred.unsafe_load(i))) == trunc(v)
        if weighted:
            var wi = w.unsafe_load(i)
            tw += wi
            if same:
                hit += wi
        elif same:
            cnt += 1
    if not weighted:
        return (Float64(cnt), Float64(m))
    return (hit, tw)


def topn_desc(x: F32Ptr, m: Int, skip: F32Ptr, has_skip: Bool, n: Int, dst: I32Ptr) raises -> Int:
    """`sorted((i for i in range(m) if not skip[i]), key=lambda i: (-x[i], i))[:n]`
    (skip[i] = a nonzero entry of the skip row); returns the count written."""
    var keys = List[UInt64](capacity=m)
    for i in range(m):
        if has_skip and skip.unsafe_load(i) != Float32(0):
            continue
        keys.append((UInt64(f32_key(-x.unsafe_load(i))) << 32) | UInt64(i))
    sort(keys)
    var c = min(n, len(keys))
    for a in range(c):
        dst.unsafe_store(a, Int32(Int(keys[a] & UInt64(0xFFFFFFFF))))
    return c
