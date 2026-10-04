# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of `core/msel_device.mojo` (lane cpu2-l4-modelsel,
2026-10-04): the same store, gather, scatter, column and offset merge over
host memory, for the core host binding (`bindings/_mojolearn_core_host.mojo`)
of a CPU-only install and as the bytes the device must match. The element
casts are the same `core/msel_convert.mojo` functions the kernels call.
"""

from std.memory import bitcast, memcpy

from core.msel_convert import (
    MSEL_I64,
    msel_itemsize,
    msel_load_f32,
    msel_load_f64_bits,
    msel_load_i64,
)

comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _F32 = MutPointer[Float32, MutAnyOrigin]


struct MselHostStore(Defaultable, Movable):
    var bufs: List[List[UInt8]]
    #: bytes held by each id; -1 for a freed id
    var nbytes: List[Int]
    var released: List[Int]
    var live: Int

    def __init__(out self):
        self.bufs = List[List[UInt8]]()
        self.nbytes = List[Int]()
        self.released = List[Int]()
        self.live = 0

    def check(self, id: Int, need: Int) raises:
        if id < 0 or id >= len(self.nbytes) or self.nbytes[id] < 0:
            raise Error(String("msel store: id ", id, " is not live"))
        if self.nbytes[id] < need:
            raise Error(String("msel store: id ", id, " holds ", self.nbytes[id], " bytes, ", need, " read"))

    def _insert(mut self, var buf: List[UInt8], n: Int) -> Int:
        var id: Int
        if len(self.released) > 0:
            id = self.released.pop()
            self.bufs[id] = buf^
            self.nbytes[id] = n
        else:
            self.bufs.append(buf^)
            self.nbytes.append(n)
            id = len(self.nbytes) - 1
        self.live += 1
        return id

    def put(mut self, src_addr: Int, n: Int) raises -> Int:
        if n < 1 or src_addr == 0:
            raise Error("msel store: put needs at least one byte at a non-null address")
        var buf = List[UInt8](length=n, fill=0)
        memcpy(dest=buf.unsafe_ptr(), src=_U8(unsafe_from_address=src_addr), count=n)
        return self._insert(buf^, n)

    def alloc(mut self, n: Int) raises -> Int:
        if n < 1:
            raise Error("msel store: alloc needs at least one byte")
        return self._insert(List[UInt8](length=n, fill=0), n)

    def ptr(mut self, id: Int, need: Int) raises -> _U8:
        self.check(id, need)
        return _U8(unsafe_from_address=Int(self.bufs[id].unsafe_ptr()))

    def read(mut self, id: Int, dst_addr: Int, n: Int) raises:
        self.check(id, n)
        if n > 0:
            if dst_addr == 0:
                raise Error("msel store: read into a null address")
            memcpy(dest=_U8(unsafe_from_address=dst_addr), src=self.ptr(id, n), count=n)

    def free(mut self, id: Int) raises:
        self.check(id, 0)
        self.bufs[id] = List[UInt8]()
        self.nbytes[id] = -1
        self.released.append(id)
        self.live -= 1


def host_take_rows(
    mut store: MselHostStore, src_id: Int, n_src: Int, row_bytes: Int,
    idx_id: Int, n_idx: Int, dst_addr: Int,
) raises -> Bool:
    """`device_take_rows` on the host: every index tested before a byte of
    `dst` is written."""
    if n_idx < 1 or row_bytes < 1:
        return True
    var src = store.ptr(src_id, n_src * row_bytes)
    var idx = store.ptr(idx_id, n_idx * 8).bitcast[Int64]()
    for r in range(n_idx):
        var s = Int(idx.unsafe_load(r))
        if s < 0 or s >= n_src:
            return False
    if dst_addr == 0:
        raise Error("msel take_rows: null destination")
    var dst = _U8(unsafe_from_address=dst_addr)
    for r in range(n_idx):
        var s = Int(idx.unsafe_load(r))
        memcpy(dest=dst + r * row_bytes, src=src + s * row_bytes, count=row_bytes)
    return True


def host_scatter_rows(
    mut store: MselHostStore, dst_id: Int, n_dst: Int, dst_code: Int,
    src_addr: Int, src_code: Int, n_src: Int, width: Int, idx_id: Int,
) raises -> Bool:
    if n_src < 1 or width < 1:
        return True
    var dst = store.ptr(dst_id, n_dst * width * 8).bitcast[UInt64]()
    var idx = store.ptr(idx_id, n_src * 8).bitcast[Int64]()
    var src = _U8(unsafe_from_address=src_addr)
    var to_int = dst_code == MSEL_I64
    var ok = True
    for r in range(n_src):
        var d = Int(idx.unsafe_load(r))
        if d < 0 or d >= n_dst:
            ok = False
            continue
        for c in range(width):
            var t = r * width + c
            var word: UInt64
            if to_int:
                word = bitcast[DType.uint64](msel_load_i64(src, src_code, t))
            else:
                word = msel_load_f64_bits(src, src_code, t)
            dst.unsafe_store(d * width + c, word)
    return ok


def host_proba_column(src_addr: Int, code: Int, n: Int, k: Int, col: Int, dst_addr: Int) raises:
    var src = _U8(unsafe_from_address=src_addr)
    var dst = _F32(unsafe_from_address=dst_addr)
    for i in range(n):
        dst.unsafe_store(i, msel_load_f32(src, code, i * k + col))


def host_rebase_offsets_i32(
    raw_addr: Int, n_raw: Int, lens_addr: Int, parts: Int, n_out: Int, dst_addr: Int,
) raises -> Bool:
    if n_out < 1 or parts < 1 or n_raw < 1:
        return False
    var raw = _I32(unsafe_from_address=raw_addr)
    var lens = _I64(unsafe_from_address=lens_addr)
    var dst = _I32(unsafe_from_address=dst_addr)
    dst.unsafe_store(0, Int32(0))
    var o = 1
    var base = Int64(0)
    var start = 0
    for s in range(parts):
        var len_s = Int(lens.unsafe_load(s))
        for j in range(1, len_s):
            var v = base + Int64(raw.unsafe_load(start + j))
            if v < Int64(-2147483648) or v > Int64(2147483647) or o >= n_out:
                return False
            dst.unsafe_store(o, Int32(v))
            o += 1
        base += Int64(raw.unsafe_load(start + len_s - 1))
        start += len_s
    return o == n_out


def host_split_table_i32(
    perm_addr: Int, m: Int, n_test: Int, n_train: Int, counts_addr: Int, table_addr: Int, sums_addr: Int,
) raises -> Bool:
    var pp = _I64(unsafe_from_address=perm_addr)
    var cp = _I64(unsafe_from_address=counts_addr)
    var tp = _I32(unsafe_from_address=table_addr)
    var sp = _I64(unsafe_from_address=sums_addr)
    for i in range(n_test + n_train):
        var g = Int(pp.unsafe_load(i))
        if g < 0 or g >= m:
            return False
    var c_tr = Int32(0)
    var c_te = Int32(0)
    for g in range(m):
        tp.unsafe_store(g, Int32(2))
    for i in range(n_test + n_train):
        var g = Int(pp.unsafe_load(i))
        if i < n_test:
            tp.unsafe_store(g, Int32(1))
            c_te += Int32(cp.unsafe_load(g))
        else:
            tp.unsafe_store(g, Int32(0))
            c_tr += Int32(cp.unsafe_load(g))
    sp.unsafe_store(0, Int64(c_tr))
    sp.unsafe_store(1, Int64(c_te))
    return True


def host_group_fold_perm_i32(
    counts_addr: Int, m: Int, n_folds: Int, perm_addr: Int, dst_addr: Int, sizes_addr: Int,
) raises -> Bool:
    var pp = _I64(unsafe_from_address=perm_addr)
    var cp = _I64(unsafe_from_address=counts_addr)
    var dp = _I32(unsafe_from_address=dst_addr)
    var sp = _I64(unsafe_from_address=sizes_addr)
    for j in range(m):
        var g = Int(pp.unsafe_load(j))
        if g < 0 or g >= m:
            return False
    var sizes = List[Int32](length=n_folds, fill=0)
    var start = 0
    for f in range(n_folds):
        var size = m // n_folds + (1 if f < m % n_folds else 0)
        for j in range(start, start + size):
            var g = Int(pp.unsafe_load(j))
            dp.unsafe_store(g, Int32(f))
            sizes[f] += Int32(cp.unsafe_load(g))
        start += size
    for f in range(n_folds):
        sp.unsafe_store(f, Int64(sizes[f]))
    return True
