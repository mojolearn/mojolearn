# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the core host binding
(`bindings/_mojolearn_core_host.mojo`): the host column of
`bindings/msel_device.mojo` (lane cpu2-l4-modelsel, 2026-10-04), same names,
signatures and refusals, over `core/msel_host.mojo`."""

from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.msel_convert import MSEL_F32, MSEL_F64, MSEL_I64, msel_is_int, msel_valid_code
from core.msel_host import (
    MselHostStore,
    host_group_fold_perm_i32,
    host_proba_column,
    host_split_table_i32,
    host_rebase_offsets_i32,
    host_scatter_rows,
    host_take_rows,
)

comptime MSEL_HOST_STORE = _Global[
    StorageType=MselHostStore, name="MojoMselHostStore", init_fn=MselHostStore.__init__
]


def msel_put_binding(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    return PythonObject(MSEL_HOST_STORE.get_or_create_ptr()[].put(Int(py=addr), Int(py=nbytes)))


def msel_alloc_binding(nbytes: PythonObject) raises -> PythonObject:
    return PythonObject(MSEL_HOST_STORE.get_or_create_ptr()[].alloc(Int(py=nbytes)))


def msel_read_binding(id: PythonObject, dst_addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    MSEL_HOST_STORE.get_or_create_ptr()[].read(Int(py=id), Int(py=dst_addr), Int(py=nbytes))
    return PythonObject(0)


def msel_free_binding(id: PythonObject) raises -> PythonObject:
    MSEL_HOST_STORE.get_or_create_ptr()[].free(Int(py=id))
    return PythonObject(0)


def msel_live_binding() raises -> PythonObject:
    return PythonObject(MSEL_HOST_STORE.get_or_create_ptr()[].live)


def msel_take_rows_binding(
    src_id: PythonObject, n_src: PythonObject, row_bytes: PythonObject,
    idx_id: PythonObject, n_idx: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    var s = Int(py=src_id)
    var ns = Int(py=n_src)
    var rb = Int(py=row_bytes)
    var ix = Int(py=idx_id)
    var ni = Int(py=n_idx)
    var d = Int(py=dst_addr)
    if ns < 0 or rb < 0 or ni < 0:
        raise Error("msel_take_rows: dimensions must be non-negative")
    var ok = True
    with GILReleased(Python()):
        ok = host_take_rows(MSEL_HOST_STORE.get_or_create_ptr()[], s, ns, rb, ix, ni, d)
    if not ok:
        raise Error("msel_take_rows: row index out of bounds")
    return PythonObject(0)


def msel_scatter_rows_binding(
    dst_id: PythonObject, n_dst: PythonObject, dst_code: PythonObject,
    src_addr: PythonObject, src_code: PythonObject, n_src: PythonObject,
    width: PythonObject, idx_id: PythonObject,
) raises -> PythonObject:
    var di = Int(py=dst_id)
    var nd = Int(py=n_dst)
    var dc = Int(py=dst_code)
    var sa = Int(py=src_addr)
    var sc = Int(py=src_code)
    var ns = Int(py=n_src)
    var w = Int(py=width)
    var ix = Int(py=idx_id)
    if nd < 0 or ns < 0 or w < 0:
        raise Error("msel_scatter_rows: dimensions must be non-negative")
    if dc != MSEL_I64 and dc != MSEL_F64:
        raise Error("msel_scatter_rows: the block is int64 or float64")
    if not msel_valid_code(sc) or (dc == MSEL_I64 and not msel_is_int(sc)):
        raise Error("msel_scatter_rows: bad source dtype code")
    if ns > 0 and w > 0 and sa == 0:
        raise Error("msel_scatter_rows: null source")
    var ok = True
    with GILReleased(Python()):
        ok = host_scatter_rows(MSEL_HOST_STORE.get_or_create_ptr()[], di, nd, dc, sa, sc, ns, w, ix)
    if not ok:
        raise Error("msel_scatter_rows: row index out of bounds")
    return PythonObject(0)


def msel_proba_column_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, k: PythonObject,
    col: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    var sa = Int(py=src_addr)
    var c = Int(py=code)
    var nn = Int(py=n)
    var kk = Int(py=k)
    var cc = Int(py=col)
    var d = Int(py=dst_addr)
    if c != MSEL_F32 and c != MSEL_F64:
        raise Error("msel_proba_column: float32 or float64 input")
    if nn < 0 or kk < 1 or cc < 0 or cc >= kk:
        raise Error("msel_proba_column: bad sizes")
    if nn > 0 and (sa == 0 or d == 0):
        raise Error("msel_proba_column: null buffer address")
    with GILReleased(Python()):
        host_proba_column(sa, c, nn, kk, cc, d)
    return PythonObject(0)


def msel_rebase_offsets_i32_binding(
    raw_addr: PythonObject, n_raw: PythonObject, lens_addr: PythonObject, parts: PythonObject,
    n_out: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    var ra = Int(py=raw_addr)
    var nr = Int(py=n_raw)
    var la = Int(py=lens_addr)
    var p = Int(py=parts)
    var no = Int(py=n_out)
    var d = Int(py=dst_addr)
    if p < 1 or nr < 1 or no < 1 or ra == 0 or la == 0 or d == 0:
        raise Error("msel_rebase_offsets_i32: bad sizes")
    var lens = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=la)
    var total = 0
    var outs = 1
    for s in range(p):
        var l = Int(lens.unsafe_load(s))
        if l < 1:
            raise Error("msel_rebase_offsets_i32: every part holds its leading offset")
        total += l
        outs += l - 1
    if total != nr or outs != no:
        raise Error("msel_rebase_offsets_i32: part lengths disagree with the sizes")
    var ok = True
    with GILReleased(Python()):
        ok = host_rebase_offsets_i32(ra, nr, la, p, no, d)
    if not ok:
        raise Error("msel_rebase_offsets_i32: a merged offset leaves int32")
    return PythonObject(0)


def msel_split_table_i32_binding(
    perm_addr: PythonObject, m: PythonObject, n_test: PythonObject, n_train: PythonObject,
    counts_addr: PythonObject, table_addr: PythonObject, sums_addr: PythonObject,
) raises -> PythonObject:
    """GroupShuffleSplit's per-group side table (the signature of
    `split_table_i32`): table[g] 1 test, 0 train, 2 neither; sums [train
    rows, test rows]."""
    var mm = Int(py=m)
    var te = Int(py=n_test)
    var tr = Int(py=n_train)
    if mm < 1 or te < 0 or tr < 0 or te + tr > mm or mm > 2147483000:
        raise Error("split_table_i32: bad sizes")
    var pa = Int(py=perm_addr)
    var ca = Int(py=counts_addr)
    var ta = Int(py=table_addr)
    var sa = Int(py=sums_addr)
    if (te + tr > 0 and pa == 0) or ca == 0 or ta == 0 or sa == 0:
        raise Error("split_table_i32: null buffer address")
    var ok = True
    with GILReleased(Python()):
        ok = host_split_table_i32(pa, mm, te, tr, ca, ta, sa)
    if not ok:
        raise Error("split_table_i32: group index out of range")
    return PythonObject(0)


def msel_group_fold_perm_i32_binding(
    counts_addr: PythonObject, m: PythonObject, n_folds: PythonObject, perm_addr: PythonObject,
    dst_addr: PythonObject, sizes_addr: PythonObject,
) raises -> PythonObject:
    """Shuffled GroupKFold (the perm branch of `group_fold_assign_i32`):
    the permuted groups in n_folds nearly equal runs."""
    var mm = Int(py=m)
    var K = Int(py=n_folds)
    if K < 1 or mm < K or mm > 2147483000:
        raise Error("group_fold_assign_i32: m >= n_folds >= 1")
    var ca = Int(py=counts_addr)
    var pa = Int(py=perm_addr)
    var da = Int(py=dst_addr)
    var sa = Int(py=sizes_addr)
    if ca == 0 or pa == 0 or da == 0 or sa == 0:
        raise Error("group_fold_assign_i32: null buffer address")
    var ok = True
    with GILReleased(Python()):
        ok = host_group_fold_perm_i32(ca, mm, K, pa, da, sa)
    if not ok:
        raise Error("group_fold_assign_i32: a permuted group is out of range")
    return PythonObject(0)
