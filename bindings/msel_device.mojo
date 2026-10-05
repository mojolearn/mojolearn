# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the GPU base binding (`bindings/_mojolearn.mojo`)
on the device (lane cpu2-l4-modelsel, 2026-10-04; `core/msel_device.mojo`).

The core host binding exports the same names and signatures from
`bindings/msel_host.mojo` (the host column, and the CPU-only install's route).
Every IDENTICAL and FAST GPU build takes these; there is no host fallback in
this binding.

    msel_put(addr, nbytes) -> id          resident copy of host bytes
    msel_alloc(nbytes) -> id              resident zero bytes
    msel_read(id, dst_addr, nbytes)       download
    msel_free(id) / msel_live() -> count
    msel_take_rows(src_id, n_src, row_bytes, idx_id, n_idx, dst_addr)
    msel_scatter_rows(dst_id, n_dst, dst_code, src_addr, src_code, n_src, width, idx_id)
    msel_proba_column(src_addr, code, n, k, col, dst_addr)
    msel_rebase_offsets_i32(raw_addr, n_raw, lens_addr, parts, n_out, dst_addr)
"""

from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.msel_convert import MSEL_F32, MSEL_F64, MSEL_I64, msel_is_int, msel_valid_code
from core.msel_device import (
    MselStore,
    device_group_fold_perm_i32,
    device_proba_column,
    device_split_table_i32,
    device_rebase_offsets_i32,
    device_scatter_rows,
    device_take_rows,
)
from core.neural_context import process_ctx

#: The base binding's process-lifetime context slot
#: (`bindings/_mojolearn.mojo::_DEVCTX_SLOT`): the same name, so the same
#: DeviceContext, whose buffers the store holds.
comptime _MSEL_SLOT = (
    "MojoCoreContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoCoreContextFast"
)
comptime _MSEL_STORE_NAME = (
    "MojoMselStoreIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoMselStoreFast"
)
comptime MSEL_STORE = _Global[StorageType=MselStore, name=_MSEL_STORE_NAME, init_fn=MselStore.__init__]


def msel_put_binding(addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    var a = Int(py=addr)
    var n = Int(py=nbytes)
    var ctx = process_ctx[_MSEL_SLOT]()
    var id: Int
    with GILReleased(Python()):
        id = MSEL_STORE.get_or_create_ptr()[].put(ctx, a, n)
    return PythonObject(id)


def msel_alloc_binding(nbytes: PythonObject) raises -> PythonObject:
    var n = Int(py=nbytes)
    var ctx = process_ctx[_MSEL_SLOT]()
    var id: Int
    with GILReleased(Python()):
        id = MSEL_STORE.get_or_create_ptr()[].alloc(ctx, n)
    return PythonObject(id)


def msel_read_binding(id: PythonObject, dst_addr: PythonObject, nbytes: PythonObject) raises -> PythonObject:
    var i = Int(py=id)
    var d = Int(py=dst_addr)
    var n = Int(py=nbytes)
    var ctx = process_ctx[_MSEL_SLOT]()
    with GILReleased(Python()):
        MSEL_STORE.get_or_create_ptr()[].read(ctx, i, d, n)
    return PythonObject(0)


def msel_free_binding(id: PythonObject) raises -> PythonObject:
    var i = Int(py=id)
    var ctx = process_ctx[_MSEL_SLOT]()
    with GILReleased(Python()):
        MSEL_STORE.get_or_create_ptr()[].free(ctx, i)
    return PythonObject(0)


def msel_live_binding() raises -> PythonObject:
    return PythonObject(MSEL_STORE.get_or_create_ptr()[].live)


def msel_take_rows_binding(
    src_id: PythonObject, n_src: PythonObject, row_bytes: PythonObject,
    idx_id: PythonObject, n_idx: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """Rows idx of the resident source into host memory (a fold's rows)."""
    var s = Int(py=src_id)
    var ns = Int(py=n_src)
    var rb = Int(py=row_bytes)
    var ix = Int(py=idx_id)
    var ni = Int(py=n_idx)
    var d = Int(py=dst_addr)
    if ns < 0 or rb < 0 or ni < 0:
        raise Error("msel_take_rows: dimensions must be non-negative")
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_take_rows(ctx, MSEL_STORE.get_or_create_ptr()[], s, ns, rb, ix, ni, d)
    if not ok:
        raise Error("msel_take_rows: row index out of bounds")
    return PythonObject(0)


def msel_scatter_rows_binding(
    dst_id: PythonObject, n_dst: PythonObject, dst_code: PythonObject,
    src_addr: PythonObject, src_code: PythonObject, n_src: PythonObject,
    width: PythonObject, idx_id: PythonObject,
) raises -> PythonObject:
    """Rows of a host buffer cast into rows idx of a resident int64 or
    float64 block (cross_val_predict's placement)."""
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
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_scatter_rows(ctx, MSEL_STORE.get_or_create_ptr()[], di, nd, dc, sa, sc, ns, w, ix)
    if not ok:
        raise Error("msel_scatter_rows: row index out of bounds")
    return PythonObject(0)


def msel_proba_column_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, k: PythonObject,
    col: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """Column `col` of an n x k float32/float64 buffer as float32."""
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
    var ctx = process_ctx[_MSEL_SLOT]()
    with GILReleased(Python()):
        device_proba_column(ctx, sa, c, nn, kk, cc, d)
    return PythonObject(0)


def msel_rebase_offsets_i32_binding(
    raw_addr: PythonObject, n_raw: PythonObject, lens_addr: PythonObject, parts: PythonObject,
    n_out: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """The parallel forest's merged int32 tree offsets."""
    var ra = Int(py=raw_addr)
    var nr = Int(py=n_raw)
    var la = Int(py=lens_addr)
    var p = Int(py=parts)
    var no = Int(py=n_out)
    var d = Int(py=dst_addr)
    if p < 1 or nr < 1 or no < 1 or ra == 0 or la == 0 or d == 0:
        raise Error("msel_rebase_offsets_i32: bad sizes")
    # the part lengths are k scalars (one per device shard): checked here
    var lens = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=la)
    var total = 0
    var outs = 1
    for s in range(p):  # small-loop(p: one length per device shard): shard bookkeeping, not data
        var l = Int(lens.unsafe_load(s))
        if l < 1:
            raise Error("msel_rebase_offsets_i32: every part holds its leading offset")
        total += l
        outs += l - 1
    if total != nr or outs != no:
        raise Error("msel_rebase_offsets_i32: part lengths disagree with the sizes")
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_rebase_offsets_i32(ctx, ra, nr, la, p, no, d)
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
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_split_table_i32(ctx, pa, mm, te, tr, ca, ta, sa)
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
    var ctx = process_ctx[_MSEL_SLOT]()
    var ok = True
    with GILReleased(Python()):
        ok = device_group_fold_perm_i32(ctx, ca, mm, K, pa, da, sa)
    if not ok:
        raise Error("group_fold_assign_i32: a permuted group is out of range")
    return PythonObject(0)
