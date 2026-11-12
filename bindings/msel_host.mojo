# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-selection helpers of the core host binding
(`bindings/_mojolearn_core_host.mojo`): the host column of
`bindings/msel_device.mojo` (lane cpu2-l4-modelsel, 2026-10-04), same names,
signatures and refusals, over `core/msel_host.mojo`."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.msel_host import host_take_rows
# lane calltable-msel: the store slot and the scatter, column, split-table
# and group-fold bodies are the runs the model-selection helper table
# publishes (core/msel_helpers_host.mojo); these exports unpack and call them.
from core.msel_helpers_host import (
    MSEL_HOST_STORE, msel_scatter_rows_host_run, msel_proba_column_host_run, msel_split_table_host_run,
    msel_group_fold_perm_host_run,
)


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


def msel_free_bytes_binding() raises -> PythonObject:
    """The host column has no device memory to bound concurrent
    cross-validation units by (bindings/msel_cv_door.mojo msel_cv_units): -1."""
    return PythonObject(-1)


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
    """`core/msel_helpers_host.mojo::msel_scatter_rows_host_run`."""
    var di = Int(py=dst_id)
    var nd = Int(py=n_dst)
    var dc = Int(py=dst_code)
    var sa = Int(py=src_addr)
    var sc = Int(py=src_code)
    var ns = Int(py=n_src)
    var w = Int(py=width)
    var ix = Int(py=idx_id)
    var status = 0
    with GILReleased(Python()):
        status = msel_scatter_rows_host_run(di, nd, dc, sa, sc, ns, w, ix)
    return PythonObject(status)


def msel_proba_column_binding(
    src_addr: PythonObject, code: PythonObject, n: PythonObject, k: PythonObject,
    col: PythonObject, dst_addr: PythonObject,
) raises -> PythonObject:
    """`core/msel_helpers_host.mojo::msel_proba_column_host_run`."""
    var sa = Int(py=src_addr)
    var c = Int(py=code)
    var nn = Int(py=n)
    var kk = Int(py=k)
    var cc = Int(py=col)
    var d = Int(py=dst_addr)
    var status = 0
    with GILReleased(Python()):
        status = msel_proba_column_host_run(sa, c, nn, kk, cc, d)
    return PythonObject(status)


def msel_split_table_i32_binding(
    perm_addr: PythonObject, m: PythonObject, n_test: PythonObject, n_train: PythonObject,
    counts_addr: PythonObject, table_addr: PythonObject, sums_addr: PythonObject,
) raises -> PythonObject:
    """GroupShuffleSplit's per-group side table (the signature of
    `split_table_i32`): `core/msel_helpers_host.mojo::msel_split_table_host_run`."""
    var mm = Int(py=m)
    var te = Int(py=n_test)
    var tr = Int(py=n_train)
    var pa = Int(py=perm_addr)
    var ca = Int(py=counts_addr)
    var ta = Int(py=table_addr)
    var sa = Int(py=sums_addr)
    var status = 0
    with GILReleased(Python()):
        status = msel_split_table_host_run(pa, mm, te, tr, ca, ta, sa)
    return PythonObject(status)


def msel_group_fold_perm_i32_binding(
    counts_addr: PythonObject, m: PythonObject, n_folds: PythonObject, perm_addr: PythonObject,
    dst_addr: PythonObject, sizes_addr: PythonObject,
) raises -> PythonObject:
    """Shuffled GroupKFold (the perm branch of `group_fold_assign_i32`):
    `core/msel_helpers_host.mojo::msel_group_fold_perm_host_run`."""
    var mm = Int(py=m)
    var K = Int(py=n_folds)
    var ca = Int(py=counts_addr)
    var pa = Int(py=perm_addr)
    var da = Int(py=dst_addr)
    var sa = Int(py=sizes_addr)
    var status = 0
    with GILReleased(Python()):
        status = msel_group_fold_perm_host_run(ca, mm, K, pa, da, sa)
    return PythonObject(status)
