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


