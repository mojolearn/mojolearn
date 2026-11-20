# SPDX-License-Identifier: Apache-2.0
"""Native DeviceRows descriptor validation and the existing parallel gather."""
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_NONFINITE, ARG_OUT_OF_RANGE


def admitted_rows_base(rank: Int, f32: Bool, c_order: Bool) raises:
    if rank != 2 or not f32 or not c_order:
        raise Error("TypeError: DeviceRows requires a float32 C-order 2-D base")


def admitted_rows_indices(rank: Int, kind: String, item: Int, c_order: Bool) raises:
    if rank != 1 or kind != "i" or item != 8 or not c_order:
        raise Error("TypeError: DeviceRows requires C-contiguous int64 row indices")


def admitted_rows_shape(rows: Int, cols: Int, estimator: String) raises:
    if rows == 0:
        raise Error("Found array with 0 sample(s) (shape=(0, " + String(cols) + ")) while a minimum of 1 is required by " + estimator + ".")
    if cols == 0:
        raise Error("Found array with 0 feature(s) (shape=(" + String(rows) + ", 0)) while a minimum of 1 is required by " + estimator + ".")


def admitted_rows_store(active: Bool, stored_rows: Int, stored_cols: Int, stored_base_rows: Int,
    rows: Int, cols: Int, base_rows: Int) raises:
    if active and (stored_rows != rows or stored_cols != cols or stored_base_rows != base_rows):
        raise Error("TypeError: DeviceRows store entries differ from the fold's base and indices")


def admitted_rows_run(base: Int, base_rows: Int, cols: Int, indices: Int,
    rows: Int, destination: Int, finite: Bool, estimator: String) raises:
    var status = admitted_gather_rows_f32(base, base_rows, cols, indices, rows, destination)
    if status == ARG_OUT_OF_RANGE:
        raise Error("IndexError: DeviceRows row index out of range for a base of " + String(base_rows) + " rows")
    if status == ARG_NONFINITE and finite:
        raise Error("Input X contains NaN or infinity (" + estimator + ").")
