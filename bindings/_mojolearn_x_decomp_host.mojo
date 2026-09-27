# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_decomp` (lane/algos-decomp, 2026-09-27).
HOST ONLY: the entry points of x_decomp/api.mojo on the host executor
(x_decomp/host.mojo), under the GPU binding's names and address contract,
plus the family read-backs."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_decomp.api import (
    chol_py, colsum_py, eigh_py, ew_py, gemm_py, lu_py, lu_solve_py, numeric_mode_py, rand_py,
    rowsum_py, sqdist_py, vendor_py,
)
from x_decomp.host import HostExec, X_DECOMP_HOST_SABOTAGE


def x_decomp_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_decomp_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_decomp_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_decomp host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_decomp_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_DECOMP_HOST_SABOTAGE)


@export
def PyInit__mojolearn_x_decomp_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_decomp_host")
        m.def_function[x_decomp_host_numeric_mode_binding]("x_decomp_host_numeric_mode")
        m.def_function[x_decomp_host_vendor_binding]("x_decomp_host_vendor")
        m.def_function[x_decomp_host_column_binding]("x_decomp_host_column")
        m.def_function[x_decomp_host_sabotage_binding]("x_decomp_host_sabotage")
        m.def_function[gemm_py[HostExec]]("x_decomp_gemm")
        m.def_function[ew_py[HostExec]]("x_decomp_ew")
        m.def_function[colsum_py[HostExec]]("x_decomp_colsum")
        m.def_function[rowsum_py[HostExec]]("x_decomp_rowsum")
        m.def_function[sqdist_py[HostExec]]("x_decomp_sqdist")
        m.def_function[rand_py[HostExec]]("x_decomp_rand")
        m.def_function[lu_py[HostExec]]("x_decomp_lu")
        m.def_function[lu_solve_py[HostExec]]("x_decomp_lu_solve")
        m.def_function[chol_py[HostExec]]("x_decomp_chol")
        m.def_function[eigh_py[HostExec]]("x_decomp_eigh")
        m.def_function[numeric_mode_py]("x_decomp_numeric_mode")
        m.def_function[vendor_py[HostExec]]("x_decomp_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_decomp_host: ", e))
