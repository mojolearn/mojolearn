# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host column of x_neighbors/kapprox_dev.mojo: the same items in a plain
loop, so the CPU binding exports the same names over the same address
contract. HOST ONLY. The Python layer never takes the kapprox device path on
this binding (`x_neighbors_sparse_rp_device()` is 0 here)."""
from std.python import PythonObject
from x_neighbors.items import FP
from x_neighbors.kapprox_items import kapprox_sparse_rp_item


def kpca_resident_binding() raises -> PythonObject:
    return PythonObject(0)


def sparse_rp_device_binding() raises -> PythonObject:
    return PythonObject(0)


def op_kapprox_sparse_rp(res: Int, kc: Int, d: Int, seed: Int, dens: Float32, scale: Float32) raises:
    var p_res = FP(unsafe_from_address=res)
    for t in range(kc * d):
        kapprox_sparse_rp_item(t, p_res, kc, d, seed, dens, scale)
