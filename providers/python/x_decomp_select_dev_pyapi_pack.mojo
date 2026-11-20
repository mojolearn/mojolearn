# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/select_dev.mojo: the device select, order
and reduce doors the GPU binding registers. Each unpacks pool ids and
integers and calls one typed `*_run` entry in x_decomp/select_dev.mojo; no
data work happens here."""
from std.python import PythonObject

from providers.python.x_decomp_resident_pyapi_unpack import _id, _n
from x_decomp.select_dev import (
    dev_argmin_all_run,
    dev_order_small_run,
    dev_pca_mle_rank_run,
    dev_reduce_run,
    dev_select_smallest_run,
)

def dev_order_small_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (n exact floats, device) = the stable ascending order of the
    first n values of the device matrix a (p = [n])."""
    var n = _n(p, 0)
    var i_a = _id(a)
    var i_dst = _id(dst)
    return PythonObject(dev_order_small_run(i_a, i_dst, n))


def dev_pca_mle_rank_py(sp: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (one exact float, device) = Minka's MLE rank of the spectrum in
    the device matrix sp (d binary64 values, 2 d float32 words), p = [d,
    n_samples]: x_decomp/pca_mle.mojo's six phases, enqueued with no sync
    (lane cpu3-python). The host column is `x_decomp_pca_mle_rank`."""
    var d = _n(p, 0)
    var n = _n(p, 1)
    var i_sp = _id(sp)
    var i_dst = _id(dst)
    return PythonObject(dev_pca_mle_rank_run(i_sp, i_dst, d, n))
