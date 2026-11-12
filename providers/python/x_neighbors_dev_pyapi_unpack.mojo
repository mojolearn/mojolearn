# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the x_neighbors kernels the GPU binding
(`bindings/_mojolearn_x_neighbors.mojo`) exports by hand: PythonObject
arguments to typed values and back, argument checks, the GIL released
around the typed call. No arithmetic on data; every number comes from the
typed kernels in x_neighbors/."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_neighbors.kapprox_dev import kpca_resident
from x_neighbors.kfeat_dev import (
    kfeat_first_negative,
    kfeat_flags,
    kfeat_pcs_draw_idn,
    kfeat_schi2_draw,
    kfeat_schi2_fit,
    kfeat_schi2_fit_idn,
    kfeat_schi2_transform,
    kfeat_schi2_weights,
)
from x_neighbors.nan_cells_device import nc_fit_lean
from x_neighbors.ocsvm_dev import op_ocsvm_alpha_init, op_ocsvm_x, op_unit_ff
from x_neighbors.ocsvm_init import oci_nu_hi, oci_nu_lo
from x_neighbors.native_descriptor_aux_device import xn_aux_execute


def _aux_call(op: Int, a_: PythonObject, i_: PythonObject, f_: PythonObject, na: Int, ni: Int,
              nf: Int, optional_ints: Int = 0) raises:
    """One auxiliary descriptor operation (x_neighbors/native_descriptor_aux_*)
    from the Python door: the fixed address, dimension and scalar words of
    the named operation, then its typed admission and owner, with the GIL released."""
    var a = List[Int]()
    var p = List[Int]()
    var f = List[Float64]()
    for k in range(na):  # small-loop(the fixed address words of the named operation)
        a.append(Int(py=a_[k]))
    for k in range(ni):  # small-loop(the fixed dimension words of the named operation)
        p.append(Int(py=i_[k]))
    # `optional_ints` trailing words the caller may pass (lane
    # a8b-device-x-neighbors-rest: the finite status address).
    for k in range(ni, min(ni + optional_ints, len(i_))):  # small-loop(at most the named operation's optional words)
        p.append(Int(py=i_[k]))
    for k in range(nf):  # small-loop(the fixed scalar words of the named operation)
        f.append(Float64(py=f_[k]))
    with GILReleased(Python()):
        _ = xn_aux_execute(op, a, p, f)
