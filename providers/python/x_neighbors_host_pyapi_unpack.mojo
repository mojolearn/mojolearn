# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the x_neighbors host kernels the CPU binding
(`bindings/_mojolearn_x_neighbors_host.mojo`) exports by hand: PythonObject
arguments to typed values and back, and argument checks. No arithmetic on
data; every number comes from the typed host kernels in x_neighbors/.
HOST ONLY."""
from std.python import PythonObject
from x_neighbors.items import FP, IP
from x_neighbors.kapprox_host import kpca_resident
from x_neighbors.kfeat_rng import op_kfeat_pcs_draw_idn_host, op_kfeat_schi2_fit_idn_host
from x_neighbors.ocsvm_host import op_ocsvm_alpha_init, op_unit_ff
from x_neighbors.ocsvm_init import oci_nu_hi, oci_nu_lo
from x_neighbors.native_descriptor_aux_host import xn_aux_execute


def _kfeat_args(p: PythonObject, what: String) raises -> Tuple[UInt32, Int, Int]:
    var seed = UInt32(Int(py=p[0]) & 0xFFFFFFFF)
    var a = Int(py=p[1])
    var b = Int(py=p[2])
    if a <= 0 or b <= 0:
        raise Error("x_neighbors_" + what + ": sizes must be positive")
    if a * b > 2147483647:
        raise Error("x_neighbors_" + what + ": more than 2^31 - 1 draws")
    return (seed, a, b)


def _aux_call(op: Int, a_: PythonObject, i_: PythonObject, f_: PythonObject, na: Int, ni: Int,
              nf: Int, optional_ints: Int = 0) raises:
    """One auxiliary descriptor operation (x_neighbors/native_descriptor_aux_*)
    from the Python door: the fixed address, dimension and scalar words of
    the named operation, then its typed admission and owner."""
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
    _ = xn_aux_execute(op, a, p, f)
