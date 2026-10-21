# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of the ExponentialLR value block
(`sequence/sched_table.mojo`): PythonObject arguments to typed values and
the typed result back. Nothing else."""
from std.python import PythonObject

from sequence.sched_table import F64P, sched_exp_block


def sched_exp_block_py(addrs: PythonObject, fp: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [out (n) float64]; fp = [base_lr, gamma] (both > 0, finite);
    ip = [e0 >= 0, n >= 0]. Returns the number of undecided (NaN) entries."""
    if len(addrs) != 1 or len(fp) != 2 or len(ip) != 2:
        raise Error("sched_exp_block: requires 1 address, 2 floats and 2 integers")
    var base = Float64(py=fp[0])
    var gamma = Float64(py=fp[1])
    var e0 = Int(py=ip[0])
    var n = Int(py=ip[1])
    return PythonObject(sched_exp_block(F64P(unsafe_from_address=Int(py=addrs[0])), base, gamma, e0, n))
