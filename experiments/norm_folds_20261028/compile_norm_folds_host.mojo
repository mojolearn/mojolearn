# SPDX-License-Identifier: Apache-2.0
# Compile-only root (lane norm-folds, 2026-10-28): every CPU-column body that
# MOJOLEARN_IDN_NORM_FIXED_LANES touches, reachable from one export, so a
# CPU build with and without the define elaborates them. Never imported by
# the library; not a test and not a measurement.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from transformer.experiments.norm_fixed_lanes import nfl_dot, nfl_square, nfl_sum
from transformer.checks.transformer_oracle import norm_into, rms_norm_into
from transformer.checks.transformer_backward_oracle import rms_norm_backward_into
from training.host.samba_ops_oracle import host_samba_rms_norm_backward, host_samba_rms_norm_forward
from training.byte_lm_host_kernels import rms_norm_fast


def _probe(m_obj: PythonObject, dm_obj: PythonObject, kind_obj: PythonObject) raises -> PythonObject:
    var m = Int(py=m_obj)
    var dm = Int(py=dm_obj)
    var x = List[Float32](length=m * dm, fill=Float32(0.5))
    var w = List[Float32](length=dm, fill=Float32(1.0))
    var b = List[Float32](length=dm, fill=Float32(0.0))
    var ss = List[Float32](length=m, fill=Float32(0.0))
    var y = List[Float32](length=m * dm, fill=Float32(0.0))
    rms_norm_into(x, w, m, dm, ss, y)
    norm_into(x, w, b, m, dm, Float32(1e-5), Int(py=kind_obj), False, ss, y)
    var dot = List[Float32](length=m, fill=Float32(0.0))
    var dx = List[Float32](length=m * dm, fill=Float32(0.0))
    var dp = List[Float32](length=m * dm, fill=Float32(0.0))
    rms_norm_backward_into(y, x, w, ss, m, dm, dot, dx, dp)
    var f = host_samba_rms_norm_forward(x, w, m, dm, Float32(1e-6))
    var g = host_samba_rms_norm_backward(y, x, w, m, dm, Float32(1e-6))
    var h = rms_norm_fast(x, w, m, dm)
    var xp = rebind[MutPointer[Float32, MutAnyOrigin]](x.unsafe_ptr())
    var s = nfl_dot(xp, xp, 0, 0, dm) + nfl_sum(xp, 0, dm) + nfl_square(xp, 0, dm, Float32(0.25))
    return PythonObject(Float64(s + f[0] + g[0][0] + h[0] + dx[0] + dot[0]))


@export
def PyInit__mojolearn_norm_folds_host_compile() abi("C") -> PythonObject:
    try:
        var mod = PythonModuleBuilder("_mojolearn_norm_folds_host_compile")
        mod.def_function[_probe]("probe")
        return mod.finalize()
    except error:
        abort(String(error))
