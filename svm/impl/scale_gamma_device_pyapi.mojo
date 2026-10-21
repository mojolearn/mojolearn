# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/impl/scale_gamma_device.mojo` (the shell): they
unpack PythonObject arguments, call the typed device kernel and box the result."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext

from core.py2mojo_linear import py2mojo_linear_flags
from svm.impl.scale_gamma_device import scale_gamma_limbs_device


def scale_gamma_limbs_device_binding(
    ctx: DeviceContext, x_addr: PythonObject, count: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """The body of each GPU binding's `scale_gamma_limbs(x, count, out)`;
    the binding hands in its family context. Returns count."""
    var n = Int(py=count)
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    if n < 0 or oa == 0 or (n > 0 and xa == 0):
        raise Error("scale_gamma_limbs: null buffer or negative count")
    with GILReleased(Python()):
        scale_gamma_limbs_device(ctx, xa, n, oa)
    return PythonObject(n)


def py2mojo_linear_flags_binding() raises -> PythonObject:
    return PythonObject(py2mojo_linear_flags())
