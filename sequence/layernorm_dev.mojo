# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LAYERNORM ON RESIDENT TENSORS (lane gap-neural-io, 2026-10-08; plan
docs/plans/gaps-2026-10-08.md Section 4; device binding only).

`layer_norm_py` uploads x (and dy) and downloads y (or dx) on every call:
at the board's 67 MB activations a forward + backward fit moved five such
arrays and the host page-faulted fresh `np.zeros` outputs, against torch's
kernel-only clock on device tensors (64x on the L40S board). Here x, y, dy
and dx are resident sequence tensors (`sequence/seq_tensor.mojo`); only the
D-float weight and bias go up and the D-float dweight and dbias come down.
The launches are `layer_norm_fwd_core` / `layer_norm_bwd_core`, the per-call
entry's own, on the same values: no bit moves."""
from std.python import PythonObject

from sequence.exec_device import DeviceExec
from sequence.ops import FP
from sequence.pyapi import fptr, fval, ival, layer_norm_bwd_core, layer_norm_fwd_core
from sequence.seq_tensor import seq_tensor_ptr


def layer_norm_dev_py(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                      fp: PythonObject) raises -> PythonObject:
    """handles = [x (M, D), y (M, D) out or 0, dy (M, D) or 0, dx (M, D) out
    or 0] (seq_tensor handles); addrs = [weight (D) or 0, bias (D) or 0,
    dweight (D) out or 0, dbias (D) out or 0] (host arrays); ip = [M, D,
    has_weight, has_bias, backward]; fp = [eps]. A forward writes y; a
    backward (dy and dx given) writes dx and, when present, dweight / dbias,
    and y only when a y handle is passed. Returns M * D."""
    if len(handles) != 4 or len(addrs) != 4 or len(ip) != 5 or len(fp) != 1:
        raise Error("layer_norm_dev: requires 4 handles, 4 addresses, 5 integer and 1 float parameters")
    var M = ival(ip, 0)
    var D = ival(ip, 1)
    var hw = ival(ip, 2) != 0
    var hb = ival(ip, 3) != 0
    var bwd = ival(ip, 4) != 0
    if M < 1 or D < 1:
        raise Error("layer_norm_dev: M and D must be >= 1")
    var n = M * D
    var X = seq_tensor_ptr(handles[0], n, "x")
    var has_y = Int(py=handles[1]) != 0
    if not bwd and not has_y:
        raise Error("layer_norm_dev: a forward needs a y tensor")
    var ex = DeviceExec()
    var W = ex.alloc(D)
    var Bb = ex.alloc(D)
    if hw:
        ex.upload(W, fptr(addrs[0], "weight"), D)
    if hb:
        ex.upload(Bb, fptr(addrs[1], "bias"), D)
    # OP_LN_FWD stores every y, mean and rstd cell: no fill
    var Y: FP
    if has_y:
        Y = seq_tensor_ptr(handles[1], n, "y")
    else:
        Y = ex._alloc(n, False)
    var mean = ex._alloc(M, False)
    var rstd = ex._alloc(M, False)
    layer_norm_fwd_core(ex, X, W, Bb, Y, mean, rstd, M, D, hw, hb, fval(fp, 0))
    if bwd:
        var DY = seq_tensor_ptr(handles[2], n, "dy")
        var DX = seq_tensor_ptr(handles[3], n, "dx")
        var DW = ex.alloc(D)
        var DB = ex.alloc(D)
        layer_norm_bwd_core(ex, X, W, DY, DX, DW, DB, mean, rstd, M, D, hw)
        if hw:
            ex.download_async(fptr(addrs[2], "dweight"), DW, D)
        if hb:
            ex.download_async(fptr(addrs[3], "dbias"), DB, D)
    ex.sync()
    return PythonObject(n)
