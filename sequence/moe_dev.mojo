# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOE FORWARD ON RESIDENT TENSORS (lane neural-io-2, 2026-10-09; device
binding only; read ~/mojolearn-evidence/neural-gaps-20261009/read.md
section 2, moe).

`moe_forward` with a weight handle still moved x up and y down on every
call: at T tokens x D features that is 2 x 4 T D bytes over PCIe (two 33 MB
arrays at 8192 x 1024), plus four `np.zeros` outputs page-faulted on the
host (y, logits, picks, weights), around 5-8 ms of a ~32 ms forward on an
L40S against torch's kernel-only clock on device tensors. Here x, y and the
three routing outputs are resident sequence tensors
(`sequence/seq_tensor.mojo`): nothing crosses the bus, only the executor's
pooled workspace (`sequence/exec_device.mojo`'s `_SeqPool`, kept across
calls) is used for the probabilities, hidden activations and grouping. The
launches are `moe_forward_core`, the host-array entry's own, on the same
values: no bit moves.

Registered unless -D MOJOLEARN_SEQ_MOE_DEVICE_IO_OFF (the grid's before
arm); `MoEBlock.forward` uses it only when the binding has it and x is a
resident tensor."""
from std.python import PythonObject

from sequence.exec_device import DeviceExec
from sequence.moe_weights import moe_weights_ptrs
from sequence.pyapi import ival, moe_forward_core
from sequence.seq_tensor import seq_tensor_ptr


def moe_forward_dev_py(handles: PythonObject, ip: PythonObject) raises -> PythonObject:
    """handles = [x (T, D), y (T, D) out, logits (T, E) out, selected (T, k)
    out as floats, weights (T, k) out] (seq_tensor handles); ip = [T, D, F,
    E, k, renormalise, weight handle (> 0, `moe_weights_put`)]. Returns
    T * D."""
    if len(handles) != 5 or len(ip) != 7:
        raise Error("moe_forward_dev: requires 5 tensor handles and 7 integer parameters")
    var T = ival(ip, 0)
    var D = ival(ip, 1)
    var F = ival(ip, 2)
    var En = ival(ip, 3)
    var k = ival(ip, 4)
    var renorm = ival(ip, 5)
    var wh = ival(ip, 6)
    if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
        raise Error("moe_forward_dev: T, D, F, E >= 1 and 1 <= k <= E")
    if wh < 1:
        raise Error("moe_forward_dev: requires a moe_weights_put handle")
    var X = seq_tensor_ptr(handles[0], T * D, "x")
    var Y = seq_tensor_ptr(handles[1], T * D, "y")
    var L = seq_tensor_ptr(handles[2], T * En, "logits")
    var Sel = seq_tensor_ptr(handles[3], T * k, "selected")
    var W = seq_tensor_ptr(handles[4], T * k, "weights")
    var w = moe_weights_ptrs(wh)
    var ex = DeviceExec()
    moe_forward_core(ex, X, Y, L, Sel, W, T, D, F, En, k, renorm, w[0], w[1], w[2])
    return PythonObject(T * D)
