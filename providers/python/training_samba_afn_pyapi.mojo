# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entries of `training/samba_afn.mojo` (the shell: PythonObject
arguments to typed values and back, nothing else; the kernels and their
host drivers stay in `training/samba_afn.mojo`)."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from training.samba_afn import (
    afn_embedding_backward_tied_host,
    afn_norm_head_forward_host,
    afn_tail_train_host,
)


# ===========================================================================
# THE PYTHON ENTRIES (registered by bindings/_mojolearn_training.mojo under
# AFN_SAMBA_FUSE; the (addresses, params) shape of the samba ops)
# ===========================================================================

comptime _AFN_NEURAL_CTX = (
    "MojoNeuralTrainingContextIdentical"
    if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojoNeuralTrainingContextFast"
)


def _afn_addrs(addresses: PythonObject, want: Int, name: String) raises -> List[Int]:
    if len(addresses) != want:
        raise Error(
            name + ": addresses must contain " + String(want) + " entries, got "
            + String(len(addresses))
        )
    var out = List[Int]()
    for i in range(want):  # small-loop(want: buffer addresses of one op): address slot admission
        var a = Int(py=addresses[i])
        if a == 0:
            raise Error(name + ": null buffer address at slot " + String(i))
        out.append(a)
    return out^


def _afn_params(params: PythonObject, want: Int, name: String) raises:
    if len(params) != want:
        raise Error(
            name + ": params must contain " + String(want) + " values, got "
            + String(len(params))
        )


def _afn_f32(addr: Int) -> MutPointer[Float32, MutUntrackedOrigin]:
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _afn_i32(addr: Int) -> MutPointer[Int32, MutUntrackedOrigin]:
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def samba_afn_norm_head_forward_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [c (m*n f32, written), x (m*k f32), norm_w (k f32),
    head_w (n*k f32)]; params = [m, n, k, eps (float)]. Returns `m * n`."""
    var a = _afn_addrs(addresses, 4, "samba_afn_norm_head_forward")
    _afn_params(params, 4, "samba_afn_norm_head_forward")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_norm_head_forward_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_f32(a[2]), _afn_f32(a[3]),
            m, n, k, eps,
        )
    return PythonObject(count)


def samba_afn_tail_train_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [loss (1 f32, written), row_loss (m f32, written), dh
    (m*k f32, written), d_norm_w (k f32, written), d_head_w (n*k f32,
    written), x (m*k f32), norm_w (k f32), head_w (n*k f32), targets (m
    i32)]; params = [m, n, k, eps (float), ignore_index, reduction (1 sum,
    2 mean), num_items, label_smoothing (float)]. Returns `count`."""
    var a = _afn_addrs(addresses, 9, "samba_afn_tail_train")
    _afn_params(params, 8, "samba_afn_tail_train")
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var eps = Float32(Float64(py=params[3]))
    var ignore_index = Int(py=params[4])
    var reduction = Int(py=params[5])
    var num_items = Int(py=params[6])
    var label_smoothing = Float32(Float64(py=params[7]))
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_tail_train_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_f32(a[2]), _afn_f32(a[3]),
            _afn_f32(a[4]), _afn_f32(a[5]), _afn_f32(a[6]), _afn_f32(a[7]),
            _afn_i32(a[8]),
            m, n, k, eps, ignore_index, reduction, num_items, label_smoothing,
        )
    return PythonObject(count)


def samba_afn_embedding_backward_tied_binding(
    addresses: PythonObject, params: PythonObject
) raises -> PythonObject:
    """addresses = [dw (vocab*width f32, written), dy (n_positions*width
    f32), ids (n_positions i32), pair (vocab*width f32, the head's weight
    gradient)]; params = [n_positions, vocab, width]. Returns
    `vocab * width`."""
    var a = _afn_addrs(addresses, 4, "samba_afn_embedding_backward_tied")
    _afn_params(params, 3, "samba_afn_embedding_backward_tied")
    var n_positions = Int(py=params[0])
    var vocab = Int(py=params[1])
    var width = Int(py=params[2])
    var count = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_AFN_NEURAL_CTX]()
        count = afn_embedding_backward_tied_host(
            ctx, _afn_f32(a[0]), _afn_f32(a[1]), _afn_i32(a[2]), _afn_f32(a[3]),
            n_positions, vocab, width,
        )
    return PythonObject(count)
