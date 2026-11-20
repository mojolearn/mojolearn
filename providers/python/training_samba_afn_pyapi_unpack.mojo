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
    afn_tail_train_host,
)
from training.training_samba_afn_run import _afn_f32, _afn_i32


# ===========================================================================
# THE PYTHON ENTRIES (registered by bindings/_mojolearn_training.mojo under
# AFN_SAMBA_FUSE; the (addresses, params) shape of the samba ops)
# ===========================================================================

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


# TOMBSTONE: SAMBA_THREE_TIER_FORWARD samba_afn_norm_head_forward_binding deleted 2026-11-09 by lane/x6-samba-on-executor; code recoverable at 5b5490960d.
# Restore: git apply experiments/removed/SAMBA_THREE_TIER_FORWARD.patch
