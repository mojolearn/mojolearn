# SPDX-License-Identifier: Apache-2.0
"""Optimizer lifetime boundary: a typed control record and its extension pin."""
# The PACK half of the former `bindings/optimizer_owner_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/optimizer_owner_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr


from bindings.optimizer_owner_boundary_unpack import (
    OptimizerOwner,
)


def optimizer_owner_new(owner: PythonObject, kind: Int, pristine: Bool) raises -> PythonObject:
    return PythonObject(alloc=OptimizerOwner(Int(owner._obj_ptr), kind, pristine))
