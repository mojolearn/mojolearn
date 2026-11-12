# SPDX-License-Identifier: Apache-2.0
"""Optimizer finalization uses its typed control and retained provider owner."""
# The PACK half of the former `bindings/optimizer_lifecycle.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/optimizer_lifecycle_unpack.mojo`.
from std.python import PythonObject
from bindings.optimizer_owner_boundary_unpack import optimizer_owner_finalize


def optimizer_finalize_binding(owner: PythonObject) raises -> PythonObject:
    try:
        optimizer_owner_finalize(owner)
    except:
        # Keep the live native record/provider for an explicit retry after a
        # failed release, matching the existing best-effort finalizer contract.
        pass
    return PythonObject(None)
