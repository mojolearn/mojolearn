# SPDX-License-Identifier: Apache-2.0
"""Optimizer finalization uses its typed control and retained provider owner."""
from std.python import PythonObject
from bindings.optimizer_owner_boundary import optimizer_owner_finalize


def optimizer_finalize_descriptor_binding(context: PythonObject, owner: PythonObject) raises -> PythonObject:
    return optimizer_finalize_binding(owner)


def optimizer_finalize_binding(owner: PythonObject) raises -> PythonObject:
    try:
        optimizer_owner_finalize(owner)
    except:
        # Keep the live native record/provider for an explicit retry after a
        # failed release, matching the existing best-effort finalizer contract.
        pass
    return PythonObject(None)
