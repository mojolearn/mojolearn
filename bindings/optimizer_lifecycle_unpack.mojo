# SPDX-License-Identifier: Apache-2.0
"""Optimizer finalization uses its typed control and retained provider owner."""
from std.python import PythonObject
from bindings.optimizer_owner_boundary_unpack import optimizer_owner_finalize


from bindings.optimizer_lifecycle_pack import (
    optimizer_finalize_binding,
)


def optimizer_finalize_descriptor_binding(context: PythonObject, owner: PythonObject) raises -> PythonObject:
    return optimizer_finalize_binding(owner)
