# SPDX-License-Identifier: Apache-2.0
# Compile-only C export of the actual device numerical descriptor. Never execute
# this root as a runtime test; root coordinates compiler flags and evidence.
from core.native_descriptor_abi import NativeDescriptorError
from hierarchy.solver_descriptor_device import hierarchy_solver_device_v1


@export
def peer1_hierarchy_solver_device_compile(operation: Int, aa: Int, na: Int,
                                          pp: Int, np: Int, ff: Int, nf: Int,
                                          result: Int, context: Int,
                                          sink: NativeDescriptorError) abi("C") -> Int:
    return hierarchy_solver_device_v1(operation, aa, na, pp, np, ff, nf, result, context, sink)
