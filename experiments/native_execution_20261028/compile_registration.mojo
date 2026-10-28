# SPDX-License-Identifier: Apache-2.0
# Compile-only coverage of the new bound native owner types; never executed.
from std.python.bindings import PythonModuleBuilder
from bindings.native_identity_key import NativeIdentityKey
from bindings.neural_block_public import NativeBlockPlan
from bindings.sequential_admission import SequentialAdmissions

def main() raises:
    var module = PythonModuleBuilder("_native_execution_compile")
    _ = module.add_type[NativeIdentityKey]("_NativeIdentityKey")
    _ = module.add_type[NativeBlockPlan]("_NativeBlockPlan")
    _ = module.add_type[SequentialAdmissions]("_SequentialAdmissions")
    _ = module.finalize()
