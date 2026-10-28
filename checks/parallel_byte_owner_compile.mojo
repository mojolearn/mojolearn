# SPDX-License-Identifier: Apache-2.0
# Compile-only root (Program pooled/offload lane, 2026-10-28): the tokenizer-host
# side of the byte-LM multi-device trainers -- the native parallel_byte_call
# door, its owner record type, the descriptor provider/resource types it
# retains, and the cross-vendor door that constructs replica owners. Never
# imported or executed; the product exports live in _mojolearn_tokenizer_host.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.parallel_byte_native import parallel_byte_c, ParallelByteOwner
from bindings.cross_vendor_native import cross_vendor_c
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, NativeDescriptorResource


@export
def PyInit_parallel_byte_owner_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("parallel_byte_owner_compile")
        module.def_py_c_function(parallel_byte_c, "parallel_byte_call")
        module.def_py_c_function(cross_vendor_c, "cross_vendor_call")
        _ = module.add_type[ParallelByteOwner]("_ParallelByteOwner")
        _ = module.add_type[NeuralDescriptorProvider]("_NeuralDescriptorProvider")
        _ = module.add_type[NativeDescriptorResource]("_NativeDescriptorResource")
        return module.finalize()
    except error:
        abort(String(error))
