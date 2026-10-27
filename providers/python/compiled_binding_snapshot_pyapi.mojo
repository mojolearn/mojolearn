# SPDX-License-Identifier: Apache-2.0
# Operation-local metadata, not another execution provider. The existing common
# provider owns dispatch/module lifetime; this immutable-by-convention admission
# record carries its freshly read scalar metadata through one public operation.
from std.python import PythonObject
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider


struct CompiledBindingSnapshot(Movable, Writable):
    var provider: PythonObject
    var module: PythonObject
    var abi: String
    var mode: Int
    var capabilities: List[Int]

    def __init__(out self, provider: PythonObject, module: PythonObject, var abi: String,
                 mode: Int, var capabilities: List[Int]) raises:
        var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
        if owner[].module is not module or owner[].name != abi:
            raise Error("compiled binding snapshot requires its admitted provider and module")
        self.provider = provider
        self.module = module
        self.abi = abi^
        self.mode = mode
        self.capabilities = capabilities^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("CompiledBindingSnapshot(", self.abi, ", mode=", self.mode, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
