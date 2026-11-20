# SPDX-License-Identifier: Apache-2.0
"""An opaque prepared owner for native GP postfix tables, never model state."""
from core.gp_native_storage import GPKernelStorage


struct GPPackedKernel(Movable, Writable):
    var storage: GPKernelStorage

    def __init__(out self, var storage: GPKernelStorage):
        self.storage = storage^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPPackedKernel(nodes=", len(self.storage.kinds), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
