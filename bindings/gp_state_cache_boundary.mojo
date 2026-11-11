# SPDX-License-Identifier: Apache-2.0
"""Prepared-only fitted GP admission state and retained public identity pins."""
from std.python import PythonObject
from bindings.gp_kernel_cache_boundary import GPKernelKey
from bindings.native_model_records import ContentRecord


struct GPPreparedState(Movable, Writable):
    var key: GPKernelKey
    var nodes: List[PythonObject]
    var classes: List[PythonObject]
    var context_key: GPKernelKey
    var fields: PythonObject
    var content: ContentRecord
    var generation: Int

    def __init__(out self, var key: GPKernelKey, var nodes: List[PythonObject],
                 var classes: List[PythonObject]):
        self.key = key^
        self.nodes = nodes^
        self.classes = classes^
        self.context_key = GPKernelKey()
        self.fields = PythonObject(None)
        self.content = ContentRecord()
        self.generation = 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPPreparedState(nodes=", len(self.nodes), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
