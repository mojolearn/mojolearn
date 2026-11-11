# SPDX-License-Identifier: Apache-2.0
"""Prepared GP numeric class table with only necessary public label pins."""
from std.python import PythonObject
from core.gp_native_label_table import GPLabelTable


struct GPPreparedClassTable(Movable, Writable):
    var labels: List[PythonObject]
    var table: GPLabelTable

    def __init__(out self, var labels: List[PythonObject], var table: GPLabelTable):
        self.labels = labels^
        self.table = table^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPPreparedClassTable(classes=", len(self.labels), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
