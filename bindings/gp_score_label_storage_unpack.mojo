# SPDX-License-Identifier: Apache-2.0
"""Private decoded score words or retained public scalar identity pins."""
from std.python import PythonObject
from core.gp_native_label_table import GP_LABEL_OBJECT


struct GPDecodedScoreLabels(Movable):
    var rows: Int
    var kind: Int
    var words: List[UInt64]
    var pins: List[PythonObject]

    def __init__(out self, rows: Int, kind: Int):
        self.rows = rows
        self.kind = kind
        self.words = List[UInt64](length=rows if kind != GP_LABEL_OBJECT else 0, fill=0)
        self.pins = List[PythonObject](capacity=rows if kind == GP_LABEL_OBJECT else 0)
