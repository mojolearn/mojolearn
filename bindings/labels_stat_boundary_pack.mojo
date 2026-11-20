# SPDX-License-Identifier: Apache-2.0
"""Pack a native reduction result for the legacy public scalar wrapper."""
from std.python import Python, PythonObject
from core.hotpath_impl import HpStat, HP_STAT_FLOAT, HP_STAT_TUPLE


def labels_stat_pack(value: HpStat) raises -> PythonObject:
    if value.kind == HP_STAT_TUPLE:
        return Python.tuple(PythonObject(value.hi), PythonObject(value.mid), PythonObject(value.low))
    if value.kind == HP_STAT_FLOAT:
        return PythonObject(value.f)
    return PythonObject(value.i)
