# SPDX-License-Identifier: Apache-2.0
"""Admit a compiled linear provider and pack its native build metadata."""
from std.python import PythonObject
from core.linear_fit_types import LINEAR_FIT_ABI, LINEAR_FIT_OPERATION_COUNT
from core.native_descriptor_call import native_descriptor_call
from providers.python.linear_entry_boundary import linear_entry_connection
from core.linear_entry_table import LINEAR_TABLE_ESTIMATORS, linear_table_entry


def linear_execution_word(entries: PythonObject, operation: Int) raises -> Int:
    var connection = linear_entry_connection(entries)
    var key = connection.key.copy()
    var entry = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 0, String(LINEAR_FIT_ABI), LINEAR_FIT_OPERATION_COUNT)
    try:
        return native_descriptor_call(entry, operation, List[Int](), List[Int](), List[Float64]())
    finally:
        _ = entries
        _ = connection
