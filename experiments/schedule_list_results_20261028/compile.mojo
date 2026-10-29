# SPDX-License-Identifier: Apache-2.0
# Compile-only public schedule operation root. Never execute.
from std.python import PythonObject
from bindings.optimizer_schedule_native import schedule_operation


def main() raises:
    var value = PythonObject(None)
    _ = schedule_operation("schedule_list", value, value, value)
    _ = schedule_operation("schedule_values", value, value, value)
