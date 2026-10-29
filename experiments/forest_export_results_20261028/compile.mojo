# SPDX-License-Identifier: Apache-2.0
# Compile-only forest-export result conversion root; never execute.
from bindings.forest_export_binding import forest_export_py_list_i64, forest_export_py_list_f64


def main() raises:
    _ = forest_export_py_list_i64(List[Int64](length=1, fill=Int64(-1)))
    _ = forest_export_py_list_f64(List[Float64](length=1, fill=Float64(0)))
