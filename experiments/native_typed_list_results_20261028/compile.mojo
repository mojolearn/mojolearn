# SPDX-License-Identifier: Apache-2.0
# Compile-only root for every changed typed result publication helper. Never run.
from bindings.py_lists import py_list_f64, py_list_f32, py_list_i32, py_list_i64, py_list_u32, py_list_u64, py_list_int


def main() raises:
    _ = py_list_f64(List[Float64](length=1, fill=Float64(0)))
    _ = py_list_f32(List[Float32](length=1, fill=Float32(0)))
    _ = py_list_i32(List[Int32](length=1, fill=Int32(-1)))
    _ = py_list_i64(List[Int64](length=1, fill=Int64(-1)))
    _ = py_list_u32(List[UInt32](length=1, fill=UInt32(1)))
    _ = py_list_u64(List[UInt64](length=1, fill=UInt64(1)))
    _ = py_list_int(List[Int](length=1, fill=1))
