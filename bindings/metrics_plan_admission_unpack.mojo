# SPDX-License-Identifier: Apache-2.0
# Snapshot the fixed opcode connection once; decode only names this plan uses.
from std.collections import Dict
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type
from bindings.native_loaded_module import native_loaded_module
from bindings.cpython_table import cpython_table


struct MetricPlanOpcodes(Movable):
    var values: PythonObject
    var codes: Dict[String, Int]
    var int_type: PythonObject

    def __init__(out self, values: PythonObject) raises:
        var bi = native_loaded_module("builtins")
        if py_type(values) is not bi.dict:
            raise Error("TypeError: metric opcodes require an exact string-keyed dictionary")
        ref cpy = Python().cpython()
        # p3-labels-metrics: PyDict_Copy from the process-wide table (it was
        # resolved by name, dlsym, per program). The per-program snapshot
        # stays: it is what protects a running program from an edited table.
        var raw = cpython_table()[].PyDict_Copy(values._obj_ptr)
        if not raw:
            raise cpy.unsafe_get_error()
        self.values = PythonObject(from_owned=raw)
        self.codes = Dict[String, Int]()
        # The int class is read once per program, not per opcode name.
        self.int_type = bi.int
        _ = values

    def get(mut self, name: String) raises -> Int:
        # Each opcode is admitted when a stage first names it (an exact int
        # under its exact string key), not by a walk of the whole table on
        # every metric call.
        if name not in self.codes:
            var value = self.values[PythonObject(name)]
            if py_type(value) is not self.int_type:
                raise Error("TypeError: metric opcode entries require exact string keys and integers")
            self.codes[name] = Int(py=value)
        return self.codes[name]


struct MetricArenaOwner(Copyable, Movable):
    # Private completed program storage. The runner already admitted/allocated
    # this owner and captured its base. No caller can resize it before reads;
    # public outputs copy their result spans into separately owned arrays.
    var store: PythonObject
    var base: Int

    def __init__(out self, store: PythonObject, base: Int):
        self.store = store
        self.base = base
