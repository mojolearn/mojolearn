# SPDX-License-Identifier: Apache-2.0
# Snapshot the fixed opcode connection once; decode only names this plan uses.
from std.collections import Dict
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_type
from bindings.native_loaded_module import native_loaded_module


struct MetricPlanOpcodes(Movable):
    var values: PythonObject
    var codes: Dict[String, Int]

    def __init__(out self, values: PythonObject) raises:
        var bi = native_loaded_module("builtins")
        if py_type(values) is not bi.dict:
            raise Error("TypeError: metric opcodes require an exact string-keyed dictionary")
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyDict_Copy")(values._obj_ptr)
        if not raw:
            raise cpy.unsafe_get_error()
        self.values = PythonObject(from_owned=raw)
        self.codes = Dict[String, Int]()
        var position = 0
        var key = PyObjectPtr()
        var value = PyObjectPtr()
        var next = cpy.lib.get_function[Int32]("PyDict_Next")
        var string_type = bi.str
        var integer_type = bi.int
        # driver-loop: over connected opcode schema entries before native plan construction
        while next(self.values._obj_ptr, MutPointer(to=position), MutPointer(to=key), MutPointer(to=value)) != 0:
            var key_type = cpy.PyObject_Type(key)
            var value_type = cpy.PyObject_Type(value)
            var admitted = key_type == string_type._obj_ptr and value_type == integer_type._obj_ptr
            cpy.Py_DecRef(key_type)
            cpy.Py_DecRef(value_type)
            if not admitted:
                raise Error("TypeError: metric opcode entries require exact string keys and integers")
        # The private copy pins immutable exact strings/ints. Public connection
        # mutation cannot affect this plan, even across input preparation calls.
        _ = values

    def get(mut self, name: String) raises -> Int:
        if name not in self.codes:
            # No hooks can run: all keys and values were individually admitted.
            # Avoid constructing native strings/decoding unused opcodes at setup.
            self.codes[name] = Int(py=self.values[PythonObject(name)])
        return self.codes[name]
