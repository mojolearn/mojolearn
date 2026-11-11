# SPDX-License-Identifier: Apache-2.0
"""GIL-held exact dictionary capture into owning native metadata containers."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type


struct GPMappingSnapshot(Movable):
    var keys: List[PythonObject]
    var values: List[PythonObject]

    def __init__(out self):
        self.keys = List[PythonObject]()
        self.values = List[PythonObject]()


def gp_mapping_snapshot[Values: Bool = True, Keys: Bool = True](value: PythonObject) raises -> GPMappingSnapshot:
    # Only metadata references are read. Holding the GIL prevents mutation by
    # another interpreter thread; PyDict_Next and reference acquisition invoke
    # no key hash/equality, iterator or application callback. The finished native
    # lists own each borrowed reference before any caller inspects its contents.
    if py_type(value) is not native_loaded_module("builtins").dict:
        raise Error("TypeError: GP metadata capture requires an exact native dictionary")
    var result = GPMappingSnapshot()
    var cursor = Array[Int, 1](fill=0)
    var items = Array[PyObjectPtr, 2](fill=PyObjectPtr())
    var api = cpython_table()
    while api[].PyDict_Next(value._obj_ptr, Int(cursor.unsafe_ptr()),
                           Int(items.unsafe_ptr()), Int(items.unsafe_ptr() + 1)):
        @__parameter
        if Keys:
            result.keys.append(PythonObject(from_borrowed=items[0]))
        @__parameter
        if Values:
            result.values.append(PythonObject(from_borrowed=items[1]))
    _ = value
    return result^


struct GPFieldUpdates(Movable):
    # Private constructor publication plan. Hash lookup retains the first field
    # name and insertion order while a repeated name replaces only its value,
    # matching the former exact Python dictionary. Public values remain pins.
    var indices: Dict[String, Int]
    var keys: List[PythonObject]
    var values: List[PythonObject]

    def __init__(out self):
        self.indices = Dict[String, Int]()
        self.keys = List[PythonObject]()
        self.values = List[PythonObject]()

    def put(mut self, name: PythonObject, value: PythonObject) raises:
        if py_type(name) is not native_loaded_module("builtins").str:
            raise Error("TypeError: GP kernel publication requires exact native field names")
        var text = String(name)
        if text in self.indices:
            self.values[self.indices[text]] = value
        else:
            self.indices[text] = len(self.keys)
            self.keys.append(name)
            self.values.append(value)
