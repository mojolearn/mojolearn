# SPDX-License-Identifier: Apache-2.0
"""GP UNPACK argument pins; execution requests use typed core values."""
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.cpython_borrowed import borrowed_item


struct GPArgumentPins(Movable):
    var values: List[PythonObject]

    def __init__(out self, source: PythonObject) raises:
        var b = native_loaded_module("builtins")
        var cls = py_type(source)
        var tuple_source = cls is b.tuple
        if not tuple_source and cls is not b.list:
            raise Error("TypeError: native GP arguments require an exact tuple or list")
        self.values = List[PythonObject](capacity=len(source))
        for i in range(len(source)):  # UNPACK fixed public argument slots under GIL
            self.values.append(PythonObject(from_borrowed=borrowed_item(source, i, tuple_source)))
        _ = source

    def __init__(out self, var values: List[PythonObject]):
        self.values = values^

    def __len__(self) -> Int:
        return len(self.values)

    def __getitem__(self, index: Int) raises -> PythonObject:
        if index < 0 or index >= len(self.values):
            raise Error("IndexError: tuple index out of range")
        return self.values[index]

    def set(mut self, index: Int, value: PythonObject) raises:
        if index < 0 or index >= len(self.values):
            raise Error("IndexError: tuple index out of range")
        self.values[index] = value

    def binding(mut self, value: PythonObject):
        if len(self.values) == 0:
            self.values.append(value)
        else:
            self.values[0] = value
