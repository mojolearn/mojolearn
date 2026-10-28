# SPDX-License-Identifier: Apache-2.0
"""Retained native identity snapshots; compare live containers without Python temporaries.

Only exact built-in containers are traversed, using documented CPython APIs while
holding the GIL. Native vectors own the compared objects so pointer reuse cannot
turn replacement into a hit. Container lengths delimit the recursive stream;
in-place edits and same-size replacement are checked, not inferred from size.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import NativeModules
from bindings.py_balanced import py_type


struct NativeIdentityKey(Movable, Writable):
    var objects: List[PythonObject]
    var sizes: List[Int]
    var list_type: PythonObject
    var tuple_type: PythonObject
    var dict_type: PythonObject
    var str_type: PythonObject

    def __init__(out self, mut mods: NativeModules) raises:
        var bi = mods.module("builtins")
        self.objects = List[PythonObject]()
        self.sizes = List[Int]()
        self.list_type = bi.list
        self.tuple_type = bi.tuple
        self.dict_type = bi.dict
        self.str_type = bi.str

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NativeIdentityKey(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def _size(self, value: PythonObject, depth: Int, limit: Int) raises -> Int:
        var cls = py_type(value)
        if cls is not self.list_type and cls is not self.tuple_type and cls is not self.dict_type:
            return -1
        if depth >= limit:
            raise Error("native identity snapshot: container nesting exceeds limit")
        return len(value)

    def add(mut self, value: PythonObject, depth: Int = 0, limit: Int = 64) raises:
        var size = self._size(value, depth, limit)
        self.objects.append(value)
        self.sizes.append(size)
        if size < 0:
            return
        ref cpy = Python().cpython()
        var cls = py_type(value)
        if cls is self.dict_type:
            var pos = Int(0)
            var key = PyObjectPtr()
            var item = PyObjectPtr()
            while cpy.lib.get_function[Int32]("PyDict_Next")(value._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                self.add(PythonObject(from_borrowed=key), depth + 1, limit)
                self.add(PythonObject(from_borrowed=item), depth + 1, limit)
        else:
            for i in range(size):  # driver-loop: retained exact container metadata
                var item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")(value._obj_ptr, i) if cls is self.list_type else cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(value._obj_ptr, i)
                self.add(PythonObject(from_borrowed=item), depth + 1, limit)

    def matches(self, value: PythonObject, mut offset: Int, depth: Int = 0, limit: Int = 64) raises -> Bool:
        if offset >= len(self.objects) or self.objects[offset] is not value:
            return False
        var size = self._size(value, depth, limit)
        if self.sizes[offset] != size:
            return False
        offset += 1
        if size < 0:
            return True
        ref cpy = Python().cpython()
        var cls = py_type(value)
        if cls is self.dict_type:
            var pos = Int(0)
            var key = PyObjectPtr()
            var item = PyObjectPtr()
            while cpy.lib.get_function[Int32]("PyDict_Next")(value._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                if not self.matches(PythonObject(from_borrowed=key), offset, depth + 1, limit) or not self.matches(PythonObject(from_borrowed=item), offset, depth + 1, limit):
                    return False
        else:
            for i in range(size):  # driver-loop: compare live retained container pointers
                var item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")(value._obj_ptr, i) if cls is self.list_type else cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(value._obj_ptr, i)
                if not self.matches(PythonObject(from_borrowed=item), offset, depth + 1, limit):
                    return False
        return True

    def _skip(self, key: PythonObject, skip: List[String]) raises -> Bool:
        if py_type(key) is self.str_type:
            var name = String(key)
            for i in range(len(skip)):  # small-loop(skip: fixed runtime-owned field names)
                if name == skip[i]:
                    return True
        return False

    def fields(mut self, fields: PythonObject, skip: List[String]) raises:
        if py_type(fields) is not self.dict_type:
            raise Error("native identity snapshot requires exact field dictionary")
        ref cpy = Python().cpython()
        var pos = Int(0)
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        while cpy.lib.get_function[Int32]("PyDict_Next")(fields._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
            var name = PythonObject(from_borrowed=key)
            if not self._skip(name, skip):
                self.add(name)
                self.add(PythonObject(from_borrowed=item), 1)
        # An explicit end marker prevents fields being confused with a following
        # stream segment when fields are removed or inserted.
        self.objects.append(fields)
        self.sizes.append(-2)

    def matches_fields(self, fields: PythonObject, skip: List[String], mut offset: Int) raises -> Bool:
        if py_type(fields) is not self.dict_type:
            return False
        ref cpy = Python().cpython()
        var pos = Int(0)
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        while cpy.lib.get_function[Int32]("PyDict_Next")(fields._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
            var name = PythonObject(from_borrowed=key)
            if not self._skip(name, skip):
                if not self.matches(name, offset) or not self.matches(PythonObject(from_borrowed=item), offset, 1):
                    return False
        if offset >= len(self.objects) or self.objects[offset] is not fields or self.sizes[offset] != -2:
            return False
        offset += 1
        return True

    def complete(self, offset: Int) -> Bool:
        return offset == len(self.objects)
