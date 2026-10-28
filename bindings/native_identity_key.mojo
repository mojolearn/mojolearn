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
from bindings.py_balanced import py_type, py_call
from bindings.native_array_snapshot import native_array_slots


struct NativeIdentityKey(Movable, Writable):
    var objects: List[PythonObject]
    var sizes: List[Int]
    var list_type: PythonObject
    var tuple_type: PythonObject
    var dict_type: PythonObject
    var str_type: PythonObject

    var array_class: PythonObject
    var array_getters: List[PythonObject]
    var array_owners: List[PythonObject]
    var array_values: List[PythonObject]
    var array_views: List[PythonObject]
    var array_view_owner_getter: PythonObject
    var array_mro: PythonObject
    var array_mro_getter: PythonObject
    var array_rows: List[PythonObject]
    var array_row_sizes: List[Int]
    var array_row_keys: List[List[PythonObject]]
    var array_row_values: List[List[PythonObject]]

    def __init__(out self, mut mods: NativeModules) raises:
        var bi = mods.module("builtins")
        self.objects = List[PythonObject]()
        self.sizes = List[Int]()
        self.list_type = bi.list
        self.tuple_type = bi.tuple
        self.dict_type = bi.dict
        self.str_type = bi.str
        self.array_class = PythonObject(None)
        self.array_getters = List[PythonObject]()
        self.array_owners = List[PythonObject]()
        self.array_values = List[PythonObject]()
        self.array_views = List[PythonObject]()
        self.array_view_owner_getter = PythonObject(None)
        self.array_mro = PythonObject(None)
        self.array_mro_getter = PythonObject(None)
        self.array_rows = List[PythonObject]()
        self.array_row_sizes = List[Int]()
        self.array_row_keys = List[List[PythonObject]]()
        self.array_row_values = List[List[PythonObject]]()

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

    def add_identity(mut self, value: PythonObject):
        self.objects.append(value)
        self.sizes.append(-1)

    def matches_identity(self, value: PythonObject, mut offset: Int) -> Bool:
        if offset >= len(self.objects) or self.objects[offset] is not value or self.sizes[offset] != -1:
            return False
        offset += 1
        return True

    def retains(self, value: PythonObject) -> Bool:
        for i in range(len(self.objects)):  # driver-loop: over retained identity references
            if self.objects[i] is value:
                return True
        for i in range(len(self.array_values)):  # driver-loop: over retained array metadata owners
            if self.array_values[i] is value:
                return True
        return False

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
            for _ in range(size):  # driver-loop: over exact dictionary metadata
                if not cpy.lib.get_function[Int32]("PyDict_Next")(value._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                    raise Error("native identity snapshot: dictionary changed during capture")
                self.add(PythonObject(from_borrowed=key), depth + 1, limit)
                self.add(PythonObject(from_borrowed=item), depth + 1, limit)
        else:
            for i in range(size):  # driver-loop: over retained exact container metadata
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
            for _ in range(size):  # driver-loop: over exact dictionary metadata
                if not cpy.lib.get_function[Int32]("PyDict_Next")(value._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                    raise Error("native identity snapshot: dictionary changed during capture")
                if not self.matches(PythonObject(from_borrowed=key), offset, depth + 1, limit) or not self.matches(PythonObject(from_borrowed=item), offset, depth + 1, limit):
                    return False
        else:
            for i in range(size):  # driver-loop: over live retained container pointers
                var item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")(value._obj_ptr, i) if cls is self.list_type else cpy.lib.get_function[PyObjectPtr]("PyTuple_GetItem")(value._obj_ptr, i)
                if not self.matches(PythonObject(from_borrowed=item), offset, depth + 1, limit):
                    return False
        return True

    def _skip(self, key: PythonObject, skip: List[String]) raises -> Bool:
        if py_type(key) is self.str_type:
            var name = String(key)
            for i in range(len(skip)):  # small-loop(skip: fixed runtime-owned field names): compare fixed exclusions only
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
        for _ in range(len(fields)):  # driver-loop: over admitted instance fields
            if not cpy.lib.get_function[Int32]("PyDict_Next")(fields._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                raise Error("native identity snapshot: fields changed during capture")
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
        for _ in range(len(fields)):  # driver-loop: over admitted instance fields
            if not cpy.lib.get_function[Int32]("PyDict_Next")(fields._obj_ptr, MutPointer(to=pos), MutPointer(to=key), MutPointer(to=item)):
                raise Error("native identity snapshot: fields changed during capture")
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

    def watch_arrays(mut self, mut mods: NativeModules, context: PythonObject) raises:
        # Array slots are public writable storage. Object identity alone cannot
        # establish that a retained view still describes the caller's Array.
        var cls = context["array_class"]
        for i in range(len(self.objects)):  # driver-loop: over retained input identities
            var value = self.objects[i]
            if py_type(value) is cls:
                var found = False
                for j in range(len(self.array_owners)):  # driver-loop: over already captured array owners
                    if self.array_owners[j] is value:
                        found = True
                        break
                if not found:
                    self.array_owners.append(value)
        if len(self.array_owners) == 0:
            return
        var slots = native_array_slots(mods, context)
        self.array_class = cls
        var bi = mods.module("builtins")
        var names = py_call(bi.list, slots.keys())
        for i in range(len(names)):  # small-loop(names: native Array slot schema): capture descriptors once
            self.array_getters.append(slots[names[i]].__get__)
        for i in range(len(self.array_owners)):  # driver-loop: over retained array metadata owners
            for j in range(len(self.array_getters)):  # small-loop(array_getters: native Array slot schema): retain slot values
                self.array_values.append(py_call(self.array_getters[j], self.array_owners[i], cls))
        # A released memoryview keeps its identity, but no longer admits a
        # fresh capture. The snapshot's independent pin must not hide that.
        self.array_view_owner_getter = bi.memoryview.__dict__["obj"].__get__
        for i in range(len(self.array_values)):  # driver-loop: over retained Array slot values
            if py_type(self.array_values[i]) is bi.memoryview:
                _ = py_call(self.array_view_owner_getter, self.array_values[i])
                self.array_views.append(self.array_values[i])
        # Keep live mapping proxies, not dictionary copies. On a hit, native
        # lookups compare every admitted class value and namespace size. This
        # detects same-size replacement, descriptor edits and inherited hooks.
        self.array_mro_getter = bi.type.__dict__["__mro__"].__get__
        self.array_mro = py_call(self.array_mro_getter, cls)
        var get_namespace = bi.type.__dict__["__dict__"].__get__
        for i in range(len(self.array_mro)):  # driver-loop: over native Array class ancestry metadata
            var row = py_call(get_namespace, self.array_mro[i])
            self.array_rows.append(row)
            self.array_row_sizes.append(len(row))
            var keys = py_call(bi.list, row.keys())
            var native_keys = List[PythonObject]()
            var native_values = List[PythonObject]()
            for j in range(len(keys)):  # small-loop(keys: Array class namespace): fixed storage metadata names only
                if py_type(keys[j]) is not self.str_type:
                    raise Error("native identity snapshot: class namespace needs exact string keys")
                native_keys.append(keys[j])
            # Admit the entire live key set before the first keyed lookup:
            # another colliding key could otherwise invoke a foreign hook.
            for j in range(len(keys)):  # driver-loop: over admitted exact namespace keys
                native_values.append(row[keys[j]])
            self.array_row_keys.append(native_keys^)
            self.array_row_values.append(native_values^)

    def arrays_match(self, mut mods: NativeModules, context: PythonObject) raises -> Bool:
        if len(self.array_owners) == 0:
            return True
        if context["array_class"] is not self.array_class or py_call(self.array_mro_getter, self.array_class) is not self.array_mro:
            return False
        ref cpy = Python().cpython()
        for i in range(len(self.array_rows)):  # small-loop(self.array_rows: Array namespaces): fixed storage class hierarchy not tensor rows
            var row = self.array_rows[i]
            if len(row) != self.array_row_sizes[i]:
                return False
            # A same-size edit can insert a foreign key whose equality hook
            # collides with an admitted string. Inspect the full live key set
            # before any lookup. The retained row is an exact mappingproxy
            # produced by type.__dict__; its iterator is CPython's native
            # dictionary-key iterator, never a caller-defined protocol.
            var iterator_raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GetIter")(row._obj_ptr)
            if not iterator_raw:
                cpy.lib.get_function[NoneType]("PyErr_Clear")()
                return False
            var iterator = PythonObject(from_owned=iterator_raw)
            for j in range(len(self.array_row_keys[i])):  # driver-loop: over live namespace key identities before lookup
                var key_raw = cpy.lib.get_function[PyObjectPtr]("PyIter_Next")(iterator._obj_ptr)
                if not key_raw:
                    cpy.lib.get_function[NoneType]("PyErr_Clear")()
                    return False
                var key = PythonObject(from_owned=key_raw)
                if py_type(key) is not self.str_type or key is not self.array_row_keys[i][j]:
                    return False
            for j in range(len(self.array_row_keys[i])):  # driver-loop: over retained native namespace keys
                var raw = cpy.lib.get_function[PyObjectPtr]("PyObject_GetItem")(row._obj_ptr, self.array_row_keys[i][j]._obj_ptr)
                if not raw:
                    cpy.lib.get_function[NoneType]("PyErr_Clear")()
                    return False
                var current = PythonObject(from_owned=raw)
                if current is not self.array_row_values[i][j]:
                    return False
        for i in range(len(self.array_views)):  # driver-loop: over retained native memoryview owners
            _ = py_call(self.array_view_owner_getter, self.array_views[i])
        var offset = 0
        for i in range(len(self.array_owners)):  # driver-loop: over retained array metadata owners
            if py_type(self.array_owners[i]) is not self.array_class:
                return False
            for j in range(len(self.array_getters)):  # small-loop(array_getters: native Array slot schema): compare live slots
                if py_call(self.array_getters[j], self.array_owners[i], self.array_class) is not self.array_values[offset]:
                    return False
                offset += 1
        return True
