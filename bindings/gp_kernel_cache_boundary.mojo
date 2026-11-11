# SPDX-License-Identifier: Apache-2.0
"""Prepared GP kernel cache: native metadata and public identity/lifetime pins."""
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules, native_loaded_module
from bindings.native_class_admission import class_record, class_admit
from bindings.native_model_records import ContentRecord, content_object, content_same, model_generation, records_stampable
from bindings.gp_svm_records import gpsvm_side_get, gpsvm_side_put
from bindings.mode_metadata_native import _fields
from bindings.py_balanced import py_call, py_type
from bindings.gp_common_native import gp_kernel_arrays
from bindings.gp_kernel_public_native import gp_kernel_snapshot

comptime GP_KERNEL_PACK_SLOT = "gp_kernel_pack_v1"
comptime GP_KERNEL_CLASS_SLOT = "gp_kernel_class_admitted_v1"


@fieldwise_init
struct GPKernelKeyItem(Movable):
    # Counts were boxed key entries. Stored objects remain pins because their
    # identity is part of the supported mutable Python kernel interface.
    var is_count: Bool
    var count: Int
    var pin: PythonObject


struct GPKernelKey(Movable):
    var items: List[GPKernelKeyItem]

    def __init__(out self):
        self.items = List[GPKernelKeyItem]()

    def pin(mut self, value: PythonObject):
        self.items.append(GPKernelKeyItem(False, 0, value))

    def count(mut self, value: Int):
        self.items.append(GPKernelKeyItem(True, value, PythonObject(None)))

    def same(self, other: Self) raises -> Bool:
        if len(self.items) != len(other.items):
            return False
        var integer = native_loaded_module("builtins").int
        for i in range(len(self.items)):  # driver-loop: native key metadata, never numeric tensor contents
            if self.items[i].is_count != other.items[i].is_count:
                return False
            if self.items[i].is_count:
                if self.items[i].count != other.items[i].count:
                    return False
            else:
                var left = self.items[i].pin
                var right = other.items[i].pin
                if left is right:
                    continue
                # Preserve the old key's exact builtin integer value rule,
                # including arbitrary-size ints. Neither operand has hooks.
                if py_type(left) is integer and py_type(right) is integer and Bool(py=(left == right)):
                    continue
                return False
        return True


def _gp_key_value(mut key: GPKernelKey, value: PythonObject, depth: Int) raises:
    key.pin(value)
    if depth >= 4 or value is None:
        return
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        key.count(len(value))
        for i in range(len(value)):  # driver-loop: admitted public kernel container identities
            _gp_key_value(key, value[i], depth + 1)
    elif cls is b.dict:
        var items = py_call(b.list, py_call(value.items))
        key.count(len(items))
        for i in range(len(items)):  # driver-loop: public mapping identities in original insertion order
            key.pin(items[i][0])
            _gp_key_value(key, items[i][1], depth + 1)
    elif not (cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes):
        key.pin(cls)


def gp_key_fields(mut key: GPKernelKey, fields: PythonObject, skip: List[String]) raises:
    # Shared admitted field capture for packed kernels and fitted GP state.
    # Preserve insertion order, the depth-four value walk and skipped-name rule.
    var b = native_loaded_module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    var count = 0
    for i in range(len(items)):  # driver-loop: captured public field metadata
        var name = items[i][0]
        if py_type(name) is b.str:
            var text = String(name)
            var skipped = False
            for j in range(len(skip)):  # small-loop(call-written field names)
                if skip[j] == text:
                    skipped = True
                    break
            if skipped:
                continue
        key.pin(name)
        _gp_key_value(key, items[i][1], 0)
        count += 1
    key.count(count)


def _gp_key(context: PythonObject, nodes: List[PythonObject]) raises -> GPKernelKey:
    var key = GPKernelKey()
    key.pin(context)
    key.pin(context["family_gp_kernel_nodes"])
    key.count(len(nodes))
    var mods = NativeModules()
    for i in range(len(nodes)):  # driver-loop: each distinct kernel object's mutable state
        var node = nodes[i]
        key.pin(node)
        key.pin(py_type(node))
        gp_key_fields(key, _fields(mods, node), List[String]())
    return key^


def _gp_content(nodes: List[PythonObject]) raises -> ContentRecord:
    var content = ContentRecord()
    for i in range(len(nodes)):  # driver-loop: preserve generation/content fast path per kernel owner
        content_object(content, nodes[i], 4)
    return content^


struct GPPreparedKernel(Movable, Writable):
    var kernel: PythonObject
    var context: PythonObject
    var table: PythonObject
    var packed: PythonObject
    var nodes: List[PythonObject]
    var classes: List[PythonObject]
    var key: GPKernelKey
    var content: ContentRecord
    var generation: Int

    def __init__(out self, kernel: PythonObject, context: PythonObject, packed: PythonObject,
                 var nodes: List[PythonObject], var classes: List[PythonObject],
                 var key: GPKernelKey, var content: ContentRecord, generation: Int) raises:
        self.kernel = kernel
        self.context = context
        self.table = context["family_gp_kernel_nodes"]
        self.packed = packed
        self.nodes = nodes^
        self.classes = classes^
        self.key = key^
        self.content = content^
        self.generation = generation

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPPreparedKernel(nodes=", len(self.nodes), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def gp_packed_kernel_reused(owner: PythonObject, kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    var mods = NativeModules()
    var held = gpsvm_side_get(GP_KERNEL_PACK_SLOT, owner)
    if held is not None:
        var admitted = False
        try:
            _ = held.downcast_value_ptr[GPPreparedKernel]()
            admitted = True
        except:
            pass  # A former tuple cache is replaced on first use.
        if admitted:
            var record = held.downcast_value_ptr[GPPreparedKernel]()
            if record[].kernel is kernel:
                var generation = model_generation(owner)
                var same_context = record[].context is context and record[].table is context["family_gp_kernel_nodes"]
                var same = same_context and generation != 0 and generation == record[].generation and content_same(record[].content)
                if not same:
                    var key = _gp_key(context, record[].nodes)
                    same = record[].key.same(key)
                    if same:
                        if records_stampable(owner):
                            record[].content = _gp_content(record[].nodes)
                            record[].generation = model_generation(owner)
                if same:
                    for i in range(len(record[].classes)):  # driver-loop: class watcher admission before cache reuse
                        if class_record(mods, GP_KERNEL_CLASS_SLOT, record[].classes[i]) is None:
                            same = False
                            break
                if same:
                    var packed = record[].packed
                    _ = held
                    return packed
    var packed = gp_kernel_arrays(kernel, context)
    # As before, optional cache admission may fail after the required packing
    # succeeded; this does not change the operation's supported kernel inputs.
    var nodes = List[PythonObject]()
    var classes = List[PythonObject]()
    try:
        var snapshot = gp_kernel_snapshot(kernel, context)
        swap(nodes, snapshot.owners)
        for i in range(len(nodes)):  # driver-loop: distinct admitted kernel classes, identity only
            var cls = py_type(nodes[i])
            var known = False
            for j in range(len(classes)):  # driver-loop: the handful of distinct kernel classes
                if cls is classes[j]:
                    known = True
                    break
            if not known:
                classes.append(cls)
    except:
        return packed
    for i in range(len(classes)):  # driver-loop: preserve class watcher publication after full admission
        class_admit(mods, GP_KERNEL_CLASS_SLOT, classes[i], PythonObject(True))
    var key = _gp_key(context, nodes)
    var content = ContentRecord()
    var generation = 0
    if records_stampable(owner):
        content = _gp_content(nodes)
        generation = model_generation(owner)
    var prepared = GPPreparedKernel(kernel, context, packed, nodes^, classes^, key^, content^, generation)
    var record = PythonObject(alloc=prepared^)
    _ = gpsvm_side_put(GP_KERNEL_PACK_SLOT, owner, record)
    return packed
