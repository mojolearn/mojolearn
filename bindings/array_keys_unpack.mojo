# SPDX-License-Identifier: Apache-2.0
# The array family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime ARRAY_OPN_INIT = 0
comptime ARRAY_OPN_FROM_BUFFER = 1
comptime ARRAY_OPN_FROM_LIST = 2
comptime ARRAY_OPN_OWNED = 3
comptime ARRAY_OPN_FROM_FLAT = 4
comptime ARRAY_OPN_FULL = 5
comptime ARRAY_OPN_FROMBYTES = 6
comptime ARRAY_OPN_VIEW_OF = 7
comptime ARRAY_OPN_AS_C = 8
comptime ARRAY_OPN_AS_ORDER = 9
comptime ARRAY_OPN_HAS_ORDER = 10
comptime ARRAY_OPN_ROW = 11
comptime ARRAY_OPN_GETITEM = 12
comptime ARRAY_OPN_LEN = 13
comptime ARRAY_OPN_REPR = 14
comptime ARRAY_OPN_FLAGS = 15
comptime ARRAY_OPN_INTERFACE = 16
comptime ARRAY_OPN_BOOL = 17
comptime ARRAY_OPN_TOBYTES = 18
comptime ARRAY_OPN_COPY = 19
comptime ARRAY_OPN_TOLIST = 20
comptime ARRAY_OPN_ITER = 21
comptime ARRAY_OPN_RESHAPE = 22
comptime ARRAY_OPN_RAVEL = 23
comptime ARRAY_OPN_ASTYPE = 24
comptime ARRAY_OPN_MIN = 25
comptime ARRAY_OPN_MAX = 26
comptime ARRAY_OPN_SUM = 27
comptime ARRAY_OPN_ARGMAX = 28
comptime ARRAY_OPN_EQ = 29
comptime ARRAY_OPN_NE = 30
comptime ARRAY_OPN_ARRAY = 31
comptime ARRAY_OPN_BUFFER = 32
comptime ARRAY_OPN_REDUCE = 33
comptime ARRAY_OPN_SETSTATE = 34


struct ArrayKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___mro__: PythonObject
    var k__addr: PythonObject
    var k__base: PythonObject
    var k__mv: PythonObject
    var k__pin: PythonObject
    var k__readonly: PythonObject
    var k__store: PythonObject
    var k_aliases: PythonObject
    var k_array_class: PythonObject
    var k_array_type: PythonObject
    var k_codes: PythonObject
    var k_dtype: PythonObject
    var k_empty_address: PythonObject
    var k_entry: PythonObject
    var k_eq_i64_f64: PythonObject
    var k_from_buffer_mmap: PythonObject
    var k_from_buffer_mmap_hooks: PythonObject
    var k_from_buffer_scalar_types: PythonObject
    var k_from_buffer_slots: PythonObject
    var k_helpers: PythonObject
    var k_infinity: PythonObject
    var k_int_bounds: PythonObject
    var k_interface_class: PythonObject
    var k_itemsize: PythonObject
    var k_itemsizes: PythonObject
    var k_minus_infinity: PythonObject
    var k_modules: PythonObject
    var k_native_codes: PythonObject
    var k_native_min: PythonObject
    var k_nbytes: PythonObject
    var k_ndim: PythonObject
    var k_newobj: PythonObject
    var k_not_implemented: PythonObject
    var k_not_table: PythonObject
    var k_numpy: PythonObject
    var k_obj: PythonObject
    var k_operator: PythonObject
    var k_order: PythonObject
    var k_partial: PythonObject
    var k_row_type_set: PythonObject
    var k_scalar_type_set: PythonObject
    var k_scalar_types: PythonObject
    var k_shape: PythonObject
    var k_size: PythonObject
    var k_strides: PythonObject
    var k_struct_pack: PythonObject
    var k_struct_unpack: PythonObject
    var k_supported: PythonObject
    var k_sys_modules: PythonObject
    var k_user_objects: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k__addr = PythonObject(None)
        self.k__base = PythonObject(None)
        self.k__mv = PythonObject(None)
        self.k__pin = PythonObject(None)
        self.k__readonly = PythonObject(None)
        self.k__store = PythonObject(None)
        self.k_aliases = PythonObject(None)
        self.k_array_class = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_codes = PythonObject(None)
        self.k_dtype = PythonObject(None)
        self.k_empty_address = PythonObject(None)
        self.k_entry = PythonObject(None)
        self.k_eq_i64_f64 = PythonObject(None)
        self.k_from_buffer_mmap = PythonObject(None)
        self.k_from_buffer_mmap_hooks = PythonObject(None)
        self.k_from_buffer_scalar_types = PythonObject(None)
        self.k_from_buffer_slots = PythonObject(None)
        self.k_helpers = PythonObject(None)
        self.k_infinity = PythonObject(None)
        self.k_int_bounds = PythonObject(None)
        self.k_interface_class = PythonObject(None)
        self.k_itemsize = PythonObject(None)
        self.k_itemsizes = PythonObject(None)
        self.k_minus_infinity = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_native_codes = PythonObject(None)
        self.k_native_min = PythonObject(None)
        self.k_nbytes = PythonObject(None)
        self.k_ndim = PythonObject(None)
        self.k_newobj = PythonObject(None)
        self.k_not_implemented = PythonObject(None)
        self.k_not_table = PythonObject(None)
        self.k_numpy = PythonObject(None)
        self.k_obj = PythonObject(None)
        self.k_operator = PythonObject(None)
        self.k_order = PythonObject(None)
        self.k_partial = PythonObject(None)
        self.k_row_type_set = PythonObject(None)
        self.k_scalar_type_set = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_size = PythonObject(None)
        self.k_strides = PythonObject(None)
        self.k_struct_pack = PythonObject(None)
        self.k_struct_unpack = PythonObject(None)
        self.k_supported = PythonObject(None)
        self.k_sys_modules = PythonObject(None)
        self.k_user_objects = PythonObject(None)


comptime _ARRAY_KEYS = SharedSlot[ArrayKeys, "MojolearnArrayKeysV1"]


def _array_keys_fill(nk: UnsafePointer[ArrayKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___mro__ = py_key("__mro__")
    nk[].k__addr = py_key("_addr")
    nk[].k__base = py_key("_base")
    nk[].k__mv = py_key("_mv")
    nk[].k__pin = py_key("_pin")
    nk[].k__readonly = py_key("_readonly")
    nk[].k__store = py_key("_store")
    nk[].k_aliases = py_key("aliases")
    nk[].k_array_class = py_key("array_class")
    nk[].k_array_type = py_key("array_type")
    nk[].k_codes = py_key("codes")
    nk[].k_dtype = py_key("dtype")
    nk[].k_empty_address = py_key("empty_address")
    nk[].k_entry = py_key("entry")
    nk[].k_eq_i64_f64 = py_key("eq_i64_f64")
    nk[].k_from_buffer_mmap = py_key("from_buffer_mmap")
    nk[].k_from_buffer_mmap_hooks = py_key("from_buffer_mmap_hooks")
    nk[].k_from_buffer_scalar_types = py_key("from_buffer_scalar_types")
    nk[].k_from_buffer_slots = py_key("from_buffer_slots")
    nk[].k_helpers = py_key("helpers")
    nk[].k_infinity = py_key("infinity")
    nk[].k_int_bounds = py_key("int_bounds")
    nk[].k_interface_class = py_key("interface_class")
    nk[].k_itemsize = py_key("itemsize")
    nk[].k_itemsizes = py_key("itemsizes")
    nk[].k_minus_infinity = py_key("minus_infinity")
    nk[].k_modules = py_key("modules")
    nk[].k_native_codes = py_key("native_codes")
    nk[].k_native_min = py_key("native_min")
    nk[].k_nbytes = py_key("nbytes")
    nk[].k_ndim = py_key("ndim")
    nk[].k_newobj = py_key("newobj")
    nk[].k_not_implemented = py_key("not_implemented")
    nk[].k_not_table = py_key("not_table")
    nk[].k_numpy = py_key("numpy")
    nk[].k_obj = py_key("obj")
    nk[].k_operator = py_key("operator")
    nk[].k_order = py_key("order")
    nk[].k_partial = py_key("partial")
    nk[].k_row_type_set = py_key("row_type_set")
    nk[].k_scalar_type_set = py_key("scalar_type_set")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_shape = py_key("shape")
    nk[].k_size = py_key("size")
    nk[].k_strides = py_key("strides")
    nk[].k_struct_pack = py_key("struct_pack")
    nk[].k_struct_unpack = py_key("struct_unpack")
    nk[].k_supported = py_key("supported")
    nk[].k_sys_modules = py_key("sys_modules")
    nk[].k_user_objects = py_key("user_objects")
    var ops = Python.dict()
    py_op_add(ops, "init", ARRAY_OPN_INIT)
    py_op_add(ops, "from_buffer", ARRAY_OPN_FROM_BUFFER)
    py_op_add(ops, "from_list", ARRAY_OPN_FROM_LIST)
    py_op_add(ops, "owned", ARRAY_OPN_OWNED)
    py_op_add(ops, "from_flat", ARRAY_OPN_FROM_FLAT)
    py_op_add(ops, "full", ARRAY_OPN_FULL)
    py_op_add(ops, "frombytes", ARRAY_OPN_FROMBYTES)
    py_op_add(ops, "view_of", ARRAY_OPN_VIEW_OF)
    py_op_add(ops, "as_c", ARRAY_OPN_AS_C)
    py_op_add(ops, "as_order", ARRAY_OPN_AS_ORDER)
    py_op_add(ops, "has_order", ARRAY_OPN_HAS_ORDER)
    py_op_add(ops, "row", ARRAY_OPN_ROW)
    py_op_add(ops, "getitem", ARRAY_OPN_GETITEM)
    py_op_add(ops, "len", ARRAY_OPN_LEN)
    py_op_add(ops, "repr", ARRAY_OPN_REPR)
    py_op_add(ops, "flags", ARRAY_OPN_FLAGS)
    py_op_add(ops, "interface", ARRAY_OPN_INTERFACE)
    py_op_add(ops, "bool", ARRAY_OPN_BOOL)
    py_op_add(ops, "tobytes", ARRAY_OPN_TOBYTES)
    py_op_add(ops, "copy", ARRAY_OPN_COPY)
    py_op_add(ops, "tolist", ARRAY_OPN_TOLIST)
    py_op_add(ops, "iter", ARRAY_OPN_ITER)
    py_op_add(ops, "reshape", ARRAY_OPN_RESHAPE)
    py_op_add(ops, "ravel", ARRAY_OPN_RAVEL)
    py_op_add(ops, "astype", ARRAY_OPN_ASTYPE)
    py_op_add(ops, "min", ARRAY_OPN_MIN)
    py_op_add(ops, "max", ARRAY_OPN_MAX)
    py_op_add(ops, "sum", ARRAY_OPN_SUM)
    py_op_add(ops, "argmax", ARRAY_OPN_ARGMAX)
    py_op_add(ops, "eq", ARRAY_OPN_EQ)
    py_op_add(ops, "ne", ARRAY_OPN_NE)
    py_op_add(ops, "array", ARRAY_OPN_ARRAY)
    py_op_add(ops, "buffer", ARRAY_OPN_BUFFER)
    py_op_add(ops, "reduce", ARRAY_OPN_REDUCE)
    py_op_add(ops, "setstate", ARRAY_OPN_SETSTATE)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def array_keys() raises -> UnsafePointer[ArrayKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _ARRAY_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _array_keys_fill(nk)
    return nk
