# SPDX-License-Identifier: Apache-2.0
# The resample family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct ResampleKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k__ARRAY_API: PythonObject
    var k___bases__: PythonObject
    var k___del__: PythonObject
    var k___get__: PythonObject
    var k___init__: PythonObject
    var k___mro__: PythonObject
    var k___new__: PythonObject
    var k__categories: PythonObject
    var k__closed: PythonObject
    var k__data: PythonObject
    var k__dtype: PythonObject
    var k__fill_value: PythonObject
    var k__freq: PythonObject
    var k__global_config: PythonObject
    var k__left: PythonObject
    var k__mask: PythonObject
    var k__mgr: PythonObject
    var k__na_value: PythonObject
    var k__ordered: PythonObject
    var k__pa_array: PythonObject
    var k__range: PythonObject
    var k__right: PythonObject
    var k__sparse_index: PythonObject
    var k__sparse_values: PythonObject
    var k__tz: PythonObject
    var k_array_type: PythonObject
    var k_bit_length: PythonObject
    var k_blocks: PythonObject
    var k_copy: PythonObject
    var k_data: PythonObject
    var k_distinguish_nan_and_na: PythonObject
    var k_family_rs_result_classes: PythonObject
    var k_fromkeys: PythonObject
    var k_future: PythonObject
    var k_modules: PythonObject
    var k_ndarray: PythonObject
    var k_resample_pandas_array_snapshots: PythonObject
    var k_resample_pandas_category_records: PythonObject
    var k_resample_pandas_dtype_classes: PythonObject
    var k_resample_pandas_dtype_factory: PythonObject
    var k_resample_pandas_interval_types: PythonObject
    var k_resample_pandas_na: PythonObject
    var k_resample_pandas_nat: PythonObject
    var k_resample_pandas_scalar_api: PythonObject
    var k_resample_pandas_scalar_classes: PythonObject
    var k_resample_pandas_sparse_records: PythonObject
    var k_resample_pandas_static: PythonObject
    var k_resample_pandas_timedelta_class: PythonObject
    var k_resample_pandas_timedelta_from: PythonObject
    var k_resample_pandas_timestamp_class: PythonObject
    var k_resample_pandas_timestamp_from: PythonObject
    var k_resample_pandas_view: PythonObject
    var k_utcoffset: PythonObject
    var k_view: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k__ARRAY_API = PythonObject(None)
        self.k___bases__ = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___get__ = PythonObject(None)
        self.k___init__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___new__ = PythonObject(None)
        self.k__categories = PythonObject(None)
        self.k__closed = PythonObject(None)
        self.k__data = PythonObject(None)
        self.k__dtype = PythonObject(None)
        self.k__fill_value = PythonObject(None)
        self.k__freq = PythonObject(None)
        self.k__global_config = PythonObject(None)
        self.k__left = PythonObject(None)
        self.k__mask = PythonObject(None)
        self.k__mgr = PythonObject(None)
        self.k__na_value = PythonObject(None)
        self.k__ordered = PythonObject(None)
        self.k__pa_array = PythonObject(None)
        self.k__range = PythonObject(None)
        self.k__right = PythonObject(None)
        self.k__sparse_index = PythonObject(None)
        self.k__sparse_values = PythonObject(None)
        self.k__tz = PythonObject(None)
        self.k_array_type = PythonObject(None)
        self.k_bit_length = PythonObject(None)
        self.k_blocks = PythonObject(None)
        self.k_copy = PythonObject(None)
        self.k_data = PythonObject(None)
        self.k_distinguish_nan_and_na = PythonObject(None)
        self.k_family_rs_result_classes = PythonObject(None)
        self.k_fromkeys = PythonObject(None)
        self.k_future = PythonObject(None)
        self.k_modules = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_resample_pandas_array_snapshots = PythonObject(None)
        self.k_resample_pandas_category_records = PythonObject(None)
        self.k_resample_pandas_dtype_classes = PythonObject(None)
        self.k_resample_pandas_dtype_factory = PythonObject(None)
        self.k_resample_pandas_interval_types = PythonObject(None)
        self.k_resample_pandas_na = PythonObject(None)
        self.k_resample_pandas_nat = PythonObject(None)
        self.k_resample_pandas_scalar_api = PythonObject(None)
        self.k_resample_pandas_scalar_classes = PythonObject(None)
        self.k_resample_pandas_sparse_records = PythonObject(None)
        self.k_resample_pandas_static = PythonObject(None)
        self.k_resample_pandas_timedelta_class = PythonObject(None)
        self.k_resample_pandas_timedelta_from = PythonObject(None)
        self.k_resample_pandas_timestamp_class = PythonObject(None)
        self.k_resample_pandas_timestamp_from = PythonObject(None)
        self.k_resample_pandas_view = PythonObject(None)
        self.k_utcoffset = PythonObject(None)
        self.k_view = PythonObject(None)


comptime _RESAMPLE_KEYS = SharedSlot[ResampleKeys, "MojolearnResampleKeysV1"]


def _resample_keys_fill(nk: UnsafePointer[ResampleKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k__ARRAY_API = py_key("_ARRAY_API")
    nk[].k___bases__ = py_key("__bases__")
    nk[].k___del__ = py_key("__del__")
    nk[].k___get__ = py_key("__get__")
    nk[].k___init__ = py_key("__init__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___new__ = py_key("__new__")
    nk[].k__categories = py_key("_categories")
    nk[].k__closed = py_key("_closed")
    nk[].k__data = py_key("_data")
    nk[].k__dtype = py_key("_dtype")
    nk[].k__fill_value = py_key("_fill_value")
    nk[].k__freq = py_key("_freq")
    nk[].k__global_config = py_key("_global_config")
    nk[].k__left = py_key("_left")
    nk[].k__mask = py_key("_mask")
    nk[].k__mgr = py_key("_mgr")
    nk[].k__na_value = py_key("_na_value")
    nk[].k__ordered = py_key("_ordered")
    nk[].k__pa_array = py_key("_pa_array")
    nk[].k__range = py_key("_range")
    nk[].k__right = py_key("_right")
    nk[].k__sparse_index = py_key("_sparse_index")
    nk[].k__sparse_values = py_key("_sparse_values")
    nk[].k__tz = py_key("_tz")
    nk[].k_array_type = py_key("array_type")
    nk[].k_bit_length = py_key("bit_length")
    nk[].k_blocks = py_key("blocks")
    nk[].k_copy = py_key("copy")
    nk[].k_data = py_key("data")
    nk[].k_distinguish_nan_and_na = py_key("distinguish_nan_and_na")
    nk[].k_family_rs_result_classes = py_key("family_rs_result_classes")
    nk[].k_fromkeys = py_key("fromkeys")
    nk[].k_future = py_key("future")
    nk[].k_modules = py_key("modules")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_resample_pandas_array_snapshots = py_key("resample_pandas_array_snapshots")
    nk[].k_resample_pandas_category_records = py_key("resample_pandas_category_records")
    nk[].k_resample_pandas_dtype_classes = py_key("resample_pandas_dtype_classes")
    nk[].k_resample_pandas_dtype_factory = py_key("resample_pandas_dtype_factory")
    nk[].k_resample_pandas_interval_types = py_key("resample_pandas_interval_types")
    nk[].k_resample_pandas_na = py_key("resample_pandas_na")
    nk[].k_resample_pandas_nat = py_key("resample_pandas_nat")
    nk[].k_resample_pandas_scalar_api = py_key("resample_pandas_scalar_api")
    nk[].k_resample_pandas_scalar_classes = py_key("resample_pandas_scalar_classes")
    nk[].k_resample_pandas_sparse_records = py_key("resample_pandas_sparse_records")
    nk[].k_resample_pandas_static = py_key("resample_pandas_static")
    nk[].k_resample_pandas_timedelta_class = py_key("resample_pandas_timedelta_class")
    nk[].k_resample_pandas_timedelta_from = py_key("resample_pandas_timedelta_from")
    nk[].k_resample_pandas_timestamp_class = py_key("resample_pandas_timestamp_class")
    nk[].k_resample_pandas_timestamp_from = py_key("resample_pandas_timestamp_from")
    nk[].k_resample_pandas_view = py_key("resample_pandas_view")
    nk[].k_utcoffset = py_key("utcoffset")
    nk[].k_view = py_key("view")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def resample_keys() raises -> UnsafePointer[ResampleKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _RESAMPLE_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _resample_keys_fill(nk)
    return nk
