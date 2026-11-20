# SPDX-License-Identifier: Apache-2.0
# The prophet family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add



struct ProphetKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___dict__: PythonObject
    var k___mro__: PythonObject
    var k_float32: PythonObject
    var k_float64: PythonObject
    var k_fourier_order: PythonObject
    var k_int64: PythonObject
    var k_ndarray: PythonObject
    var k_period: PythonObject
    var k_prior_scale: PythonObject
    var k_scalar_types: PythonObject
    var k_seasonalities: PythonObject
    var k_seasonality_prior_scale: PythonObject
    var k_xseq_forecast_ownership: PythonObject
    var k_xseq_lifecycle_context: PythonObject
    var k_xseq_prophet_big_i64: PythonObject
    var k_xseq_prophet_class: PythonObject
    var k_xseq_prophet_complex_dtype: PythonObject
    var k_xseq_prophet_complex_scalars: PythonObject
    var k_xseq_prophet_complex_warning: PythonObject
    var k_xseq_prophet_datetime_ns: PythonObject
    var k_xseq_prophet_datetime_type: PythonObject
    var k_xseq_prophet_extended_scalars: PythonObject
    var k_xseq_prophet_extra_scalars: PythonObject
    var k_xseq_prophet_layouts: PythonObject
    var k_xseq_prophet_object_dtype: PythonObject
    var k_xseq_prophet_promote_types: PythonObject
    var k_xseq_prophet_scalar_inputs: PythonObject
    var k_xseq_prophet_string_type: PythonObject
    var k_xseq_prophet_timedelta_type: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k_float32 = PythonObject(None)
        self.k_float64 = PythonObject(None)
        self.k_fourier_order = PythonObject(None)
        self.k_int64 = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_period = PythonObject(None)
        self.k_prior_scale = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_seasonalities = PythonObject(None)
        self.k_seasonality_prior_scale = PythonObject(None)
        self.k_xseq_forecast_ownership = PythonObject(None)
        self.k_xseq_lifecycle_context = PythonObject(None)
        self.k_xseq_prophet_big_i64 = PythonObject(None)
        self.k_xseq_prophet_class = PythonObject(None)
        self.k_xseq_prophet_complex_dtype = PythonObject(None)
        self.k_xseq_prophet_complex_scalars = PythonObject(None)
        self.k_xseq_prophet_complex_warning = PythonObject(None)
        self.k_xseq_prophet_datetime_ns = PythonObject(None)
        self.k_xseq_prophet_datetime_type = PythonObject(None)
        self.k_xseq_prophet_extended_scalars = PythonObject(None)
        self.k_xseq_prophet_extra_scalars = PythonObject(None)
        self.k_xseq_prophet_layouts = PythonObject(None)
        self.k_xseq_prophet_object_dtype = PythonObject(None)
        self.k_xseq_prophet_promote_types = PythonObject(None)
        self.k_xseq_prophet_scalar_inputs = PythonObject(None)
        self.k_xseq_prophet_string_type = PythonObject(None)
        self.k_xseq_prophet_timedelta_type = PythonObject(None)


comptime _PROPHET_KEYS = SharedSlot[ProphetKeys, "MojolearnProphetKeysV1"]


def _prophet_keys_fill(nk: UnsafePointer[ProphetKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k_float32 = py_key("float32")
    nk[].k_float64 = py_key("float64")
    nk[].k_fourier_order = py_key("fourier_order")
    nk[].k_int64 = py_key("int64")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_period = py_key("period")
    nk[].k_prior_scale = py_key("prior_scale")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_seasonalities = py_key("seasonalities")
    nk[].k_seasonality_prior_scale = py_key("seasonality_prior_scale")
    nk[].k_xseq_forecast_ownership = py_key("xseq_forecast_ownership")
    nk[].k_xseq_lifecycle_context = py_key("xseq_lifecycle_context")
    nk[].k_xseq_prophet_big_i64 = py_key("xseq_prophet_big_i64")
    nk[].k_xseq_prophet_class = py_key("xseq_prophet_class")
    nk[].k_xseq_prophet_complex_dtype = py_key("xseq_prophet_complex_dtype")
    nk[].k_xseq_prophet_complex_scalars = py_key("xseq_prophet_complex_scalars")
    nk[].k_xseq_prophet_complex_warning = py_key("xseq_prophet_complex_warning")
    nk[].k_xseq_prophet_datetime_ns = py_key("xseq_prophet_datetime_ns")
    nk[].k_xseq_prophet_datetime_type = py_key("xseq_prophet_datetime_type")
    nk[].k_xseq_prophet_extended_scalars = py_key("xseq_prophet_extended_scalars")
    nk[].k_xseq_prophet_extra_scalars = py_key("xseq_prophet_extra_scalars")
    nk[].k_xseq_prophet_layouts = py_key("xseq_prophet_layouts")
    nk[].k_xseq_prophet_object_dtype = py_key("xseq_prophet_object_dtype")
    nk[].k_xseq_prophet_promote_types = py_key("xseq_prophet_promote_types")
    nk[].k_xseq_prophet_scalar_inputs = py_key("xseq_prophet_scalar_inputs")
    nk[].k_xseq_prophet_string_type = py_key("xseq_prophet_string_type")
    nk[].k_xseq_prophet_timedelta_type = py_key("xseq_prophet_timedelta_type")
    var ops = Python.dict()
    nk[].ops = ops
    nk[].ready = True


@always_inline
def prophet_keys() raises -> UnsafePointer[ProphetKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _PROPHET_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _prophet_keys_fill(nk)
    return nk
