# SPDX-License-Identifier: Apache-2.0
"""Lifetime pins for typed linear model records at the Python boundary.

The shared weak model-record table owns this envelope. RUN receives a copy
of its native state, and cannot access the Python owners retained here.
Public fitted buffers remain writable; each operation reconciles their
admitted descriptors before taking its immutable native snapshot.
"""
from std.python import PythonObject
from std.memory import ArcPointer
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.native_owner_registry import borrowed_native_owner, register_native_owner
from bindings.classical_arrays import classical_address
from bindings.native_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from core.linear_model_state import LinearModelState
from core.linear_registry import LINEAR_STATES
from core.state_registry import StateHandle

comptime _LINEAR_STATE_SLOT = "linear_model_state_v1"


struct LinearStateOwner(Movable, Writable):
    var handle: StateHandle
    var coefficients: PythonObject
    var means: PythonObject

    def __init__(out self, state: LinearModelState, coefficients: PythonObject, means: PythonObject) raises:
        self.handle = LINEAR_STATES.insert(state.copy())
        self.coefficients = coefficients
        self.means = means

    def __del__(deinit self):
        try:
            LINEAR_STATES.release(self.handle)
        except:
            pass  # Preserve quiet finalization; stale explicit release is refused.
        # Registry removal precedes destruction of the borrowed buffer pins.
        _ = self.coefficients
        _ = self.means

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LinearStateOwner(handle=", self.handle.index, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def linear_state_owner(model: PythonObject) raises -> PythonObject:
    return records_get(RECORDS_PREPARED, model, _LINEAR_STATE_SLOT)


def linear_state_publish(model: PythonObject, coefficients: PythonObject, means: PythonObject,
    features: Int, intercept: Float64, ymean: Float64, copied: Bool) raises:
    var state = LinearModelState()
    state.coef = Int(py=classical_address(coefficients))
    state.coef_count = features
    state.features = features
    state.intercept = intercept
    # Loaded archives restore coefficients/intercept without a training mean
    # buffer. None stays an owner-boundary sentinel; native state uses null.
    state.mean = 0 if means is None else Int(py=classical_address(means))
    state.ymean = ymean
    state.has_coef = True
    state.fitted = True
    state.input_copied = copied
    var owner = PythonObject(alloc=LinearStateOwner(state, coefficients, means))
    _ = records_put(RECORDS_PREPARED, model, _LINEAR_STATE_SLOT, owner)


def linear_state_clear(model: PythonObject) raises:
    records_drop(RECORDS_PREPARED, model, _LINEAR_STATE_SLOT)


@fieldwise_init
struct LinearStateCall(Movable):
    var owner: PythonObject
    var lease: ArcPointer[LinearModelState]


def linear_state_refresh(model: PythonObject, var owner: PythonObject,
    state: LinearModelState, coefficients: PythonObject,
    prior: Optional[ArcPointer[LinearModelState]]) raises -> LinearStateCall:
    # UNPACK passes its already acquired snapshot. Unchanged public weights
    # need one registry acquisition, not repeated lock/lookups of that state.
    if owner is None:
        owner = PythonObject(alloc=LinearStateOwner(state, coefficients, PythonObject(None)))
        _ = records_put(RECORDS_PREPARED, model, _LINEAR_STATE_SLOT, owner)
        var handle = owner.downcast_value_ptr[LinearStateOwner]()[].handle.copy()
        return LinearStateCall(owner, LINEAR_STATES.acquire(handle))
    if not prior:
        raise Error("native linear state: missing admitted snapshot")
    var native = owner.downcast_value_ptr[LinearStateOwner]()
    var lease = prior.value().copy()
    if (lease[].coef != state.coef or lease[].coef_count != state.coef_count
            or lease[].features != state.features or lease[].intercept != state.intercept
            or lease[].has_coef != state.has_coef):
        LINEAR_STATES.replace(native[].handle, state.copy())
        lease = LINEAR_STATES.acquire(native[].handle)
    native[].coefficients = coefficients
    return LinearStateCall(owner, lease^)


def linear_state_owner_pins(value: Int, index: Int) abi("C") -> Int:
    if not value or index < -1 or index > 1:
        return -1
    try:
        var owner = borrowed_native_owner(value)
        var native = owner.downcast_value_ptr[LinearStateOwner]()
        var result = 2
        if index == 0:
            result = Int(native[].coefficients._obj_ptr)
        elif index == 1:
            result = Int(native[].means._obj_ptr)
        _ = owner
        return result
    except:
        return -1


def register_linear_state_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[LinearStateOwner]()
    except error:
        if not py_type_absent(error, String(reflect[LinearStateOwner].name())):
            raise error
        _ = module.add_type[LinearStateOwner]("_LinearStateOwner")
        register_native_owner(lookup_py_type_object[LinearStateOwner](), module.module, linear_state_owner_pins)
        return
    py_type_attach(module.module, "_LinearStateOwner", existing)
    register_native_owner(existing, module.module, linear_state_owner_pins)
