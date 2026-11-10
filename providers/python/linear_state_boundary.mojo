# SPDX-License-Identifier: Apache-2.0
"""Lifetime pins for typed linear model records at the Python boundary.

The shared weak model-record table owns this envelope. RUN receives a copy
of its native state, and cannot access the Python owners retained here.
Public fitted buffers remain writable; each operation reconciles their
admitted descriptors before taking its immutable native snapshot.
"""
from std.python import PythonObject
from bindings.classical_arrays import classical_address
from bindings.native_model_records import records_get, records_put, records_drop, RECORDS_PREPARED
from core.linear_model_state import LinearModelState

comptime _LINEAR_STATE_SLOT = "linear_model_state_v1"


struct LinearStateOwner(Movable, Writable):
    var state: LinearModelState
    var coefficients: PythonObject
    var means: PythonObject

    def __init__(out self, state: LinearModelState, coefficients: PythonObject, means: PythonObject):
        self.state = state.copy()
        self.coefficients = coefficients
        self.means = means

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LinearStateOwner(features=", self.state.features, ")")

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
