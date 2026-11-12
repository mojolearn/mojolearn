# SPDX-License-Identifier: Apache-2.0
"""Boundary pins for PCA's native cached admission; no model state in Python containers."""
from std.python import PythonObject
from core.pca_query_state import PCAQueryState, PCAQueryIdentity


struct PCAQueryPins(Copyable, Movable):
    var identity: PCAQueryIdentity
    var pins: List[PythonObject]

    def __init__(out self, valid: Bool = True):
        self.identity = PCAQueryIdentity(valid)
        self.pins = List[PythonObject]()

    def append(mut self, value: PythonObject):
        self.identity.words.append(Int(value._obj_ptr))
        self.pins.append(value)

    def absent(mut self):
        self.identity.words.append(0)


@fieldwise_init
struct PCAQueryAdmission(Copyable, Movable):
    var fields: PythonObject
    var state: PCAQueryState


@fieldwise_init
struct PCAQueryCache(Movable, Writable):
    var state: PCAQueryState
    var key: PCAQueryPins
    var context_key: PCAQueryPins
    var generation: Int
    var mode: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PCAQueryCache(", self.state.features, ", ", self.state.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
