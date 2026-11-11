# SPDX-License-Identifier: Apache-2.0
"""Requested public score storage, or private native score/label storage."""
from std.python import Python, PythonObject
from core.ocsvm_plan import OcQueryState
from bindings.family_common import family_array
from bindings.classical_arrays import classical_address


struct OcOutput(Movable):
    var public_value: PythonObject
    var state: OcQueryState

    def __init__(out self, count: Int, labels: Bool, public_output: Bool, context: PythonObject) raises:
        self.public_value = family_array(Python.tuple(PythonObject(count)), "<f4", context) if public_output else PythonObject(None)
        self.state = OcQueryState(count, labels, not public_output)
        if public_output:
            self.state.external = Int(py=classical_address(self.public_value))
