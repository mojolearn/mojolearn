# SPDX-License-Identifier: Apache-2.0
"""Public GP prediction pins or private native score storage at UNPACK/PACK."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.os import getenv
from core.gp_native_score import GPRScoreValues, gpr_score_prepare, gp_r2_value
from bindings.arena_prog_native import _family_entries
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT
from bindings.family_common import family_array
from bindings.classical_arrays import classical_address


struct GPRPredictionRoom(Movable):
    var rows: Int
    var mean: PythonObject
    var stddev: PythonObject
    var score: Optional[GPRScoreValues]

    def __init__(out self, rows: Int, private_mean: Bool, context: PythonObject) raises:
        self.rows = rows
        self.mean = PythonObject(None)
        self.stddev = PythonObject(None)
        self.score = None
        if private_mean:
            self.score = GPRScoreValues(rows)
        else:
            self.mean = family_array(Python.tuple(rows), "<f4", context)

    def mean_address(self) raises -> Int:
        if self.score:
            return self.score.value().prediction_address()
        return Int(py=classical_address(self.mean))


def gpr_score_result(values: GPRScoreValues, target: PythonObject,
                     context: PythonObject) raises -> Float64:
    var target_address = Int(py=classical_address(target))
    # Finite/copy refusal precedes the metrics-image selection, as in _Reg.
    var program = gpr_score_prepare(values, target_address)
    var prog_context = context["family_metrics_prog_context"]
    var entries = _family_entries(PythonObject(None), "x_metrics", prog_context)
    var provider = neural_descriptor_connect_factory_once(entries["metrics_program_native_api_v1"],
        String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var ranges = String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0"
    var result = Float64(0.0)
    try:
        with GILReleased(Python()):
            result = gp_r2_value(program.run(entry, ranges))
    finally:
        _ = provider
        _ = entries
        _ = values
        _ = target
        _ = program
    return result
