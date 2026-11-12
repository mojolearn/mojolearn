# SPDX-License-Identifier: Apache-2.0
"""Explicit GP normalization boundary: two image pins and admitted buffers."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.os import getenv
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type, py_get
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.scaler_native_api import SCALER_NATIVE_ABI, PREP_PROGRAM_ABI, ScalerDriver, PrepProgramDriver
from core.gp_native_normalization import GPRNormalizeRequest, GPRNormalizeResult, gp_normalize_targets


struct GPNormalizationConnection(Movable):
    var scaler_provider: PythonObject
    var program_provider: PythonObject
    var scaler: ScalerDriver
    var program: PrepProgramDriver
    var direct: Bool

    def __init__(out self, entries: PythonObject, expected: Int) raises:
        if py_type(entries) is native_loaded_module("builtins").str:
            raise Error("ImportError: " + String(entries))
        var scaler_module = py_get(entries, "native_scaler_provider")
        if scaler_module is None:
            raise Error("ImportError: HostStandardScaler training requires _mojolearn_preprocessing_host: " + String(py_get(entries, "scaler_training_error", "standard_fit is unavailable")))
        self.scaler_provider = neural_descriptor_connect_once(scaler_module, "scaler_native_api_v1", String(SCALER_NATIVE_ABI), 4)
        self.scaler = ScalerDriver(self.scaler_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        self.program_provider = neural_descriptor_connect_once(entries["native_program_provider"], "x_prep_native_api_v1", String(PREP_PROGRAM_ABI), 2)
        self.program = PrepProgramDriver(self.program_provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        if self.scaler.metadata(0) != expected or self.program.mode() != expected:
            raise Error("RuntimeError: Scaler native numeric mode disagrees with requested mode")
        self.direct = self.scaler.metadata(1) != 0 and String(getenv("MOJOLEARN_SCALER_DIRECT", "1").strip()) != "0"


def gp_normalization_run(connection: GPNormalizationConnection, request: GPRNormalizeRequest) raises -> GPRNormalizeResult:
    try:
        with GILReleased(Python()):
            return gp_normalize_targets(connection.scaler, connection.program, request)
    finally:
        _ = connection
