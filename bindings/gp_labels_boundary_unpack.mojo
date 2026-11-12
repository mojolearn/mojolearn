# SPDX-License-Identifier: Apache-2.0
"""GP label provider image pins around the existing typed label descriptor."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_once
from core.labels_native_api import LABELS_NATIVE_ABI_NAME, LABELS_NATIVE_FACTORY, LABELS_NATIVE_OPERATION_COUNT, LabelsNativeDriver, LabelsPrepareRequest, LabelsGatherRequest


struct GPLabelsConnection(Movable):
    var provider: PythonObject
    var driver: LabelsNativeDriver

    def __init__(out self, base: PythonObject) raises:
        self.provider = neural_descriptor_connect_once(base, LABELS_NATIVE_FACTORY, String(LABELS_NATIVE_ABI_NAME), LABELS_NATIVE_OPERATION_COUNT)
        self.driver = LabelsNativeDriver(self.provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)


def gp_labels_prepare(connection: GPLabelsConnection, request: LabelsPrepareRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return connection.driver.prepare(request)
    finally:
        _ = connection


def gp_labels_gather(connection: GPLabelsConnection, kind: Int, request: LabelsGatherRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.gather(kind, request)
    finally:
        _ = connection
