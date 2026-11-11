# SPDX-License-Identifier: Apache-2.0
"""Prepared query cache/pins and one direct native HDBSCAN query call.

This opaque record lives only in the existing weak model record table, never
in estimator fields or archives. Its Python values are boundary lifetime pins;
all operation/count/buffer/snapshot state is typed Mojo. It never retains its
model and therefore creates no owner-to-model reference cycle.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.hdbscan_query_api import (
    HDBSCAN_QUERY_ABI, HDBSCAN_QUERY_COUNT, HDBSCAN_QUERY_HOST,
    HdbscanQueryDriver, HdbscanQueryDimensions, HdbscanQueryRequest, HdbscanQueryPlan,
)
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


@fieldwise_init
struct HdbscanQueryConnection(Movable):
    var provider: PythonObject
    var driver: HdbscanQueryDriver
    var host: Bool


def hdbscan_query_connection(binding: PythonObject) raises -> HdbscanQueryConnection:
    var factory = neural_descriptor_entry(binding, "hdbscan_query_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(HDBSCAN_QUERY_ABI), HDBSCAN_QUERY_COUNT)
    var driver = HdbscanQueryDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var host = driver.metadata(HDBSCAN_QUERY_HOST)
    if host != 0 and host != 1:
        raise Error("HDBSCAN query provider: invalid host snapshot capability")
    return HdbscanQueryConnection(provider, driver^, host != 0)


@fieldwise_init
struct HdbscanQueryRecord(Movable, Writable):
    var binding: PythonObject
    var provider: PythonObject
    var driver: HdbscanQueryDriver
    var host: Bool
    var dimensions: HdbscanQueryDimensions
    var raw: PythonObject
    var core_distances: PythonObject
    var labels: PythonObject
    var parents: PythonObject
    var lambdas: PythonObject
    var deaths: PythonObject
    var clusters: PythonObject
    var children: PythonObject
    var indices: PythonObject
    var offsets: PythonObject

    def write_to(self, mut writer: Some[Writer]):
        writer.write("HdbscanQueryRecord")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def hdbscan_query_run(connection: HdbscanQueryConnection, request: HdbscanQueryRequest,
    operation: Int) raises:
    # Caller pins every borrowed input/output through this whole helper.
    # The constructor snapshots exactly the provider-specific inputs before
    # GIL release. Its native allocations die before caller cache publication.
    var plan = HdbscanQueryPlan(request, operation, connection.host)
    try:
        with GILReleased(Python()):
            _ = connection.driver.run(plan)
    finally:
        _ = connection.provider
        _ = plan
