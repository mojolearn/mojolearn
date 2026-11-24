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
    HDBSCAN_QUERY_ABI, HDBSCAN_QUERY_COUNT, HDBSCAN_QUERY_HOST, HDBSCAN_QUERY_PREPARE,
    HdbscanQueryDriver, HdbscanQueryDimensions, HdbscanQueryRequest, HdbscanQueryPlan,
)
from providers.python.neural_descriptor_pyapi_door import (
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
    var driver = HdbscanQueryDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
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
    #: Lane vdf-misc4: the device image's fitted snapshot this record owns
    #: (0: none, the host image). Released when the record dies: a refit
    #: drops the model's query records, so does a binding change or the
    #: model's end.
    var handle: Int

    def __del__(deinit self):
        if self.handle != 0:
            try:
                self.driver.release(self.handle)
            except:
                pass
        _ = self.provider

    def write_to(self, mut writer: Some[Writer]):
        writer.write("HdbscanQueryRecord")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def hdbscan_query_prepare(connection: HdbscanQueryConnection, request: HdbscanQueryRequest,
    operation: Int) raises -> Int:
    """Lane vdf-misc4: the fitted arrays of `request` taken ONCE onto the
    device image's device for query `operation`; the snapshot's handle (0 on
    the host image, which keeps its per-call route)."""
    if connection.host:
        return 0
    var plan = HdbscanQueryPlan(request, HDBSCAN_QUERY_PREPARE, False, 0, operation)
    var handle = 0
    try:
        with GILReleased(Python()):
            handle = connection.driver.run(plan)
    finally:
        _ = connection.provider
        _ = plan
    return handle


def hdbscan_query_run(connection: HdbscanQueryConnection, request: HdbscanQueryRequest,
    operation: Int, handle: Int = 0) raises:
    # Caller pins every borrowed input/output through this whole helper.
    # The constructor snapshots exactly the provider-specific inputs before
    # GIL release. Its native allocations die before caller cache publication.
    # With a `handle` (lane vdf-misc4) nothing fitted is copied: the device
    # query reads the snapshot.
    var plan = HdbscanQueryPlan(request, operation, connection.host, handle)
    try:
        with GILReleased(Python()):
            _ = connection.driver.run(plan)
    finally:
        _ = connection.provider
        _ = plan
