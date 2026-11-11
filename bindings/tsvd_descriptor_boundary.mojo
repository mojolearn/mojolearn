# SPDX-License-Identifier: Apache-2.0
"""Native TSVD provider selection and compiled image pins at the boundary."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.pca_native_api import pca_vendor_name
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_NATIVE_COUNT, TSVD_MODE, TSVD_VENDOR, TSVD_CAPABILITIES, TSVD_QFIX,
    TSVDNativeDriver, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest,
)
from bindings.py_balanced import py_call, py_type, py_get
from bindings.native_class_admission import dict_keys_admitted, dict_keys_admit
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


@fieldwise_init
struct TSVDConnection(Movable):
    var provider: PythonObject
    var driver: TSVDNativeDriver
    var mode: Int
    var vendor: Int
    var capabilities: Int
    var qfix: Int


struct TSVDConnections(Defaultable, Movable):
    var records: List[TSVDConnection]

    def __init__(out self):
        self.records = List[TSVDConnection]()


comptime TSVD_CONNECTIONS = SharedSlot[TSVDConnections, "MojolearnTSVDConnectionsV1"]


def tsvd_factory(context: PythonObject, table_name: String, mode: String) raises -> PythonObject:
    var table = context[PythonObject(table_name)]
    if py_type(table) is not py_type(Python.dict()):
        raise Error("TypeError: TruncatedSVD binding tables require native dictionaries")
    if not dict_keys_admitted(table):
        var keys = py_call(py_type(Python.list()), table)
        for index in range(len(keys)):  # driver-loop: fixed connected mode names before keyed lookup
            if py_type(keys[index]) is not py_type(PythonObject("")):
                raise Error("TypeError: TruncatedSVD binding tables require native string keys")
        dict_keys_admit(table)
    return table[PythonObject(mode)]


def tsvd_connect(factory: PythonObject, mode: Int, vendor: String) raises -> Int:
    if py_type(factory) is py_type(PythonObject("")):
        raise Error(String(factory))
    var provider = neural_descriptor_connect_factory_once(factory, String(TSVD_NATIVE_ABI), TSVD_NATIVE_COUNT)
    var cache = TSVD_CONNECTIONS.get_or_create_ptr()
    var index = -1
    for candidate in range(len(cache[].records)):  # driver-loop: connected native provider images
        if cache[].records[candidate].provider is provider:
            index = candidate
            break
    if index < 0:
        var driver = TSVDNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
        var native_mode = driver.metadata(TSVD_MODE)
        var native_vendor = driver.metadata(TSVD_VENDOR)
        var capabilities = driver.metadata(TSVD_CAPABILITIES)
        var qfix = driver.metadata(TSVD_QFIX)
        index = len(cache[].records)
        cache[].records.append(TSVDConnection(provider, driver^, native_mode, native_vendor, capabilities, qfix))
    if pca_vendor_name(cache[].records[index].vendor) != vendor:
        raise Error("RuntimeError: TruncatedSVD native vendor differs from selected execution")
    if cache[].records[index].mode != mode:
        raise Error("RuntimeError: TruncatedSVD native mode differs from selected execution")
    return index


def tsvd_connection(mode: String, context: PythonObject) raises -> Int:
    return tsvd_connect(tsvd_factory(context, "tsvd_bindings", mode),
        1 if mode == "identical" else 0, String(context["vendor"]))


def tsvd_capabilities(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].capabilities


def tsvd_qfix(connection: Int) raises -> Int:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].qfix


def tsvd_provider(connection: Int) raises -> PythonObject:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].provider


def tsvd_driver(connection: Int) raises -> TSVDNativeDriver:
    return TSVD_CONNECTIONS.get_or_create_ptr()[].records[connection].driver.copy()


def tsvd_run_fit(connection: Int, request: TSVDFitRequest) raises -> TSVDFitResult:
    var pin = tsvd_provider(connection)
    var driver = tsvd_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.fit(request)
    finally:
        _ = pin


def tsvd_run_query(connection: Int, request: TSVDQueryRequest) raises -> Int:
    var pin = tsvd_provider(connection)
    var driver = tsvd_driver(connection)
    try:
        with GILReleased(Python()):
            return driver.query(request)
    finally:
        _ = pin
