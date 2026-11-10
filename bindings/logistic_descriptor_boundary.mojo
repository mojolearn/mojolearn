# SPDX-License-Identifier: Apache-2.0
"""Logistic descriptor admission and Python image lifetime pins only."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.native_descriptor_call import native_descriptor_call
from core.logistic_native_api import (
    LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, QN_NATIVE_MODE, QN_NATIVE_HOST,
    QN_NATIVE_MULTICLASS_AVAILABLE, QnNativeDriver, QnFitRequest, QnQueryRequest, QnLinkRequest,
    LOGISTIC_FOLD_ABI, QN_FOLD_COUNT, QnFoldDriver, QnFoldStore, QnFoldResult,
)
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


@fieldwise_init
struct LogisticConnection(Movable):
    var owner: PythonObject
    var driver: QnNativeDriver
    var mode: Int
    var host: Bool
    var multiclass: Bool


struct LogisticConnections(Defaultable, Movable):
    var records: List[LogisticConnection]

    def __init__(out self):
        self.records = List[LogisticConnection]()


comptime LOGISTIC_CONNECTIONS = SharedSlot[LogisticConnections, "MojolearnLogisticBoundaryConnectionsV1"]


def logistic_connection(entries: PythonObject) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(entries["logistic_native_api_v1"],
        String(LOGISTIC_NATIVE_ABI), QN_NATIVE_COUNT)
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    for i in range(len(cache[].records)):  # driver-loop: connected native extension images
        if cache[].records[i].owner is provider:
            return i
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var mode = native_descriptor_call(entry, QN_NATIVE_MODE, [], [], [])
    var host = native_descriptor_call(entry, QN_NATIVE_HOST, [], [], []) != 0
    var multiclass = native_descriptor_call(entry, QN_NATIVE_MULTICLASS_AVAILABLE, [], [], []) != 0
    var index = len(cache[].records)
    cache[].records.append(LogisticConnection(provider, QnNativeDriver(entry), mode, host, multiclass))
    return index


def logistic_compiled_mode(entries: PythonObject) raises -> Int:
    var index = logistic_connection(entries)
    return LOGISTIC_CONNECTIONS.get_or_create_ptr()[].records[index].mode


def logistic_multiclass_available(entries: PythonObject) raises -> Bool:
    var index = logistic_connection(entries)
    return LOGISTIC_CONNECTIONS.get_or_create_ptr()[].records[index].multiclass


def logistic_run_fit(entries: PythonObject, request: QnFitRequest) raises -> Int:
    var index = logistic_connection(entries)
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    var provider = cache[].records[index].owner
    var driver = cache[].records[index].driver.copy()
    var iterations = 0
    try:
        with GILReleased(Python()):
            iterations = driver.fit(request)
    finally:
        _ = provider
    return iterations


def logistic_run_query(entries: PythonObject, operation: Int, request: QnQueryRequest) raises:
    var index = logistic_connection(entries)
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    var provider = cache[].records[index].owner
    var driver = cache[].records[index].driver.copy()
    try:
        with GILReleased(Python()):
            driver.query(operation, request)
    finally:
        _ = provider


def logistic_run_link(entries: PythonObject, operation: Int, request: QnLinkRequest) raises:
    var index = logistic_connection(entries)
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    var provider = cache[].records[index].owner
    var driver = cache[].records[index].driver.copy()
    try:
        with GILReleased(Python()):
            driver.link(operation, request)
    finally:
        _ = provider


def logistic_run_log(entries: PythonObject, source: Int, dst: Int, count: Int) raises:
    var index = logistic_connection(entries)
    var cache = LOGISTIC_CONNECTIONS.get_or_create_ptr()
    var provider = cache[].records[index].owner
    var driver = cache[].records[index].driver.copy()
    try:
        with GILReleased(Python()):
            driver.logarithm(source, dst, count)
    finally:
        _ = provider


def logistic_run_fold_fit(factory: PythonObject, store: QnFoldStore, request: QnFitRequest) raises -> QnFoldResult:
    var provider = neural_descriptor_connect_factory_once(factory, String(LOGISTIC_FOLD_ABI), QN_FOLD_COUNT)
    var driver = QnFoldDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var words = List[Int32](length=2, fill=0)
    var iterations = 0
    try:
        with GILReleased(Python()):
            iterations = driver.fit(store, request, Int(words.unsafe_ptr()))
    finally:
        _ = provider
        _ = words
    return QnFoldResult(iterations, Int(words[0]), Int(words[1]))


def logistic_run_fold_query(factory: PythonObject, operation: Int, store: QnFoldStore, request: QnQueryRequest) raises -> QnFoldResult:
    var provider = neural_descriptor_connect_factory_once(factory, String(LOGISTIC_FOLD_ABI), QN_FOLD_COUNT)
    var driver = QnFoldDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    var words = List[Int32](length=2, fill=0)
    try:
        with GILReleased(Python()):
            driver.query(operation, store, request, Int(words.unsafe_ptr()))
    finally:
        _ = provider
        _ = words
    return QnFoldResult(0, Int(words[0]), Int(words[1]))
