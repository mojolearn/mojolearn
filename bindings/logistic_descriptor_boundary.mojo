# SPDX-License-Identifier: Apache-2.0
"""Logistic descriptor admission and Python image lifetime pins only (door).

The typed connection records are in `glm/logistic_descriptor_boundary_run.mojo`."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.logistic_native_api import (
    LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, QnFitRequest, QnQueryRequest, QnLinkRequest,
    LOGISTIC_FOLD_ABI, QN_FOLD_COUNT, QnFoldDriver, QnFoldStore, QnFoldResult,
)
from glm.logistic_descriptor_boundary_run import (
    logistic_connection_index, logistic_connection_add, logistic_mode, logistic_multiclass, logistic_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


comptime LOGISTIC_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnLogisticProviderPinsV1"]


def logistic_connection(entries: PythonObject) raises -> Int:
    var provider = neural_descriptor_connect_factory_once(entries["logistic_native_api_v1"],
        String(LOGISTIC_NATIVE_ABI), QN_NATIVE_COUNT)
    var index = logistic_connection_index(provider_word(provider))
    if index >= 0:
        return index
    # The typed record and the door pin are appended together, under the GIL.
    index = logistic_connection_add(provider_word(provider),
        provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry)
    LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    return index


def logistic_compiled_mode(entries: PythonObject) raises -> Int:
    return logistic_mode(logistic_connection(entries))


def logistic_multiclass_available(entries: PythonObject) raises -> Bool:
    return logistic_multiclass(logistic_connection(entries))


def logistic_run_fit(entries: PythonObject, request: QnFitRequest) raises -> Int:
    var index = logistic_connection(entries)
    var provider = LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers[index]
    var driver = logistic_driver(index)
    var iterations = 0
    try:
        with GILReleased(Python()):
            iterations = driver.fit(request)
    finally:
        _ = provider
    return iterations


def logistic_run_query(entries: PythonObject, operation: Int, request: QnQueryRequest) raises:
    var index = logistic_connection(entries)
    var provider = LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers[index]
    var driver = logistic_driver(index)
    try:
        with GILReleased(Python()):
            driver.query(operation, request)
    finally:
        _ = provider


def logistic_run_link(entries: PythonObject, operation: Int, request: QnLinkRequest) raises:
    var index = logistic_connection(entries)
    var provider = LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers[index]
    var driver = logistic_driver(index)
    try:
        with GILReleased(Python()):
            driver.link(operation, request)
    finally:
        _ = provider


def logistic_run_log(entries: PythonObject, source: Int, dst: Int, count: Int) raises:
    var index = logistic_connection(entries)
    var provider = LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers[index]
    var driver = logistic_driver(index)
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
