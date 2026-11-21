# SPDX-License-Identifier: Apache-2.0
"""Logistic descriptor admission and Python image lifetime pins only (door).

The typed connection records are in `glm/logistic_descriptor_boundary_run.mojo`."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.shared_slot import SharedSlot
from core.logistic_native_api import (
    LOGISTIC_NATIVE_ABI, QN_NATIVE_COUNT, QnFitRequest, QnQueryRequest,
    LOGISTIC_FOLD_ABI, QN_FOLD_COUNT, QnFoldDriver, QnFoldStore, QnFoldResult,
)
from glm.logistic_descriptor_boundary_run import (
    logistic_connection_index, logistic_connection_add, logistic_driver,
)
from bindings.py_provider_pins import ProviderPins, provider_word
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from bindings.logistic_keys_unpack import logistic_keys
from bindings.py_balanced import py_item


comptime LOGISTIC_PROVIDER_PINS = SharedSlot[ProviderPins, "MojolearnLogisticProviderPinsV1"]


def logistic_connection(entries: PythonObject) raises -> Int:
    var nk = logistic_keys()
    var provider = neural_descriptor_connect_factory_once(py_item(entries, nk[].k_logistic_native_api_v1),
        String(LOGISTIC_NATIVE_ABI), QN_NATIVE_COUNT)
    var index = logistic_connection_index(provider_word(provider))
    if index >= 0:
        return index
    # The typed record and the door pin are appended together, under the GIL.
    index = logistic_connection_add(provider_word(provider),
        provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers.append(provider)
    return index


# TOMBSTONE: `logistic_compiled_mode`, `logistic_multiclass_available`, `logistic_run_fit` and `logistic_run_link`
# (the by-entries forms LogisticRegression's door called per public operation: each admitted the provider and
# scanned the connection list again; the estimator's route owner resolves the connection once and calls its
# driver, providers/python/logistic_route_boundary_door.mojo; `logistic_run_link` had no caller) deleted
# 2026-11-20 by lane dev2-logistic-device; code recoverable at 5c252f1573. `logistic_run_query` and
# `logistic_run_log` stay: the explicit host inference owner (bindings/host_logistic_native_door.mojo) calls them.


def logistic_run_query(entries: PythonObject, operation: Int, request: QnQueryRequest) raises:
    var index = logistic_connection(entries)
    var provider = LOGISTIC_PROVIDER_PINS.get_or_create_ptr()[].providers[index]
    var driver = logistic_driver(index)
    try:
        with GILReleased(Python()):
            driver.query(operation, request)
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
    var driver = QnFoldDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
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
    var driver = QnFoldDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    var words = List[Int32](length=2, fill=0)
    try:
        with GILReleased(Python()):
            driver.query(operation, store, request, Int(words.unsafe_ptr()))
    finally:
        _ = provider
        _ = words
    return QnFoldResult(0, Int(words[0]), Int(words[1]))
