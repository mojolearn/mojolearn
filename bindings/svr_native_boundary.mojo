# SPDX-License-Identifier: Apache-2.0
"""Exact SVR provider connection and boundary storage/image pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.svr_native_api import SVR_NATIVE_ABI, SVR_NATIVE_COUNT, SvrNativeDriver, SvrFitRequest, SvrFitPlan, SvrQueryRequest
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once


@fieldwise_init
struct SvrConnection(Movable):
    var provider: PythonObject
    var driver: SvrNativeDriver


def svr_connection(binding: PythonObject) raises -> SvrConnection:
    var factory = neural_descriptor_entry(binding, "svr_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(SVR_NATIVE_ABI), SVR_NATIVE_COUNT)
    return SvrConnection(provider, SvrNativeDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))


def svr_run_fit(connection: SvrConnection, request: SvrFitRequest, mut plan: SvrFitPlan) raises:
    try:
        with GILReleased(Python()):
            plan.count = connection.driver.fit(request)
    finally:
        _ = connection
        _ = plan


def svr_run_query(connection: SvrConnection, request: SvrQueryRequest, resident: Bool) raises:
    try:
        with GILReleased(Python()):
            _ = connection.driver.query(request, resident)
    finally:
        _ = connection
