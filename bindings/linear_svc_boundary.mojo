# SPDX-License-Identifier: Apache-2.0
"""Exact compiled provider admission and lifetime pins for typed LinearSVC."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.linear_svc_native_api import (
    LINEAR_SVC_NATIVE_ABI, LINEAR_SVC_NATIVE_COUNT, LinearSvcDriver,
    LinearSvcFitRequest, LinearSvcFitPlan, LinearSvcWeightsState,
)
from core.logistic_native_api import QnQueryRequest, QnFitRequest
from core.qn_single_fit import QnSingleFitPlan
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)


@fieldwise_init
struct LinearSvcConnection(Movable):
    var provider: PythonObject
    var driver: LinearSvcDriver


@fieldwise_init
struct LinearSvcWeights(Movable):
    # Python is retained only for admitted input storage. Generated ones,
    # class rows and multiplication output live in native lists.
    var input_pin: PythonObject
    var state: LinearSvcWeightsState


def linear_svc_connection(binding: PythonObject) raises -> LinearSvcConnection:
    var factory = neural_descriptor_entry(binding, "linear_svc_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory,
        String(LINEAR_SVC_NATIVE_ABI), LINEAR_SVC_NATIVE_COUNT)
    return LinearSvcConnection(provider,
        LinearSvcDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry))


def linear_svc_run_fit(connection: LinearSvcConnection, request: LinearSvcFitRequest,
    mut plan: LinearSvcFitPlan) raises:
    try:
        with GILReleased(Python()):
            connection.driver.fit(request, plan)
    finally:
        _ = connection
        _ = plan


def linear_svc_run_query(connection: LinearSvcConnection, operation: Int,
    request: QnQueryRequest) raises:
    try:
        with GILReleased(Python()):
            connection.driver.qn().query(operation, request)
    finally:
        _ = connection


@fieldwise_init
struct LinearSvrFitResult(Movable):
    # Fitted public storage is retained at PACK; metadata is native.
    var weights: PythonObject
    var state: QnSingleFitPlan


def linear_svr_run_fit(connection: LinearSvcConnection, request: QnFitRequest,
    mut plan: QnSingleFitPlan) raises:
    try:
        with GILReleased(Python()):
            plan.run(connection.driver.qn(), request)
    finally:
        _ = connection
        _ = plan


@fieldwise_init
struct LinearSvcPrediction(Movable):
    var classes: PythonObject
    var codes: List[Int64]
