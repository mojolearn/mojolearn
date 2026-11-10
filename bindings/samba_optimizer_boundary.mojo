# SPDX-License-Identifier: Apache-2.0
"""Samba optimizer descriptor connections and boundary lifetime pins."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.causal_lm_public import _native_function
from bindings.optimizer_owner_boundary import OptimizerOwner
from core.native_descriptor_abi import PRIMITIVE_ABI_NAME
from core.samba_optimizer_plan import SambaOptimizerPlan, samba_optimizer_run
from training.optimizer_native_api import OPTIMIZER_NATIVE_ABI, OPT_NATIVE_COUNT
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once


def samba_optimizer_provider(entries: PythonObject) raises -> PythonObject:
    return neural_descriptor_connect_factory_once(_native_function(entries, "optimizer_native_api_v1"),
        String(OPTIMIZER_NATIVE_ABI), OPT_NATIVE_COUNT)


def samba_optimizer_primitives(entries: PythonObject) raises -> PythonObject:
    # This is the existing packed optimizer endpoint. It does not change the
    # independently selected Samba model/session path or add tensor packing.
    return neural_descriptor_connect_factory_once(_native_function(entries, "neural_primitives_api_v1"),
        String(PRIMITIVE_ABI_NAME), 29)


def samba_optimizer_execute(provider: PythonObject, operation: Int, addresses: List[Int],
    plan: SambaOptimizerPlan, handle: Int, io: Int) raises -> Int:
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var result = 0
    try:
        with GILReleased(Python()):
            result = samba_optimizer_run(entry, operation, addresses, plan, handle, io)
    finally:
        _ = provider
        _ = addresses
    return result


def samba_optimizer_transfer(record: OptimizerOwner, first: Int, second: Int,
    count: Int, upload: Bool) raises:
    if record.control.handle < 0 or not record.control.driver:
        raise Error("RuntimeError: Samba optimizer transfer requires its native resident owner")
    var driver = record.control.driver.value().copy()
    var handle = record.control.handle
    try:
        with GILReleased(Python()):
            if upload:
                _ = driver.upload(handle, first, second, count)
            else:
                _ = driver.download(handle, first, second, count)
    finally:
        _ = record.provider
