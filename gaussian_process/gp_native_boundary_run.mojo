# SPDX-License-Identifier: Apache-2.0
"""The typed GP connection (RUN): the resolved native GP table entry, its
driver, the published key's tier and the capability word, and the typed
driver call for each request. The door (`bindings/gp_native_boundary_door.mojo`)
reads the key from the connected module, pins the module and releases the GIL
around `gp_native_run`."""
from core.entry_table import EntryRef
from core.gp_entry_table import GPEntryKey, gp_table_resolve
from core.gp_native_api import (
    GP_CAPABILITIES, GPDriver,
    GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest,
    GPRGradientRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest,
    GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GPThetaRequest,
    GPQueryResult,
)


struct GPNative(Movable):
    # The mode is the published key's tier (the image's GLOBAL_NUMERIC_MODE,
    # the word GP_MODE answered); the capability word is one native metadata
    # call through the resolved entry.
    var entry: EntryRef
    var driver: GPDriver
    var mode: Int
    var capabilities: Int

    def __init__(out self, key: GPEntryKey) raises:
        var entry = gp_table_resolve(key)
        var driver = GPDriver(entry.call)
        var capabilities = driver.metadata(GP_CAPABILITIES)
        self.entry = entry
        self.driver = driver^
        self.mode = key.mode
        self.capabilities = capabilities


def gp_require(native: GPNative, capability: Int, entry: String) raises:
    if (native.capabilities & capability) == 0:
        raise Error("ImportError: the connected GP provider does not support " + entry)


def gp_native_run(native: GPNative, request: GPRFitRequest) raises -> Int:
    return native.driver.gpr_fit(request)


def gp_native_run(native: GPNative, request: GPRQueryRequest) raises -> GPQueryResult:
    return native.driver.gpr_query(request)


def gp_native_run(native: GPNative, request: GPRCovarianceRequest) raises -> Int:
    return native.driver.gpr_covariance(request)


def gp_native_run(native: GPNative, request: GPRSampleRequest) raises -> Int:
    return native.driver.gpr_sample(request)


def gp_native_run(native: GPNative, request: GPRGradientRequest) raises -> Int:
    return native.driver.gpr_gradient(request)


def gp_native_run(native: GPNative, request: GPROptimizeRequest) raises -> Int:
    return native.driver.gpr_optimize(request)


def gp_native_run(native: GPNative, request: GPCFitRequest) raises -> Int:
    return native.driver.gpc_fit(request)


def gp_native_run(native: GPNative, request: GPCFitAllRequest) raises -> Int:
    return native.driver.gpc_fit_all(request)


def gp_native_run(native: GPNative, request: GPCQueryRequest) raises -> Int:
    return native.driver.gpc_query(request)


def gp_native_run(native: GPNative, request: GPCQueryAllRequest) raises -> GPQueryResult:
    return native.driver.gpc_query_all(request)


def gp_native_run(native: GPNative, request: GPCCombineRequest) raises -> Int:
    return native.driver.gpc_combine(request)


def gp_native_run(native: GPNative, request: GPThetaRequest) raises -> Int:
    return native.driver.theta(request)
