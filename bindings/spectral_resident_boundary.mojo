# SPDX-License-Identifier: Apache-2.0
"""Image and public storage pins around the native Spectral resident owner."""
from std.memory import ArcPointer
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.spectral_native_api import SpectralPrepareRequest, SpectralResidentQueryRequest
from core.spectral_native_state import SpectralResidentState
from bindings.spectral_native_boundary import SpectralConnection, SpectralStatePins


@fieldwise_init
struct SpectralResidentLease(Copyable, Movable):
    var native: ArcPointer[SpectralResidentState]
    var provider: PythonObject
    var state: SpectralStatePins
    var training: PythonObject

    def __del__(deinit self):
        # Dict operations may drop this reference with the GIL held. Never
        # release the GIL implicitly during a metadata mutation. Explicit
        # retirement below releases a final Arc outside the table operation.
        _ = self.native^
        _ = self.provider
        _ = self.state^
        _ = self.training


def spectral_resident_retire(var lease: SpectralResidentLease):
    var provider = lease.provider
    var state = lease.state.copy()
    var training = lease.training
    var native = lease.native.copy()
    _ = lease^
    try:
        with GILReleased(Python()):
            _ = native^
    except:
        pass
    _ = provider
    _ = state
    _ = training


def spectral_resident_prepare(connection: SpectralConnection, state: SpectralStatePins,
                              training: PythonObject, request: SpectralPrepareRequest) raises -> SpectralResidentLease:
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = connection.driver.prepare(request)
        var owner = SpectralResidentState(connection.driver.copy(), handle)
        handle = -1
        var native = ArcPointer[SpectralResidentState](owner^)
        return SpectralResidentLease(native^, connection.provider, state.copy(), training)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                connection.driver.release(handle)
        _ = connection.provider
        _ = state
        _ = training


def spectral_resident_query(lease: SpectralResidentLease, request: SpectralResidentQueryRequest) raises:
    try:
        with GILReleased(Python()):
            _ = lease.native[].driver.query_resident(request)
    finally:
        _ = lease
