# SPDX-License-Identifier: Apache-2.0
"""Boundary image/storage pins around the Python-free native resident Arc."""
from std.memory import ArcPointer
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.kmeans_variants_api import KMeansVariantsDriver
from core.kmeans_variants_resident_state import VariantsResidentState


@fieldwise_init
struct VariantsResidentLease(Copyable, Movable):
    var native: ArcPointer[VariantsResidentState]
    var provider: PythonObject
    var centers: PythonObject

    def __del__(deinit self):
        # A Dict may copy/move/drop an ordinary value while updating metadata.
        # Never release the GIL from that implicit destruction. Explicit
        # cache/query retirement below releases it outside the Dict operation.
        # The image remains pinned through a last-reference error-path drop.
        _ = self.native^
        _ = self.provider
        _ = self.centers


def variants_resident_retire(var lease: VariantsResidentLease):
    var provider = lease.provider
    var centers = lease.centers
    var native = lease.native.copy()
    _ = lease^  # Cannot be the final native reference: `native` retains it.
    try:
        with GILReleased(Python()):
            _ = native^
    except:
        pass
    _ = provider
    _ = centers


def variants_resident_prepare(driver: KMeansVariantsDriver, provider: PythonObject,
                              centers: PythonObject, address: Int,
                              clusters: Int, features: Int) raises -> VariantsResidentLease:
    var handle = -1
    try:
        with GILReleased(Python()):
            handle = driver.prepare(address, clusters, features)
        var state = VariantsResidentState(driver.copy(), handle)
        handle = -1  # The native state now owns error-path cleanup as well.
        var lease = ArcPointer[VariantsResidentState](state^)
        return VariantsResidentLease(lease^, provider, centers)
    finally:
        if handle >= 0:
            with GILReleased(Python()):
                driver.release(handle)
        _ = provider
        _ = centers
