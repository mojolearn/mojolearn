# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN half of `bindings/_mojolearn_core_host.mojo` (wave 8 run split): the
int-type address slot its `_index` reads, the host KMeans native table and its
slot, and `kmeans_fit_binding`'s host fit (the deferred finite test of X, then
the fit), which the door calls with the GIL released. Typed Mojo only: no
Python import."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.host_finite_scan import host_all_finite_f32
from core.hostptr import f32_ptr
from core.core_host_impl import kmeans_fit_host
from core.shared_slot import SharedSlot
from core.kmeans_native_api import KMEANS_NATIVE_ABI
from core.kmeans_native_host import kmeans_host_native_v1
from core.native_descriptor_abi import NativeDescriptorCall


struct _CoreHostIntType(Defaultable, Movable):
    # Address of CPython's int type (static, immortal), 0 until read.
    var address: Int

    def __init__(out self):
        self.address = 0


comptime _CORE_HOST_INT_TYPE = SharedSlot[_CoreHostIntType, "MojolearnCoreHostIntTypeV2"]


struct KMeansHostNativeTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KMEANS_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kmeans_host_native_v1)


comptime KMEANS_NATIVE_TABLE = SharedSlot[KMeansHostNativeTable, ("MojolearnKMeansHostNativeIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnKMeansHostNativeFastV1")]


def kmeans_fit_host_run(
    x_address: Int, centroids_address: Int, lp: MutPointer[UInt32, MutUntrackedOrigin],
    weights_address: Int, ns: Int, nf: Int, nc: Int, nw: Int, mi: Int, tl: Float64, sd: UInt64,
    ninit: Int, ii: Int, mm: Int, ovs: Float64, test: Bool,
    mut status: Int, mut inertia: Float64, mut n_iter: Int, mut sum_scale: Float64,
    mut weight_scale: Float64,
) raises:
    """`kmeans_fit_binding`'s host fit: with `test`, X's finite word first
    (`status` 1 skips the fit); then the fit's four returns."""
    if test and ns >= 1 and nf >= 1:
        # parallel: the deferred finite test of X (SIMD scan across the host's cores)
        status = 0 if host_all_finite_f32(f32_ptr(x_address), ns * nf) else 1
    if status == 0:
        var r = kmeans_fit_host(
            x_address, centroids_address, lp, weights_address,
            ns, nf, nc, nw, mi, tl, sd, ninit, ii, mm, ovs,
        )
        inertia = r[0]
        n_iter = r[1]
        sum_scale = r[2]
        weight_scale = r[3]
