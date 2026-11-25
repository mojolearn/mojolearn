# SPDX-License-Identifier: Apache-2.0
"""The GaussianMixture GPU image's resident store and its scope transport entry (lane vdf-scope-cluster2b,
2026-11-24).

A `DeviceStore` (core/device_store.mojo) on the mixture context (mixture/estimator.mojo `_binding_ctx`), so a slot
is a buffer the GMM kernels read directly. `mixture_store_v1` is the image's transport entry with the x_prep
numbering (core/family_store_transport.mojo): the `resident()` scope keeps a model-selection X here ONCE
(`arena_scope_store_put`, prefix `MIXTURE_STORE_PREFIX`) and each fold's rows are taken on the device
(`arena_store_take_rows`). The binding exports it as the capsule `mixture_store_v1`; the member route carries it
in `route.second` (providers/python/member_mixture_boundary_door.mojo) when the image answers
`MIXTURE_CAP_DEV_STORE`. The fit and the scores read a slot through `MIXTURE_NATIVE_FIT_STORE` /
`MIXTURE_NATIVE_SCORE_STORE` of the image's descriptor (mixture/gmm_native_device.mojo).

GPU image only: the host images keep their host path and never answer MIXTURE_CAP_DEV_STORE.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_store import DeviceStore
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.family_store_transport import family_store_entry
from core.native_descriptor_call import native_descriptor_report
from x_prep.transport_abi import XP_TRANSPORT_ABI
from mixture.estimator import _binding_ctx

#: The scope cache's family prefix of this store.
comptime MIXTURE_STORE_PREFIX = "mixture"

comptime MIXTURE_STORE = SharedSlot[DeviceStore, ("MojolearnMixtureStoreIdenticalV2" if GLOBAL_NUMERIC_MODE == 1
                                                  else "MojolearnMixtureStoreFastV2"), 2]


def mixture_store_ctx() raises -> DeviceContext:
    """The context every slot of the store is made on: the GMM kernels' own."""
    return _binding_ctx()


def mixture_store_buffer(slot: Int, words: Int) raises -> DeviceBuffer[DType.float32]:
    """A reference to slot `slot`'s buffer (holding at least `words` words), copied out under the store's lock: it
    keeps the device memory alive past a concurrent free while the caller's kernels read it."""
    return MIXTURE_STORE.get_or_create_ptr()[]._held(slot, words)


def mixture_store_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
                     context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The store's transport entry (ARENA_STORE_PUT / FREE / TAKE_ROWS / LIVE)."""
    try:
        var ctx = mixture_store_ctx()
        return family_store_entry(MIXTURE_STORE.get_or_create_ptr()[], ctx, "mixture", operation, aa, na, pp, np,
                                  nf, result, context, sink)
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct MixtureStoreAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(XP_TRANSPORT_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(mixture_store_v1)


comptime MIXTURE_STORE_API = SharedSlot[MixtureStoreAPI, ("MojolearnMixtureStoreAPIIdenticalV1"
                                                          if GLOBAL_NUMERIC_MODE == 1
                                                          else "MojolearnMixtureStoreAPIFastV1")]
