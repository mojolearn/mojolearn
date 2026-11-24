# SPDX-License-Identifier: Apache-2.0
"""TSVD fold fitting keeps the base image's existing resident store and kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.shared_slot import SharedSlot
from core.msel_device_x import msel_fold_entry_context
from core.pca_native_api import pca_vendor_code
from core.pca_native_fold import _fold_pointer
from decomposition.pca_fold import tsvd_fit_fold, tsvd_transform_fold, tsvd_transform_fold_resident, inverse_transform_fold
from core.fitted_resident_store import (
    FittedResidentStore, fitted_lease_open, fitted_lease_close, fitted_resident_view,
)
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_CAP_FOLD, TSVD_CAP_RESIDENT, TSVD_FOLD,
    TSVDNativeKernels, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest, tsvd_native_dispatch,
)


#: The fold entry's own resident fitted store (lane vdf-scope-neural5): a lease's device copy of `components_`
#: lives on THIS image's fold-entry context (`msel_fold_entry_context`), so it is never the estimators image's
#: store (decomposition/tsvd_resident_state.mojo, another context). One store per numeric tier.
comptime TSVD_FOLD_RESIDENT_STORE = SharedSlot[FittedResidentStore, (
    "MojolearnTsvdFoldResidentIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    else "MojolearnTsvdFoldResidentFastV1")]


def _fold_resident_store() raises -> UnsafePointer[FittedResidentStore, MutAnyOrigin]:
    return TSVD_FOLD_RESIDENT_STORE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct TSVDFoldKernels(TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return TSVD_CAP_FOLD | TSVD_CAP_RESIDENT

    @staticmethod
    def qfix() -> Int:
        return 0

    @staticmethod
    def fit(r: TSVDFitRequest) raises -> TSVDFitResult:
        if r.route != TSVD_FOLD or not r.fold.active():
            raise Error("TruncatedSVD fold provider requires an exact resident fold fit")
        var cp = _fold_pointer(r.components)
        var sp = _fold_pointer(r.singular)
        var vp = _fold_pointer(r.variance)
        var rp = _fold_pointer(r.ratio)
        var ctx = msel_fold_entry_context()
        var words = tsvd_fit_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
            cp, sp, vp, rp, r.rows, r.features, r.count)
        return TSVDFitResult(Int(words[0]), Int(words[1]))

    @staticmethod
    def query(r: TSVDQueryRequest) raises -> Int:
        """TruncatedSVD.transform of a resident fold (lane vdf-scope-neural4): `r.fold` names the fold; the
        components are read from the lease's device copy (`r.resident`, `r.lease` > 0: uploaded at the lease's
        first query and again only when the host words' address or size changed, lane vdf-scope-neural5), else
        uploaded for this call. Returns the finite word, or 2 when an index is out of range (the member's
        `door_rows_refuse` reads both)."""
        if not r.fold.active():
            raise Error("TruncatedSVD fold provider requires a resident fold")
        if r.resident and r.lease < 0:
            raise Error("tsvd_transform_resident: the lease must be 0 or a lease, got " + String(r.lease))
        var cp = _fold_pointer(r.components)
        var dst = _fold_pointer(r.dst)
        var ctx = msel_fold_entry_context()
        if r.inverse:
            # Lane vdf-scope-neural5: inverse_transform of a fold of scores (no mean: the host entry's
            # `inverse_transform_host(..., False)` body over the gathered scores), the lease's copy or this
            # call's upload of the components. Returns 2 for an index out of range, else 0 (no finite test,
            # as the host inverse).
            var in_range: Int32
            if r.resident and r.lease > 0:
                var view = fitted_resident_view(_fold_resident_store(), ctx, r.lease, 0, [r.components],
                                                [r.count * r.features])
                var kept = view.bufs[0].copy()
                in_range = inverse_transform_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                    kept, cp, dst, r.rows, r.features, r.count, False)
                _ = kept^
                _ = view^
            else:
                var comps = ctx.enqueue_create_buffer[DType.float32](r.count * r.features)
                ctx.enqueue_copy(dst_buf=comps, src_ptr=cp)
                in_range = inverse_transform_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                    comps, cp, dst, r.rows, r.features, r.count, False)
                _ = comps^
            return 2 if in_range != 0 else 0
        var words: Tuple[Int32, Int32]
        if r.resident and r.lease > 0:
            var view = fitted_resident_view(_fold_resident_store(), ctx, r.lease, 0, [r.components],
                                            [r.count * r.features])
            var components = view.bufs[0].copy()
            words = tsvd_transform_fold_resident(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                components, dst, r.rows, r.features, r.count)
            _ = components^
            _ = view^
        else:
            words = tsvd_transform_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
                cp, dst, r.rows, r.features, r.count)
        if words[1] != 0:
            return 2
        return Int(words[0])

    @staticmethod
    def open() raises -> Int:
        """A lease of the fold entry's resident store (the member keeps one per fitted state)."""
        return fitted_lease_open(_fold_resident_store())

    @staticmethod
    def close(lease: Int) raises:
        fitted_lease_close(_fold_resident_store(), lease)


def tsvd_fold_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return tsvd_native_dispatch[TSVDFoldKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct TSVDFoldAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TSVD_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsvd_fold_native_v1)


comptime TSVD_FOLD_SLOT = "MojolearnTSVDFoldIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnTSVDFoldFastV1"
comptime TSVD_FOLD_API = SharedSlot[TSVDFoldAPI, TSVD_FOLD_SLOT]
