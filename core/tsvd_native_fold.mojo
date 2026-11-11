# SPDX-License-Identifier: Apache-2.0
"""TSVD fold fitting keeps the base image's existing resident store and kernels."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.shared_slot import SharedSlot
from core.msel_device_x import msel_fold_entry_context
from core.pca_native_api import pca_vendor_code
from core.pca_native_fold import _fold_pointer
from decomposition.pca_fold import tsvd_fit_fold
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_CAP_FOLD, TSVD_FOLD,
    TSVDNativeKernels, TSVDFitRequest, TSVDFitResult, TSVDQueryRequest, tsvd_native_dispatch,
)


struct TSVDFoldKernels(TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return TSVD_CAP_FOLD

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
        raise Error("TruncatedSVD fold provider does not provide queries")

    @staticmethod
    def open() raises -> Int:
        raise Error("TruncatedSVD fold provider does not own fitted leases")

    @staticmethod
    def close(lease: Int) raises:
        raise Error("TruncatedSVD fold provider does not own fitted leases")


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
