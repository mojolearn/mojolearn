# SPDX-License-Identifier: Apache-2.0
"""PCA resident folds execute in the base image that owns their device store."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.shared_slot import SharedSlot
from core.msel_device_x import msel_fold_entry_context
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from decomposition.pca_fold import pca_fit_fold, pca_transform_fold
from core.pca_native_api import (
    PCA_NATIVE_ABI, PCA_CAP_EXACT, PCA_CAP_FULL, PCA_CAP_FOLD,
    PCANativeKernels, PCAFitRequest, PCAFitResult, PCAQueryRequest, PCAQueryResult,
    pca_native_dispatch, pca_vendor_code,
)


def _fold_pointer(address: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address <= 0 or address % 4 != 0:
        raise Error("pca_transform_fold: pointers must be positive and float32-aligned")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)


struct PCAFoldKernels(PCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR))

    @staticmethod
    def capabilities() -> Int:
        return PCA_CAP_EXACT | PCA_CAP_FULL | PCA_CAP_FOLD

    @staticmethod
    def fit(r: PCAFitRequest) raises -> PCAFitResult:
        if not r.fold.active() or r.randomized or r.projection != 0:
            raise Error("PCA fold provider requires an exact resident fit without projection")
        var cp = _fold_pointer(r.components)
        var mp = _fold_pointer(r.mean)
        var ep = _fold_pointer(r.variance)
        var rp = _fold_pointer(r.ratio)
        var sp = _fold_pointer(r.singular)
        var ctx = msel_fold_entry_context()
        var fitted = pca_fit_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
            cp, mp, ep, rp, sp, r.rows, r.features, r.count, r.dense, r.keep_mode, r.fraction)
        return PCAFitResult(fitted.keep, fitted.noise, Int(fitted.finite), Int(fitted.in_range))

    @staticmethod
    def query(r: PCAQueryRequest) raises -> PCAQueryResult:
        if not r.fold.active() or r.inverse or r.whiten:
            raise Error("PCA fold provider requires an unwhitened resident transform")
        var mp = _fold_pointer(r.mean)
        var cp = _fold_pointer(r.components)
        var dst = _fold_pointer(r.dst)
        var ctx = msel_fold_entry_context()
        var words = pca_transform_fold(ctx, r.fold.base, r.fold.rows, r.fold.indices,
            mp, cp, dst, r.rows, r.features, r.count)
        return PCAQueryResult(Int(words[0]), Int(words[1]))


def pca_fold_native_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return pca_native_dispatch[PCAFoldKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct PCAFoldAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(PCA_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(pca_fold_native_v1)


comptime PCA_FOLD_SLOT = "MojolearnPCAFoldIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnPCAFoldFastV1"
comptime PCA_FOLD_API = SharedSlot[PCAFoldAPI, PCA_FOLD_SLOT]
