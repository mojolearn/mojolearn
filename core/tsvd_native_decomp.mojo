# SPDX-License-Identifier: Apache-2.0
"""Native TSVD randomized/TSQR requests retain their decomposition executor."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_decomp.exec_trait import Exec
from x_decomp.cells import F32Ptr
from x_decomp.pca_randomized import RDev
from x_decomp.tsvd_owner import tsvd_randomized_fit_run, tsvd_tsqr_components_run
from x_decomp.qfix import TSVD_QFIX as TSVD_QFIX_ENABLED
from core.pca_native_api import pca_vendor_code
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.tsvd_native_api import (
    TSVD_NATIVE_ABI, TSVD_CAP_RANDOMIZED, TSVD_CAP_TSQR, TSVD_CAP_HOST,
    TSVD_RANDOMIZED, TSVD_TSQR, TSVDNativeKernels, TSVDFitRequest, TSVDFitResult,
    TSVDQueryRequest, tsvd_native_dispatch,
)


def _decomp_pointer(address: Int) raises -> F32Ptr:
    if address == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=address)


def _decomp_dimension(value: Int) raises:
    if value < 0 or value > 2147483647:
        raise Error("x_decomp: dimension out of range")


struct TSVDDecompKernels[E: Exec, D: RDev, RES: Bool](TSVDNativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR)) if Self.RES else 1

    @staticmethod
    def capabilities() -> Int:
        return TSVD_CAP_RANDOMIZED | (TSVD_CAP_TSQR if Self.RES else TSVD_CAP_HOST)

    @staticmethod
    def qfix() -> Int:
        return 2 if Self.RES and TSVD_QFIX_ENABLED else 0

    @staticmethod
    def fit(r: TSVDFitRequest) raises -> TSVDFitResult:
        if r.route != TSVD_RANDOMIZED and r.route != TSVD_TSQR:
            raise Error("TruncatedSVD decomposition provider does not provide this fit route")
        if r.route == TSVD_TSQR and not Self.RES:
            raise Error("TruncatedSVD host decomposition provider does not provide TSQR components")
        _decomp_dimension(r.rows)
        _decomp_dimension(r.features)
        _decomp_dimension(r.count)
        var xp = _decomp_pointer(r.source)
        var cp = _decomp_pointer(r.components)
        var sp = _decomp_pointer(r.singular)
        if r.route == TSVD_RANDOMIZED:
            var vp = _decomp_pointer(r.variance)
            var rp = _decomp_pointer(r.ratio)
            var word = List[Int32](length=1, fill=0)
            tsvd_randomized_fit_run[Self.E, Self.D, Self.RES](xp, cp, sp, vp, rp,
                r.rows, r.features, r.count, r.oversamples, r.iterations, r.auto_iterations, r.seed,
                Int(word.unsafe_ptr()) if r.test_finite else 0)
            return TSVDFitResult(Int(word[0]), 0)
        var word = List[Int32](length=1, fill=0)
        tsvd_tsqr_components_run[Self.E, Self.D, Self.RES](xp, cp, sp, r.rows, r.features, r.count,
            Int(word.unsafe_ptr()) if r.test_finite else 0)
        return TSVDFitResult(Int(word[0]), 0)

    @staticmethod
    def query(r: TSVDQueryRequest) raises -> Int:
        raise Error("TruncatedSVD decomposition provider does not provide queries")

    @staticmethod
    def open() raises -> Int:
        raise Error("TruncatedSVD decomposition provider does not own resident leases")

    @staticmethod
    def close(lease: Int) raises:
        raise Error("TruncatedSVD decomposition provider does not own resident leases")


def tsvd_decomp_native_v1[E: Exec, D: RDev, RES: Bool](operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return tsvd_native_dispatch[TSVDDecompKernels[E, D, RES]](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct TSVDDecompAPI[E: Exec, D: RDev, RES: Bool](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(TSVD_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(tsvd_decomp_native_v1[Self.E, Self.D, Self.RES])
