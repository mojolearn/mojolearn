# SPDX-License-Identifier: Apache-2.0
"""Native KernelPCA eigen dispatch to the existing decomposition executor."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from x_decomp.exec_trait import Exec
from x_decomp.cells import F32Ptr
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.kpca_owner import kpca_eig_run
from core.pca_native_api import pca_vendor_code
from core.kpca_native_api import KPCAEigenRequest, KPCANativeKernels, KPCA_CAP_EIGEN


def _pointer(address: Int) raises -> F32Ptr:
    if address == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=address)


def _dimension(value: Int) raises:
    if value < 0 or value > 2147483647:
        raise Error("x_decomp: dimension out of range")


struct KPCAEigenKernels[E: Exec, D: RDev, RES: Bool, X: DecompDev](KPCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR)) if Self.RES else 1

    @staticmethod
    def capabilities() -> Int:
        return KPCA_CAP_EIGEN

    @staticmethod
    def eigen(r: KPCAEigenRequest) raises -> Int:
        var centered = _pointer(r.centered)
        var source = _pointer(r.source)
        _dimension(r.rows)
        _dimension(r.features)
        var values = _pointer(r.values)
        var vectors = _pointer(r.vectors)
        var columns = _pointer(r.columns)
        var total = _pointer(r.total)
        return kpca_eig_run[Self.E, Self.D, Self.RES, Self.X](centered, source, r.resident,
            r.rows, r.features, r.gamma, r.requested, r.lanczos, r.remove_zero,
            values, vectors, columns, total)
