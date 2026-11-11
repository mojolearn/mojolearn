# SPDX-License-Identifier: Apache-2.0
"""Native metadata from the same image that owns the KernelPCA neighbor ops."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.pca_native_api import pca_vendor_code
from core.kpca_native_api import KPCAEigenRequest, KPCANativeKernels, KPCA_CAP_NEIGHBORS, KPCA_CAP_RESIDENT


struct KPCANeighborInfo[DEVICE: Bool, RESIDENT: Bool](KPCANativeKernels):
    @staticmethod
    def mode() -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def vendor() -> Int:
        return pca_vendor_code(String(COMPILED_VENDOR)) if Self.DEVICE else 1

    @staticmethod
    def capabilities() -> Int:
        return KPCA_CAP_NEIGHBORS | (KPCA_CAP_RESIDENT if Self.RESIDENT else 0)

    @staticmethod
    def eigen(request: KPCAEigenRequest) raises -> Int:
        raise Error("KernelPCA neighbor provider does not own the eigensolver")
