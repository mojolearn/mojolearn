# SPDX-License-Identifier: Apache-2.0
"""PCA whitening's existing scalar and pointer admission, before execution."""
from core.pca_native_api import PCAQueryRequest
from core.hostptr import f32_ptr


def pca_whiten_dimensions(r: PCAQueryRequest) raises:
    if r.rows < 1 or r.features < 1 or r.count < 1 or r.count > r.features or r.fit_rows < 2:
        raise Error("PCA whitening requires positive rows/features/components and fit_rows>=2")
    if r.rows > 2147483647 or r.features > 2147483647 or r.count > 2147483647 or r.fit_rows > 2147483647:
        raise Error("PCA whitening dimensions exceed Int32")
    if r.rows > 2147483647 // r.features or r.count > 2147483647 // r.features:
        raise Error("PCA whitening matrix cell count exceeds Int32")


def pca_whiten_device_pointer(address: Int, count: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if address <= 0 or address % 4 != 0:
        raise Error("PCA whitening requires positive aligned FP32 pointers")
    if address > 9223372036854775807 - count * 4:
        raise Error("PCA whitening pointer span overflows signed Int")
    return f32_ptr(address)
