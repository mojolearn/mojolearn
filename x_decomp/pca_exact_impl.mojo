# SPDX-License-Identifier: Apache-2.0
"""Typed PCA state checks and widening moved word for word from bindings/pca_exact.mojo."""
from std.memory import bitcast
from core.host_finite_scan import host_all_finite_f32


def pca_check_state(ptr: MutPointer[Float32, MutUntrackedOrigin], size: Int, name: String,
                    finite: Bool, nonnegative: Bool) raises:
    if finite and not nonnegative:
        # A finite-only check is the one host finite scan; with the sign
        # test the loop below keeps the order of the two refusals.
        if not host_all_finite_f32(ptr, size):
            raise Error("PCA whitening requires finite " + name)
        return
    for i in range(size):  # native input marshaling: validation of externally mutable fitted arrays
        if finite and (bitcast[DType.uint32](ptr[i]) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            raise Error("PCA whitening requires finite " + name)
        if nonnegative and ptr[i] < 0:
            raise Error("PCA whitening requires nonnegative singular values/variance")


def pca_widen_spectrum(src: MutPointer[Float32, MutUntrackedOrigin], dst: MutPointer[Float64, MutUntrackedOrigin], count: Int):
    for i in range(count):  # native input marshaling: the MLE kernel's binary64 spectrum ABI
        dst[i] = Float64(src[i])
