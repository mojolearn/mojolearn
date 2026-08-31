# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Pointer-type tags shared across the control plane.

PORT OF `catboost/cuda/cuda_lib/fwd.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

`EPtrType` is the tag that decides which of their four `TMemoryCopyKind`
specialisations a copy resolves to (`cuda_base.h:133-166`). It is also what
makes `TCudaHostBufferPtr` a distinct type from `TCudaBufferPtr`, which is
how `TSplitPointsKernel` can dereference `PartitionsCpu` on the host while
`PartitionsGpu` stays device-only.
"""


struct EPtrType(Copyable, ImplicitlyCopyable, Movable):
    """Where a pointer's memory lives.

    Their enum, same three values, same meaning:

        CudaDevice  device memory, host may not dereference it
        CudaHost    page-locked host memory, BOTH sides may dereference it
        Host        ordinary pageable host memory
    """

    var value: Int32

    comptime CudaDevice = Self(0)
    comptime CudaHost = Self(1)
    comptime Host = Self(2)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


def is_host_ptr(type: EPtrType) -> Bool:
    """`cuda_base.h:115`. Everything that is not device memory is host memory.
    """
    return type != EPtrType.CudaDevice
