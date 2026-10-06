# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host storage allocation and address adapters, with no data computation.

GPU API owners may allocate staging storage through this module without
importing CPU vector kernels, task policy, thread pools, or host copies.
The caller owns each allocation and must write every element before reading.
"""

#: The pointer every `_into` entry takes: the host bindings' own type
#: (`bindings/hostptr.mojo::f32_ptr`), so a binding hands the caller's
#: address straight through and a List door rebinds its storage to it.
comptime HostF32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime HostF64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime HostU32Ptr = MutPointer[UInt32, MutUntrackedOrigin]
comptime HostI64Ptr = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def host_list_ptr(x: List[Float32]) -> HostF32Ptr:
    """A List's storage as `HostF32Ptr`, the `rebind` `core/gram_multi_gpu.
    mojo` performs on its shard list. The List must outlive every read
    and write through the result; the callers here keep it in a local
    until after the join."""
    return rebind[HostF32Ptr](x.unsafe_ptr())


@always_inline
def host_list_ptr_u32(x: List[UInt32]) -> HostU32Ptr:
    """`host_list_ptr` for the k-NN index output."""
    return rebind[HostU32Ptr](x.unsafe_ptr())


def host_f32_uninit(n: Int) -> List[Float32]:
    """A float32 list of `n` elements whose every element the caller writes
    before reading (lane neural-pass8): no fill, so the page-touching memset
    of a `List(length=n, fill=0.0)` is paid once by the writes instead of
    twice. Not for a list any element of which could be read unwritten."""
    var out = List[Float32]()
    if n > 0:
        out.resize(unsafe_uninit_length=n)
    return out^
