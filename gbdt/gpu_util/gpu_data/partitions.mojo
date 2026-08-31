# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in the root DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""A leaf is a contiguous range of the index array.

PORT OF `catboost/cuda/cuda_util/gpu_data/partitions.h` at CatBoost
`54a8143a`. Transliterated. Do not improve.

    struct TDataPartition {
        ui32 Offset;
        ui32 Size;
    };

**That is the entire leaf-membership representation, and it is the single
biggest structural difference from mojotrees.** There is no leaf index array
per row and no per-leaf row list. A row's leaf is decided by WHERE IT SITS in
the index array, so the histogram kernel takes `(offset, size)` and walks a
contiguous span. Membership is positional.

What gets permuted after a split is the `UInt32` index array and the stat
columns. The binned feature matrix is NEVER permuted: it is read indirectly
through `cindex[feature.offset + indices[i]]` and does not move for the life
of the fit.

================================ DEVIATION BLOCK ======================
`partsCpu` IS PINNED HOST MEMORY THE DEVICE WRITES, AND THE PINNED
TOOLCHAIN HAS NO COUNTERPART FOR IT.

**What theirs does.** Every partition descriptor exists twice.
`TPointsSubsets` holds `Partitions`, an ordinary device buffer, beside

    NCudaLib::TCudaBuffer<TDataPartition,
                          NCudaLib::TStripeMapping,
                          NCudaLib::EPtrType::CudaHost> PartitionsCpu;

at `split_properties_helper.h:49`. `EPtrType::CudaHost` allocates through
`cudaHostAlloc(&ptr, size, cudaHostAllocPortable)` (`cuda_base.cpp:6`), so
the block is page-locked and, under unified virtual addressing, carries one
address that the host and the device both dereference.

The DEVICE writes it. `UpdatePartitionsAfterSplitImpl` takes
`TDataPartition* partsCpu` and stores the whole struct to it in the same
branch that stores the device copy (`split_points.cu:372` and `:379`, and
`:428` and `:435` in the single-leaf form). `TWriteInitPartitions::Run`
writes the same allocation from the host at the root and then copies it out
to the device (`split_properties_helper.cpp:62-66`), which is the identical
aliasing seen from the other side.

The HOST reads it with no copy and no synchronize. `TSplitPointsKernel::Run`
dereferences `PartitionsCpu.Get()` as a plain pointer to find the largest
leaf it is about to sort (`split_points.cpp:56-62`), `SortByFlagsInLeaves`
reads `partsCpu[leafId]` once per leaf to size each radix sort
(`split_points.cu:667`), and `TSplitPointsSingleLeafKernel::Run` takes the
leaf size the same way (`split_points.cpp:236`). None of those is a
transfer, and that is the entire reason the second allocation exists.

Their one BLOCKING partition read per level is `RebuildLeavesSizes`, which
is `currentParts.Read(partsCpu)` over the whole leaf range
(`split_properties_helper.cpp:800-813`), called once per multi-leaf
`MakeSplit` at `:950`. Even that one is a host-to-host memcpy rather than a
device transfer, because both ends are host memory.

**What ours does.** `update_partitions_after_split_kernel` takes
`host_offset` and `host_size` and writes them exactly where theirs writes
`partsCpu`, so the device half is a faithful port. The driver allocates them
with `enqueue_create_buffer`, so they are a SECOND DEVICE ARRAY that no host
read ever touches. We pay their write and collect none of their benefit, and
the host still learns a leaf size through an `enqueue_copy` and a drain.

**The search, written down so nobody repeats it.** The pin is `mojo 1.0.0`
with `max 26.5.0` (`pixi.toml`, confirmed against `conda-meta` in the solved
environment).

- `DeviceContext.enqueue_create_host_buffer` allocates the right KIND of
  memory. Its reference page says the allocation is "page-locked (pinned)"
  and "accessible by the device". It is still not passable to a kernel.
  `HostBuffer` conforms to `AnyType`, `Copyable`, `Deinitable`,
  `ImplicitlyCopyable`, `Movable`, `Sized` and `Writable`, and NOT to
  `DevicePassable`, which is the trait a kernel argument has to satisfy.
  Handing a kernel `HostBuffer.unsafe_ptr()` anyway is what UNWIRED.md
  measured, and the kernel wrote nothing, silently, 64 cells of 64 wrong.
  A host virtual address is not a device address here.
- `DeviceBuffer.map_to_host` does see kernel output and was measured 2x
  slower than the copy it would replace, so it is not a route either.
- Every allocating entry point on `DeviceContext` was enumerated against the
  published reference. There are three, `enqueue_create_buffer`,
  `create_buffer_sync` and `enqueue_create_host_buffer`. Nothing named for
  managed, unified, mapped or zero-copy memory exists on `DeviceContext`,
  and the modules under `max.gpu.host` are `compile`,
  `constant_memory_mapping`, `device_attribute`, `device_context`,
  `device_graph`, `dim`, `func_attribute`, `info`, `launch_attribute` and
  `nvidia`. None of them allocates.
- `max.driver.DevicePinnedBuffer` and `max.driver.CompletionFlag` are Python
  driver types rather than Mojo ones. `CompletionFlag` genuinely is "pinned
  host memory mapped into a device's address space", and it is eight bytes
  wide and is a signalling primitive, not a buffer.

**The one thing that would close it, and it is not shipped yet.**
`DeviceBuffer.unsafe_host_ptr()` is in the MAX NIGHTLY reference and in the
nightly release notes, worded for this exact case. "On devices with unified
memory (Apple silicon), it returns a CPU-addressable pointer to the buffer,
so the host can read a kernel's output after `DeviceContext.synchronize()`
without an `enqueue_copy` round trip. Reads through it are uncached, so it
suits small control records rather than bulk readback." A partition array is
a small control record. The method is absent from the `26.5` reference page
for `DeviceBuffer` and absent from the shipped release notes, so this
checkout cannot call it. When the pin moves, `hp_off` and `hp_size` stop
being dead weight and become the READ side, and the `enqueue_copy` of
`p_sz` at the foot of every level goes away.

**What we do until then.** Pay the copy ONCE PER LEVEL, which is their
count. One blocking partition read per level is fidelity
(`split_properties_helper.cpp:950`); anything more is ours.

**The ordering, which is the dangerous part.** A host that reads a size the
device has not finished writing gets a plausible number rather than a fault,
and plausible wrong sizes build a valid-looking tree. The only thing that
orders our read is that the `enqueue_copy` is enqueued after
`update_partitions_after_split_kernel` on the same queue, and the drain that
follows waits for both. That is the same guarantee `currentParts.Read` gets
from being stream-ordered behind the split kernel launched at
`split_properties_helper.cpp:920-934` and read at `:950`. Do not move a
partition read in front of the kernel that writes it, and do not read
`hp_off` or `hp_size` at all until they are backed by memory the host can
address.
======================================================================
"""


@fieldwise_init
struct DataPartition(Copyable, Movable):
    """`TDataPartition`. Two `ui32`, nothing else."""

    var offset: UInt32
    var size: UInt32

    @staticmethod
    def empty() -> Self:
        return Self(0, 0)


@fieldwise_init
struct FeatureInBlock(Copyable, Movable):
    """`TFeatureInBlock`, the per-feature descriptor the histogram kernel reads.

    Ported from its use sites in `hist_binary.cu` and
    `compute_hist_loop_two_stats.cuh` rather than from a header, so the field
    set is what those kernels actually touch:

    - `compressed_index_offset`: where this feature's group starts in the
      compressed index.
    - `folds`: how many folds the feature uses. `folds == 0` means the
      feature is absent from this block, and the kernels test it before
      writing anything.
    - `fold_offset_in_group`: where this feature's bins start in the flat
      per-leaf histogram the scorer reads.
    - `bin_count`: `folds`, kept separately because the writeback loop bounds
      on it.
    """

    var compressed_index_offset: UInt32
    var folds: UInt32
    var fold_offset_in_group: UInt32
    var bin_count: UInt32
