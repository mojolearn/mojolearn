"""`TIndexWrapper`: one `ui32` carrying an index AND a segment-start flag.

MIRRORS `catboost/cuda/cuda_util/kernel/index_wrapper.cuh` at CatBoost
`54a8143a`, the whole file (30 lines).

## Why it is here and not at its mirror address

Its address is `gbdt/gpu_util/kernel/index_wrapper.mojo`, under the
directory that mirrors `cuda_util`. That directory belongs to another lane
this round. **Move this file there when the lanes rejoin**; nothing about
it is CTR-specific, and `cuda_util`'s own gather/scatter/scan kernels
(`ScatterWithMask`, `SegmentedScanAndScatterNonNegativeVector`) read the
same packing.

## The packing, which is not what it looks like

    UpdateMask(b)     Idx |= b << 31
    IsSegmentStart()  Idx >> 31
    Index()           Idx & 0x3FFFFFFF          <-- THIRTY bits, not 31
    Value()           Idx

`Index()` masks THIRTY bits, so bit 30 is cleared on every read and the
usable row space is 2^30, not 2^31. That is not a typo of theirs to fix:
`TCtrBinBuilder::Mask` is the same `0x3FFFFFFF` constant
(`ctr_bins_builder.h:265`) and every `GatherWithMask` / `ScatterWithMask`
call site passes it explicitly, so the two agree by construction. Widening
`Index()` to `0x7FFFFFFF` here would leave those call sites narrower and
the disagreement would show up as a wrong gather on datasets above 2^30
rows and nowhere below it.

The SIGN of a float carries the same flag on the weight/statistic vectors:
`GatherTrivialWeights` and `WriteMask` negate the value at a segment start
so that `SegmentedScanAndScatterNonNegativeVector` can find segment
boundaries in a float buffer with no second array. `ExtractSignBit` in
`FillBinarizedTargetsStats` is the read side of that.
"""


comptime CTR_INDEX_MASK = UInt32(0x3FFFFFFF)
"""`TCtrBinBuilder<T>::Mask` (`ctr_bins_builder.h:265`), and the same
constant `TIndexWrapper::Index()` applies (`index_wrapper.cuh:23`)."""

comptime CTR_SEGMENT_START_BIT = UInt32(1) << 31
"""The bit `UpdateMask` sets (`index_wrapper.cuh:14`)."""


@always_inline
def index_of(value: UInt32) -> UInt32:
    """`TIndexWrapper::Index()` (`index_wrapper.cuh:22-24`)."""
    return value & CTR_INDEX_MASK


@always_inline
def is_segment_start(value: UInt32) -> Bool:
    """`TIndexWrapper::IsSegmentStart()` (`index_wrapper.cuh:18-20`).

    Theirs returns `Idx >> 31` as a `ui32` and every caller uses it as a
    truth value.
    """
    return (value >> 31) != UInt32(0)


@always_inline
def with_mask(value: UInt32, is_on_border: Bool) -> UInt32:
    """`TIndexWrapper::UpdateMask(bool)` followed by `Value()`
    (`index_wrapper.cuh:14-16`, `:26-28`).

    An OR, not an assignment: a flag already set is never cleared by a
    later `UpdateMask(false)`, and `UpdateBordersMaskImpl` depends on that
    because its own first test is `currentIndex.IsSegmentStart()`.
    """
    if is_on_border:
        return value | CTR_SEGMENT_START_BIT
    return value
