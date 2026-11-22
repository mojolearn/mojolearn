# SPDX-License-Identifier: Apache-2.0
"""Ragged stable float-word sorting using the caller's public key policy.

Offsets are admitted shape metadata. Values remain device resident. Scratch
is supplied/reused by the caller, and every segment keeps its original
position for equal-word ties. The bounded quadratic arm is an executable
small control, never a large-data production fallback.
"""
from max.gpu import block_idx,thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from x_prep.common import canon,word_order
from core.stable_radix_sort import stable_radix_counts_len,stable_radix_bsum_len

def _rank_control(src: MutPointer[Float32,MutAnyOrigin],dst: MutPointer[Float32,MutAnyOrigin],permutation: MutPointer[UInt32,MutAnyOrigin],base: Int32,n_in: Int32,categories: Int32):
    var i = Int(block_idx.x)*128+Int(thread_idx.x)
    var n = Int(n_in)
    if i>=n:
        return
    var value = ftz(src[Int(base)+i])
    if categories!=Int32(0):
        value = canon(value)
    var key = word_order(bitcast[DType.uint32](value))
    var rank = 0
    for j in range(n):
        var other = ftz(src[Int(base)+j])
        if categories!=Int32(0):
            other = canon(other)
        var other_key = word_order(bitcast[DType.uint32](other))
        if other_key<key or (other_key==key and j<i):
            rank+=1
    dst[Int(base)+rank] = value
    permutation[Int(base)+rank] = UInt32(Int(base)+i)

# I19 new candidate remains default off. Qualification is pending: native
# compilation is not four-column identity or NVIDIA+AMD full-operation speed.
def enqueue_ragged_float_sort(ctx: DeviceContext,mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],mut permutation: DeviceBuffer[DType.uint32],
    offsets: List[Int32],categories: Bool,mut keys: DeviceBuffer[DType.uint32],
    mut positions: DeviceBuffer[DType.uint32],mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_positions: DeviceBuffer[DType.uint32],mut counts: DeviceBuffer[DType.int32],
    mut bsum: DeviceBuffer[DType.int32]) raises:
    if len(offsets)<2 or offsets[0]!=Int32(0):
        raise Error("ragged float sort: offsets must start at zero")
    var total = Int(offsets[len(offsets)-1])
    var largest = 0
    for s in range(len(offsets)-1):
        if offsets[s]<0 or offsets[s+1]<offsets[s]:
            raise Error("ragged float sort: offsets are not monotone")
        largest=max(largest,Int(offsets[s+1]-offsets[s]))
    if total>len(src) or total>len(dst) or total>len(permutation):
        raise Error("ragged float sort: output capacity below logical length")
    if largest>len(keys) or largest>len(positions) or largest>len(temp_keys) or largest>len(temp_positions) or len(counts)<stable_radix_counts_len(largest) or len(bsum)<stable_radix_bsum_len(largest):
        raise Error("ragged float sort: caller scratch capacity too short")
    # I19 AMD MI325X scoped WINNER (2026-10-06, source5b467815b): radix/rank
    # 0.683/0.465/0.356 at total65537/98305/131071,32segments (max2049/3073/4096).
    # Complete ragged-quantile caller, one same-process warmup/score per arm;
    # existing binaries and accepted identity evidence reused without revalidation.
    # Original1M+ totals exceeded the rank control's4096-per-segment domain;
    # those baseline refusals and large candidate-only receipts remain retained.
    # This bounded component win does not qualify larger workloads or promotion.
    # Evidence: overnight-ab-20261006/amd/normalized-measurements.json, I19;
    # raw attempts: amd/live/repairs. Default radix admission remains OFF.
    # TOMBSTONE: MOJOLEARN_IDN_RADIX_TILE128 (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 51ec4db2c0.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RADIX_TILE128.patch
    # TOMBSTONE: MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE (no A/B recorded for the opt-in arm) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 36a2cc0af1.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE.patch
    # TOMBSTONE: MOJOLEARN_IDN_RAGGED_FLOAT_RADIX (AMD-only component screening (I19 MI325X), never an NVIDIA + AMD full-operation A/B) deleted 2026-11-15 by lane/expdel2-07-base; code recoverable at 6e8a669206.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_FLOAT_RADIX.patch
    if largest>4096:
        raise Error("ragged float sort: rank control restricted to bounded small rows")
    for s in range(len(offsets)-1):
        var base = offsets[s]
        var n = offsets[s+1]-base
        if n==0:
            continue
        ctx.enqueue_function[_rank_control](src.unsafe_ptr(),dst.unsafe_ptr(),permutation.unsafe_ptr(),base,n,Int32(1 if categories else 0),grid_dim=((Int(n)+127)//128,1,1),block_dim=(128,1,1))
