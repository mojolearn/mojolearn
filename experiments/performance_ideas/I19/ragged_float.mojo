# SPDX-License-Identifier: Apache-2.0
"""Ragged stable float-word sorting using the caller's public key policy.

Offsets are admitted shape metadata. Values remain device resident. Scratch
is supplied/reused by the caller, and every segment keeps its original
position for equal-word ties. The bounded quadratic arm is an executable
small control, never a large-data production fallback.
"""
from std.gpu import block_idx,thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from x_prep.common import canon,word_order
from x_prep.dradix import radix_key,radix_word
from core.stable_radix_digits import stable_nibble_pairs_u32
from core.stable_radix_sort import stable_radix_sort_pairs_u32,stable_radix_counts_len,stable_radix_bsum_len

def _keys(src: MutPointer[Float32,MutAnyOrigin],keys: MutPointer[UInt32,MutAnyOrigin],positions: MutPointer[UInt32,MutAnyOrigin],base: Int32,n_in: Int32,categories: Int32):
    var i = Int(block_idx.x)*128+Int(thread_idx.x)
    if i>=Int(n_in):
        return
    var value = ftz(src[Int(base)+i])
    if categories!=Int32(0):
        value = canon(value)
    keys[i] = radix_key(bitcast[DType.uint32](value))
    positions[i] = UInt32(Int(base)+i)

def _words(keys: MutPointer[UInt32,MutAnyOrigin],positions: MutPointer[UInt32,MutAnyOrigin],dst: MutPointer[Float32,MutAnyOrigin],permutation: MutPointer[UInt32,MutAnyOrigin],base: Int32,n_in: Int32):
    var i = Int(block_idx.x)*128+Int(thread_idx.x)
    if i<Int(n_in):
        dst[Int(base)+i] = bitcast[DType.float32](radix_word(keys[i]))
        permutation[Int(base)+i] = positions[i]

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
    # NEVER RUN — PENDING MEASUREMENT
    comptime if not is_defined["MOJOLEARN_IDN_RAGGED_FLOAT_RADIX"]():
        if largest>4096:
            raise Error("ragged float sort: rank control restricted to bounded small rows")
    for s in range(len(offsets)-1):
        var base = offsets[s]
        var n = offsets[s+1]-base
        if n==0:
            continue
        # NEVER RUN — PENDING MEASUREMENT
        comptime if is_defined["MOJOLEARN_IDN_RAGGED_FLOAT_RADIX"]():
            ctx.enqueue_function[_keys](src.unsafe_ptr(),keys.unsafe_ptr(),positions.unsafe_ptr(),base,n,Int32(1 if categories else 0),grid_dim=((Int(n)+127)//128,1,1),block_dim=(128,1,1))
            # NEVER RUN — PENDING MEASUREMENT
            comptime if is_defined["MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE"]():
                # NEVER RUN — PENDING MEASUREMENT
                comptime if is_defined["MOJOLEARN_IDN_RADIX_TILE128"]():
                    stable_nibble_pairs_u32[128](ctx,Int(n),32,keys,positions,temp_keys,temp_positions,counts,bsum)
                else:
                    stable_nibble_pairs_u32[256](ctx,Int(n),32,keys,positions,temp_keys,temp_positions,counts,bsum)
            else:
                stable_radix_sort_pairs_u32(ctx,Int(n),32,keys,positions,temp_keys,temp_positions,counts,bsum)
            ctx.enqueue_function[_words](keys.unsafe_ptr(),positions.unsafe_ptr(),dst.unsafe_ptr(),permutation.unsafe_ptr(),base,n,grid_dim=((Int(n)+127)//128,1,1),block_dim=(128,1,1))
        else:
            ctx.enqueue_function[_rank_control](src.unsafe_ptr(),dst.unsafe_ptr(),permutation.unsafe_ptr(),base,n,Int32(1 if categories else 0),grid_dim=((Int(n)+127)//128,1,1),block_dim=(128,1,1))
