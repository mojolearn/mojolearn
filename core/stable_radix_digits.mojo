# SPDX-License-Identifier: Apache-2.0
"""Bounded stable four-bit radix scheduling for explicit I19 experiments.

No production default changes. Integer counts/ranks preserve the unique
stable (UInt32 key, original position) order. Digit width and tile size
control shared memory footprint, independently of dataset/board dimensions.
Native compile and device qualification pending.
"""
from std.gpu import block_idx,thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer,DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from core.fast_radix_sort import frs_exclusive_scan,frs_scan_blocks

comptime NIBBLE_BINS=16

def _count[TPB: Int](keys: MutPointer[UInt32,MutAnyOrigin],counts: MutPointer[Int32,MutAnyOrigin],n: Int32,shift: Int32,tiles: Int32):
    var tile=Int(block_idx.x); var t=Int(thread_idx.x); var i=tile*TPB+t
    var digits=stack_allocation[TPB,Int32,address_space=AddressSpace.SHARED]()
    digits[t]=Int32((keys[i]>>UInt32(shift))&UInt32(15)) if i<Int(n) else Int32(-1)
    barrier()
    if t<NIBBLE_BINS:
        var count=Int32(0)
        for j in range(TPB):
            if digits[j]==Int32(t):
                count+=1
        counts[t*Int(tiles)+tile]=count

def _scatter[TPB: Int](keys: MutPointer[UInt32,MutAnyOrigin],values: MutPointer[UInt32,MutAnyOrigin],out_keys: MutPointer[UInt32,MutAnyOrigin],out_values: MutPointer[UInt32,MutAnyOrigin],counts: MutPointer[Int32,MutAnyOrigin],n: Int32,shift: Int32,tiles: Int32):
    var tile=Int(block_idx.x); var t=Int(thread_idx.x); var i=tile*TPB+t
    var digits=stack_allocation[TPB,Int32,address_space=AddressSpace.SHARED]()
    var key=keys[i] if i<Int(n) else UInt32(0)
    var digit=Int32((key>>UInt32(shift))&UInt32(15)) if i<Int(n) else Int32(-1)
    digits[t]=digit
    barrier()
    if i<Int(n):
        var rank=Int32(0)
        for j in range(t):
            if digits[j]==digit:
                rank+=1
        var dst=Int(counts[Int(digit)*Int(tiles)+tile]+rank)
        out_keys[dst]=key; out_values[dst]=values[i]

def nibble_counts_len(size: Int,tile: Int) -> Int:
    return NIBBLE_BINS*((size+tile-1)//tile)

def stable_nibble_pairs_u32[TPB: Int](ctx: DeviceContext,size: Int,key_bits: Int,
    mut keys: DeviceBuffer[DType.uint32],mut values: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],mut temp_values: DeviceBuffer[DType.uint32],
    mut counts: DeviceBuffer[DType.int32],mut bsum: DeviceBuffer[DType.int32]) raises:
    comptime assert TPB==128 or TPB==256
    if size<0 or size>2147483647 or key_bits<0 or key_bits>32:
        raise Error("nibble radix: invalid size/key width")
    var m=nibble_counts_len(size,TPB)
    if m>2147483647 or len(keys)<size or len(values)<size or len(temp_keys)<size or len(temp_values)<size or len(counts)<m or len(bsum)<frs_scan_blocks(m):
        raise Error("nibble radix: caller capacity/index bound")
    if size==0:
        return
    var passes=max(2,(key_bits+3)//4)
    passes+=passes%2  # even ping-pong passes return to original storage
    var tiles=(size+TPB-1)//TPB
    for pass_index in range(passes):  # small-loop(passes<=8: launch metadata)
        var shift=Int32(pass_index*4)
        if pass_index%2==0:
            ctx.enqueue_function[_count[TPB]](keys.unsafe_ptr(),counts.unsafe_ptr(),Int32(size),shift,Int32(tiles),grid_dim=(tiles,1,1),block_dim=(TPB,1,1))
            frs_exclusive_scan(ctx,counts,m,bsum)
            ctx.enqueue_function[_scatter[TPB]](keys.unsafe_ptr(),values.unsafe_ptr(),temp_keys.unsafe_ptr(),temp_values.unsafe_ptr(),counts.unsafe_ptr(),Int32(size),shift,Int32(tiles),grid_dim=(tiles,1,1),block_dim=(TPB,1,1))
        else:
            ctx.enqueue_function[_count[TPB]](temp_keys.unsafe_ptr(),counts.unsafe_ptr(),Int32(size),shift,Int32(tiles),grid_dim=(tiles,1,1),block_dim=(TPB,1,1))
            frs_exclusive_scan(ctx,counts,m,bsum)
            ctx.enqueue_function[_scatter[TPB]](temp_keys.unsafe_ptr(),temp_values.unsafe_ptr(),keys.unsafe_ptr(),values.unsafe_ptr(),counts.unsafe_ptr(),Int32(size),shift,Int32(tiles),grid_dim=(tiles,1,1),block_dim=(TPB,1,1))
