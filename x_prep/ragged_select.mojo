# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded order statistics without sorted/permutation materialization.

The rank-count control is the only route (the I19 radix arm is deleted). Unsigned key decisions and stable original-position
secondary keys are exact integers. No estimator/default profile changes.
"""
from max.gpu import block_idx,thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer,DeviceContext
from checks.numerics import ftz
from x_prep.common import canon
from x_prep.dradix import radix_key,radix_word

# TOMBSTONE: MOJOLEARN_IDN_RAGGED_RADIX_SELECT (opt-in arm, never run) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_RADIX_SELECT.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old-path umbrella, EXP:2438; every IDENTICAL default kept) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-36-neural-series-linear.patch; record in docs/TOMBSTONES.md.
comptime SELECT_TPB=128

@always_inline
def _select_key(v: Float32,categories: Int32) -> UInt32:
    var value=ftz(v)
    if categories!=0:
        value=canon(value)
    return radix_key(bitcast[DType.uint32](value))

def _select_kernel(src: MutPointer[Float32,MutAnyOrigin],desc: MutPointer[Int32,MutAnyOrigin],values: MutPointer[Float32,MutAnyOrigin],positions: MutPointer[Int32,MutAnyOrigin],categories: Int32):
    var task=Int(block_idx.x); var t=Int(thread_idx.x)
    var base=Int(desc[task*3]); var n=Int(desc[task*3+1]); var r=Int(desc[task*3+2])
    # TOMBSTONE: MOJOLEARN_IDN_RAGGED_RADIX_SELECT (opt-in arm, never run) deleted 2026-11-15 by lane/expdel2-36-neural-series-linear; code recoverable at 145be10898.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_RAGGED_RADIX_SELECT.patch; record in docs/TOMBSTONES.md.
    if t==0:
        for i in range(n):
            var key=_select_key(src[base+i],categories); var rank=0
            for j in range(n):
                var other=_select_key(src[base+j],categories)
                if other<key or (other==key and j<i):
                    rank+=1
            if rank==r:
                values[task]=bitcast[DType.float32](radix_word(key))
                positions[task]=Int32(base+i)
                break

