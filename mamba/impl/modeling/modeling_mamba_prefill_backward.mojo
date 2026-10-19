# SPDX-License-Identifier: Apache-2.0
"""The Mamba-1 zero-state prefill VJP's result type and its conv1d weight
layout kernel. The device VJP is assembled on the tape
(autograd/rules_mamba.mojo::mamba1_tape_backward), which launches this file's
kernel; the CPU column's hand walk is mamba/host/mamba1_prefill_chain.mojo.
"""
from std.gpu import block_dim, block_idx, thread_idx
from mamba.checks.mamba_fixture import D_CONV


struct Mamba1PrefillGradients(Movable):
    """Independent host arrays; field order preserves the native dump inventory."""
    var out_proj_weight: List[Float32]
    var D: List[Float32]
    var A_log: List[Float32]
    var dt_proj_weight: List[Float32]
    var dt_proj_bias: List[Float32]
    var x_proj_weight: List[Float32]
    var conv1d_weight: List[Float32]
    var conv1d_bias: List[Float32]
    var in_proj_weight: List[Float32]
    var norm_weight: List[Float32]
    var x: List[Float32]
    var stage_dB: List[Float32]
    var stage_dC: List[Float32]

    def __init__(out self):
        self.out_proj_weight = List[Float32]()
        self.D = List[Float32]()
        self.A_log = List[Float32]()
        self.dt_proj_weight = List[Float32]()
        self.dt_proj_bias = List[Float32]()
        self.x_proj_weight = List[Float32]()
        self.conv1d_weight = List[Float32]()
        self.conv1d_bias = List[Float32]()
        self.in_proj_weight = List[Float32]()
        self.norm_weight = List[Float32]()
        self.x = List[Float32]()
        self.stage_dB = List[Float32]()
        self.stage_dC = List[Float32]()


def m1_conv_w_interleave_kernel(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    t0_ptr: MutPointer[Float32, MutAnyOrigin],
    t1_ptr: MutPointer[Float32, MutAnyOrigin],
    t2_ptr: MutPointer[Float32, MutAnyOrigin],
    t3_ptr: MutPointer[Float32, MutAnyOrigin],
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    di_in: Int32,
):
    """cpu3-seq (2026-10-04): the conv1d weight gradient's layout copy on
    the device, one thread per channel `d`: the four tap gradients land at
    `out[d * D_CONV + p_k]` (`p_k = mamba_reduction_tap(RED_CONV_W_TAPk)`),
    every other cell keeps the zero fill. A copy, no arithmetic, so no bit
    moves; it replaces the host loop that downloaded the four taps and
    interleaved them."""
    var d = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if d >= Int(di_in):
        return
    var base = d * D_CONV
    out_ptr.unsafe_store(base + Int(p0), t0_ptr.unsafe_load(d))
    out_ptr.unsafe_store(base + Int(p1), t1_ptr.unsafe_load(d))
    out_ptr.unsafe_store(base + Int(p2), t2_ptr.unsafe_load(d))
    out_ptr.unsafe_store(base + Int(p3), t3_ptr.unsafe_load(d))


# TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (hand Mamba-1 prefill backward glue, superseded by the tape) deleted 2026-10-19 by e378999c7; code recoverable at e378999c7^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
# (the CPU column keeps the same walk as generator input: mamba/host/mamba1_prefill_chain.mojo)
