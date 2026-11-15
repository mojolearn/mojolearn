# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Apple FAST optimizer/loss candidates of lane afn-optim (2026-10-03)
are gone; what stays is the scratch pool IDENTICAL's resident optimizer step
uses (`training/estimator.mojo`) and the False aliases and refusing entry
points that other files still import (the integrator removes them with
their importers' dead branches).
"""
from std.sys.compile import is_defined
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.step_phase import step_count_device_alloc
from core.shared_slot import SharedSlot


# ===========================================================================
# THE GUARDS
# ===========================================================================

#: FAST tier on an Apple host: the only build any candidate compiles into.
comptime AFN_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)
# TOMBSTONE: MOJOLEARN_AFN_OPTIM_ALL (NEUTRAL: M3 rab2-combo sgd/adam/adamw = TRAIN_OPT_FAST_PIPE_DOWN alone, the bundle
# adds nothing and moves the digest; EXPERIMENTS.md 2026-10-05 reconcile) deleted 2026-11-15 by lane/expdel-04-neural;
# code recoverable at 54b6a264fc. Each candidate keeps its own define.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_OPTIM_ALL.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_FUSE_SCAN (DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality
# metric) deleted 2026-10-09 on lane/owed-deletions-D2; code recoverable at b639a2bd2. Also retires its aliases
# MOJOLEARN_AFN26_OPT_FUSE_SCAN and the AFN_OPTIM_ALL share. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_FUSE_SCAN.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_VEC4 (DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality metric)
# deleted 2026-10-09 on lane/owed-deletions-D2 (with alias MOJOLEARN_AFN26_OPT_VEC4); code recoverable at
# b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_VEC4.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_RESIDENT_STATE (DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality
# metric) deleted 2026-10-09 on lane/owed-deletions-D2 (with alias MOJOLEARN_AFN26_OPT_RESIDENT_STATE); code
# recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_RESIDENT_STATE.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_CLIP_FUSE (MERGED-UNMEASURED opt-in, EXP:647, never measured) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_CLIP_FUSE.patch
# TOMBSTONE: MOJOLEARN_AFN26_OPT_CLIP_FUSE (unmeasured alias of AFN_OPT_CLIP_FUSE) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_OPT_CLIP_FUSE.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_MULTITENSOR (MERGED-UNMEASURED opt-in, EXP:649, never measured) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_MULTITENSOR.patch
# TOMBSTONE: MOJOLEARN_AFN26_OPT_MULTITENSOR (unmeasured alias of AFN_OPT_MULTITENSOR) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_OPT_MULTITENSOR.patch
# TOMBSTONE: MOJOLEARN_AFN_LOSS_FUSED (MERGED-UNMEASURED opt-in, EXP:630, never measured) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_LOSS_FUSED.patch
# TOMBSTONE: MOJOLEARN_AFN26_LOSS_FUSED (unmeasured alias of AFN_LOSS_FUSED) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_LOSS_FUSED.patch
# TOMBSTONE: MOJOLEARN_AFN26_OPT_BLOCK128 (unmeasured opt-in block width) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_OPT_BLOCK128.patch
# TOMBSTONE: MOJOLEARN_AFN26_OPT_BLOCK512 (unmeasured opt-in block width) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_OPT_BLOCK512.patch
# TOMBSTONE: MOJOLEARN_AFN26_LOSS_BLOCK128 (unmeasured opt-in block width) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_LOSS_BLOCK128.patch
# TOMBSTONE: MOJOLEARN_AFN26_LOSS_BLOCK512 (unmeasured opt-in block width) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AFN26_LOSS_BLOCK512.patch
# (Sweep 2026-11-15: AFN_OPT_CLIP_FUSE, AFN_LOSS_FUSED, AFN_OPT_ANY (False) and the always-refusing
# afn_optimizer_step / afn_ce_loss_resident removed with their readers' dead arms in training/estimator.mojo and
# training/checks/optimizer.mojo.)


# ===========================================================================
# THE SCRATCH POOL (IDENTICAL's resident step, training/estimator.mojo)
# ===========================================================================
# TOMBSTONE: MOJOLEARN_AFN_OPT_RESIDENT_STATE (DROPPED-noise, rab19) deleted 2026-10-09 on lane/owed-deletions-D2:
# the FAST users of this pool, its Int32 and pinned-host slots (afn_scratch_i32, afn_scratch_host_i32) and
# afn_scratch_f32; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_RESIDENT_STATE.patch


struct _AfnScratch(Defaultable, Movable):
    """Process-wide scratch slots, created once per shape and reused by
    every step: `f32[k]` device buffers, each with the length it was
    created at. A slot is recreated only when a call names a different
    length."""

    var f32: List[DeviceBuffer[DType.float32]]
    var f32_n: List[Int]

    def __init__(out self):
        self.f32 = List[DeviceBuffer[DType.float32]]()
        self.f32_n = List[Int]()


#: one pool per tier's binding (lane idn-opt-resident: IDENTICAL's resident
#: step and clip take their small scratch from it too, training/estimator.mojo)
comptime _AFN_SCRATCH_NAME = (
    "MojolearnAfnOptimScratchFastV1" if GLOBAL_NUMERIC_MODE == NUMERIC_FAST else "MojolearnAfnOptimScratchIdenticalV1"
)
comptime AFN_SCRATCH = SharedSlot[_AfnScratch, _AFN_SCRATCH_NAME]

#: Float32 slots (0, 1 and 10..12 were the FAST RESIDENT_STATE slots; numbers kept).
comptime AFN_SF_DENOM = 2  # the resident host entry's eight buffers ...
comptime AFN_SF_Q = 3
comptime AFN_SF_SUMSQ = 4
comptime AFN_SF_NORMS = 5
comptime AFN_SF_TOTAL = 6
comptime AFN_SF_OUT2 = 7
comptime AFN_SF_WS = 8
comptime AFN_SF_SAB = 9  # ... through here
def afn_scratch_f32_ensure(ctx: DeviceContext, slot: Int, n: Int) raises:
    """Create or resize Float32 slot `slot` to `n` elements (at least one).
    A caller that needs the `DeviceBuffer` itself (the resident host entry
    hands main's step its eight small buffers) reads
    `AFN_SCRATCH.get_or_create_ptr()[].f32[slot]` after this."""
    var want = n if n > 0 else 1
    var s = AFN_SCRATCH.get_or_create_ptr()
    while len(s[].f32) <= slot:
        step_count_device_alloc()
        s[].f32.append(ctx.enqueue_create_buffer[DType.float32](1))
        s[].f32_n.append(0)
    if s[].f32_n[slot] != want:
        step_count_device_alloc()
        s[].f32[slot] = ctx.enqueue_create_buffer[DType.float32](want)
        s[].f32_n[slot] = want


def afn_scratch_view(
    ctx: DeviceContext, slot: Int, n: Int
) raises -> DeviceBuffer[DType.float32]:
    """A sub-buffer VIEW of `n` floats over Float32 slot `slot` (created at
    `n` when absent or sized differently): a handle main's launchers take
    as a `DeviceBuffer`, with no Metal allocation behind it after the first
    step (a view of a pooled buffer is free; the pool owns the memory)."""
    var want = n if n > 0 else 1
    afn_scratch_f32_ensure(ctx, slot, want)
    var s = AFN_SCRATCH.get_or_create_ptr()
    return s[].f32[slot].create_sub_buffer[DType.float32](0, want)


# ===========================================================================
# THE REMOVED ENTRY POINTS (imported by dead branches of other files)
# ===========================================================================
# The earlier inline markers of the deleted kernels and step bodies, kept verbatim:
# TOMBSTONE: MOJOLEARN_AFN_OPT_FUSE_SCAN (DROPPED-noise, rab19 adam within +-2%) deleted 2026-10-09 on
# lane/owed-deletions-D2: the SCAN/FOUR refusal-scan arms of this kernel (was afn_scan4_kernel); code
# recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_FUSE_SCAN.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_VEC4 (DROPPED-noise, rab19 adam within +-2%) deleted 2026-10-09 on
# lane/owed-deletions-D2: the W = 4 arm and its scalar tail; code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_VEC4.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_RESIDENT_STATE (DROPPED-noise, rab19) deleted 2026-10-09 on lane/owed-deletions-D2:
# the pooled arm; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_RESIDENT_STATE.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_FUSE_SCAN (DROPPED-noise, rab19 adam within +-2%) deleted 2026-10-09 on
# lane/owed-deletions-D2: the fused refusal-scan arms; code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_FUSE_SCAN.patch
# TOMBSTONE: MOJOLEARN_AFN_OPT_RESIDENT_STATE (DROPPED-noise, rab19) deleted 2026-10-09 on lane/owed-deletions-D2:
# the pooled loss scratch; code recoverable at b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_AFN_OPT_RESIDENT_STATE.patch


