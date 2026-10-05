# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST candidates for the optimizer step and the cross-entropy loss
(lane afn-optim, 2026-10-03).

Everything here is reached only when the build is FAST, the host has an
Apple GPU and one of the `MOJOLEARN_AFN_*` defines below is named (the
`AFN_*` aliases). IDENTICAL compiles main's `identical_optimizer_step` and
`identical_ce_loss_resident` unchanged: the dispatch into this file sits
inside `comptime if` guards on those aliases, and every kernel here is
parametric, so an IDENTICAL build instantiates none of it.

The mechanism of every candidate is FEWER LAUNCHES AND FEWER WAITS on
Apple, where a launch costs ~20 us host time plus ~0.25 us per live Metal
buffer and a launch + readback + wait round trip costs ~180 us (the lane
brief's measured costs). The arithmetic per element is the FAST tier's
own: `checks/numerics` helpers are plain operations under FAST (`ftz` is
the identity, `identical_mul_add` is `a * b + c`, `identical_sqrt` is the
stdlib sqrt), so these kernels spell the plain operations directly; f32
accumulation everywhere; fold orders are free (the FAST tier makes no bit
claim), no approximation anywhere.

Candidates (one define each, default OFF, `MOJOLEARN_AFN_OPTIM_ALL` turns
on every one):

  MOJOLEARN_AFN_OPT_FUSE_SCAN      the four non-finite refusal scans of
      `opt_refuse_device_inputs` (4 launches, 1 host readback, 1 wait,
      then the update and a second wait) become ONE scan launch over all
      four buffers, one fold launch that writes a device gate, and an
      update launch that reads the gate and does nothing when the step is
      refused. One wait per step, after which the host reads 32 bytes of
      cells and raises the oracle's message when the gate is set. The
      state is never partially updated.
  MOJOLEARN_AFN_OPT_CLIP_FUSE      the global-norm clip: the squared norm
      is folded in free order (block partials inside the scan launch when
      FUSE_SCAN is on, else one partials launch), one fold launch turns it
      into `total_norm` and the clamped `coef` on the device, and the
      update kernel multiplies the gradient by `coef` as it loads it and
      writes the clipped gradient back. No per-tensor GEMVs, no sqrt
      launch, no finish launch, no scale launch, no host read of the norm
      inside the step (main's clip pays five waits).
  MOJOLEARN_AFN_OPT_MULTITENSOR    SGD's per-tensor launches (one per
      tensor, for the per-tensor momentum flag) become ONE launch over the
      flat model: a device table of the tensor offsets and flags, and each
      thread finds its tensor by a binary search over the offsets. Adam is
      already one launch over the flat model in main; it is untouched.
  MOJOLEARN_AFN_OPT_VEC4           the Adam/AdamW update does four
      consecutive elements per thread with 4-wide loads and stores (a
      quarter of the threads, wider memory transactions), the same
      per-element arithmetic. The tail of the model runs the scalar form.
  MOJOLEARN_AFN_OPT_RESIDENT_STATE the per-step scratch (the scan
      partials, the clip partials, the gate cells and their pinned mirror,
      the SGD table, the eight small buffers the resident host entry
      allocates every step, the loss's row/flag scratch) lives in a
      process-wide pool and is created once per shape, so a step allocates
      nothing. The moments were already resident; the step scalars are
      kernel arguments (no upload), so no device-side schedule is needed.
  MOJOLEARN_AFN_LOSS_FUSED         cross-entropy forward + backward in ONE
      launch, one block per row: the row max, the sum of exponentials, the
      row loss (with label smoothing when spelled) and `dlogits` are
      written together, with the non-finite scan of the row and the target
      range check folded into the same pass (per-row flag cells); a second
      one-block launch folds the row losses into the scalar loss and the
      flag cells into the gate. Two launches, one wait, two scratch
      buffers (vs ~12 launches, 2 vendor matmuls, 5 waits, ~16 buffers and
      a host-built ones vector per call in main).
"""
from std.ffi import _Global
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import exp, fma, log, max, sqrt
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_scan import NONFINITE_NONE, device_classify_nonfinite
from core.step_phase import (
    step_count_d2h,
    step_count_device_alloc,
    step_count_host_alloc,
    step_count_launch,
    step_count_sync,
)
from training.checks.loss_contract import (
    REDUCTION_NONE,
    CeConfig,
    ce_divisor,
    ce_nonfinite_message,
    ce_one_minus_eps,
    ce_refuse_shape,
    ce_refuse_targets,
    ce_smoothing_targets,
    neg_by_bits,
)
from training.checks.optimizer_contract import (
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    StepScalars,
    clip_eps,
    opt_nonfinite_message,
    refuse_nonfinite_scalar,
)


# ===========================================================================
# THE GUARDS
# ===========================================================================

#: FAST tier on an Apple host: the only build any candidate compiles into.
comptime AFN_APPLE_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_COLUMN_CPU"]()
)
#: AFN_OPTIM_ALL OUTCOME (M3 afc_ab_def, full board size, 1 run per arm,
#: 2026-10-04, lane/apple-fast-rec-ab2 @ 40027eb8e): sgd 331.2 -> 329.2 ms, adam
#: 336.3 -> 325.6, adamw 331.6 -> 326.4 (gain 1-3%); the output digest changes
#: and the lane has no quality metric. HOLD (no quality evidence): stays off.
#: AFN_OPT_FUSE_SCAN / AFN_OPT_VEC4 / AFN_OPT_RESIDENT_STATE OUTCOME (M3
#: afc_ab_def, full board size, 1 run per arm, 2026-10-05, rab19): adam within
#: +-2%, the output digest changes, no quality metric. DROPPED: each stays off
#: (opt-in only).
comptime AFN_OPTIM_ALL = is_defined["MOJOLEARN_AFN_OPTIM_ALL"]()
comptime AFN_OPT_FUSE_SCAN = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_OPT_FUSE_SCAN"]()
)
comptime AFN_OPT_CLIP_FUSE = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_OPT_CLIP_FUSE"]()
)
comptime AFN_OPT_MULTITENSOR = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_OPT_MULTITENSOR"]()
)
comptime AFN_OPT_VEC4 = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_OPT_VEC4"]()
)
comptime AFN_OPT_RESIDENT_STATE = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_OPT_RESIDENT_STATE"]()
)
comptime AFN_LOSS_FUSED = AFN_APPLE_FAST and (
    AFN_OPTIM_ALL or is_defined["MOJOLEARN_AFN_LOSS_FUSED"]()
)
#: Any optimizer candidate: `identical_optimizer_step` dispatches here.
comptime AFN_OPT_ANY = (
    AFN_OPT_FUSE_SCAN
    or AFN_OPT_CLIP_FUSE
    or AFN_OPT_MULTITENSOR
    or AFN_OPT_VEC4
    or AFN_OPT_RESIDENT_STATE
)

#: Threads per block for every launch here (Metal's threadgroup limit is
#: 1024; 256 is the block main's optimizer and scans run at).
comptime AFN_TPB = 256
#: The grid cap of the scan and partial-sum launches; above it every
#: thread grid-strides. The fold launch reads at most this many partials
#: per buffer with one block.
comptime AFN_SCAN_BLOCKS = 512
#: The width of the vectorized Adam update.
comptime AFN_VEC = 4

#: Gate cell layout (Int32 x AFN_CELLS): 0..3 the smallest non-finite index
#: in param, grad, m, v (or NONFINITE_NONE); 4 the gate (nonzero refuses
#: the update); 5 nonzero when the clip's total norm is non-finite.
comptime AFN_CELLS = 8
comptime AFN_CELL_GATE = 4
comptime AFN_CELL_CLIP_BAD = 5
#: Float cells (Float32 x AFN_FCELLS): 0 total_norm, 1 coef.
comptime AFN_FCELLS = 4

#: Loss cells (Int32 x 4): 0 the smallest non-finite logit index, 1 the
#: first row with a target out of range, 2 the gate.
comptime AFN_LOSS_CELLS = 4
#: the loss fold's first level: at most this many blocks, one partial each.
comptime AFN_CE_PARTS = 64


def _afn_grid(n: Int) -> Int:
    """Blocks for `n` elements at `AFN_TPB`, never 0."""
    if n <= 0:
        return 1
    return (n + AFN_TPB - 1) // AFN_TPB


def _afn_scan_grid(n: Int) -> Int:
    """Blocks for a grid-striding scan over `n` elements: one per
    `AFN_TPB`, capped at `AFN_SCAN_BLOCKS`."""
    var b = _afn_grid(n)
    if b > AFN_SCAN_BLOCKS:
        b = AFN_SCAN_BLOCKS
    return b


@always_inline
def _nonfinite_bits(x: Float32) -> Bool:
    """NaN or infinity BY BITS (the scan's predicate; Metal flushes compare
    operands, bits do not flush)."""
    return (bitcast[DType.uint32](x) & UInt32(0x7FFFFFFF)) >= UInt32(
        0x7F800000
    )


# ===========================================================================
# THE SCRATCH POOL (MOJOLEARN_AFN_OPT_RESIDENT_STATE)
# ===========================================================================


struct _AfnScratch(Defaultable, Movable):
    """Process-wide scratch slots, created once per shape and reused by
    every step: `i32[k]` / `f32[k]` device buffers and `hi[k]` pinned host
    mirrors, each with the length it was created at. A slot is recreated
    only when a call names a different length."""

    var i32: List[DeviceBuffer[DType.int32]]
    var i32_n: List[Int]
    var f32: List[DeviceBuffer[DType.float32]]
    var f32_n: List[Int]
    var hi: List[HostBuffer[DType.int32]]
    var hi_n: List[Int]

    def __init__(out self):
        self.i32 = List[DeviceBuffer[DType.int32]]()
        self.i32_n = List[Int]()
        self.f32 = List[DeviceBuffer[DType.float32]]()
        self.f32_n = List[Int]()
        self.hi = List[HostBuffer[DType.int32]]()
        self.hi_n = List[Int]()


#: one pool per tier's binding (lane idn-opt-resident: IDENTICAL's resident
#: step and clip take their small scratch from it too, training/estimator.mojo)
comptime _AFN_SCRATCH_NAME = (
    "MojoAfnOptimScratchFast" if GLOBAL_NUMERIC_MODE == NUMERIC_FAST else "MojoAfnOptimScratchIdentical"
)
comptime AFN_SCRATCH = _Global[
    StorageType=_AfnScratch,
    name=_AFN_SCRATCH_NAME,
    init_fn=_AfnScratch.__init__,
]

#: Int32 slots.
comptime AFN_SI_PART = 0  # scan partials, 4 * blocks
comptime AFN_SI_CELLS = 1  # the gate cells
comptime AFN_SI_TABLE = 2  # the SGD offsets + flags table
comptime AFN_SI_LOSS_BAD = 3  # the loss's per-row flag cells, 2 * N
comptime AFN_SI_LOSS_CELLS = 4  # the loss's gate cells
comptime AFN_SI_LOSS_PART = 5  # the loss's per-block flag partials, 2 * AFN_CE_PARTS
#: Float32 slots.
comptime AFN_SF_SUMS = 0  # clip partials, blocks
comptime AFN_SF_FCELLS = 1  # total_norm, coef
comptime AFN_SF_DENOM = 2  # the resident host entry's eight buffers ...
comptime AFN_SF_Q = 3
comptime AFN_SF_SUMSQ = 4
comptime AFN_SF_NORMS = 5
comptime AFN_SF_TOTAL = 6
comptime AFN_SF_OUT2 = 7
comptime AFN_SF_WS = 8
comptime AFN_SF_SAB = 9  # ... through here
comptime AFN_SF_LOSS_ROW = 10  # the loss's row losses, N
comptime AFN_SF_LOSS_LOSS = 11  # the loss's scalar
comptime AFN_SF_LOSS_PART = 12  # the loss's per-block sums, AFN_CE_PARTS
#: Pinned Int32 mirrors.
comptime AFN_SH_CELLS = 0
comptime AFN_SH_LOSS_CELLS = 1


def afn_scratch_i32(
    ctx: DeviceContext, slot: Int, n: Int
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """The pointer of Int32 slot `slot`, created at `n` elements (at least
    one) when absent or sized differently. The pool owns the buffer."""
    var want = n if n > 0 else 1
    var s = AFN_SCRATCH.get_or_create_ptr()
    while len(s[].i32) <= slot:
        step_count_device_alloc()
        s[].i32.append(ctx.enqueue_create_buffer[DType.int32](1))
        s[].i32_n.append(0)
    if s[].i32_n[slot] != want:
        step_count_device_alloc()
        s[].i32[slot] = ctx.enqueue_create_buffer[DType.int32](want)
        s[].i32_n[slot] = want
    return s[].i32[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def afn_scratch_f32(
    ctx: DeviceContext, slot: Int, n: Int
) raises -> MutPointer[Float32, MutAnyOrigin]:
    """`afn_scratch_i32` for a Float32 slot."""
    afn_scratch_f32_ensure(ctx, slot, n)
    var s = AFN_SCRATCH.get_or_create_ptr()
    return s[].f32[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


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


def afn_scratch_host_i32(
    ctx: DeviceContext, slot: Int, n: Int
) raises -> MutPointer[Int32, MutAnyOrigin]:
    """`afn_scratch_i32` for a pinned host mirror (created with one wait,
    once per shape)."""
    var want = n if n > 0 else 1
    var s = AFN_SCRATCH.get_or_create_ptr()
    while len(s[].hi) <= slot:
        step_count_host_alloc()
        s[].hi.append(ctx.enqueue_create_host_buffer[DType.int32](1))
        s[].hi_n.append(0)
    if s[].hi_n[slot] != want:
        step_count_host_alloc()
        s[].hi[slot] = ctx.enqueue_create_host_buffer[DType.int32](want)
        s[].hi_n[slot] = want
        step_count_sync()
        ctx.synchronize()
    return s[].hi[slot].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


# ===========================================================================
# THE OPTIMIZER KERNELS
# ===========================================================================


def afn_scan4_kernel[SCAN: Bool, FOUR: Bool, CLIP: Bool](
    part: MutPointer[Int32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    m_state: MutPointer[Float32, MutAnyOrigin],
    v_state: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """ONE grid-striding pass over the step's inputs. Under `SCAN`, each
    block writes the smallest non-finite index it saw in `param`, `grad`,
    `m_state` and (`FOUR`) `v_state` to `part[k * blocks + block]` (or
    `NONFINITE_NONE`): main's four `nonfinite_partial_kernel` launches as
    one. Under `CLIP`, each block also writes its partial sum of squared
    gradients to `sums[block]` (free fold order, f32)."""
    var n = Int(n_in)
    var red = stack_allocation[
        AFN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var redf = stack_allocation[
        AFN_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var blocks = Int(grid_dim.x)
    var stride = blocks * AFN_TPB
    var i = Int(block_idx.x) * AFN_TPB + tid
    var best = SIMD[DType.int32, 4](NONFINITE_NONE)
    var acc = Float32(0.0)
    while i < n:
        var gv = grad.unsafe_load(i)
        comptime if SCAN:
            if best[0] == NONFINITE_NONE and _nonfinite_bits(
                param.unsafe_load(i)
            ):
                best[0] = Int32(i)
            if best[1] == NONFINITE_NONE and _nonfinite_bits(gv):
                best[1] = Int32(i)
            if best[2] == NONFINITE_NONE and _nonfinite_bits(
                m_state.unsafe_load(i)
            ):
                best[2] = Int32(i)
            comptime if FOUR:
                if best[3] == NONFINITE_NONE and _nonfinite_bits(
                    v_state.unsafe_load(i)
                ):
                    best[3] = Int32(i)
        comptime if CLIP:
            acc = fma(gv, gv, acc)
        i += stride
    comptime if SCAN:
        comptime for k in range(4):
            comptime if k < 3 or FOUR:
                red.unsafe_store(tid, best[k])
                barrier()
                var active = AFN_TPB // 2
                while active > 0:
                    if tid < active:
                        var o = red.unsafe_load(tid + active)
                        if o < red.unsafe_load(tid):
                            red.unsafe_store(tid, o)
                    barrier()
                    active = active // 2
                if tid == 0:
                    part.unsafe_store(
                        k * blocks + Int(block_idx.x), red.unsafe_load(0)
                    )
                barrier()
    comptime if CLIP:
        redf.unsafe_store(tid, acc)
        barrier()
        var activef = AFN_TPB // 2
        while activef > 0:
            if tid < activef:
                redf.unsafe_store(
                    tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + activef)
                )
            barrier()
            activef = activef // 2
        if tid == 0:
            sums.unsafe_store(Int(block_idx.x), redf.unsafe_load(0))


def afn_fold_kernel[SCAN: Bool, FOUR: Bool, CLIP: Bool](
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    out2: MutPointer[Float32, MutAnyOrigin],
    part: MutPointer[Int32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    blocks_in: Int32,
    max_norm: Float32,
    eps_clip: Float32,
):
    """ONE block. Under `SCAN`, the minimum over the scan partials per
    buffer into `cells[0..3]` and the gate `cells[4]` (nonzero when any
    buffer holds a non-finite value). Under `CLIP`, the sum of the partials
    (free order), `total_norm = sqrt(sum)`, `coef = min(1, max_norm /
    (total_norm + eps))` into `fcells[0..1]` and `out2[0..1]` (the cells
    main's callers read `total_norm` and `coef` from), and `cells[5]` plus
    the gate when `total_norm` is non-finite (main refuses that scalar on
    the host; here the update is withheld on the device and the host
    raises after its one wait)."""
    var blocks = Int(blocks_in)
    var red = stack_allocation[
        AFN_TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var redf = stack_allocation[
        AFN_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var tid = Int(thread_idx.x)
    var gate = Int32(0)
    var bad = Int32(0)
    var hits = SIMD[DType.int32, 4](NONFINITE_NONE)
    comptime if SCAN:
        comptime for k in range(4):
            comptime if k < 3 or FOUR:
                var best = NONFINITE_NONE
                var b = tid
                while b < blocks:
                    var o = part.unsafe_load(k * blocks + b)
                    if o < best:
                        best = o
                    b += AFN_TPB
                red.unsafe_store(tid, best)
                barrier()
                var active = AFN_TPB // 2
                while active > 0:
                    if tid < active:
                        var o2 = red.unsafe_load(tid + active)
                        if o2 < red.unsafe_load(tid):
                            red.unsafe_store(tid, o2)
                    barrier()
                    active = active // 2
                var hit = red.unsafe_load(0)
                hits[k] = hit
                if hit != NONFINITE_NONE:
                    gate = Int32(1)
                barrier()
    comptime if CLIP:
        var acc = Float32(0.0)
        var bb = tid
        while bb < blocks:
            acc = acc + sums.unsafe_load(bb)
            bb += AFN_TPB
        redf.unsafe_store(tid, acc)
        barrier()
        var activef = AFN_TPB // 2
        while activef > 0:
            if tid < activef:
                redf.unsafe_store(
                    tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + activef)
                )
            barrier()
            activef = activef // 2
        var tn = sqrt(redf.unsafe_load(0))
        var denom = tn + eps_clip
        var coef = max_norm / denom
        if not (coef < Float32(1.0)):
            coef = Float32(1.0)
        if _nonfinite_bits(tn):
            bad = Int32(1)
            gate = Int32(1)
        if tid == 0:
            fcells.unsafe_store(0, tn)
            fcells.unsafe_store(1, coef)
            out2.unsafe_store(0, tn)
            out2.unsafe_store(1, coef)
    if tid == 0:
        cells.unsafe_store(0, hits[0])
        cells.unsafe_store(1, hits[1])
        cells.unsafe_store(2, hits[2])
        cells.unsafe_store(3, hits[3])
        cells.unsafe_store(AFN_CELL_GATE, gate)
        cells.unsafe_store(AFN_CELL_CLIP_BAD, bad)


@always_inline
def _afn_adam_apply[W: Int, CLIP: Bool](
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    m_state: MutPointer[Float32, MutAnyOrigin],
    v_state: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    coef: Float32,
    is_adamw: Bool,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    c1: Float32,
    c2: Float32,
    step_size: Float32,
    rt_bc2: Float32,
    decay_mul: Float32,
):
    """Contract 7.2's arithmetic on `W` consecutive elements from `i`, the
    FAST tier's spelling (plain operations, `fma` where main fuses). Under
    `CLIP` the gradient is scaled by `coef` as loaded and the clipped
    gradient is stored back (main's `clip_scale_kernel`, folded in)."""
    comptime V = SIMD[DType.float32, W]
    var g = grad.unsafe_load[width=W](i)
    var p = param.unsafe_load[width=W](i)
    var mp = m_state.unsafe_load[width=W](i)
    var vp = v_state.unsafe_load[width=W](i)
    comptime if CLIP:
        g = g * V(coef)
        grad.unsafe_store[width=W](i, g)
    if weight_decay != Float32(0.0):
        if is_adamw:
            p = p * V(decay_mul)
        else:
            g = fma(V(weight_decay), p, g)
    var m = fma(V(c1), g, mp * V(beta1))
    var v = fma(V(c2), g * g, vp * V(beta2))
    var dn = sqrt(v) / V(rt_bc2) + V(eps)
    var q = m / dn
    var p_out = fma(V(-step_size), q, p)
    param.unsafe_store[width=W](i, p_out)
    m_state.unsafe_store[width=W](i, m)
    v_state.unsafe_store[width=W](i, v)


def afn_adam_kernel[W: Int, GATED: Bool, CLIP: Bool](
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    m_state: MutPointer[Float32, MutAnyOrigin],
    v_state: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    is_adamw_in: Int32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    c1: Float32,
    c2: Float32,
    step_size: Float32,
    rt_bc2: Float32,
    decay_mul: Float32,
):
    """The Adam/AdamW update, `W` elements per thread, one launch over the
    flat model. Under `GATED` every thread reads the gate cell first and
    returns when the step was refused (the fold kernel ran before this
    launch on the same queue). Under `CLIP` the coefficient comes from the
    float cells the fold kernel wrote."""
    comptime if GATED:
        if cells.unsafe_load(AFN_CELL_GATE) != Int32(0):
            return
    var n = Int(n_in)
    var base = (Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)) * W
    if base >= n:
        return
    var coef = Float32(1.0)
    comptime if CLIP:
        coef = fcells.unsafe_load(1)
    var is_adamw = is_adamw_in != Int32(0)
    comptime if W == 1:
        _afn_adam_apply[1, CLIP](
            param, grad, m_state, v_state, base, coef, is_adamw, beta1,
            beta2, eps, weight_decay, c1, c2, step_size, rt_bc2, decay_mul,
        )
    else:
        if base + W <= n:
            _afn_adam_apply[W, CLIP](
                param, grad, m_state, v_state, base, coef, is_adamw, beta1,
                beta2, eps, weight_decay, c1, c2, step_size, rt_bc2,
                decay_mul,
            )
        else:
            var i = base
            while i < n:
                _afn_adam_apply[1, CLIP](
                    param, grad, m_state, v_state, i, coef, is_adamw, beta1,
                    beta2, eps, weight_decay, c1, c2, step_size, rt_bc2,
                    decay_mul,
                )
                i += 1


@always_inline
def _afn_sgd_apply[CLIP: Bool](
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    coef: Float32,
    initialized: Bool,
    nesterov: Bool,
    momentum: Float32,
    c_damp: Float32,
    weight_decay: Float32,
    neg_lr: Float32,
):
    """Contract 7.3's arithmetic on element `i`, the FAST spelling."""
    var g = grad.unsafe_load(i)
    var p = param.unsafe_load(i)
    comptime if CLIP:
        g = g * coef
        grad.unsafe_store(i, g)
    if weight_decay != Float32(0.0):
        g = fma(weight_decay, p, g)
    var b = buf.unsafe_load(i)
    if momentum != Float32(0.0):
        if not initialized:
            b = g
        else:
            b = fma(c_damp, g, momentum * b)
        buf.unsafe_store(i, b)
        if nesterov:
            g = fma(momentum, b, g)
        else:
            g = b
    param.unsafe_store(i, fma(neg_lr, g, p))


def afn_sgd_kernel[GATED: Bool, CLIP: Bool](
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    begin_in: Int32,
    count_in: Int32,
    buf_initialized_in: Int32,
    nesterov_in: Int32,
    momentum: Float32,
    c_damp: Float32,
    weight_decay: Float32,
    neg_lr: Float32,
):
    """Main's per-tensor SGD launch shape (one launch per tensor, the
    tensor's momentum flag a scalar), with the gate and the clip
    coefficient read from the cells."""
    comptime if GATED:
        if cells.unsafe_load(AFN_CELL_GATE) != Int32(0):
            return
    var count = Int(count_in)
    var local = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if local >= count:
        return
    var coef = Float32(1.0)
    comptime if CLIP:
        coef = fcells.unsafe_load(1)
    _afn_sgd_apply[CLIP](
        param, grad, buf, Int(begin_in) + local, coef,
        buf_initialized_in != Int32(0), nesterov_in != Int32(0), momentum,
        c_damp, weight_decay, neg_lr,
    )


def afn_sgd_multi_kernel[GATED: Bool, CLIP: Bool](
    param: MutPointer[Float32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    buf: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    table: MutPointer[Int32, MutAnyOrigin],
    j_count_in: Int32,
    n_in: Int32,
    nesterov_in: Int32,
    momentum: Float32,
    c_damp: Float32,
    weight_decay: Float32,
    neg_lr: Float32,
):
    """MOJOLEARN_AFN_OPT_MULTITENSOR: ONE launch over the flat model.
    `table[0 .. J]` holds the tensor offsets (ascending, `table[J] == n`)
    and `table[J + 1 .. 2J]` the per-tensor momentum flags; a thread finds
    its tensor by a binary search over the offsets (`log2 J` loads of a
    tiny, cached table) and takes that tensor's flag."""
    comptime if GATED:
        if cells.unsafe_load(AFN_CELL_GATE) != Int32(0):
            return
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var j_count = Int(j_count_in)
    var lo = 0
    var hi = j_count
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if Int(table.unsafe_load(mid)) <= i:
            lo = mid
        else:
            hi = mid
    var flag = table.unsafe_load(j_count + 1 + lo)
    var coef = Float32(1.0)
    comptime if CLIP:
        coef = fcells.unsafe_load(1)
    _afn_sgd_apply[CLIP](
        param, grad, buf, i, coef, flag != Int32(0), nesterov_in != Int32(0),
        momentum, c_damp, weight_decay, neg_lr,
    )


# ===========================================================================
# THE OPTIMIZER STEP (the device half, after main's refusal scan and clip
# when their fused forms are off)
# ===========================================================================


def afn_optimizer_step(
    ctx: DeviceContext,
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    mut out2: DeviceBuffer[DType.float32],
    mut buf_initialized: List[Bool],
    offsets: List[Int],
    cfg: OptimizerConfig,
    sc: StepScalars,
) raises:
    """One step under the `AFN_OPT_*` candidates that are on. The caller
    (`identical_optimizer_step`) has already run main's refusal scan when
    `AFN_OPT_FUSE_SCAN` is off and main's clip when `AFN_OPT_CLIP_FUSE` is
    off, and has checked `offsets`. Synchronizes ONCE before it returns;
    with the fused scan or the fused clip on, the host reads the gate cells
    after that wait and raises the oracle's message when the step was
    refused (the kernels withheld every write)."""
    var j_count = len(offsets) - 1
    var n = offsets[j_count]
    var is_sgd = cfg.kind == OPT_SGD
    var want_clip = cfg.max_norm > Float32(0.0)
    var clipf = AFN_OPT_CLIP_FUSE and want_clip
    var blocks = _afn_scan_grid(n)

    # ---- scratch: pooled under RESIDENT_STATE, else fresh and kept alive
    # past the wait by the lists below.
    var keep_i = List[DeviceBuffer[DType.int32]]()
    var keep_f = List[DeviceBuffer[DType.float32]]()
    var keep_h = List[HostBuffer[DType.int32]]()
    var part_n = 4 * blocks if AFN_OPT_FUSE_SCAN else 1
    var sums_n = blocks if clipf else 1
    var table_n = 2 * j_count + 1 if (AFN_OPT_MULTITENSOR and is_sgd) else 1
    var part: MutPointer[Int32, MutAnyOrigin]
    var sums: MutPointer[Float32, MutAnyOrigin]
    var cells: MutPointer[Int32, MutAnyOrigin]
    var fcells: MutPointer[Float32, MutAnyOrigin]
    var table: MutPointer[Int32, MutAnyOrigin]
    var host: MutPointer[Int32, MutAnyOrigin]
    comptime if AFN_OPT_RESIDENT_STATE:
        part = afn_scratch_i32(ctx, AFN_SI_PART, part_n)
        cells = afn_scratch_i32(ctx, AFN_SI_CELLS, AFN_CELLS)
        table = afn_scratch_i32(ctx, AFN_SI_TABLE, table_n)
        sums = afn_scratch_f32(ctx, AFN_SF_SUMS, sums_n)
        fcells = afn_scratch_f32(ctx, AFN_SF_FCELLS, AFN_FCELLS)
        host = afn_scratch_host_i32(ctx, AFN_SH_CELLS, AFN_CELLS)
    else:
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](part_n))
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](AFN_CELLS))
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](table_n))
        step_count_device_alloc()
        keep_f.append(ctx.enqueue_create_buffer[DType.float32](sums_n))
        step_count_device_alloc()
        keep_f.append(ctx.enqueue_create_buffer[DType.float32](AFN_FCELLS))
        step_count_host_alloc()
        keep_h.append(ctx.enqueue_create_host_buffer[DType.int32](AFN_CELLS))
        part = keep_i[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        cells = keep_i[1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        table = keep_i[2].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        sums = keep_f[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        fcells = keep_f[1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        host = keep_h[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    # ---- the one scan / partial-sum pass and the fold into the cells.
    var fused = AFN_OPT_FUSE_SCAN or clipf
    if fused:
        comptime if AFN_OPT_FUSE_SCAN:
            if clipf:
                if is_sgd:
                    _afn_launch_scan[True, False, True](
                        ctx, part, sums, param, grad, m_state, v_state, n, blocks
                    )
                    _afn_launch_fold[True, False, True](
                        ctx, cells, fcells, out2, part, sums, blocks, cfg.max_norm
                    )
                else:
                    _afn_launch_scan[True, True, True](
                        ctx, part, sums, param, grad, m_state, v_state, n, blocks
                    )
                    _afn_launch_fold[True, True, True](
                        ctx, cells, fcells, out2, part, sums, blocks, cfg.max_norm
                    )
            else:
                if is_sgd:
                    _afn_launch_scan[True, False, False](
                        ctx, part, sums, param, grad, m_state, v_state, n, blocks
                    )
                    _afn_launch_fold[True, False, False](
                        ctx, cells, fcells, out2, part, sums, blocks, cfg.max_norm
                    )
                else:
                    _afn_launch_scan[True, True, False](
                        ctx, part, sums, param, grad, m_state, v_state, n, blocks
                    )
                    _afn_launch_fold[True, True, False](
                        ctx, cells, fcells, out2, part, sums, blocks, cfg.max_norm
                    )
        else:
            # the fused clip alone: the partial sums and the coefficient
            _afn_launch_scan[False, False, True](
                ctx, part, sums, param, grad, m_state, v_state, n, blocks
            )
            _afn_launch_fold[False, False, True](
                ctx, cells, fcells, out2, part, sums, blocks, cfg.max_norm
            )

    # ---- the update.
    var p_ptr = param.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var g_ptr = grad.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var m_ptr = m_state.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var v_ptr = v_state.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var tab = List[Int32]()
    if is_sgd:
        var nest = Int32(1) if cfg.nesterov else Int32(0)
        comptime if AFN_OPT_MULTITENSOR:
            tab.reserve(table_n)
            for j in range(j_count + 1):
                tab.append(Int32(offsets[j]))
            for j in range(j_count):
                tab.append(Int32(1) if buf_initialized[j] else Int32(0))
            comptime if AFN_OPT_RESIDENT_STATE:
                var s = AFN_SCRATCH.get_or_create_ptr()
                ctx.enqueue_copy(
                    dst_buf=s[].i32[AFN_SI_TABLE], src_ptr=tab.unsafe_ptr()
                )
            else:
                ctx.enqueue_copy(dst_buf=keep_i[2], src_ptr=tab.unsafe_ptr())
            _afn_launch_sgd_multi(
                ctx, p_ptr, g_ptr, m_ptr, cells, fcells, table, j_count, n,
                nest, cfg, sc, fused, clipf,
            )
        else:
            for j in range(j_count):
                var begin = offsets[j]
                var count = offsets[j + 1] - begin
                if count <= 0:
                    continue
                var init_flag = Int32(1) if buf_initialized[j] else Int32(0)
                _afn_launch_sgd_one(
                    ctx, p_ptr, g_ptr, m_ptr, cells, fcells, begin, count,
                    init_flag, nest, cfg, sc, fused, clipf,
                )
    else:
        var is_adamw = Int32(1) if cfg.kind == OPT_ADAMW else Int32(0)
        _afn_launch_adam(
            ctx, p_ptr, g_ptr, m_ptr, v_ptr, cells, fcells, n, is_adamw,
            cfg, sc, fused, clipf,
        )

    # ---- the one wait, then the gate.
    if fused:
        comptime if AFN_OPT_RESIDENT_STATE:
            var s2 = AFN_SCRATCH.get_or_create_ptr()
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=host, src_buf=s2[].i32[AFN_SI_CELLS])
        else:
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=host, src_buf=keep_i[1])
    step_count_sync()
    ctx.synchronize()
    _ = tab^
    if fused:
        _afn_refuse_from_cells(ctx, host, param, grad, m_state, v_state, is_sgd)
    if is_sgd and cfg.momentum != Float32(0.0):
        for j in range(j_count):
            buf_initialized[j] = True
    _ = keep_i^
    _ = keep_f^
    _ = keep_h^
    _ = param
    _ = grad
    _ = m_state
    _ = v_state
    _ = out2


def _afn_launch_scan[SCAN: Bool, FOUR: Bool, CLIP: Bool](
    ctx: DeviceContext,
    part: MutPointer[Int32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    n: Int,
    blocks: Int,
) raises:
    comptime kern = afn_scan4_kernel[SCAN, FOUR, CLIP]
    step_count_launch()
    ctx.enqueue_function[kern](
        part,
        sums,
        param.unsafe_ptr(),
        grad.unsafe_ptr(),
        m_state.unsafe_ptr(),
        v_state.unsafe_ptr(),
        Int32(n),
        grid_dim=(blocks, 1, 1),
        block_dim=(AFN_TPB, 1, 1),
    )


def _afn_launch_fold[SCAN: Bool, FOUR: Bool, CLIP: Bool](
    ctx: DeviceContext,
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    mut out2: DeviceBuffer[DType.float32],
    part: MutPointer[Int32, MutAnyOrigin],
    sums: MutPointer[Float32, MutAnyOrigin],
    blocks: Int,
    max_norm: Float32,
) raises:
    comptime kern = afn_fold_kernel[SCAN, FOUR, CLIP]
    step_count_launch()
    ctx.enqueue_function[kern](
        cells,
        fcells,
        out2.unsafe_ptr(),
        part,
        sums,
        Int32(blocks),
        max_norm,
        clip_eps(),
        grid_dim=(1, 1, 1),
        block_dim=(AFN_TPB, 1, 1),
    )


def _afn_launch_adam(
    ctx: DeviceContext,
    p_ptr: MutPointer[Float32, MutAnyOrigin],
    g_ptr: MutPointer[Float32, MutAnyOrigin],
    m_ptr: MutPointer[Float32, MutAnyOrigin],
    v_ptr: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    is_adamw: Int32,
    cfg: OptimizerConfig,
    sc: StepScalars,
    gated: Bool,
    clipf: Bool,
) raises:
    """One launch of `afn_adam_kernel` at the width, gate and clip form the
    step needs (the four runtime combinations are four instantiations)."""
    comptime W = AFN_VEC if AFN_OPT_VEC4 else 1
    var threads = (n + W - 1) // W
    var grid = _afn_grid(threads)
    if gated:
        if clipf:
            comptime k_gc = afn_adam_kernel[W, True, True]
            step_count_launch()
            ctx.enqueue_function[k_gc](
                p_ptr, g_ptr, m_ptr, v_ptr, cells, fcells, Int32(n), is_adamw,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, sc.c1, sc.c2,
                sc.step_size, sc.rt_bc2, sc.decay_mul,
                grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
        else:
            comptime k_g = afn_adam_kernel[W, True, False]
            step_count_launch()
            ctx.enqueue_function[k_g](
                p_ptr, g_ptr, m_ptr, v_ptr, cells, fcells, Int32(n), is_adamw,
                cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, sc.c1, sc.c2,
                sc.step_size, sc.rt_bc2, sc.decay_mul,
                grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
    else:
        comptime k_p = afn_adam_kernel[W, False, False]
        step_count_launch()
        ctx.enqueue_function[k_p](
            p_ptr, g_ptr, m_ptr, v_ptr, cells, fcells, Int32(n), is_adamw,
            cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, sc.c1, sc.c2,
            sc.step_size, sc.rt_bc2, sc.decay_mul,
            grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )


def _afn_launch_sgd_one(
    ctx: DeviceContext,
    p_ptr: MutPointer[Float32, MutAnyOrigin],
    g_ptr: MutPointer[Float32, MutAnyOrigin],
    m_ptr: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    begin: Int,
    count: Int,
    init_flag: Int32,
    nest: Int32,
    cfg: OptimizerConfig,
    sc: StepScalars,
    gated: Bool,
    clipf: Bool,
) raises:
    var grid = _afn_grid(count)
    if gated:
        if clipf:
            comptime k_gc = afn_sgd_kernel[True, True]
            step_count_launch()
            ctx.enqueue_function[k_gc](
                p_ptr, g_ptr, m_ptr, cells, fcells, Int32(begin), Int32(count),
                init_flag, nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
                sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
        else:
            comptime k_g = afn_sgd_kernel[True, False]
            step_count_launch()
            ctx.enqueue_function[k_g](
                p_ptr, g_ptr, m_ptr, cells, fcells, Int32(begin), Int32(count),
                init_flag, nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
                sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
    else:
        comptime k_p = afn_sgd_kernel[False, False]
        step_count_launch()
        ctx.enqueue_function[k_p](
            p_ptr, g_ptr, m_ptr, cells, fcells, Int32(begin), Int32(count),
            init_flag, nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
            sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )


def _afn_launch_sgd_multi(
    ctx: DeviceContext,
    p_ptr: MutPointer[Float32, MutAnyOrigin],
    g_ptr: MutPointer[Float32, MutAnyOrigin],
    m_ptr: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    fcells: MutPointer[Float32, MutAnyOrigin],
    table: MutPointer[Int32, MutAnyOrigin],
    j_count: Int,
    n: Int,
    nest: Int32,
    cfg: OptimizerConfig,
    sc: StepScalars,
    gated: Bool,
    clipf: Bool,
) raises:
    var grid = _afn_grid(n)
    if gated:
        if clipf:
            comptime k_gc = afn_sgd_multi_kernel[True, True]
            step_count_launch()
            ctx.enqueue_function[k_gc](
                p_ptr, g_ptr, m_ptr, cells, fcells, table, Int32(j_count),
                Int32(n), nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
                sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
        else:
            comptime k_g = afn_sgd_multi_kernel[True, False]
            step_count_launch()
            ctx.enqueue_function[k_g](
                p_ptr, g_ptr, m_ptr, cells, fcells, table, Int32(j_count),
                Int32(n), nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
                sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
            )
    else:
        comptime k_p = afn_sgd_multi_kernel[False, False]
        step_count_launch()
        ctx.enqueue_function[k_p](
            p_ptr, g_ptr, m_ptr, cells, fcells, table, Int32(j_count),
            Int32(n), nest, cfg.momentum, sc.c_damp, cfg.weight_decay,
            sc.neg_lr, grid_dim=(grid, 1, 1), block_dim=(AFN_TPB, 1, 1),
        )


def _afn_refuse_from_cells(
    ctx: DeviceContext,
    host: MutPointer[Int32, MutAnyOrigin],
    mut param: DeviceBuffer[DType.float32],
    mut grad: DeviceBuffer[DType.float32],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    is_sgd: Bool,
) raises:
    """After the wait: the oracle's refusal in the oracle's order (`param`,
    `grad`, then the moments), with its message, from the gate cells. The
    one 4 B classification readback runs on the error path only."""
    if host.unsafe_load(AFN_CELL_GATE) == Int32(0):
        return
    var count = 3 if is_sgd else 4
    for k in range(count):
        var idx = host.unsafe_load(k)
        if idx == NONFINITE_NONE:
            continue
        var hit = Int(idx)
        var is_nan: Bool
        var name: String
        if k == 0:
            is_nan = device_classify_nonfinite(ctx, param, hit)
            name = String("param")
        elif k == 1:
            is_nan = device_classify_nonfinite(ctx, grad, hit)
            name = String("grad")
        elif k == 2:
            is_nan = device_classify_nonfinite(ctx, m_state, hit)
            name = String("momentum_buffer") if is_sgd else String("exp_avg")
        else:
            is_nan = device_classify_nonfinite(ctx, v_state, hit)
            name = String("exp_avg_sq")
        raise Error(opt_nonfinite_message(name, hit, is_nan))
    if host.unsafe_load(AFN_CELL_CLIP_BAD) != Int32(0):
        # the fold kernel saw a non-finite total norm: the scalar main
        # refuses on the host, refused with the same name
        refuse_nonfinite_scalar(
            String("clip.total_norm"), bitcast[DType.float32](UInt32(0x7FC00000))
        )
    raise Error("optimizer step: the device gate refused the step")


# ===========================================================================
# THE FUSED CROSS-ENTROPY (MOJOLEARN_AFN_LOSS_FUSED)
# ===========================================================================


def afn_ce_row_kernel[TPB: Int](
    row_out: MutPointer[Float32, MutAnyOrigin],
    dlogits: MutPointer[Float32, MutAnyOrigin],
    bad_nf: MutPointer[Int32, MutAnyOrigin],
    bad_tg: MutPointer[Int32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
    ignore_index_in: Int32,
    want_grad_in: Int32,
    smoothing_in: Int32,
    t_target: Float32,
    t_other: Float32,
    divisor: Float32,
    one_minus_eps: Float32,
    eps: Float32,
):
    """One block per row, seams L1 through L16 of the loss contract in one
    pass, the FAST spelling. The block strides over the vocabulary three
    times: max (with the non-finite scan of the row by bits), sum of
    exponentials (and the sum of shifted logits when smoothing is
    spelled), and, under `want_grad`, `dlogits`. Thread 0 writes the row's
    loss and its two flag cells (smallest non-finite flat index or
    `NONFINITE_NONE`; the row index when its target is out of range, else
    `NONFINITE_NONE`). The fold orders are the block's trees; f32
    throughout; `exp` and `log` are the stdlib's, which is what the FAST
    tier's `identical_exp` / `identical_log` are."""
    var row = Int(block_idx.x)
    var n_rows = Int(n_rows_in)
    if row >= n_rows:
        return
    var vocab = Int(vocab_in)
    var base = row * vocab
    var tid = Int(thread_idx.x)
    var redf = stack_allocation[
        TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var red = stack_allocation[
        TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    # ---- pass 1: the row maximum and the non-finite scan
    var mx = bitcast[DType.float32](UInt32(0xFF800000))
    var bad = NONFINITE_NONE
    var v = tid
    while v < vocab:
        var x = logits.unsafe_load(base + v)
        if bad == NONFINITE_NONE and _nonfinite_bits(x):
            bad = Int32(base + v)
        mx = max(mx, x)
        v += TPB
    redf.unsafe_store(tid, mx)
    red.unsafe_store(tid, bad)
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            redf.unsafe_store(
                tid, max(redf.unsafe_load(tid), redf.unsafe_load(tid + active))
            )
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    var m = redf.unsafe_load(0)
    var y = Int(targets.unsafe_load(row))
    var ignored = y == Int(ignore_index_in)
    if tid == 0:
        bad_nf.unsafe_store(row, red.unsafe_load(0))
        var tg = NONFINITE_NONE
        if not ignored and (y < 0 or y >= vocab):
            tg = Int32(row)
        bad_tg.unsafe_store(row, tg)
    barrier()
    # ---- pass 2: the denominator (and the smoothing sum)
    var smoothing = smoothing_in != Int32(0)
    var se = Float32(0.0)
    var ss = Float32(0.0)
    v = tid
    while v < vocab:
        var s = logits.unsafe_load(base + v) - m
        se = se + exp(s)
        if smoothing:
            ss = ss + s
        v += TPB
    redf.unsafe_store(tid, se)
    barrier()
    active = TPB // 2
    while active > 0:
        if tid < active:
            redf.unsafe_store(
                tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + active)
            )
        barrier()
        active = active // 2
    var denom = redf.unsafe_load(0)
    barrier()
    var sum_shift = Float32(0.0)
    if smoothing:
        redf.unsafe_store(tid, ss)
        barrier()
        active = TPB // 2
        while active > 0:
            if tid < active:
                redf.unsafe_store(
                    tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + active)
                )
            barrier()
            active = active // 2
        sum_shift = redf.unsafe_load(0)
        barrier()
    var logdenom = log(denom)
    # ---- the row loss (thread 0)
    if tid == 0:
        var yy = y
        if ignored or yy < 0 or yy >= vocab:
            yy = 0
        var lp = (logits.unsafe_load(base + yy) - m) - logdenom
        var nll = neg_by_bits(lp)
        var out = Float32(0.0)
        if not ignored:
            if smoothing:
                var logp_sum = sum_shift - Float32(vocab) * logdenom
                var smooth = neg_by_bits(logp_sum / Float32(vocab))
                out = one_minus_eps * nll + eps * smooth
            else:
                out = nll
        row_out.unsafe_store(row, out)
    # ---- pass 3: dlogits
    if want_grad_in != Int32(0):
        v = tid
        while v < vocab:
            var d = Float32(0.0)
            if not ignored:
                var w = exp(logits.unsafe_load(base + v) - m) / denom
                var t = t_other
                if v == y:
                    t = t_target
                d = (w - t) / divisor
            dlogits.unsafe_store(base + v, d)
            v += TPB


def afn_ce_part_kernel[TPB: Int](
    psum: MutPointer[Float32, MutAnyOrigin],
    pnf: MutPointer[Int32, MutAnyOrigin],
    ptg: MutPointer[Int32, MutAnyOrigin],
    row: MutPointer[Float32, MutAnyOrigin],
    bad_nf: MutPointer[Int32, MutAnyOrigin],
    bad_tg: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    want_total_in: Int32,
):
    """The loss fold's first level: block `b` folds rows `b * TPB + tid`,
    stride `grid * TPB` (free order), into `psum[b]` (the row-loss sum
    under `want_total`), `pnf[b]` and `ptg[b]` (the flag minima).
    `afn_ce_fold_kernel` folds the partials."""
    var n_rows = Int(n_rows_in)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var stride = Int(grid_dim.x) * TPB
    var redf = stack_allocation[
        TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var rnf = stack_allocation[
        TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var rtg = stack_allocation[
        TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var acc = Float32(0.0)
    var nf = NONFINITE_NONE
    var tg = NONFINITE_NONE
    var want_total = want_total_in != Int32(0)
    var r = blk * TPB + tid
    while r < n_rows:
        if want_total:
            acc = acc + row.unsafe_load(r)
        var a = bad_nf.unsafe_load(r)
        if a < nf:
            nf = a
        var b = bad_tg.unsafe_load(r)
        if b < tg:
            tg = b
        r += stride
    redf.unsafe_store(tid, acc)
    rnf.unsafe_store(tid, nf)
    rtg.unsafe_store(tid, tg)
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            redf.unsafe_store(
                tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + active)
            )
            var o = rnf.unsafe_load(tid + active)
            if o < rnf.unsafe_load(tid):
                rnf.unsafe_store(tid, o)
            var o2 = rtg.unsafe_load(tid + active)
            if o2 < rtg.unsafe_load(tid):
                rtg.unsafe_store(tid, o2)
        barrier()
        active = active // 2
    if tid == 0:
        psum.unsafe_store(blk, redf.unsafe_load(0))
        pnf.unsafe_store(blk, rnf.unsafe_load(0))
        ptg.unsafe_store(blk, rtg.unsafe_load(0))


def afn_ce_fold_kernel[TPB: Int](
    loss_out: MutPointer[Float32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    row: MutPointer[Float32, MutAnyOrigin],
    bad_nf: MutPointer[Int32, MutAnyOrigin],
    bad_tg: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    divisor: Float32,
    want_total_in: Int32,
):
    """ONE block over `afn_ce_part_kernel`'s partials (at most
    AFN_CE_PARTS of them; `row`, `bad_nf`, `bad_tg` are the partials and
    `n_rows_in` their count): the batch fold (free order) divided
    once by `divisor` into `loss_out[0]` (seam L13) under `want_total`, and
    the minimum over the per-row flag cells into `cells[0]` (non-finite
    logit index), `cells[1]` (first bad-target row) and the gate
    `cells[2]`."""
    var n_rows = Int(n_rows_in)
    var tid = Int(thread_idx.x)
    var redf = stack_allocation[
        TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var red = stack_allocation[
        TPB,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var acc = Float32(0.0)
    var nf = NONFINITE_NONE
    var tg = NONFINITE_NONE
    var want_total = want_total_in != Int32(0)
    var r = tid
    while r < n_rows:
        if want_total:
            acc = acc + row.unsafe_load(r)
        var a = bad_nf.unsafe_load(r)
        if a < nf:
            nf = a
        var b = bad_tg.unsafe_load(r)
        if b < tg:
            tg = b
        r += TPB
    redf.unsafe_store(tid, acc)
    red.unsafe_store(tid, nf)
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            redf.unsafe_store(
                tid, redf.unsafe_load(tid) + redf.unsafe_load(tid + active)
            )
            var o = red.unsafe_load(tid + active)
            if o < red.unsafe_load(tid):
                red.unsafe_store(tid, o)
        barrier()
        active = active // 2
    var total = redf.unsafe_load(0)
    var nf_min = red.unsafe_load(0)
    barrier()
    red.unsafe_store(tid, tg)
    barrier()
    active = TPB // 2
    while active > 0:
        if tid < active:
            var o2 = red.unsafe_load(tid + active)
            if o2 < red.unsafe_load(tid):
                red.unsafe_store(tid, o2)
        barrier()
        active = active // 2
    if tid == 0:
        var tg_min = red.unsafe_load(0)
        cells.unsafe_store(0, nf_min)
        cells.unsafe_store(1, tg_min)
        var gate = Int32(0)
        if nf_min != NONFINITE_NONE or tg_min != NONFINITE_NONE:
            gate = Int32(1)
        cells.unsafe_store(2, gate)
        if want_total:
            loss_out.unsafe_store(0, total / divisor)


def afn_ce_loss_resident(
    ctx: DeviceContext,
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    row_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut dlogits: DeviceBuffer[DType.float32],
    mut logits: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    count: Int,
    reduction: Int,
    want_grad: Int,
    cfg: CeConfig,
) raises:
    """`identical_ce_loss_resident`'s contract under MOJOLEARN_AFN_LOSS_FUSED:
    the same inputs, `dlogits` written on the device when `want_grad`,
    `row_ptr` and `loss_ptr` written on the host, the shape / non-finite /
    target refusals raised with the oracle's messages. Two launches, one
    wait."""
    var vocab = cfg.vocab
    ce_refuse_shape(n_rows, n_rows * vocab, cfg)
    if n_rows < 1 or vocab < 1:
        loss_ptr.unsafe_store(0, Float32(0.0))
        return
    var want_total = reduction != REDUCTION_NONE
    var divisor = Float32(1.0)
    if want_total:
        divisor = ce_divisor(reduction, count, cfg.num_items)
    var tv = ce_smoothing_targets(cfg.eps, vocab)
    var one_minus = ce_one_minus_eps(cfg.eps)
    var smoothing = Int32(1) if cfg.smoothing_is_spelled() else Int32(0)

    var keep_i = List[DeviceBuffer[DType.int32]]()
    var keep_f = List[DeviceBuffer[DType.float32]]()
    var keep_h = List[HostBuffer[DType.int32]]()
    var row: MutPointer[Float32, MutAnyOrigin]
    var loss: MutPointer[Float32, MutAnyOrigin]
    var bad: MutPointer[Int32, MutAnyOrigin]
    var cells: MutPointer[Int32, MutAnyOrigin]
    var psum: MutPointer[Float32, MutAnyOrigin]
    var pflag: MutPointer[Int32, MutAnyOrigin]
    var host: MutPointer[Int32, MutAnyOrigin]
    comptime if AFN_OPT_RESIDENT_STATE:
        row = afn_scratch_f32(ctx, AFN_SF_LOSS_ROW, n_rows)
        loss = afn_scratch_f32(ctx, AFN_SF_LOSS_LOSS, 1)
        bad = afn_scratch_i32(ctx, AFN_SI_LOSS_BAD, 2 * n_rows)
        cells = afn_scratch_i32(ctx, AFN_SI_LOSS_CELLS, AFN_LOSS_CELLS)
        psum = afn_scratch_f32(ctx, AFN_SF_LOSS_PART, AFN_CE_PARTS)
        pflag = afn_scratch_i32(ctx, AFN_SI_LOSS_PART, 2 * AFN_CE_PARTS)
        host = afn_scratch_host_i32(ctx, AFN_SH_LOSS_CELLS, AFN_LOSS_CELLS)
    else:
        step_count_device_alloc()
        keep_f.append(ctx.enqueue_create_buffer[DType.float32](n_rows))
        step_count_device_alloc()
        keep_f.append(ctx.enqueue_create_buffer[DType.float32](1))
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](2 * n_rows))
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](AFN_LOSS_CELLS))
        step_count_device_alloc()
        keep_f.append(ctx.enqueue_create_buffer[DType.float32](AFN_CE_PARTS))
        step_count_device_alloc()
        keep_i.append(ctx.enqueue_create_buffer[DType.int32](2 * AFN_CE_PARTS))
        step_count_host_alloc()
        keep_h.append(
            ctx.enqueue_create_host_buffer[DType.int32](AFN_LOSS_CELLS)
        )
        row = keep_f[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        loss = keep_f[1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        bad = keep_i[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        cells = keep_i[1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        psum = keep_f[2].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        pflag = keep_i[2].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        host = keep_h[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    # ---- the two launches: a 32-thread block per row when the vocabulary
    # is tiny (the MLP head), else 256.
    if vocab <= 32:
        _afn_launch_ce[32](
            ctx, row, loss, bad, cells, psum, pflag, dlogits, logits, targets, n_rows,
            vocab, cfg.ignore_index, want_grad, smoothing, tv[0], tv[1],
            divisor, one_minus, cfg.eps, want_total,
        )
    else:
        _afn_launch_ce[AFN_TPB](
            ctx, row, loss, bad, cells, psum, pflag, dlogits, logits, targets, n_rows,
            vocab, cfg.ignore_index, want_grad, smoothing, tv[0], tv[1],
            divisor, one_minus, cfg.eps, want_total,
        )

    # ---- the readbacks and the one wait
    comptime if AFN_OPT_RESIDENT_STATE:
        var s = AFN_SCRATCH.get_or_create_ptr()
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=row_ptr, src_buf=s[].f32[AFN_SF_LOSS_ROW])
        if want_total:
            step_count_d2h()
            ctx.enqueue_copy(
                dst_ptr=loss_ptr, src_buf=s[].f32[AFN_SF_LOSS_LOSS]
            )
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=host, src_buf=s[].i32[AFN_SI_LOSS_CELLS])
    else:
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=row_ptr, src_buf=keep_f[0])
        if want_total:
            step_count_d2h()
            ctx.enqueue_copy(dst_ptr=loss_ptr, src_buf=keep_f[1])
        step_count_d2h()
        ctx.enqueue_copy(dst_ptr=host, src_buf=keep_i[1])
    step_count_sync()
    ctx.synchronize()
    if host.unsafe_load(2) != Int32(0):
        _afn_ce_refuse(ctx, host, logits, targets, n_rows, cfg)
    if not want_total:
        loss_ptr.unsafe_store(0, Float32(0.0))
    _ = keep_i^
    _ = keep_f^
    _ = keep_h^
    _ = dlogits
    _ = logits
    _ = targets


def _afn_launch_ce[TPB: Int](
    ctx: DeviceContext,
    row: MutPointer[Float32, MutAnyOrigin],
    loss: MutPointer[Float32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin],
    cells: MutPointer[Int32, MutAnyOrigin],
    psum: MutPointer[Float32, MutAnyOrigin],
    pflag: MutPointer[Int32, MutAnyOrigin],
    mut dlogits: DeviceBuffer[DType.float32],
    mut logits: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    vocab: Int,
    ignore_index: Int,
    want_grad: Int,
    smoothing: Int32,
    t_target: Float32,
    t_other: Float32,
    divisor: Float32,
    one_minus: Float32,
    eps: Float32,
    want_total: Bool,
) raises:
    comptime row_kern = afn_ce_row_kernel[TPB]
    comptime part_kern = afn_ce_part_kernel[TPB]
    comptime fold_kern = afn_ce_fold_kernel[TPB]
    step_count_launch()
    ctx.enqueue_function[row_kern](
        row,
        dlogits.unsafe_ptr(),
        bad,
        bad + n_rows,
        logits.unsafe_ptr(),
        targets.unsafe_ptr(),
        Int32(n_rows),
        Int32(vocab),
        Int32(ignore_index),
        Int32(want_grad),
        smoothing,
        t_target,
        t_other,
        divisor,
        one_minus,
        eps,
        grid_dim=(n_rows, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    # The batch fold in two levels: up to AFN_CE_PARTS blocks fold the
    # rows into partials, then one block folds the partials.
    var n_parts = min((n_rows + TPB - 1) // TPB, AFN_CE_PARTS)
    step_count_launch()
    ctx.enqueue_function[part_kern](
        psum,
        pflag,
        pflag + AFN_CE_PARTS,
        row,
        bad,
        bad + n_rows,
        Int32(n_rows),
        Int32(1) if want_total else Int32(0),
        grid_dim=(n_parts, 1, 1),
        block_dim=(TPB, 1, 1),
    )
    step_count_launch()
    ctx.enqueue_function[fold_kern](
        loss,
        cells,
        psum,
        pflag,
        pflag + AFN_CE_PARTS,
        Int32(n_parts),
        divisor,
        Int32(1) if want_total else Int32(0),
        grid_dim=(1, 1, 1),
        block_dim=(TPB, 1, 1),
    )


def _afn_ce_refuse(
    ctx: DeviceContext,
    host: MutPointer[Int32, MutAnyOrigin],
    mut logits: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    cfg: CeConfig,
) raises:
    """The error path: the oracle's refusals in the oracle's order (the
    non-finite logit first, then the targets walk) with its messages. The
    classification readback and the targets download happen here only."""
    var nf = host.unsafe_load(0)
    if nf != NONFINITE_NONE:
        var idx = Int(nf)
        var is_nan = device_classify_nonfinite(ctx, logits, idx)
        raise Error(ce_nonfinite_message("logits", idx, is_nan))
    step_count_host_alloc()
    var ht = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
    step_count_sync()
    ctx.synchronize()
    step_count_d2h()
    ctx.enqueue_copy(dst_ptr=ht.unsafe_ptr(), src_buf=targets)
    step_count_sync()
    ctx.synchronize()
    var ht_l = List[Int32](capacity=n_rows)
    var rows_n = n_rows
    var r = 0
    while r < rows_n:
        ht_l.append(ht.unsafe_ptr().unsafe_load(r))
        r += 1
    _ = ht^
    ce_refuse_targets(ht_l, cfg)
    raise Error("ce: the device gate refused the call")
