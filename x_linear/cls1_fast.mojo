# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-cls1 (2026-10-03): FAST + Apple build-time candidates
for the M3 FAST board rows where we trailed the best opponent
(bayesian-ridge, ard, ridge-clf taxi). Every switch is the FAST + Apple
default since the M3 A/B (n=1, quality identical: bayesian-ridge taxi
102 -> 15.3 ms with all three BAYES switches, ard taxi 14.4 -> 8.3 ms with
all three ARD switches, ridge-clf taxi 120 -> 19.0 ms with CODES), off with
-D <NAME>_OFF, and compiles to nothing on IDENTICAL or off Apple. Notes:
docs/apple-fast/notes/gap-cls1.md.

  * MOJOLEARN_BAYES_FAST_CLS1_STATS: BayesianRidge's means, Gram and X'y
    from the shared grid Gram (x_linear/fast_gram.mojo) in place of the
    one-block moments tile and `bayes_xty_kernel` (one thread a column over
    every row).
  * MOJOLEARN_BAYES_FAST_CLS1_PARTS / MOJOLEARN_ARD_FAST_CLS1_PARTS: the
    FOLD_BLOCK row-block partials (target sum, target variance, the
    iteration's squared residuals) a whole block per row block (strided
    loads, a shared-memory tree) in place of ONE thread walking 4,096 rows.
  * MOJOLEARN_BAYES_FAST_CLS1_BATCH: the guarded BayesianRidge iterations
    queued C1_BATCH at a time: the row pass and the step read the
    stop / trust words on the device, one witness check and one state read
    per batch instead of two synchronizations per iteration.
  * MOJOLEARN_ARD_FAST_CLS1_BATCH: ARD's iterations already no-op after the
    stop word; the witness check (a synchronize) now runs once per AG_BATCH
    iterations instead of every iteration.
  * MOJOLEARN_ARD_FAST_CLS1_STATS: ARD's moments of [X | y] from the shared
    grid Gram instead of the one-block moments tile.
  * MOJOLEARN_RIDGE_FAST_CLS1_CODES: RidgeClassifier hands the int32 class
    codes; the +-1 targets are built on the device (`c1_codes_targets_kernel`)
    instead of two Python list passes over every row.

FAST promises quality, not bits: the block trees are pairwise sums.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_linear.ops import FP, IP, ld, st, ldi, i2f
from x_linear.tops import fold_parts, fold_blocks, FOLD_BLOCK
from x_linear.witness import witness_end

comptime C1_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()

comptime BAYES_CLS1_STATS = C1_FAST_APPLE and not is_defined["MOJOLEARN_BAYES_FAST_CLS1_STATS_OFF"]()
comptime BAYES_CLS1_PARTS = C1_FAST_APPLE and not is_defined["MOJOLEARN_BAYES_FAST_CLS1_PARTS_OFF"]()
comptime BAYES_CLS1_BATCH = C1_FAST_APPLE and not is_defined["MOJOLEARN_BAYES_FAST_CLS1_BATCH_OFF"]()
comptime ARD_CLS1_STATS = C1_FAST_APPLE and not is_defined["MOJOLEARN_ARD_FAST_CLS1_STATS_OFF"]()
comptime ARD_CLS1_PARTS = C1_FAST_APPLE and not is_defined["MOJOLEARN_ARD_FAST_CLS1_PARTS_OFF"]()
comptime ARD_CLS1_BATCH = C1_FAST_APPLE and not is_defined["MOJOLEARN_ARD_FAST_CLS1_BATCH_OFF"]()
comptime RIDGE_CLS1_CODES = C1_FAST_APPLE and not is_defined["MOJOLEARN_RIDGE_FAST_CLS1_CODES_OFF"]()

comptime C1_TPB = 256
comptime C1_LOG_TPB = 8
comptime C1_BATCH = 8
"""BayesianRidge iterations queued per witness check (BAYES_CLS1_BATCH)."""
comptime C1_BAYES_STATE = 16
"""State words the batched BayesianRidge loop reads (8: the iteration count)."""


def cls1_flags() -> Int:
    """Bit 1: RIDGE_CLS1_CODES (the Python side reads it from the binding,
    so an A/B of two builds needs no env)."""
    var f = 0
    comptime if RIDGE_CLS1_CODES:
        f |= 1
    return f


@always_inline
def _block_sum(v: Float32) -> Float32:
    """The block's sum of v (every thread calls it; C1_TPB threads)."""
    var sh = stack_allocation[C1_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    sh[tid] = v
    barrier()
    comptime for lv in range(C1_LOG_TPB):
        comptime h = C1_TPB >> (lv + 1)
        if tid < h:
            sh[tid] = sh[tid] + sh[tid + h]
        barrier()
    var r = sh[0]
    barrier()
    return r


@always_inline
def _gate(state: FP, done_w: Int32, need_w: Int32, need_nz: Int32) -> Bool:
    """done_w < 0: always live; else live while state[done_w] == 0 and
    (state[need_w] != 0) == (need_nz != 0)."""
    if Int(done_w) < 0:
        return True
    if ld(state, Int(done_w)) != Float32(0):
        return False
    return (ld(state, Int(need_w)) != Float32(0)) == (need_nz != 0)


def c1_sq_parts_kernel(rows: FP, n: Int32, state: FP, done_w: Int32, need_w: Int32, need_nz: Int32,
                       parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Block b: sum of rows[i]^2 over row block b (FOLD_BLOCK rows), gated."""
    var nn = Int(n)
    var b = Int(block_idx.x)
    if b < fold_blocks(nn) and _gate(state, done_w, need_w, need_nz):
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        var acc = Float32(0)
        var i = Int(thread_idx.x)
        while i < cnt:
            var r = ld(rows, lo + i)
            acc += r * r
            i += C1_TPB
        var s = _block_sum(acc)
        if Int(thread_idx.x) == 0:
            st(parts, b, s)
    witness_end(wf, woff, nonce)


def c1_sum_parts_kernel(y: FP, n: Int32, parts: FP, state: FP, wsum_w: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Block b: sum of y over row block b; block 0 thread 0 also writes n
    into state[wsum_w] when wsum_w >= 0."""
    var nn = Int(n)
    var b = Int(block_idx.x)
    if b < fold_blocks(nn):
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        var acc = Float32(0)
        var i = Int(thread_idx.x)
        while i < cnt:
            acc += ld(y, lo + i)
            i += C1_TPB
        var s = _block_sum(acc)
        if Int(thread_idx.x) == 0:
            st(parts, b, s)
            if b == 0 and Int(wsum_w) >= 0:
                st(state, Int(wsum_w), i2f(nn))
    witness_end(wf, woff, nonce)


def c1_dev_parts_kernel(y: FP, n: Int32, yparts: FP, vparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Block b: sum of (y_i - m)^2 over row block b, m the mean from yparts."""
    var nn = Int(n)
    var nb = fold_blocks(nn)
    var b = Int(block_idx.x)
    if b < nb:
        var m = fold_parts(yparts, 0, nb) / i2f(nn)
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        var acc = Float32(0)
        var i = Int(thread_idx.x)
        while i < cnt:
            var r = ld(y, lo + i) - m
            acc += r * r
            i += C1_TPB
        var s = _block_sum(acc)
        if Int(thread_idx.x) == 0:
            st(vparts, b, s)
    witness_end(wf, woff, nonce)


def c1_codes_targets_kernel(codes: IP, n: Int32, t_n: Int32, dst: FP):
    """Row i: the +-1 targets of class code codes[i] (int32): T == 1, +1 for code 1; else +1 in column code."""
    var i = Int(block_idx.x) * C1_TPB + Int(thread_idx.x)
    var tn = Int(t_n)
    if i < Int(n):
        var c = Int(ldi(codes, i))
        if tn == 1:
            st(dst, i, Float32(1) if c == 1 else Float32(-1))
        else:
            for t in range(tn):
                st(dst, i * tn + t, Float32(1) if c == t else Float32(-1))
