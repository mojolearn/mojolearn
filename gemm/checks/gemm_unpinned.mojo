# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE UNPINNED CONTROL ARM. A MEASUREMENT INSTRUMENT, NOT A PRODUCT.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** It was
written on 2026-08-25 by a lane with no execution rights at all -- no `mojo`,
no `pixi`, no build, no test. Every claim below about what it computes is a
claim about the SOURCE, and every claim about what it costs is a PREDICTION
recorded before measurement (`gemm/UNPINNED_CONTROL.md` section 5). Nothing
here may be quoted as a result. The first person to build it should expect to
fix compile errors, and `gemm/UNPINNED_CONTROL.md`'s Appendix names the five
constructions most likely to be the ones that fail.

WHAT THIS IS FOR
-----------------
`bench/gemm_price_main.mojo::_timed_shape_with_device`, DEVIATION 1092,
2026-08-25:

> The arm that WOULD answer it does not exist anywhere in this repository:
> the identical kernel's OWN tiled plan with ONLY the fold-order pin removed,
> everything else held. One variable. Until that is built and run, the cost
> of `mojolearn.identical.gemm.fp32.v1` as distinct from the cost of writing
> our own GEMM is UNMEASURED.

**DEVIATION 1130: this file is that arm.** It takes
`gemm/checks/gemm_identical.mojo`'s own execution plans -- the same tile
shapes, the same block dims, the same grid, the same swizzles, the same
staging windows, the same barrier count, the same workspace, the same plan
dispatcher, reached BY IMPORT and not by copy -- and removes the numerical
plan of `gemm/IDENTICAL_FP32_CONTRACT.md` and nothing else.

WHAT IT DELETES, AND WHAT IT KEEPS (the clause list, table in the .md)
----------------------------------------------------------------------
DELETED, all six clauses of the pin:

1. the leaf partition AS AN ARITHMETIC BOUNDARY (contract 6);
2. serial ascending accumulation within a leaf, with no sub-partition
   (contract 7.1) -- this arm accumulates into `NACC` independent registers,
   which is verbatim the construction 7.1 forbids;
3. `identical_mul_add` (contract 4) -- this arm writes `acc + x * y` and lets
   the backend contract it or not, per backend, per context;
4. `ftz` at all seven seams 5a..5g (contract 5);
5. the fixed balanced adjacent-pair fold tree with bit-for-bit carries
   (contract 7.2) -- the TILE and FLAT plans have no fold at all, and the
   SPLITK plan folds by STRIDE with no carry rule;
6. `+0.0` leaf seeding and the unconditional fold at `P == 1` (contract 7.3,
   9.2) -- there is one accumulator set seeded `+0.0` for the whole `k` here,
   not one per leaf, so the seed is a different clause and `P == 1` is not a
   case at all.

KEPT, bit for bit, because they are the confound this arm exists to remove:

- `choose_gemm_plan`, imported. The same shape picks the same plan.
- `contract_partition`, imported. **The leaf boundaries are still walked**
  -- see DEVIATION 1131 below.
- `gemm_operand_strides`, imported. Same addressing, same three orientations.
- `_tile_grid`, `_leaf_bounds`, `FLAT_TPB`, `SPLITK_LEAF_TPB`,
  `SPLITK_FOLD_TPB`, `SWIZZLE_*`, `CONTRACT_MAX_LEAVES`, all imported. **Not
  one tile constant is respelled here**, because a copied constant is a
  constant that drifts, and a drifted tile constant would put the tiling back
  into the experiment and make the number worthless.
- the staging loops, the `barrier()` placement and the out-of-range masking
  of the tiled kernel, character for character.

DEVIATION 1131: THE LEAF LOOP SURVIVES AS A SCHEDULE, NOT AS ARITHMETIC
------------------------------------------------------------------------
The outer `for t in range(p_count)` loop and the `_leaf_bounds` clamp are
still here, and they are here ON PURPOSE even though this arm has no leaves.

The leaf partition does two jobs in the pinned kernel. One is numerical, and
it says where a fold-tree boundary falls. The other is a SCHEDULE, in that
the staging window is clamped to the leaf's end, so at any `k` where the leaf
length is not a multiple of `KS` the pinned kernel pays an extra short staging
round trip and an extra pair of barriers at every leaf boundary. Delete the
leaf loop and this arm would move BOTH -- the arithmetic and the memory
traffic -- and the confound would be back.

So the accumulators live ACROSS leaves and across staging windows, and the
leaf boundary is visible to the loop nest and invisible to the arithmetic.
The barrier count, the shared-memory transaction count and the DRAM
transaction count are identical to the pinned arm's at every shape,
INCLUDING the ragged tail, and the only surviving difference is what happens
to the values once they are in registers.

DEVIATION 1132: `NACC`, AND WHY THERE ARE TWO ARMS RATHER THAN ONE
-------------------------------------------------------------------
Contract 7.1: *"No sub-partition of a leaf is permitted, and that is the
clause a register-tiled or vectorized kernel is most likely to violate:
accumulating four `Veclen` lanes into four registers and adding them at the
end is a balanced tree of depth 2 hidden inside what the contract calls one
leaf, and it is a different answer."*

That clause is the one that looks most expensive on paper and it is a term
the other five cannot show on their own, so this file exposes it as a
parameter and ships TWO named entry points, which BOTH live in this one
binary and can therefore alternate call by call inside one thermal window
(`[[the M4 drifts 1.7x in 20 minutes]]`), namely

    unpinned_gemm_into            NACC = UNPINNED_NACC (4)
        The honest unpinned kernel. Anybody writing a GEMM without a bit
        contract writes this one, because a single dependent FMA chain
        leaves the pipeline mostly empty.

    unpinned_gemm_into_one_acc    NACC = 1
        The STRICT control. Same single dependent chain the pinned kernel
        has, with only the flush, the contraction pin and the fold removed.
        This one isolates the per-step tax; the difference between the two
        isolates the ILP the pin forbids.

Neither is more correct than the other. The pair is the decomposition, and
reporting only one of them would be choosing which half of the answer to
show.

DEVIATION 1133: THE SPLITK ARM FOLDS BY STRIDE AND KEEPS THE PING-PONG
-----------------------------------------------------------------------
The unpinned split-K fold is `buf[q] + buf[q + half]` -- the stride pairing
that `core/pinned_reduce.mojo::pinned_block_sum` ships and that contract 7.2
clause 1 forbids, with no carry rule and no flush. It is what a fold looks
like when nobody is pinning it.

It KEEPS the pinned kernel's ping-pong buffers even though a stride fold can
run in place, and it allocates the same `2 * CONTRACT_MAX_LEAVES` floats of
threadgroup memory even though it uses half. Threadgroup footprint is an
occupancy input on all three vendors, so halving it would be a second
variable. The level count, and therefore the barrier count, is
`ceil(log2 P)` under both pairings, so that is held too.

`PLAN_SPLITK_STAGED` is REFUSED rather than unpinned (DEVIATION 1134). It is
a realization OF the tree -- one global launch per logical LEVEL, addressed
by `fold_level_base` -- and with the tree gone there is nothing for it to be
a realization of. `choose_gemm_plan` never returns it, so refusing it costs
this arm no shape.

DEVIATION 1135: THE GUARD, AND WHY IT IS A RUNTIME REFUSAL
------------------------------------------------------------
**NOTHING MAY ROUTE TO THIS FILE.** It computes a product that no profile
names, that no oracle checks and that is not reproducible across vendors or
across launches -- which is the entire point of it. It is not a fallback, it
is not a fast path, and a caller that reaches it has a bug.

`unpinned_gemm_with_plan` is the single door and it REFUSES unless the binary
was built with

    -D MOJOLEARN_GEMM_UNPINNED_ARM=1

The refusal is at runtime and not a `comptime assert`, because a comptime
assert would make the file uncompilable in a default build and then it could
not sit in the tree beside the arm it is a control for. The two other guards
are that no shipped file imports it (`bindings/`, `python/mojolearn/` and
`core/` must never gain an import of this module) and that
`unpinned_gemm_banner()` exists so that a bench PRINTS which arm produced a
number -- `[[the shared checkout's mode flip]]`: read the mode back from the
run, never from the source.

DEVIATION 1136: THE UNPINNING IS SOURCE-LEVEL, NOT A MODE FLIP
----------------------------------------------------------------
This file does not import `identical_mul_add`, `ftz` or `GLOBAL_NUMERIC_MODE`
and does not care which mode the binary is in. That is deliberate and it is
what makes the experiment possible at all.

`GLOBAL_NUMERIC_MODE` is comptime, so one process cannot hold both the pinned
and the unpinned spelling of the SAME kernel -- which is why
`tools/gemm_price.sh` has to compare IDENTICAL against FAST across two
binaries and two thermal windows. By writing the unpinned arithmetic out
longhand in a separate kernel, this file puts the pinned arm and the unpinned
arm in ONE binary, where they alternate call by call and the drift the shell
has to work around at the mode level is defeated outright.

Consequence, stated because somebody will otherwise assume it. Build this in
the IDENTICAL binary. In a FAST binary the `device` arm is already unpinned
at seams 4 and 5 and the comparison degenerates.

WHAT THIS ARM CANNOT TELL YOU
------------------------------
**The number it enables is "what the pin costs at OUR level of engineering",
not "what the pin costs in principle."** An unpinned kernel wearing the
identical kernel's tiling is still not a TUNED kernel. No tensor cores, no
double buffering, no register blocking in `m` or `n`, no vectorized shared
loads, no `KS` chosen for this machine. A properly tuned unpinned GEMM would
be faster than this one, so this arm gives a LOWER BOUND on the pin's cost
and never an upper one, and `device / unpinned` must never be quoted as "the
price of identity" without that sentence attached.

`[[mojo-buffer-freed-at-last-use]]`: every launcher here takes buffers the
CALLER owns and the caller keeps alive past `ctx.synchronize()`, exactly as
`gemm_identical.mojo` does.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

from gemm.checks.gemm_oracle import CONTRACT_MAX_LEAVES
from gemm.checks.gemm_identical import (
    FLAT_TPB,
    PLAN_FLAT,
    PLAN_SPLITK,
    PLAN_SPLITK_STAGED,
    PLAN_TILE_4_4_32_TSP,
    PLAN_TILE_8_32_32,
    PLAN_TILE_16_16_8_REV,
    PLAN_TILE_16_16_32,
    PLAN_TILE_32_8_16,
    SPLITK_FOLD_TPB,
    SPLITK_LEAF_TPB,
    SWIZZLE_NONE,
    SWIZZLE_REVERSE,
    SWIZZLE_TRANSPOSE,
    _leaf_bounds,
    _tile_grid,
    choose_gemm_plan,
    contract_partition,
    gemm_operand_strides,
    gemm_plan_name,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)


# ===========================================================================
# THE GUARD (DEVIATION 1135)
# ===========================================================================

#: OFF in every build that does not name it. See the header. This arm is a
#: measurement instrument and a caller that reaches it has a bug.
comptime UNPINNED_ARM = is_defined["MOJOLEARN_GEMM_UNPINNED_ARM"]()

#: How many independent accumulators the default unpinned arm keeps per
#: output cell in the `k` direction. Contract 7.1 forbids any value but 1;
#: this arm exists to price that clause, and 4 is the smallest value that
#: covers a typical FMA issue-to-use latency without changing the tile.
#:
#: **A POWER OF TWO, AND THE SIMD REGISTER IS `SIMD[float32, NACC]`.**
#: `unpinned_gemm_with_plan` asserts both at comptime.
comptime UNPINNED_NACC = 4


def unpinned_gemm_arm_enabled() -> Bool:
    """Whether this binary can run the unpinned arm at all."""
    return UNPINNED_ARM


def unpinned_gemm_banner(nacc: Int) -> String:
    """The line a bench MUST print beside any number from this arm.

    `[[the shared checkout's mode flip]]`: a correctly labelled measurement
    of the wrong arm is worse than no measurement, so the arm identifies
    itself from the running binary rather than from a comment.
    """
    var head = String("UNPINNED CONTROL ARM (NOT a profile, NOT reproducible")
    head += ", NACC=" + String(nacc) + ")"
    if not UNPINNED_ARM:
        return head + " -- DISABLED in this binary"
    return head + " -- ENABLED"


def unpinned_gemm_workspace_floats(m: Int, n: Int, k: Int, plan: Int) -> Int:
    """Call-through to `identical_gemm_workspace_floats`. EQUAL BY
    CONSTRUCTION, and that is the point of it existing rather than the
    caller being told to use the other one.

    The unpinned SPLITK arm materializes the same `m * n * P` level-0
    partials at the same `stride = P`, so a workspace sized for the pinned
    arm serves this one exactly and the two arms cannot be given different
    allocations by accident. If this function ever stops being a
    call-through, the workspace has become a second variable and the
    experiment is over.
    """
    return identical_gemm_workspace_floats(m, n, k, plan)


def unpinned_gemm_workspace_max_floats(m: Int, n: Int, k: Int) -> Int:
    return identical_gemm_workspace_max_floats(m, n, k)


# ===========================================================================
# PLAN FLAT, UNPINNED
# ===========================================================================


def unpinned_gemm_flat_kernel[
    NACC: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
):
    """`identical_gemm_flat_kernel` with the numerical plan removed.

    Same grid, same `FLAT_TPB`, same one-thread-per-cell ownership, same
    linear cell index, same strides, same leaf walk. The differences are the
    whole of the experiment at this plan:

        pinned                            unpinned
        acc reset to +0.0 per leaf        NACC accumulators for all of k
        identical_mul_add(...)            acc + x * y, codegen's choice
        ftz at 5a, 5b, 5c, 5d, 5g         no flush anywhere
        _fold_push / _fold_drain          none; the tree does not exist
        16-lane SIMD fold stack + occ     NACC lanes

    The last row is a register-count difference and it goes the way that
    FAVORS this arm. The pinned kernel carries `GEMM_FOLD_SLOTS = 16`
    floats of fold stack per thread whether or not `P` needs them. That is a
    cost OF the pin, so it is attributable rather than confounding -- but it
    is an OCCUPANCY effect and not an arithmetic one, and
    `gemm/UNPINNED_CONTROL.md` section 4 says so, as confound C1, where a
    reader will find it.

    `leaf_in` and `p_in` are still passed and still come from
    `contract_partition(k)`, so the leaf walk is the pinned kernel's walk.
    They reach the SCHEDULE and nothing else (DEVIATION 1131).
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)

    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n

    # ONE seed for the whole `k`, not one per leaf. Contract 7.1's `+0.0`
    # leaf seed does not exist here because leaves do not exist here.
    var accs = SIMD[DType.float32, NACC](0.0)
    var a_row = i * a_si
    var b_col = j * b_sj

    for t in range(p_count):
        var bounds = _leaf_bounds(t, leaf, k)
        var p = bounds[0]
        var pe = bounds[1]
        # THE UNPINNED INNER LOOP. `NACC` independent chains, then whatever
        # the tail leaves over into lane 0. No flush, no fma pin, and the
        # accumulators are NOT reset at the leaf boundary.
        while p + NACC <= pe:
            comptime for u in range(NACC):
                accs[u] = (
                    accs[u]
                    + a.unsafe_load(a_row + (p + u) * a_sp)
                    * b.unsafe_load((p + u) * b_sp + b_col)
                )
            p += NACC
        while p < pe:
            accs[0] = (
                accs[0]
                + a.unsafe_load(a_row + p * a_sp)
                * b.unsafe_load(p * b_sp + b_col)
            )
            p += 1

    # The cross-accumulator combine. ASCENDING BY LANE, and that order is
    # arbitrary. No clause of any profile fixes it, this arm makes no
    # identity claim, and a reader must not mistake this loop for a
    # specification of anything.
    var out = accs[0]
    comptime for u in range(1, NACC):
        out = out + accs[u]
    c.unsafe_store(cell, out)


# ===========================================================================
# PLAN TILE_*, UNPINNED
# ===========================================================================


def unpinned_gemm_tiled_kernel[
    TM: Int, TN: Int, KS: Int, NACC: Int
](
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    swizzle_in: Int32,
):
    """`identical_gemm_tiled_kernel` with the numerical plan removed.

    **THE STAGING HALF OF THIS KERNEL IS A CHARACTER-FOR-CHARACTER COPY OF
    THE PINNED ONE** -- the two `while` loops that fill `as_` and `bs_`, the
    `barrier()` before and after the accumulation, the tile bijection, the
    out-of-range store mask, the `TM * KS` and `KS * TN` allocations. That
    copy is not laziness, it is the requirement. Same `TM`, `TN`, `KS`, same
    `NTH`, same threadgroup footprint, same number of DRAM transactions,
    same number of barriers, at every shape and at the ragged tail.

    It is copied CODE and not imported code because Mojo has no way to share
    a loop body between two kernels that differ in their middle. The
    CONSTANTS are imported (`_tile_grid`, `_leaf_bounds`, `SWIZZLE_*`) and
    the tile shapes arrive as parameters from `unpinned_gemm_with_plan`,
    which reads the same five `_launch_tiled` call sites the pinned
    dispatcher does. **If `gemm_identical.mojo`'s staging loop is ever
    edited, this one has to be edited in the same commit or the arms differ
    in the traffic and the number is worthless.** That is a maintenance
    hazard and `gemm/UNPINNED_CONTROL.md` section 4 lists it as confound C2
    rather than pretending it away.

    Every thread of the block still reaches every `barrier()`, for the
    pinned kernel's reason, which is that the loop bounds are functions of
    `t`, `leaf`, `k` and `KS`, all of them block-uniform, and a thread whose
    cell is out of range still stages and still barriers.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var swizzle = Int(swizzle_in)

    comptime NTH = TM * TN
    var as_ = stack_allocation[
        TM * KS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var bs_ = stack_allocation[
        KS * TN, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()

    var tiles_i = (m + TM - 1) // TM
    var tiles_j = (n + TN - 1) // TN
    var n_tiles = tiles_i * tiles_j
    var raw = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
    if raw >= n_tiles:
        return
    var tile = raw
    if swizzle == SWIZZLE_REVERSE:
        tile = n_tiles - 1 - raw
    elif swizzle == SWIZZLE_TRANSPOSE:
        tile = (raw % tiles_i) * tiles_j + (raw // tiles_i)
    var ti = tile // tiles_j
    var tj = tile - ti * tiles_j
    var i0 = ti * TM
    var j0 = tj * TN

    var tid = Int(thread_idx.x)
    var r = tid // TN
    var s = tid - r * TN
    var gi = i0 + r
    var gj = j0 + s

    var accs = SIMD[DType.float32, NACC](0.0)

    for t in range(p_count):
        var bounds = _leaf_bounds(t, leaf, k)
        var pe = bounds[1]
        var p = bounds[0]
        while p < pe:
            var chunk = pe - p
            if chunk > KS:
                chunk = KS
            var ia = tid
            while ia < TM * KS:
                var rr = ia // KS
                var cc = ia - rr * KS
                var v = Float32(0.0)
                if i0 + rr < m and cc < chunk:
                    v = a.unsafe_load((i0 + rr) * a_si + (p + cc) * a_sp)
                as_[unsafe_offset=ia] = v
                ia += NTH
            var ib = tid
            while ib < KS * TN:
                var cc2 = ib // TN
                var ss = ib - cc2 * TN
                var v2 = Float32(0.0)
                if j0 + ss < n and cc2 < chunk:
                    v2 = b.unsafe_load((p + cc2) * b_sp + (j0 + ss) * b_sj)
                bs_[unsafe_offset=ib] = v2
                ib += NTH
            barrier()
            # THE UNPINNED INNER LOOP, reading the SAME staged operands the
            # pinned kernel reads, from the same threadgroup addresses.
            var q = 0
            while q + NACC <= chunk:
                comptime for u in range(NACC):
                    accs[u] = (
                        accs[u]
                        + as_[unsafe_offset = r * KS + q + u]
                        * bs_[unsafe_offset = (q + u) * TN + s]
                    )
                q += NACC
            while q < chunk:
                accs[0] = (
                    accs[0]
                    + as_[unsafe_offset = r * KS + q]
                    * bs_[unsafe_offset = q * TN + s]
                )
                q += 1
            barrier()
            p += chunk

    var out = accs[0]
    comptime for u in range(1, NACC):
        out = out + accs[u]
    if gi < m and gj < n:
        c.unsafe_store(gi * n + gj, out)


# ===========================================================================
# PLAN SPLITK, UNPINNED (DEVIATION 1133)
# ===========================================================================


def unpinned_gemm_leaf_kernel[
    NACC: Int
](
    ws: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
    stride_in: Int32,
):
    """Level 0, unpinned. Same grid, same `SPLITK_LEAF_TPB`, same flat work
    index, same `ws[cell * stride + t]` address, same `stride = P`.

    The chunk boundaries here are NOT a schedule convenience -- split-K needs
    a decomposition of `k` to have any parallelism at all, and this arm keeps
    the pinned one's so that the number of partials, the workspace bytes and
    the second launch's width are all unchanged. What it drops inside a
    chunk is the flush, the contraction pin and the single-accumulator rule.

    `t` is still the logical index and still comes from the work index, not
    from `block_idx`, so each partial still has one writer and one address.
    That is not a contract clause being honored -- this arm honors none --
    it is a data race being avoided.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var stride = Int(stride_in)

    var total = m * n * p_count
    var gid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if gid >= total:
        return
    var cell = gid // p_count
    var t = gid - cell * p_count
    var i = cell // n
    var j = cell - i * n

    var bounds = _leaf_bounds(t, leaf, k)
    var accs = SIMD[DType.float32, NACC](0.0)
    var a_row = i * a_si
    var b_col = j * b_sj
    var p = bounds[0]
    var pe = bounds[1]
    while p + NACC <= pe:
        comptime for u in range(NACC):
            accs[u] = (
                accs[u]
                + a.unsafe_load(a_row + (p + u) * a_sp)
                * b.unsafe_load((p + u) * b_sp + b_col)
            )
        p += NACC
    while p < pe:
        accs[0] = (
            accs[0]
            + a.unsafe_load(a_row + p * a_sp)
            * b.unsafe_load(p * b_sp + b_col)
        )
        p += 1
    var part = accs[0]
    comptime for u in range(1, NACC):
        part = part + accs[u]
    ws.unsafe_store(cell * stride + t, part)


def unpinned_gemm_fold_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    p_in: Int32,
    stride_in: Int32,
):
    """One BLOCK per output cell, STRIDE pairing, no carry rule, no flush.

    This is `core/pinned_reduce.mojo::pinned_block_sum`'s shape and contract
    7.2 clause 1's fixture F8 -- deliberately, because it is what a fold
    looks like when nobody pins it. It is a balanced tree of the same DEPTH
    and a different ANSWER.

    HELD FIXED against `identical_gemm_fold_kernel`, all of it on purpose:

    - `grid_dim = (m * n, 1, 1)` and `block_dim = SPLITK_FOLD_TPB`;
    - `2 * CONTRACT_MAX_LEAVES` floats of threadgroup memory, of which this
      kernel uses half. A stride fold can run IN PLACE, which would halve the
      footprint and change occupancy on all three vendors, so the ping-pong
      allocation stays and the second half is simply not read. Wasting
      threadgroup memory on purpose is the cheap way to hold an occupancy
      input still (DEVIATION 1133);
    - the level count. `w -> ceil(w/2)` here and `w -> pairs (+1 if odd)`
      there are the same sequence, so both folds run `ceil(log2 P)` levels
      and `ceil(log2 P)` barriers.

    What moves is WHICH node is added to WHICH, and the seven flushes.
    """
    var mn = Int(mn_in)
    var p_count = Int(p_in)
    var stride = Int(stride_in)
    var cell = Int(block_idx.x)
    if cell >= mn:
        return
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)

    var buf = stack_allocation[
        2 * CONTRACT_MAX_LEAVES,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var cur = 0
    var nxt = CONTRACT_MAX_LEAVES

    var q0 = tid
    while q0 < p_count:
        buf[unsafe_offset = cur + q0] = ws.unsafe_load(cell * stride + q0)
        q0 += nth
    barrier()

    var w = p_count
    while w > 1:
        var half = (w + 1) // 2
        var q = tid
        while q < half:
            var v = buf[unsafe_offset = cur + q]
            if q + half < w:
                v = v + buf[unsafe_offset = cur + q + half]
            buf[unsafe_offset = nxt + q] = v
            q += nth
        barrier()
        var swap = cur
        cur = nxt
        nxt = swap
        w = half
    if tid == 0:
        var root = Float32(0.0)
        if p_count > 0:
            root = buf[unsafe_offset=cur]
        c.unsafe_store(cell, root)


# ===========================================================================
# THE HOST ENTRY POINTS
# ===========================================================================


def _launch_unpinned_tiled[
    TM: Int, TN: Int, KS: Int, NACC: Int
](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
    p_count: Int,
    st: Tuple[Int, Int, Int, Int],
    swizzle: Int,
    two_d: Bool,
) raises:
    """`_launch_tiled`'s twin. The grid comes from the IMPORTED `_tile_grid`
    with the same arguments, so the two arms cannot disagree about it."""
    comptime kern = unpinned_gemm_tiled_kernel[TM, TN, KS, NACC]
    var g = _tile_grid(m, n, TM, TN, two_d)
    ctx.enqueue_function[kern](
        c.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        Int32(swizzle),
        grid_dim=(g[0], g[1], 1),
        block_dim=(TM * TN, 1, 1),
    )


def unpinned_gemm_with_plan[
    NACC: Int
](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
    plan: Int,
) raises:
    """THE SINGLE DOOR, and the guard is the first thing in it.

    Structurally the same function as `identical_gemm_with_plan`: the
    partition from `k` (imported), the strides from the orientation
    (imported), then a switch on the plan whose five tiled branches carry
    the SAME `TM, TN, KS, swizzle, two_d` tuples as the pinned dispatcher's
    five. Those five tuples are the one thing in this file that is
    transcribed rather than imported, because they are call-site arguments
    and not named constants. `gemm/UNPINNED_CONTROL.md` section 4 records
    that as confound C3 and section 7 asks the run to print the plan name
    from the imported `gemm_plan_name` so a mismatch is visible.
    """
    comptime assert NACC >= 1, "unpinned_gemm_with_plan: NACC must be >= 1"
    comptime assert (NACC & (NACC - 1)) == 0, (
        "unpinned_gemm_with_plan: NACC must be a power of two, because the"
        " accumulator set is one SIMD register of NACC lanes"
    )

    if not UNPINNED_ARM:
        raise Error(
            "gemm_unpinned.mojo is a MEASUREMENT INSTRUMENT and this binary"
            " was not built with -D MOJOLEARN_GEMM_UNPINNED_ARM=1. It"
            " computes a product no profile names, no oracle checks and no"
            " two launches need agree on. Nothing may route to it: if you"
            " reached this line from a shipped path, that is the bug."
        )
    if m <= 0 or n <= 0:
        return
    var part = contract_partition(k)
    var leaf = part[0]
    var p_count = part[1]
    var st = gemm_operand_strides(op, m, n, k)

    if plan == PLAN_SPLITK_STAGED:
        # DEVIATION 1134. The staged plan is a realization OF the fold tree,
        # one launch per LOGICAL LEVEL at `fold_level_base` addresses. With
        # the tree gone it has nothing to realize, and unpinning it would
        # mean inventing a staged stride fold that the pinned arm has no
        # counterpart for -- a new kernel, not a control. `choose_gemm_plan`
        # never returns it, so no shape is lost.
        raise Error(
            "unpinned_gemm_with_plan: PLAN_SPLITK_STAGED has no unpinned"
            " counterpart. It is a realization of the contract's fold tree"
            " and this arm has no tree. choose_gemm_plan never selects it;"
            " if a gate wants it priced, that is the PINNED arm's"
            " measurement (contract 13.6.2), not this one's."
        )

    if plan == PLAN_SPLITK and p_count > 0:
        var stride = p_count
        comptime leafk = unpinned_gemm_leaf_kernel[NACC]
        ctx.enqueue_function[leafk](
            ws.unsafe_ptr(),
            a.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(leaf),
            Int32(p_count),
            Int32(st[0]),
            Int32(st[1]),
            Int32(st[2]),
            Int32(st[3]),
            Int32(stride),
            grid_dim=(
                (m * n * p_count + SPLITK_LEAF_TPB - 1) // SPLITK_LEAF_TPB,
                1,
                1,
            ),
            block_dim=(SPLITK_LEAF_TPB, 1, 1),
        )
        ctx.enqueue_function[unpinned_gemm_fold_kernel](
            c.unsafe_ptr(),
            ws.unsafe_ptr(),
            Int32(m * n),
            Int32(p_count),
            Int32(stride),
            grid_dim=(m * n, 1, 1),
            block_dim=(SPLITK_FOLD_TPB, 1, 1),
        )
        return
    if plan == PLAN_TILE_16_16_32:
        _launch_unpinned_tiled[16, 16, 32, NACC](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == PLAN_TILE_8_32_32:
        _launch_unpinned_tiled[8, 32, 32, NACC](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, True
        )
        return
    if plan == PLAN_TILE_32_8_16:
        _launch_unpinned_tiled[32, 8, 16, NACC](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_REVERSE, False
        )
        return
    if plan == PLAN_TILE_16_16_8_REV:
        _launch_unpinned_tiled[16, 16, 8, NACC](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_TRANSPOSE, True
        )
        return
    if plan == PLAN_TILE_4_4_32_TSP:
        _launch_unpinned_tiled[4, 4, 32, NACC](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_TRANSPOSE, False
        )
        return
    # PLAN_FLAT, and the fallback for SPLITK at `k == 0`.
    comptime flatk = unpinned_gemm_flat_kernel[NACC]
    ctx.enqueue_function[flatk](
        c.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        grid_dim=((m * n + FLAT_TPB - 1) // FLAT_TPB, 1, 1),
        block_dim=(FLAT_TPB, 1, 1),
    )


def unpinned_gemm_into_nacc[
    NACC: Int
](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """The general form. **`choose_gemm_plan` is the IMPORTED one** -- the
    same shape picks the same plan in both arms, which is the single most
    important thing this file holds fixed, because a plan difference would
    make the comparison a comparison of plans.
    """
    unpinned_gemm_with_plan[NACC](
        ctx, c, a, b, ws, m, n, k, op, choose_gemm_plan(m, n, k)
    )


def unpinned_gemm_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """**THE ARM.** Signature mirrors `identical_gemm_into` argument for
    argument, so `bench/gemm_price_main.mojo` can add it as a fourth device
    arm beside `device`, `vendor` and `pinned` with no other change to that
    file: same `ws`, sized by the same helper, same asynchronous contract,
    same caller-owns-the-buffers rule.

    `NACC = UNPINNED_NACC`, the honest unpinned kernel. Pair every number
    from it with one from `unpinned_gemm_into_one_acc` or the reader cannot
    tell the per-step tax from the ILP tax (DEVIATION 1132).
    """
    unpinned_gemm_into_nacc[UNPINNED_NACC](ctx, c, a, b, ws, m, n, k, op)


def unpinned_gemm_into_one_acc(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """THE STRICT CONTROL, `NACC = 1`. One dependent chain, exactly as long
    as the pinned kernel's, with only the flush, the contraction pin and the
    fold removed.

    This is the arm that answers "what do the seams cost" with the
    dependency graph held still. The gap between it and
    `unpinned_gemm_into` answers "what does contract 7.1's
    no-sub-partition clause cost", which is a different question.
    `gemm/UNPINNED_CONTROL.md` prediction 2 says that gap will be under 10%
    on a GPU, because a GPU hides a dependency chain with thread-level
    parallelism rather than with instruction-level parallelism, and names
    that as the prediction most likely to be wrong.
    """
    unpinned_gemm_into_nacc[1](ctx, c, a, b, ws, m, n, k, op)


def unpinned_gemm_plan_name(m: Int, n: Int, k: Int) -> String:
    """The plan this arm will run at this shape, spelled by the IMPORTED
    `gemm_plan_name`. A bench prints this beside the pinned arm's plan; if
    the two lines ever differ, the run is not the experiment and the number
    must be thrown away rather than explained."""
    return gemm_plan_name(choose_gemm_plan(m, n, k))
