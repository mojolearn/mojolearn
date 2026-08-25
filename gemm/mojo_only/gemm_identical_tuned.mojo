"""THE TUNED TWIN of `mojolearn.identical.gemm.fp32.v1`. EXPERIMENTAL.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** It was
written on 2026-08-25 by a lane with no execution rights of any kind -- no
`mojo`, no `pixi`, no build, no test, no benchmark. Every claim below about
what it computes is a claim about the SOURCE, and every claim about what it
costs is a PREDICTION recorded before measurement
(`gemm/TUNING_PLAN.md` section 6). Nothing here may be quoted as a result.
The first person to build it should expect to fix compile errors, and
`gemm/TUNING_PLAN.md`'s Appendix names the constructions most likely to be
the ones that fail.

**NOTHING MAY ROUTE TO THIS FILE UNTIL `gemm_device_check.mojo` PASSES ON
IT.** The gate that judges this kernel is per-cell bits against
`gemm_oracle` at 62 shapes, launch invariance over every plan, batch
invariance across a dispatch boundary and across composition, and six
build-define sabotages. Until that gate has been extended to enumerate the
plans below and has been RUN, this file is a proposal about how to go faster
and not a realization of the profile. `gemm/TUNING_PLAN.md`'s
`OWED, AND WHY I DID NOT DO IT HERE` lists the edits that wiring requires,
none of which this lane was permitted to make.

WHY THIS FILE EXISTS
---------------------
`mojolearn.identical.gemm.fp32.v1` runs at 2 to 5 percent of FP32 peak on
Apple, on an H100 and on an MI325X. Vendor libraries reach 62 to 94 percent
at the same precision. The single-variable control arm
(`gemm/mojo_only/gemm_unpinned.mojo`, `gemm/UNPINNED_CONTROL.md`) measured
the whole cost of the fold-order pin at 1.55x at the tiled `t512` rows, of
which 1.25x is the seams alone and roughly 0x is the fold tree. **So the pin
is 1.55x of a 12x to 25x gap and the remaining 8x to 16x is kernel
engineering.** This file is the kernel engineering.

THE ONE CONSTRAINT: THE BITS MUST NOT MOVE
--------------------------------------------
Every technique below moves BYTES. None of them reassociates ARITHMETIC.
`gemm/TUNING_PLAN.md` argues each one from the contract's own clauses; the
three that carry the argument are restated here because a reader of the
kernel should not have to open another file to know why it is allowed to
exist.

1. **Register blocking supplies the instruction-level parallelism that
   contract 7.1 forbids, WITHOUT VIOLATING IT** (DEVIATION 1251). Contract
   7.1 forbids sub-partitioning ONE leaf of ONE cell into several
   accumulators. It says nothing about a thread owning several CELLS. This
   kernel gives each thread `RPT * CPT` output cells and one accumulator
   each, so there are `RPT * CPT` independent FMA chains in flight and every
   single one of them is the contract's serial ascending chain, seeded
   `+0.0`, flushed after every step. The unpinned control arm bought
   1.24x by breaking 7.1 with `NACC = 4`; this kernel buys the same kind of
   thing legally, by being wider in `m` and `n` instead of in `k`.
2. **The operand flushes 5a and 5b are HOISTED out of the inner cell loop**
   (DEVIATION 1252). `ftz` is a pure, idempotent function of one float, so
   flushing a staged value once and using it `CPT` times is bit-for-bit the
   same as flushing it at each use. Contract 5's table says "each
   `A_eff[i,p]` AS LOADED" -- one load, one flush -- which is exactly what
   this does. **The accumulator flush 5c is NOT hoisted and cannot be:** it
   is one flush per cell per `k` step and it is the largest single cost item
   in the profile (contract section 5). Nothing here touches it.
3. **Staging, double buffering, vectorized copies and the shared-memory
   layout are a bit-exact copy of the operands.** Contract 13.5's second
   escape and `identical_gemm_tiled_kernel`'s own argument: threadgroup
   memory carries OPERANDS, never partial sums, and a copy is a copy. The
   value this kernel flushes and multiplies is the same float
   `A_eff[i, p]` the FLAT plan loads straight from DRAM.

HOW THE ACCUMULATORS ARE HELD, AND WHY THEY SHOULD LAND IN REGISTERS
----------------------------------------------------------------------
`bench/results/GEMM_ROUND_2026-08-19.md` records this repository porting
RAFT's `Policy4x4<float>` register tiling and getting a kernel SLOWER than
the naive 16x16 one it replaced, across the board, because
`stack_allocation` with no address space is thread-local MEMORY and every
accumulator access became a load and a store. Moving the accumulators to
`SIMD` values fixed it.

**Every per-thread accumulator in this file is a lane of a `SIMD` value with
a COMPTIME lane index, and there is no `stack_allocation` in this file that
is not `address_space = AddressSpace.SHARED`.** Specifically:

    acc   SIMD[float32, RPT * CPT]         the output accumulators
    fstk  SIMD[float32, FS * RPT * CPT]    the fold stack, all cells
    ra    SIMD[float32, RPT * VEC]         the A fragment of one k chunk
    rb    SIMD[float32, CPT * VEC]         the B fragment of one k chunk
    pa    SIMD[float32, ASLOTS * VEC]      the prefetch registers for A
    pb    SIMD[float32, BSLOTS * VEC]      the prefetch registers for B

Every index into every one of them comes from a `comptime for` bound, so the
lane is a compile-time constant and the value is SSA. This is the same
spelling `cluster/ported/distance/fused_distance_nn/simt_kernel.mojo` ships
and runs today -- that file IS the fixed version of the failed round -- and
the same spelling `mojo_only/numerics.mojo::ftz_simd` and
`gemm_identical.mojo::_fold_push` use. **It is not a guarantee.** A SIMD of
width 128 (`FS = 8`, `RPT * CPT = 16`) is a large value and a register
allocator may still spill it; that is DEVIATION 1253's whole subject and
`gemm/TUNING_PLAN.md` section 7 says a register or occupancy readout is
required before any number from this file is believed.

DEVIATION 1253: THE FOLD STACK IS WHAT BOUNDS THE REGISTER TILE
-----------------------------------------------------------------
This is the finding of the round and it is a cost of the pin that no timing
arm has priced.

`gemm/UNPINNED_CONTROL.md` confound C1 records that the pinned kernel
carries `GEMM_FOLD_SLOTS = 16` float lanes of fold stack per thread whether
or not `P` needs them, and calls it an occupancy effect. At one cell per
thread that is a footnote. **At `RPT * CPT` cells per thread it is a hard
limit on how much register blocking is available at all**, because the stack
is per CELL: a 4x4 register tile at a fold depth of 8 needs 128 float lanes
of stack beside its 16 accumulators, and no vendor in the matrix has that
many architectural registers per thread to spare at a useful occupancy.

An unpinned kernel has no such term. It carries ONE accumulator set for the
whole `k` range, so it can take the widest register tile the machine
supports. The pinned kernel cannot. **That is a real, structural, priced-at-
zero-so-far cost of contract 7.2, and it acts on the largest single lever
this round has.**

The mitigation, and it is a mitigation and not a fix, is DEVIATION 1254.

DEVIATION 1254: THE FOLD DEPTH `FS` IS COMPTIME AND THE HOST DISPATCHES ON `P`
-------------------------------------------------------------------------------
`_fold_push`'s slot `d` holds a completed node covering `2^d` consecutive
leaves, and `occ` is a binary counter over the leaves pushed so far. After
`t` pushes `occ == t`, so the highest slot index ever touched while pushing
`P` leaves is `floor(log2(P))`. **A stack of `FS` slots therefore covers
every `P <= 2^FS - 1`, exactly**, and `_fold_push_tile` still returns False
on overflow so the bound is checked rather than trusted.

So this file carries plans at three fold depths and the dispatcher picks
from `P`:

    FS = 1     P == 1      no stack at all; contract 7.3's case, where the
                           single leaf partial reaches the output through
                           seam 5g and through nothing else. A 4x4 register
                           tile is free here and this is the widest plan.
    FS = 8     P <= 255    covers `k` up to 32,640 at `K_LEAF_MIN = 128`,
                           which is every transformer row in the shape table
                           (`k = 4096` gives `P = 32`, `k = 14336` gives
                           `P = 112`). Register tile narrowed to 4x2.
    FS = 16    P <= 1024   the profile cap. Register tile narrowed to 2x2.

**Dispatching a KERNEL VARIANT on `P` is not dispatching the NUMERICAL TREE
on `P`.** Contract 13.5's last sentence forbids the second and this is the
first: every variant computes the same `(d, q)` node addressing of contract
7.2.2 over the same `P` leaves in the same pairing, and `P` is still a pure
function of `k` through the IMPORTED `contract_partition`. Nothing here can
move a leaf boundary, and there is no `if P < 32 use the serial fold`
anywhere and there never may be. What varies with `P` is how many registers
the compiled kernel reserves for a tree it computes identically either way.

**A DISPATCH ON `P` IS A DISPATCH ON `k` AND ON NOTHING ELSE, AND THAT IS
THE CLAUSE THAT MATTERS.** Contract 6.1 is the batch-invariance clause: the
numerical plan may see `k` and may not see `m` or `n`. `choose_tuned_gemm_
plan` reads `m` and `n` too -- it must, to pick a tile -- but `m` and `n`
reach only the TILE SHAPE, and the tile shape cannot reach the arithmetic
because one thread owns each cell for its whole `k` range and no float
crosses a thread boundary anywhere in this file.

DEVIATION 1255: THE LEAF LOOP IS FLATTENED INTO A UNIFORM WINDOW SEQUENCE
--------------------------------------------------------------------------
`identical_gemm_tiled_kernel` nests a staging `while` inside a leaf `for`
and clamps each staging window to the leaf's end. Double buffering across
that nest is awkward, because the window that must be PREFETCHED while
window `t` computes may belong to leaf `t + 1`.

This kernel flattens it: window `w` of `w_total = P * ceil(L / KS)` maps to
leaf `w // wpl` and offset `(w % wpl) * KS` inside it, by
`_tuned_window`, which reads `L`, `P`, `k` and the comptime `KS` and
NOTHING ELSE. The sequence is block-uniform, so every thread reaches every
`barrier()`, and the trip count no longer depends on how ragged the last
leaf is.

The cost is a few EMPTY windows on the short last leaf -- at most
`wpl - 1 <= 7` of them, once per launch per tile -- which stage zeros and
compute nothing. **A staged zero that is never read is staging hygiene and
not operand padding**, which is `identical_gemm_tiled_kernel`'s own
distinction and contract section 8's: the accumulation loop runs `chunk`
steps, never `KS`, so no `0.0 * 0.0` product ever enters an accumulator and
the signed-zero and NaN cases of section 9 are untouched.

DEVIATION 1256: PING-PONG PAGES, AND ONE BARRIER PER WINDOW INSTEAD OF TWO
---------------------------------------------------------------------------
With `PAGES == 2` the loop is

    R->S window w into page w%2 ; barrier ; G->R window w+1 ; compute page w%2

and the barrier count halves, because the write of page `(w+1)%2` at the top
of iteration `w+1` conflicts only with the compute of iteration `w-1`, which
every thread finished before the barrier at the top of iteration `w`. With
`PAGES == 1` the shape degenerates to the existing kernel's two barriers.

`PAGES` is NOT chosen here. It comes from
`kernel_matrix.mojo::lib_smem_pages_for[TARGET_COLUMN, page_bytes]`, whose
docstring already records the answer: *"2 for RAFT's double buffer, 1 where
it will not fit. Apple's 32 KB is the only column that forces 1 at
Policy4x4."* At `KS = 16` the pages fit twice on every column including
Apple; at `KS = 32` they do not on Apple and do on NVIDIA and AMD. **That is
a kernel-matrix row doing its job and it is why there is a `KS = 16` plan
and a `KS = 32` plan rather than one number chosen on an M4.**

DEVIATION 1257: VECTORIZED GLOBAL LOADS, GUARDED, WITH A BIT-IDENTICAL
FALLBACK
-----------------------------------------------------------------------
`_tuned_g2r` issues one `unsafe_load[width=VEC]` per `VEC` operand elements
when the operand is contiguous along `p` (`k_stride == 1`) and the window is
full; otherwise it issues `VEC` scalar loads. **Both paths write the same
floats into the same shared slots**, so the choice is invisible to the
arithmetic and a machine on which the guard never fires is slower and not
wrong.

Which orientations vectorize is a property of contract section 3's strides
and not of this file. `OP_NT` -- every transformer row in
`bench/gemm_shapes.mojo` -- has `a_sp == 1` AND `b_sp == 1`, so BOTH copies
vectorize. `OP_TN` -- the four Gram and OLS rows -- has `a_sp == m` and
`b_sp == n`, so NEITHER does, and `gram.128sq.x100003` is the one tiled row
in the table that gets no benefit from this technique at all. Said here
because a speedup that appears at eighteen rows and not at two is a finding
about the strides and not a flaw in the measurement.

DEVIATION 1258: THE PADDED SHARED STRIDE IS RAFT'S, NOT A ROUNDING
-------------------------------------------------------------------
`SSTRIDE = KS + VEC`, which is `simt_kernel.mojo`'s
`smem_stride = kblk + veclen` and carries that file's comment verbatim:
padding, not a rounding. Both operands are staged `[outer][k]` with `k`
innermost, so a thread reads `VEC` consecutive floats per row per chunk, and
the pad breaks the power-of-two row stride that would otherwise put every
thread of a group on the same bank. **The pad is a shared-memory ADDRESS
change. It moves no value and the accumulation reads the same floats in the
same order at any stride.**

DEVIATION 1250: WHAT THIS FILE IS NOT
---------------------------------------
It is not the shipped profile. `gemm/mojo_only/gemm_identical.mojo` is, it
is gated on three vendors, and this lane did not touch it. It is not a
replacement for the eight plans there: it serves TILE-able shapes and
DELEGATES everything else -- split-K shapes, `k == 0`, and any output too
small for a 32x32 tile -- straight back to `identical_gemm_into`, so a bench
that calls `tuned_gemm_into` gets a correct answer at every shape and must
print `tuned_gemm_plan_name` per shape to know which arm produced it.
**A row where the tuned arm delegated is a row that measured the pinned
kernel twice.**

DEVIATION 1263: THE TILE FOLD IS UNPROVEN UNTIL A CHECK RUNS
--------------------------------------------------------------
`gemm_identical.mojo::_fold_push` is PROVEN equal to
`gemm_oracle.fold_balanced_tree` for every `P` in `1 .. 2049` by
`check_stack_fold_is_the_contract_tree`. `_fold_push_tile` below is a
LANE-BY-LANE replica of it -- same expression, same `occ` arithmetic, same
merge condition, same `ftz` placement, with the scalar operand replaced by a
lane of a SIMD -- and `occ` is cell-independent, so every lane executes
exactly `_fold_push`'s program. **That is an argument and not a proof.** The
scalar one has a proof because a hand-written stack fold is exactly the kind
of thing that is right until it is not, and this one needs the same
treatment: `gemm/TUNING_PLAN.md` OWED item 3 asks for
`check_tuned_stack_fold_is_the_contract_tree` over the same `1 .. 2049`
sweep at every `FS` this file compiles.

DEVIATION 1264: THE GUARD
---------------------------
`tuned_gemm_with_plan` REFUSES unless the binary was built with

    -D MOJOLEARN_GEMM_TUNED_ARM=1

The refusal is at runtime rather than a `comptime assert`, for
`gemm_unpinned.mojo`'s reason: a comptime assert would make the file
uncompilable in a default build and then it could not sit in the tree beside
the kernel it is a candidate twin of. The other guards are that no shipped
file imports it -- `bindings/`, `python/mojolearn/` and `core/` must never
gain an import of this module -- and that `tuned_gemm_banner()` exists so a
bench PRINTS which arm produced a number. `[[the shared checkout's mode
flip]]`: read the arm back from the run, never from the source.

`SAB_NODE_ORDER` HAS NO ARM HERE, and that is not an omission. It sabotages
the SPLITK leaf kernel's workspace ADDRESS -- "the partial is written at the
address its BLOCK arrived in rather than at its logical leaf index" -- and
this file has no workspace and no split-K arm to sabotage. The other five
switches are imported and live. `gemm/TUNING_PLAN.md` section 8 says what a
gate must therefore assert about a `SAB_NODE_ORDER` build of this file
(namely: that it is bit-identical to an unsabotaged one, because the switch
cannot reach it).

`[[mojo-buffer-freed-at-last-use]]`: every launcher here takes buffers the
CALLER owns and the caller keeps alive past `ctx.synchronize()`, exactly as
`gemm_identical.mojo` does.

DEVIATIONS 1250 through 1264. The lane's range is 1250 through 1289 and
1265 through 1289 are reserved for whoever builds, gates and runs this.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined

from gemm.mojo_only.gemm_oracle import CONTRACT_MAX_LEAVES
from gemm.mojo_only.gemm_identical import (
    PLAN_SPLITK,
    PLAN_SPLITK_STAGED,
    SAB_FOLD_SERIAL,
    SAB_FOLD_STRIDE,
    SAB_LEAF_READS_LAUNCH,
    SAB_LEAF_ROTATE,
    SAB_PAD_PLUS_ZERO,
    SWIZZLE_NONE,
    SWIZZLE_REVERSE,
    SWIZZLE_TRANSPOSE,
    _leaf_at,
    _leaf_bounds,
    _tile_grid,
    choose_gemm_plan,
    contract_partition,
    gemm_operand_strides,
    gemm_sabotage_name,
    identical_gemm_into,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from mojo_only.kernel_matrix import (
    K_LIB_GEMM_CONTRACTION,
    PINNED_ACC_COLS_PER_TH,
    PINNED_ACC_ROWS_PER_TH,
    PINNED_KBLK,
    PINNED_VECLEN,
    TARGET_COLUMN,
    column_name,
    lib_block_size_for,
    lib_smem_pages_for,
)
from mojo_only.numerics import ftz, identical_mul_add


# ===========================================================================
# THE SCHEDULING CONSTANTS, ALL FROM `mojo_only/kernel_matrix.mojo`
# ===========================================================================
# DEVIATION 1260. Every number below is a kernel-matrix ROW read through a
# comptime accessor, not a constant chosen in this file, with the single
# exception of `TUNED_TC` -- which says so in its own docstring and which
# `gemm/TUNING_PLAN.md` OWED item 10 asks to be moved into the matrix.
#
# `[[ALWAYS GPU-agnostic]]`: one source for Metal, CUDA and HIP, and vendor
# divergence is a kernel-matrix ROW and never an inline `if apple`. There is
# no vendor branch anywhere in this file. `PAGES` differs between Apple and
# the other two columns at `KS = 32` and it differs because
# `lib_smem_pages_for` reads `column_shared_limit`, one directory over.


#: Threads per block. `lib_block_size_for[K_LIB_GEMM_CONTRACTION, ...]`,
#: which is RAFT's `Policy4x4` "AccThRows * AccThCols threads cover one
#: tile" and resolves to 256 on every column today. It is NOT one of the
#: rows `lib_block_bounds_a_float_fold` names, so IDENTICAL does not remap
#: it to the identity floor -- correctly, because in THIS kernel the block
#: size bounds no float fold at all: one thread owns each cell for its whole
#: `k` range and no partial sum crosses a thread boundary.
comptime TUNED_TPB = lib_block_size_for[K_LIB_GEMM_CONTRACTION, TARGET_COLUMN]()

#: The width of a staged copy, in floats. `PINNED_VECLEN`, RAFT's own
#: `Veclen = 4`.
#:
#: **THE MATRIX LABELS THIS ROW NUMERIC AND IN THIS FILE IT IS SCHEDULING,
#: AND THAT DIFFERENCE IS WORTH ONE PARAGRAPH.** In
#: `simt_kernel.mojo` the value is numeric because RAFT accumulates a
#: `Veclen` chunk per accumulator and the chunk width therefore decides a
#: summation order. Here it decides only how many floats one load
#: instruction moves: contract 7.1 fixes the accumulation to ONE accumulator
#: per cell walking `p` ascending, the inner `comptime for e in range(VEC)`
#: walks that chunk ascending into that one accumulator, and a `VEC` of 1, 2
#: or 4 accumulates the same terms in the same order. Read from the matrix
#: rather than respelled so a vendor measurement lands in one place.
comptime TUNED_VECLEN = PINNED_VECLEN

#: Thread columns per block: `AccThCols`. RAFT's `Policy4x4` is 16 at
#: `Nthreads = 256` and `simt_kernel.mojo` inherits it.
#:
#: **THIS IS THE ONE SCHEDULING NUMBER THIS FILE NAMES ITSELF AND THAT IS A
#: DEFECT, NOT A DESIGN.** The matrix pins the per-thread tile
#: (`PINNED_ACC_ROWS_PER_TH`, `PINNED_ACC_COLS_PER_TH`) and the block size,
#: but has no row for the THREAD GRID that connects them, so there is
#: nowhere for a vendor measurement of it to land. `gemm/TUNING_PLAN.md`
#: OWED item 6 asks for a `lib_acc_th_cols` row; until it exists this line
#: is a constant in a kernel, which is what the matrix's own header calls
#: the failure it was built to prevent.
comptime TUNED_TC = 16

#: The default `KS`. `PINNED_KBLK`, RAFT's `Kblk = 32`. Used by the `K32`
#: plans; the dispatcher's default plans use 16, because 16 is what lets
#: `lib_smem_pages_for` return 2 on Apple. Which of the two wins is a
#: MEASUREMENT and `gemm/TUNING_PLAN.md` section 5 predicts neither.
comptime TUNED_KBLK = PINNED_KBLK

#: The per-thread register tile of the WIDE plans. `PINNED_ACC_ROWS_PER_TH`
#: and `PINNED_ACC_COLS_PER_TH`, RAFT's `Policy4x4` 4x4. Same paragraph as
#: `TUNED_VECLEN`: the matrix labels these rows NUMERIC because in
#: `simt_kernel.mojo` they decide an accumulation geometry, and in THIS file
#: they decide only which cells a thread owns. Contract 7.1 fixes one
#: accumulator per cell walking `p` ascending, so a 4x4 tile and a 2x2 tile
#: accumulate the same terms into the same accumulator in the same order.
comptime TUNED_RPT = PINNED_ACC_ROWS_PER_TH
comptime TUNED_CPT = PINNED_ACC_COLS_PER_TH

#: The resolved output tiles, DERIVED and never written down twice. The
#: dispatcher compares `m` and `n` against these rather than against a
#: literal 64, so a column that resolves a different block size moves the
#: kernel and the dispatch threshold together. A tile constant that appears
#: in two places is a tile constant that drifts.
comptime TUNED_TR = TUNED_TPB // TUNED_TC
comptime TUNED_BM_WIDE = TUNED_RPT * TUNED_TR
comptime TUNED_BN_WIDE = TUNED_CPT * TUNED_TC
comptime TUNED_BM_NARROW = (TUNED_RPT // 2) * TUNED_TR
comptime TUNED_BN_NARROW = (TUNED_CPT // 2) * TUNED_TC


# ===========================================================================
# THE GUARD (DEVIATION 1264)
# ===========================================================================

#: OFF in every build that does not name it. See the header. This file is an
#: UNGATED candidate kernel and a caller that reaches it before
#: `gemm_device_check.mojo` has passed on it has a bug.
comptime TUNED_ARM = is_defined["MOJOLEARN_GEMM_TUNED_ARM"]()


def tuned_gemm_arm_enabled() -> Bool:
    """Whether this binary can run the tuned arm at all."""
    return TUNED_ARM


def tuned_gemm_banner() -> String:
    """The line a bench MUST print beside any number from this arm.

    `[[the shared checkout's mode flip]]`: a correctly labelled measurement
    of the wrong arm is worse than no measurement, so the arm identifies
    itself from the running binary rather than from a comment. It reports
    the resolved kernel-matrix column too, because every scheduling constant
    above comes from that column and a SIMULATED column
    (`-D MOJOLEARN_COLUMN_AMD=1` on an M4, which
    `kernel_matrix.column_is_simulated` exists to name) compiles a geometry
    this device will not run well -- and at `KS = 32` it compiles a
    DIFFERENT NUMBER OF SHARED PAGES, which is the one place in this file
    where a wrong column changes more than a block size.
    """
    var head = String("GEMM TUNED ARM (EXPERIMENTAL, UNGATED, column=")
    head += column_name(TARGET_COLUMN) + ", tpb=" + String(TUNED_TPB)
    head += ", veclen=" + String(TUNED_VECLEN) + ")"
    if not TUNED_ARM:
        head += " -- DISABLED in this binary"
    else:
        head += " -- ENABLED"
    return head + " -- sabotage=" + gemm_sabotage_name()


# ===========================================================================
# THE FOLD, IN REGISTERS, FOR A TILE OF CELLS (contract 7.2, DEVIATION 1263)
# ===========================================================================


def _fold_push_tile[
    NC: Int, FS: Int
](
    mut stack: SIMD[DType.float32, FS * NC],
    mut occ: Int,
    value: SIMD[DType.float32, NC],
) -> Bool:
    """`gemm_identical.mojo::_fold_push`, run on `NC` cells at once.

    Layout: `stack[d * NC + e]` is slot `d` of cell `e`. `occ` is ONE mask
    for all `NC` cells, and it is correct that there is only one: `occ` is a
    pure function of how many leaves have been pushed, which is the same
    number for every cell of the tile, and contract 7.2's tree is a pure
    function of `P` alone. If `occ` had to be per cell, the tree would be
    reading something other than `P` and that would be the bug.

    Returns False on overflow, exactly as `_fold_push` does. Slot
    `floor(log2(P))` is the highest ever touched, so `FS` slots cover every
    `P <= 2^FS - 1`; the return value is how the caller finds out that the
    dispatcher got that wrong rather than silently dropping a leaf.

    **THE MERGE EXPRESSION IS CHARACTER FOR CHARACTER `_fold_push`'s**, with
    the operand replaced by a lane: `val = ftz(ftz(stack[d]) + ftz(val))`,
    the occupied slot holding the EARLIER leaves and therefore standing on
    the left. Contract 5 rows 5e (both children flushed as read) and 5f (the
    node's own result flushed), contract 7.2 clauses 1 and 2.

    An unpaired leaf never merges -- it sits in its slot until drain, which
    is contract 7.2 clause 3's CARRY realized as no instruction at all. No
    padding, and no `+0.0` is ever added (clause 4).

    DEVIATION 1263: this spelling is ARGUED equal to `fold_balanced_tree`
    and not PROVEN equal to it. The scalar one is proven, over every `P` in
    `1 .. 2049`, and this one needs the same sweep before any output of this
    file is trusted.
    """
    comptime NS = FS
    comptime NE = NC
    var val = value
    var placed = False
    comptime for d in range(NS):
        if not placed:
            if ((occ >> d) & 1) == 1:
                comptime if SAB_FOLD_STRIDE:
                    # SABOTAGE: pair the far end of the stack instead of the
                    # adjacent node. Contract 7.2 clause 1, fixture F8.
                    comptime for e in range(NE):
                        val[e] = ftz(
                            ftz(val[e]) + ftz(stack[(FS - 1 - d) * NC + e])
                        )
                else:
                    comptime for e in range(NE):
                        val[e] = ftz(
                            ftz(stack[d * NC + e]) + ftz(val[e])
                        )
                occ = occ - (1 << d)
            else:
                comptime for e in range(NE):
                    stack[d * NC + e] = val[e]
                occ = occ + (1 << d)
                placed = True
    return placed


def _fold_drain_tile[
    NC: Int, FS: Int
](
    stack: SIMD[DType.float32, FS * NC], occ: Int
) -> SIMD[DType.float32, NC]:
    """`gemm_identical.mojo::_fold_drain`, run on `NC` cells at once.

    Lowest level first, and the same argument: at the end of the push
    sequence the occupied slots are the set bits of `P`, slot `d` holds the
    pairwise sum of a full `2^d` block, and the contract's root unrolls as
    `full_block(d_max) + (the same construction on the remainder)`, so the
    remainder accumulates first and each larger block joins on its left.

    `P == 1` performs NO addition (contract 7.3): one slot is occupied,
    `have` is False on the only iteration that fires, and the value is
    returned unchanged, so a `-0.0` partial stays `-0.0` (contract 9.2(b)).
    """
    comptime NS = FS
    comptime NE = NC
    var have = False
    var acc = SIMD[DType.float32, NC](0.0)
    comptime for d in range(NS):
        if ((occ >> d) & 1) == 1:
            if have:
                comptime for e in range(NE):
                    acc[e] = ftz(ftz(stack[d * NC + e]) + ftz(acc[e]))
            else:
                comptime for e in range(NE):
                    acc[e] = stack[d * NC + e]
                have = True
    return acc


# ===========================================================================
# THE WINDOW SEQUENCE (DEVIATION 1255)
# ===========================================================================


def _tuned_windows_per_leaf[KS: Int](leaf: Int) -> Int:
    """`ceil(L / KS)`, at least 1. Block-uniform, and a pure function of `L`
    and the comptime `KS`."""
    var w = (leaf + KS - 1) // KS
    if w < 1:
        return 1
    return w


def _tuned_window[
    KS: Int
](w: Int, wpl: Int, leaf: Int, k: Int, p_count: Int) -> Tuple[Int, Int, Int]:
    """`(p0, chunk, is_last_window_of_its_leaf)` for flat window `w`.

    Window `w` belongs to fold POSITION `t = w // wpl` and sits at offset
    `(w % wpl) * KS` inside that position's leaf. `_leaf_at` maps the
    position to the logical leaf -- the identity, and rotated only under
    `SAB_LEAF_ROTATE` -- and `_leaf_bounds` is the IMPORTED clamp, so the
    only inputs to a leaf boundary here are `leaf` and `k`, which is
    contract section 6's clause.

    **`chunk` may be 0.** Every position gets `wpl` windows whether or not
    its leaf needs them, so the trip count is block-uniform and independent
    of how ragged the last leaf is; the short last leaf simply ends with
    empty windows that stage zeros and accumulate nothing. That costs at
    most `wpl - 1` empty windows per tile per launch, and it buys a
    `barrier()` sequence that every thread provably reaches.

    The third element is 1 exactly at `w % wpl == wpl - 1`, so the fold push
    fires exactly once per leaf whatever the raggedness.
    """
    var t = w // wpl
    var ww = w - t * wpl
    var lb = _leaf_bounds(_leaf_at(t, p_count), leaf, k)
    var p0 = lb[0] + ww * KS
    var chunk = lb[1] - p0
    if chunk > KS:
        chunk = KS
    if chunk < 0:
        chunk = 0
    var last = 0
    if ww == wpl - 1:
        last = 1
    return (p0, chunk, last)


# ===========================================================================
# THE GLOBAL -> REGISTER STAGE (DEVIATION 1257)
# ===========================================================================


def _tuned_g2r[
    SLOTS: Int, VEC: Int, KV: Int, ROWS: Int, NTH: Int
](
    src: MutPointer[Float32, MutAnyOrigin],
    outer_stride: Int,
    k_stride: Int,
    base_outer: Int,
    outer_limit: Int,
    p0: Int,
    chunk: Int,
    tid: Int,
) -> SIMD[DType.float32, SLOTS * VEC]:
    """One window of one operand, DRAM to registers. ONE function for A and
    for B, because contract section 3 already made them one.

        A_eff[i, p] = a[i * a_si + p * a_sp]     outer = i, outer_stride = a_si
        B_eff[p, j] = b[j * b_sj + p * b_sp]     outer = j, outer_stride = b_sj

    `gemm_operand_strides` is what turns the three orientations into those
    two lines, and it is imported. There is no `if op ==` in this file.

    THE VECTOR GUARD. One `unsafe_load[width=VEC]` serves `VEC` elements
    when the operand is contiguous along `p` (`k_stride == 1`) and the whole
    vector lies inside the window and inside the matrix; otherwise `VEC`
    scalar loads with the stride, masked. **The two paths write the same
    floats into the same slots.** A machine or an orientation on which the
    guard never fires is slower here and never different, so this technique
    cannot move a bit and can only fail to help. Contract sections 2 and 3.

    Slots past the window are left `+0.0`. They are NEVER READ: the
    accumulation loop below runs `chunk` steps and never `KS`, which is
    `identical_gemm_tiled_kernel`'s own staging-hygiene distinction and
    contract section 8's ban on operand padding.

    `[[Mojo int widening sign-extends]]` does not bite here: every index is
    `Int` throughout and nothing is narrowed.
    """
    comptime NSLOT = SLOTS
    comptime NV = VEC
    var out = SIMD[DType.float32, SLOTS * VEC](0.0)
    comptime for s in range(NSLOT):
        var idx = tid + s * NTH
        if idx < ROWS * KV:
            var rr = idx // KV
            var cc = (idx - rr * KV) * VEC
            var oi = base_outer + rr
            if oi < outer_limit:
                if k_stride == 1 and cc + VEC <= chunk:
                    var vv = src.unsafe_load[width=VEC](
                        oi * outer_stride + p0 + cc
                    )
                    comptime for e in range(NV):
                        out[s * VEC + e] = vv[e]
                else:
                    comptime for e in range(NV):
                        if cc + e < chunk:
                            out[s * VEC + e] = src.unsafe_load(
                                oi * outer_stride + (p0 + cc + e) * k_stride
                            )
    return out


# ===========================================================================
# THE TUNED PLANS (EXECUTION PLAN ONLY -- none of these can move a bit)
# ===========================================================================
# Every plan below computes `gemm_oracle`, exactly as
# `gemm_identical.mojo`'s eight do, and for the same structural reason: one
# thread owns each output cell for its whole `k` range, `contract_partition`
# is the only producer of `(L, P)`, and threadgroup memory carries operands
# and never partial sums. They differ in the tile, the register block, the
# staging window, the number of shared pages and the depth of the fold
# stack. `check_device_is_launch_invariant`, extended to enumerate these,
# is what would turn that paragraph from a claim into a gate.

# DEVIATION 1259: the plan set, and the DELEGATE fallback that keeps
# `tuned_gemm_into` TOTAL. Six tuned plans serve TILE-able shapes; every
# other shape -- split-K, `k == 0`, and any output too small for the narrow
# tile -- routes straight back to `identical_gemm_into`, so a caller always
# gets a correct answer and a bench must print which arm produced it.

comptime TUNED_PLAN_DELEGATE = 0
comptime TUNED_PLAN_R4C4_K16_S1 = 1
comptime TUNED_PLAN_R4C4_K32_S1 = 2
comptime TUNED_PLAN_R4C2_K16_S8 = 3
comptime TUNED_PLAN_R2C2_K16_S16 = 4
comptime TUNED_PLAN_R2C2_K16_S1 = 5
comptime TUNED_PLAN_R4C4_K16_S8 = 6
comptime TUNED_GEMM_PLAN_COUNT = 7


def tuned_gemm_plan_name(plan: Int) -> String:
    """The name a bench MUST print per shape.

    A row whose plan is DELEGATE measured `identical_gemm_into` and not this
    file, and reporting it as a tuned number would be the most ordinary way
    for this whole round to produce a lie.
    """
    if plan == TUNED_PLAN_DELEGATE:
        return String("DELEGATE -> identical_gemm_into (NOT the tuned arm)")
    var rpt = 0
    var cpt = 0
    var ks = 16
    var tail = String("")
    if plan == TUNED_PLAN_R4C4_K16_S1:
        rpt = TUNED_RPT
        cpt = TUNED_CPT
        tail = String("P==1")
    elif plan == TUNED_PLAN_R4C4_K32_S1:
        rpt = TUNED_RPT
        cpt = TUNED_CPT
        ks = TUNED_KBLK
        tail = String("P==1, sweep only, NEVER dispatched")
    elif plan == TUNED_PLAN_R4C2_K16_S8:
        rpt = TUNED_RPT
        cpt = TUNED_CPT // 2
        tail = String("P<=255")
    elif plan == TUNED_PLAN_R2C2_K16_S16:
        rpt = TUNED_RPT // 2
        cpt = TUNED_CPT // 2
        tail = String("P<=1024, the profile cap")
    elif plan == TUNED_PLAN_R2C2_K16_S1:
        rpt = TUNED_RPT // 2
        cpt = TUNED_CPT // 2
        tail = String("P==1, narrow output")
    elif plan == TUNED_PLAN_R4C4_K16_S8:
        rpt = TUNED_RPT
        cpt = TUNED_CPT
        tail = String("P<=255, sweep only, NEVER dispatched")
    else:
        return String("TUNED PLAN?")
    # BUILT FROM THE RESOLVED CONSTANTS, not from a literal. A name that
    # says `64x64` while the matrix resolved something else is the
    # `[[the shared checkout's mode flip]]` defect wearing a plan label.
    var name = String("TUNED ") + String(rpt * TUNED_TR)
    name += "x" + String(cpt * TUNED_TC)
    name += " reg" + String(rpt) + "x" + String(cpt)
    name += " KS=" + String(ks)
    name += " fold=" + String(tuned_plan_fold_slots(plan))
    name += " tpb=" + String(TUNED_TPB)
    return name + " (" + tail + ")"


def tuned_plan_fold_slots(plan: Int) -> Int:
    """`FS` for a plan. `P <= 2^FS - 1` is the coverage; see DEVIATION
    1254 and `_fold_push_tile`'s overflow return."""
    if plan == TUNED_PLAN_R4C4_K16_S1:
        return 1
    if plan == TUNED_PLAN_R4C4_K32_S1:
        return 1
    if plan == TUNED_PLAN_R2C2_K16_S1:
        return 1
    if plan == TUNED_PLAN_R4C2_K16_S8:
        return 8
    if plan == TUNED_PLAN_R4C4_K16_S8:
        return 8
    if plan == TUNED_PLAN_R2C2_K16_S16:
        return 16
    return 0


def tuned_plan_max_leaves(plan: Int) -> Int:
    """The largest `P` a plan's fold stack covers: `2^FS - 1`.

    `_fold_push`'s `occ` is a binary counter over the leaves pushed so far,
    so after `t` pushes it equals `t` and the highest slot index touched
    while pushing `P` leaves is `floor(log2 P)`. At `P = 2^FS` the last push
    merges through every slot and needs slot `FS`, which is one past the
    end -- so the bound is `2^FS - 1` and not `2^FS`. That off-by-one is
    exactly the shape of contract 7.2.2's warning about `fold_node_total`
    not being bounded by `2P`, and it is checked rather than trusted:
    `_fold_push_tile` returns False on the overflow.
    """
    var fs = tuned_plan_fold_slots(plan)
    if fs <= 0:
        return 0
    return (1 << fs) - 1


# ===========================================================================
# THE TUNED TILED KERNEL
# ===========================================================================


def tuned_gemm_tiled_kernel[
    RPT: Int, CPT: Int, TC: Int, KS: Int, FS: Int, PAGES: Int
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
    """One block owns a `BM x BN` output tile and ALL of its `k` leaves.

    RAFT's structure (`pairwise_distance_base.cuh:139-149`), contract 13.5's
    second workspace escape -- NO GLOBAL SCRATCH AT ANY SHAPE -- and
    `identical_gemm_tiled_kernel`'s arrangement with four things changed and
    nothing else:

        identical_gemm_tiled_kernel        this kernel
        1 cell per thread                  RPT x CPT cells per thread
        1 shared page, 2 barriers/window   PAGES pages, 1 barrier at PAGES=2
        scalar global->shared copies       VEC-wide, guarded, same floats
        ftz(operand) at every use          ftz(operand) once per staged load

    **THE ARITHMETIC IS UNCHANGED AND THE ARGUMENT IS THREE LINES.** Every
    output cell has exactly one accumulator. That accumulator is seeded
    `+0.0` at the start of each logical leaf, takes one `identical_mul_add`
    per `p` ASCENDING, is flushed after every step (5c), is flushed again as
    the leaf partial (5d) and enters the same `_fold_push` tree the shipped
    kernel uses. `RPT`, `CPT`, `TC`, `KS` and `PAGES` decide which thread
    owns which cell, how many DRAM and shared transactions serve it and how
    many barriers separate them. None of them appears in any expression that
    reaches `leaf_begin`, `leaf_end` or a tree level, and `leaf_in` and
    `p_in` arrive from `contract_partition(k)` and from nowhere else.

    Every thread of the block reaches every `barrier()`. `w_total`, `wpl`,
    `chunk` and `last` are functions of `leaf`, `k`, `p_count` and the
    comptime `KS`, all block-uniform; a thread whose cell is out of range
    still stages, still barriers, and only its final STORE is masked; and
    the one early `return` (`raw >= n_tiles`) is block-uniform and happens
    before any barrier. `[[metal-hardware-gaps]]`: no block reduction
    primitive is used anywhere, so nothing here carries the "Block size must
    be greater than warp size" constraint.
    """
    comptime NTH = TUNED_TPB
    comptime TR = NTH // TC
    comptime BM = RPT * TR
    comptime BN = CPT * TC
    comptime VEC = TUNED_VECLEN
    comptime KV = KS // VEC
    comptime SSTRIDE = KS + VEC
    comptime APAGE = BM * SSTRIDE
    comptime BPAGE = BN * SSTRIDE
    comptime NCELL = RPT * CPT
    comptime NR = RPT
    comptime NCOL = CPT
    comptime ASLOTS = (BM * KV + NTH - 1) // NTH
    comptime BSLOTS = (BN * KV + NTH - 1) // NTH

    comptime assert KS % VEC == 0, (
        "tuned_gemm_tiled_kernel: KS must be a VEC multiple, so a staged"
        " window is a whole number of vector slots"
    )
    comptime assert NTH % TC == 0, (
        "tuned_gemm_tiled_kernel: TC must divide the block size"
    )
    comptime assert FS >= 1 and (FS & (FS - 1)) == 0, (
        "tuned_gemm_tiled_kernel: FS must be a power of two, because the"
        " fold stack is one SIMD register of FS * NCELL lanes"
    )
    comptime assert (NCELL & (NCELL - 1)) == 0, (
        "tuned_gemm_tiled_kernel: RPT * CPT must be a power of two, same"
        " reason"
    )
    comptime assert PAGES == 1 or PAGES == 2, (
        "tuned_gemm_tiled_kernel: PAGES comes from lib_smem_pages_for and"
        " that row returns 1 or 2"
    )

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

    comptime if SAB_LEAF_READS_LAUNCH:
        # SABOTAGE: the leaf boundary reads the LAUNCH. Contract section 6's
        # first sentence forbids exactly this. Same arithmetic as the
        # shipped kernel's arm so the two fail the same fixture.
        leaf = leaf * Int(block_dim.x) // 64
        if leaf < 1:
            leaf = 1
        p_count = (k + leaf - 1) // leaf
        if k <= 0:
            p_count = 0

    var as_ = stack_allocation[
        PAGES * APAGE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var bs_ = stack_allocation[
        PAGES * BPAGE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var tiles_i = (m + BM - 1) // BM
    var tiles_j = (n + BN - 1) // BN
    var n_tiles = tiles_i * tiles_j
    var raw = Int(block_idx.y) * Int(grid_dim.x) + Int(block_idx.x)
    if raw >= n_tiles:
        return
    # The tile permutation: an EXACT bijection, so every tile runs once.
    var tile = raw
    if swizzle == SWIZZLE_REVERSE:
        tile = n_tiles - 1 - raw
    elif swizzle == SWIZZLE_TRANSPOSE:
        tile = (raw % tiles_i) * tiles_j + (raw // tiles_i)
    var ti = tile // tiles_j
    var tj = tile - ti * tiles_j
    var i0 = ti * BM
    var j0 = tj * BN

    var tid = Int(thread_idx.x)
    # RAFT's accumulation assignment (`contractions.cuh:96-102`): thread
    # `tid` owns rows `accrow + u * TR` and columns `acccol + v * TC`. The
    # ownership is STRIDED rather than contiguous, so consecutive threads
    # write consecutive columns of `C` and the store coalesces.
    var accrow = tid // TC
    var acccol = tid - accrow * TC

    var acc = SIMD[DType.float32, NCELL](0.0)
    var fstk = SIMD[DType.float32, FS * NCELL](0.0)
    var occ = 0
    var serial = SIMD[DType.float32, NCELL](0.0)

    if p_count <= 0:
        # `k == 0`, contract section 8: every cell is `+0.0`, and the
        # implementation must WRITE it rather than skip the store. The
        # dispatcher delegates this shape, so this arm is defensive.
        comptime for u0 in range(NR):
            comptime for v0 in range(NCOL):
                var zi = i0 + accrow + u0 * TR
                var zj = j0 + acccol + v0 * TC
                if zi < m and zj < n:
                    c.unsafe_store(zi * n + zj, Float32(0.0))
        return

    var wpl = _tuned_windows_per_leaf[KS](leaf)
    var w_total = p_count * wpl

    # ---- PROLOGUE: window 0, DRAM to registers.
    var w0 = _tuned_window[KS](0, wpl, leaf, k, p_count)
    var pa = _tuned_g2r[ASLOTS, VEC, KV, BM, NTH](
        a, a_si, a_sp, i0, m, w0[0], w0[1], tid
    )
    var pb = _tuned_g2r[BSLOTS, VEC, KV, BN, NTH](
        b, b_sj, b_sp, j0, n, w0[0], w0[1], tid
    )

    var w = 0
    while w < w_total:
        var win = _tuned_window[KS](w, wpl, leaf, k, p_count)
        var chunk = win[1]
        var pgw = w % PAGES

        # ---- REGISTERS TO SHARED, into page `w % PAGES`.
        comptime for sa in range(ASLOTS):
            var ia = tid + sa * NTH
            if ia < BM * KV:
                var rra = ia // KV
                var cca = (ia - rra * KV) * VEC
                var va = SIMD[DType.float32, VEC](0.0)
                comptime for ea in range(VEC):
                    va[ea] = pa[sa * VEC + ea]
                as_.unsafe_store(pgw * APAGE + rra * SSTRIDE + cca, va)
        comptime for sb in range(BSLOTS):
            var ib = tid + sb * NTH
            if ib < BN * KV:
                var rrb = ib // KV
                var ccb = (ib - rrb * KV) * VEC
                var vb = SIMD[DType.float32, VEC](0.0)
                comptime for eb in range(VEC):
                    vb[eb] = pb[sb * VEC + eb]
                bs_.unsafe_store(pgw * BPAGE + rrb * SSTRIDE + ccb, vb)
        barrier()

        # ---- PREFETCH window w+1 while page `w % PAGES` is being consumed.
        # DEVIATION 1256. At PAGES == 2 the write of page `(w+1) % 2` at the
        # top of the NEXT iteration conflicts only with the compute of
        # iteration `w-1`, which every thread finished before the barrier
        # just executed -- so one barrier per window is sufficient and the
        # DRAM latency of window `w+1` overlaps the arithmetic of window
        # `w`.
        comptime if PAGES == 2:
            if w + 1 < w_total:
                var wn = _tuned_window[KS](w + 1, wpl, leaf, k, p_count)
                pa = _tuned_g2r[ASLOTS, VEC, KV, BM, NTH](
                    a, a_si, a_sp, i0, m, wn[0], wn[1], tid
                )
                pb = _tuned_g2r[BSLOTS, VEC, KV, BN, NTH](
                    b, b_sj, b_sp, j0, n, wn[0], wn[1], tid
                )

        # ---- ACCUMULATE. Contract 7.1, 4 and 5, character for character
        # the FLAT plan's step with the two loads served from threadgroup
        # memory and the operand flushes hoisted (DEVIATION 1252).
        var abase = pgw * APAGE + accrow * SSTRIDE
        var bbase = pgw * BPAGE + acccol * SSTRIDE
        if chunk == KS:
            # THE FULL-WINDOW PATH. `VEC`-wide shared reads, then `VEC`
            # ascending `p` steps out of them. `e` ascends inside `kc`
            # ascending, so every accumulator sums its terms in exactly the
            # order the scalar path below sums them in.
            comptime for kc in range(KV):
                var ra = SIMD[DType.float32, RPT * VEC](0.0)
                comptime for u in range(NR):
                    var ta = as_.unsafe_load[width=VEC](
                        abase + u * TR * SSTRIDE + kc * VEC
                    )
                    comptime for e in range(VEC):
                        ra[u * VEC + e] = ta[e]
                var rb = SIMD[DType.float32, CPT * VEC](0.0)
                comptime for v in range(NCOL):
                    var tb = bs_.unsafe_load[width=VEC](
                        bbase + v * TC * SSTRIDE + kc * VEC
                    )
                    comptime for e2 in range(VEC):
                        rb[v * VEC + e2] = tb[e2]
                comptime for e3 in range(VEC):
                    # 5b, ONCE per staged B value rather than once per use.
                    var bfl = SIMD[DType.float32, CPT](0.0)
                    comptime for v2 in range(NCOL):
                        bfl[v2] = ftz(rb[v2 * VEC + e3])
                    comptime for u2 in range(NR):
                        # 5a, likewise.
                        var afl = ftz(ra[u2 * VEC + e3])
                        comptime for v3 in range(NCOL):
                            # 4 (one fused rounding) and 5c (the accumulator
                            # flushed after EVERY step). 5c is per cell per
                            # step and is NOT hoisted, because it cannot be.
                            acc[u2 * CPT + v3] = ftz(
                                identical_mul_add(
                                    afl, bfl[v3], acc[u2 * CPT + v3]
                                )
                            )
        else:
            # THE RAGGED PATH, and the EMPTY one: `chunk` may be 0. Scalar
            # shared reads, same ascending `p`, same seams. No zero slot is
            # ever read, so there is no operand padding (contract 8).
            for cc in range(chunk):
                var bfl2 = SIMD[DType.float32, CPT](0.0)
                comptime for v4 in range(NCOL):
                    bfl2[v4] = ftz(
                        bs_.unsafe_load(bbase + v4 * TC * SSTRIDE + cc)
                    )
                comptime for u3 in range(NR):
                    var afl2 = ftz(
                        as_.unsafe_load(abase + u3 * TR * SSTRIDE + cc)
                    )
                    comptime for v5 in range(NCOL):
                        acc[u3 * CPT + v5] = ftz(
                            identical_mul_add(
                                afl2, bfl2[v5], acc[u3 * CPT + v5]
                            )
                        )

        comptime if PAGES == 1:
            # One page: the compute above must finish before the next
            # window overwrites it, and the prefetch cannot overlap. This is
            # `identical_gemm_tiled_kernel`'s two-barrier shape exactly.
            barrier()
            if w + 1 < w_total:
                var wn1 = _tuned_window[KS](w + 1, wpl, leaf, k, p_count)
                pa = _tuned_g2r[ASLOTS, VEC, KV, BM, NTH](
                    a, a_si, a_sp, i0, m, wn1[0], wn1[1], tid
                )
                pb = _tuned_g2r[BSLOTS, VEC, KV, BN, NTH](
                    b, b_sj, b_sp, j0, n, wn1[0], wn1[1], tid
                )

        # ---- THE LEAF BOUNDARY. Fires exactly once per logical leaf.
        if win[2] == 1:
            var part = SIMD[DType.float32, NCELL](0.0)
            comptime for pe in range(NCELL):
                part[pe] = ftz(acc[pe])  # 5d, the leaf partial as written
            comptime if SAB_FOLD_SERIAL:
                # SABOTAGE: the SUPERSEDED serial ascending fold.
                comptime for se in range(NCELL):
                    serial[se] = ftz(ftz(serial[se]) + ftz(part[se]))
            else:
                comptime if FS == 1:
                    # `P == 1` by dispatch (`tuned_plan_max_leaves` is 1 at
                    # `FS = 1` and the host refuses otherwise). Contract 7.3:
                    # the single leaf partial reaches the output through seam
                    # 5g and through nothing else. There is no "skip the fold
                    # at P == 1" optimization to get wrong -- at `P == 1` the
                    # rule and the optimization are the same rule -- and this
                    # branch exists to spend NO registers on a stack the tree
                    # does not have.
                    comptime for fe in range(NCELL):
                        fstk[fe] = part[fe]
                    occ = 1
                else:
                    _ = _fold_push_tile[NCELL, FS](fstk, occ, part)
            acc = SIMD[DType.float32, NCELL](0.0)

        w = w + 1

    comptime if SAB_PAD_PLUS_ZERO:
        # SABOTAGE: pad the level-0 width up to the next power of two with
        # `+0.0` instead of carrying the odd tail. Contract 7.2 clause 4.
        comptime if FS > 1:
            var padded = 1
            while padded < p_count:
                padded = padded * 2
            for _pad in range(p_count, padded):
                _ = _fold_push_tile[NCELL, FS](
                    fstk, occ, SIMD[DType.float32, NCELL](0.0)
                )

    var outv = SIMD[DType.float32, NCELL](0.0)
    comptime if FS == 1:
        comptime for oe in range(NCELL):
            outv[oe] = fstk[oe]
    else:
        outv = _fold_drain_tile[NCELL, FS](fstk, occ)
    comptime if SAB_FOLD_SERIAL:
        outv = serial

    comptime for u4 in range(NR):
        comptime for v6 in range(NCOL):
            var gi = i0 + accrow + u4 * TR
            var gj = j0 + acccol + v6 * TC
            if gi < m and gj < n:
                # 5g: the output cell as stored.
                c.unsafe_store(gi * n + gj, ftz(outv[u4 * CPT + v6]))


# ===========================================================================
# THE HOST ENTRY POINTS
# ===========================================================================


def tuned_gemm_workspace_floats(m: Int, n: Int, k: Int, plan: Int) -> Int:
    """Floats of global scratch a tuned plan needs: ZERO for every one of
    them, because every one is contract 13.5's second escape.

    DELEGATE is the exception and it needs whatever the shipped dispatcher
    needs, so this calls through rather than returning 0 and letting a
    delegated SPLITK shape write past the end of a 1-float buffer.
    `identical_gemm_into`'s docstring records that exact defect costing this
    lane a run.
    """
    if plan == TUNED_PLAN_DELEGATE:
        return identical_gemm_workspace_floats(
            m, n, k, choose_gemm_plan(m, n, k)
        )
    return 0


def tuned_gemm_workspace_max_floats(m: Int, n: Int, k: Int) -> Int:
    """Call-through to `identical_gemm_workspace_max_floats`, ALWAYS.

    A tuned plan needs none, but the dispatcher may DELEGATE at any shape
    and a caller must not have to know which. Sizing for the tuned plan and
    getting a delegated SPLITK dispatch is an out-of-bounds write that a
    small shape will not show you.
    """
    return identical_gemm_workspace_max_floats(m, n, k)


def choose_tuned_gemm_plan(m: Int, n: Int, k: Int) -> Int:
    """Pick a tuned EXECUTION plan, or DELEGATE.

    **Reads `m`, `n` and `k` and is allowed to** (contract 6.1: the
    execution plan may look at `m`, `n`, the device, the occupancy and
    anything else it likes, because under section 7 none of that can reach
    the arithmetic). What it may NOT do -- and structurally cannot, because
    it returns a plan id and nothing else and every plan calls
    `contract_partition(k)` -- is influence a leaf boundary or a tree level.

    THE TWO RULES, AND THEY ARE DIFFERENT KINDS OF RULE:

    - `m` and `n` pick the TILE. A 64x64 tile needs an output big enough to
      fill it; below 32x32 the masking waste dominates and the shipped
      dispatcher's narrow tiles are better, so those shapes delegate.
    - `P` picks the FOLD DEPTH, and `P` is a pure function of `k`
      (DEVIATION 1254). This is a dispatch on `k`, which section 6.1 permits
      the numerical plan itself, let alone the execution plan.

    SPLITK shapes delegate outright. They are the small-`m n`, enormous-`k`
    aspect -- `gram.32x32x1M` -- where a fused arm has only `m * n` threads
    of parallelism and register blocking has nothing to block. Contract 13.5
    already says which arm those want and this file does not second-guess
    it.
    """
    if not TUNED_ARM:
        return TUNED_PLAN_DELEGATE
    if m <= 0 or n <= 0:
        return TUNED_PLAN_DELEGATE
    var part = contract_partition(k)
    var p_count = part[1]
    if p_count <= 0:
        return TUNED_PLAN_DELEGATE
    var base = choose_gemm_plan(m, n, k)
    if base == PLAN_SPLITK or base == PLAN_SPLITK_STAGED:
        return TUNED_PLAN_DELEGATE
    if m >= TUNED_BM_WIDE and n >= TUNED_BN_WIDE:
        if p_count == 1:
            return TUNED_PLAN_R4C4_K16_S1
        if p_count <= tuned_plan_max_leaves(TUNED_PLAN_R4C2_K16_S8):
            return TUNED_PLAN_R4C2_K16_S8
        return TUNED_PLAN_R2C2_K16_S16
    if m >= TUNED_BM_NARROW and n >= TUNED_BN_NARROW:
        if p_count == 1:
            return TUNED_PLAN_R2C2_K16_S1
        return TUNED_PLAN_R2C2_K16_S16
    return TUNED_PLAN_DELEGATE


def _launch_tuned[
    RPT: Int, CPT: Int, TC: Int, KS: Int, FS: Int
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
    """`_launch_tiled`'s twin, with `PAGES` resolved from the matrix.

    The grid comes from the IMPORTED `_tile_grid` with the tile this plan's
    parameters imply, so there is no second opinion about how many blocks a
    shape needs.
    """
    comptime NTH = TUNED_TPB
    comptime TR = NTH // TC
    comptime BM = RPT * TR
    comptime BN = CPT * TC
    comptime VEC = TUNED_VECLEN
    comptime SSTRIDE = KS + VEC
    #: Both pages of both operands, in bytes. `lib_smem_pages_for` answers 2
    #: when twice this fits under `column_shared_limit(TARGET_COLUMN)` and 1
    #: when it does not, which at `KS = 32` is the Apple column and only the
    #: Apple column. `[[ALWAYS GPU-agnostic]]`: that divergence is a matrix
    #: row, not an `if apple`.
    comptime PAGE_BYTES = (BM + BN) * SSTRIDE * 4
    comptime PAGES = lib_smem_pages_for[TARGET_COLUMN, PAGE_BYTES]()
    comptime kern = tuned_gemm_tiled_kernel[RPT, CPT, TC, KS, FS, PAGES]
    var g = _tile_grid(m, n, BM, BN, two_d)
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
        block_dim=(NTH, 1, 1),
    )


def tuned_gemm_with_plan(
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

    Structurally `identical_gemm_with_plan`: the partition from `k`
    (imported), the strides from the orientation (imported), then a switch
    that chooses geometry. After `part` is computed there is no path back
    into the arithmetic -- every branch below passes the SAME `leaf` and
    `p_count` to its kernel.

    THE FOLD-DEPTH REFUSAL is the one thing this function does that its
    shipped twin does not. A plan whose `FS` cannot cover `P` would drop a
    leaf silently -- `_fold_push_tile` would return False and the caller
    would have discarded it -- so the shape is refused here, loudly, before
    a launch. That is the check on DEVIATION 1254's arithmetic, and it is
    deliberately not a fallback to another plan: a dispatcher that quietly
    repairs its own bad arithmetic is a dispatcher whose bad arithmetic
    nobody finds.
    """
    if not TUNED_ARM:
        raise Error(
            "gemm_identical_tuned.mojo is an EXPERIMENTAL, UNGATED candidate"
            " kernel and this binary was not built with"
            " -D MOJOLEARN_GEMM_TUNED_ARM=1. It has never been compiled, has"
            " never been run, and gemm_device_check.mojo has never judged"
            " it. Nothing may route to it: if you reached this line from a"
            " shipped path, that is the bug."
        )
    if m <= 0 or n <= 0:
        return
    var part = contract_partition(k)
    var leaf = part[0]
    var p_count = part[1]
    var st = gemm_operand_strides(op, m, n, k)

    if plan == TUNED_PLAN_DELEGATE:
        identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
        return

    var cap = tuned_plan_max_leaves(plan)
    if p_count > cap:
        raise Error(
            String(
                "tuned_gemm_with_plan: plan's fold stack holds P <= "
            )
            + String(cap)
            + " and this shape has P = "
            + String(p_count)
            + " (k = "
            + String(k)
            + ", L = "
            + String(leaf)
            + "). A stack of FS slots covers P <= 2^FS - 1 exactly"
            " (DEVIATION 1254). Pick a deeper plan; do NOT change the"
            " partition, which is contract section 6 and not a knob."
        )

    if plan == TUNED_PLAN_R4C4_K16_S1:
        _launch_tuned[TUNED_RPT, TUNED_CPT, TUNED_TC, 16, 1](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == TUNED_PLAN_R4C4_K32_S1:
        _launch_tuned[TUNED_RPT, TUNED_CPT, TUNED_TC, TUNED_KBLK, 1](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == TUNED_PLAN_R4C2_K16_S8:
        _launch_tuned[TUNED_RPT, TUNED_CPT // 2, TUNED_TC, 16, 8](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == TUNED_PLAN_R2C2_K16_S16:
        _launch_tuned[TUNED_RPT // 2, TUNED_CPT // 2, TUNED_TC, 16, 16](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == TUNED_PLAN_R2C2_K16_S1:
        _launch_tuned[TUNED_RPT // 2, TUNED_CPT // 2, TUNED_TC, 16, 1](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    if plan == TUNED_PLAN_R4C4_K16_S8:
        _launch_tuned[TUNED_RPT, TUNED_CPT, TUNED_TC, 16, 8](
            ctx, c, a, b, m, n, k, leaf, p_count, st, SWIZZLE_NONE, False
        )
        return
    raise Error(
        "tuned_gemm_with_plan: unknown tuned plan id. The ids are"
        " TUNED_PLAN_DELEGATE and TUNED_PLAN_* above; there are"
        " TUNED_GEMM_PLAN_COUNT of them."
    )


def tuned_gemm_into(
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
    """`identical_gemm_into`'s signature, ARGUMENT FOR ARGUMENT, so a bench
    can add this as one more arm with no other change.

    ASYNCHRONOUS: nothing here waits, and the caller must keep every buffer
    alive past its own `ctx.synchronize()`. `ws` must hold at least
    `tuned_gemm_workspace_max_floats(m, n, k)` floats -- which is the
    SHIPPED kernel's requirement, because this dispatcher may DELEGATE at
    any shape and the caller must not have to know which.

    **A bench MUST print `tuned_gemm_plan_name(choose_tuned_gemm_plan(m, n,
    k))` beside every number from this arm.** A delegated row measured
    `identical_gemm_into` twice and reporting it as a tuned result is the
    most ordinary way this round could produce a lie.
    """
    tuned_gemm_with_plan(
        ctx, c, a, b, ws, m, n, k, op, choose_tuned_gemm_plan(m, n, k)
    )


def tuned_gemm(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """The synchronizing form, for a check that does not want to own a
    workspace. `identical_gemm`'s shape and its `[[mojo-buffer-freed-at-
    last-use]]` argument verbatim: a `DeviceBuffer` created here is dead at
    its `.unsafe_ptr()`, so the `_ = ws` at the end is what keeps this one
    alive past the wait.
    """
    var nws = tuned_gemm_workspace_max_floats(m, n, k)
    var ws = ctx.enqueue_create_buffer[DType.float32](nws)
    ctx.synchronize()
    tuned_gemm_into(ctx, c, a, b, ws, m, n, k, op)
    ctx.synchronize()
    _ = ws


def tuned_gemm_fold_depth_covers_the_profile() -> Bool:
    """The deepest tuned plan must cover `CONTRACT_MAX_LEAVES`.

    Asserted against the imported profile constant rather than against a
    comment, exactly as `check_fold_stack_depth_covers_the_profile` does for
    the shipped kernel's `GEMM_FOLD_LEVELS`. If `MAX_LEAVES` is ever raised
    this returns False and the deepest plan needs another slot -- which is
    a compile-time change here and a v2 of the contract there.
    """
    return tuned_plan_max_leaves(TUNED_PLAN_R2C2_K16_S16) >= CONTRACT_MAX_LEAVES
