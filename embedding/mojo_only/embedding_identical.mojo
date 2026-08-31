# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device embedding of profile `mojolearn.identical.embedding.fp32.v1`.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** No GPU has
run a kernel from it, no gate has ever failed against it, no sabotage arm has
ever been built, no card has ever been emitted, and every claim in its
docstrings was derived on paper or read out of source on 2026-08-25. Written
by the embedding lane, DEVIATIONS 1300-1339. The contract is
`embedding/IDENTICAL_EMBEDDING_CONTRACT.md`; the answer is
`embedding/mojo_only/embedding_oracle.mojo`, bit for bit.

WHAT IS OWED
------------
  - `embedding/mojo_only/embedding_check.mojo` and
    `embedding/mojo_only/embedding_fixture.mojo` DO NOT EXIST, so **every
    sabotage switch below has never been compiled, let alone shown to fail a
    gate. A switch that has never fired is a comment.** Contract OWED item 1.
  - **`PLAN_SORT` IS NOT WRITTEN.** Contract 6.2 specifies it completely --
    the total-order key, the stable-by-id realization, the launch-invariance
    argument and the three ways to get it wrong -- and the device plan that
    would realize it is
    `gbdt/gpu_util/kernel/radix_sort.mojo::launch_radix_sort_bins`, keyed on
    the ids with the positions as the payload. That is a cross-lane import
    (`svm/mojo_only/device_select.mojo` is the precedent) needing six scratch
    buffers and a `REORDER_BLOCK` geometry this lane has not verified
    against, and writing an unverified device sort would have added a second
    thing that can be wrong. **Contract clause (d), plan invariance, is the
    strongest evidence available that the arithmetic does not read the plan,
    and it cannot run until this exists.** Contract OWED item 2.
  - `emb_run_begin_kernel` is a SINGLE-THREADED serial integer scan over `V`.
    At `V = 128256` that is 128,256 dependent integer additions in one
    thread. **A scheduling embarrassment and not a numerical one** -- integer
    addition is exactly associative, so swapping in
    `gbdt/gpu_util/kernel/scan.mojo` cannot move a bit. Contract OWED item 10.
  - No card is emitted from here. `core/identity_trace.mojo` recording is the
    check file's job and the check file does not exist.
  - No performance number exists and contract section 12 forbids quoting one.

ONE SOURCE, THREE VENDORS, AND NO `if apple` ANYWHERE
------------------------------------------------------
`[[always-gpu-agnostic]]`. There is not one branch on a vendor in this file
and there is not one place where there could be. The only place a column is
consulted at all is `_emb_max_tpb`, which resolves a BLOCK SIZE, and the
block size is a SCHEDULING row here for a reason that is structural rather
than promised:

  1. **NO FLOAT EVER CROSSES A THREAD BOUNDARY.** One thread owns one output
     cell `(v, j)`, walks its own run and adds. No shared memory on the float
     path, no warp primitive, no atomic, no cross-block float reduction.
     That is the construction every identical kernel in this repository uses
     -- `modeling_llama.mojo`, `modeling_mamba.mojo` and
     `training/mojo_only/optimizer.mojo` all say so in as many words -- and
     contract 0.2 is the argument that it survives a scatter-shaped gradient.
  2. **EVERY CROSS-THREAD OPERATION IN THIS FILE IS AN INTEGER ONE.** The
     counts, the run offsets and the permutation are integers, integer
     addition is exactly associative, and nothing in them rounds or flushes.
     A block size cannot reach an integer prefix count.
  3. **The forward is elementwise.** One thread owns one `(t, j)` cell,
     reads it, and writes it.

So `mojo_only/kernel_matrix.mojo` is consulted for the one thing it is for --
`column_max_block_size`, the vendor's dispatch cap, which the portable
baseline column exposed as a real limit on 2026-08-21 -- and for
`IDENTITY_FLOOR_BLOCK`, so that a `NUMERIC_IDENTICAL` build launches ONE
geometry on every vendor the way `block_size_for` does. DEVIATION 1320. A
vendor divergence in this lane would be a kernel-matrix ROW, never an inline
branch, and today there is no such row because there is no such divergence.

THERE IS NO MULTIPLY IN THIS FILE (contract 4.1, DEVIATION 1317)
-----------------------------------------------------------------
The backward is a pure sum and the forward is a pure copy. `identical_mul_add`
occupies no seam of this profile, so gemm section 4's multiply-add policy is
satisfied VACUOUSLY rather than followed, and the trap that has bitten this
repository repeatedly -- FMA contraction ACROSS expressions -- has nothing
here to contract. `identical_mul_add(dy, 1.0, acc)` would be BIT EQUAL
(`dy * 1.0` is exact, so the fma is one rounding of `dy + acc`), which is the
loss lane's ones-vector argument; the contract pins the plain add and
`embedding_oracle.mojo::emb_fold_via_fma_diagnostic` is what makes the
equivalence an assertion rather than a claim.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Everything, with the pin compiled away. `ftz` becomes the identity. The
kernels, the launches, the routing and the geometry are UNCHANGED -- so FAST
is a correct embedding, it is the same code on the same path (not dead code
and not a separate kernel), and it makes no identity claim at all.

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is the `_into` form.
It enqueues and returns. The CALLER owns every buffer -- including `counts`,
`run_begin` and `perm` -- and must keep every one of them alive past its own
`ctx.synchronize()`. A `DeviceBuffer` is dead at its `.unsafe_ptr()`, so a
training step that allocates the run scratch, calls the backward, and returns
without waiting has freed the scratch before the kernel ran. There is
deliberately no synchronizing form, because a training step chains many of
these and one wait per stage is the wrong shape.


WHY THIS FILE IS NOT CALLED `embedding.mojo`. DEVIATION 1326. It was, and it
did not compile: `module 'embedding' cannot import itself`, because a file
named `embedding.mojo` inside a package named `embedding` shadows the package,
so `from embedding.mojo_only.embedding_oracle import ...` resolves the
top-level name to THIS FILE. `gemm/README.md` documents the sibling of this
collision (that directory is not called `linalg` because MAX ships a `linalg`
package and wins); there the collision is with a DEPENDENCY, here it is with
our own package. Every other lane already avoids it by convention --
`mamba_check`, `transformer_oracle`, `gemm_identical` -- and none is a bare
`<lane>.mojo`. Do not rename this back.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
)
from mojo_only.kernel_matrix import (
    IDENTITY_FLOOR_BLOCK,
    TARGET_COLUMN,
    column_max_block_size,
)
from embedding.mojo_only.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    refuse_nonfinite,
)


# ===========================================================================
# THE SABOTAGE SWITCHES (contract 11.1)
# ===========================================================================
# OFF in every build that does not name them, exactly as
# `gemm/mojo_only/gemm_identical.mojo` and `training/mojo_only/loss.mojo` do
# it. Each one is a specific way to get a clause wrong that a plausible
# implementation could reach by accident, and each exists so a gate can be
# SHOWN to fail. A gate that has never failed is a gate nobody has tested.
#
# **EVERY SWITCH CARRIES ITS PREDICTED INERT SET, and the check must assert
# the inert set AS A MASK rather than merely observing that the arm moved
# something.** `IDENTICAL_BACKWARD_PLAN.md` section 5.0 is the discipline:
# two of its six routes are inert under `BWD_UNTRANSPOSED` for structural
# reasons, and a gate that reported a 4-of-6 result as a 6-of-6 one would be
# `[[reached-but-inert]]` in its purest form.
#
# TWO FIXTURES IN THIS LANE PASS EVERY ARM BELOW WHILE GATING NOTHING, and
# contract 11.2 requires both as NEGATIVE CONTROLS with their masks asserted:
#   F-NODUP    all ids distinct, so every run has R <= 1 and there is no
#              accumulation order to get wrong.
#   F-DUPSAME  duplicates present, every duplicate carrying a BITWISE EQUAL
#              `dY` row, so a permutation of a constant sequence is the same
#              sequence and the order clauses are STILL inert.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_EMB_SABOTAGE_FOLD_DESCENDING=1 \
#         -I . embedding/mojo_only/embedding_check.mojo

#: Contract 5.1 clause 1. The run is walked DESCENDING. **INERT on every run
#: of length <= 1, on every run whose contributors are bitwise equal, and on
#: every exactly-representable fixture.** Fixture F-ORDER3 separates it:
#: `{0x3F800000, 0x33800000, 0x33800000}` gives `0x3F800000` ascending and
#: `0x3F800001` descending.
comptime SAB_FOLD_DESCENDING = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_DESCENDING"
]()
#: Contract 5.1 clause 2 and 5.3. The run is folded by
#: `gemm/IDENTICAL_FP32_CONTRACT.md` 7.2's FIXED BALANCED TREE instead of the
#: chain -- which is the loss lane's DEVIATION 1152 answer applied to this
#: axis, and the candidate contract 5.2(b) refuses.
#: **INERT AT EVERY RUN OF LENGTH <= 3, PROVABLY** (contract 5.4): at R = 2
#: and R = 3 the balanced tree IS the ascending chain, node for node. Fixture
#: F-TREE4 is the smallest separator,
#: `{0x3F800000, 0x33800000, 0x33800000, 0x33800000}` -> `0x3F800000` chain
#: against `0x3F800001` tree. **`R_max` is a property of the DATA**, so a gate
#: must ASSERT `emb_max_run_length >= 4` rather than hope for it.
comptime SAB_FOLD_BALANCED_TREE = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_BALANCED_TREE"
]()
#: Contract 5.1 clause 3 and 9.2(c). The chain is SEEDLESS -- it starts from
#: the first contributor, which is what `gemm/IDENTICAL_FP32_CONTRACT.md`
#: 9.2(b)'s v1 tree does. **INERT at every input except a cell whose SOLE
#: contributor is `-0.0`**, where the seeded chain gives `0x00000000` and the
#: seedless one `0x80000000`. Fixture F-NEGZERO1.
comptime SAB_SEED_SEEDLESS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SEED_SEEDLESS"
]()
#: Contract 5.5. `R == 1` skips the addition and stores the contributor.
#: **The same inert mask as `SAB_SEED_SEEDLESS` and that is exactly why BOTH
#: exist** -- they are two different wrong spellings that agree on the same
#: single input, and a gate that carried only one would not know which clause
#: it had proved. This is deliberately NOT gemm 7.3's `P == 1` case; there
#: the tree is seedless so the rule and the optimization coincide, and here
#: they diverge.
comptime SAB_SINGLE_RUN_BYPASS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SINGLE_RUN_BYPASS"
]()
#: Contract 5.5. An empty run's store is SKIPPED rather than written.
#: **ALWAYS INERT if the gate pre-fills `dW` with zeros -- which is what a
#: fresh allocation may or may not contain -- and never inert if it POISONS.
#: The gate must poison.** At the shipped shape this is most of the output:
#: at `V = 128256` and `T = 4096` at least 124,160 of 128,256 rows are empty.
comptime SAB_EMPTY_ROW_SKIPPED = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_EMPTY_ROW_SKIPPED"
]()
#: Contract 5.5 and 9.2. An empty run stores `-0.0`. Never inert on
#: `emb.dw`, and there is no downstream in this profile to launder it --
#: `dW` IS the output. That is a real difference from the loss lane's
#: `L_IGNORED_ROW_NEG_ZERO`, which must move `ce.row` and must NOT move
#: `ce.total`.
comptime SAB_EMPTY_ROW_NEG_ZERO = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_EMPTY_ROW_NEG_ZERO"
]()
#: Contract 5.1's last paragraph. The fold reads `block_dim.x` and rotates
#: its starting offset by it, so the SAME data folded at two block sizes
#: gives two answers. **Never inert on a run of length >= 2.** It is the
#: falsifier for "no launch quantity reaches the arithmetic" and it reaches
#: the same clause `gemm_identical.mojo`'s `SAB_LEAF_READS_LAUNCH` does one
#: layer down.
comptime SAB_FOLD_READS_LAUNCH = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_FOLD_READS_LAUNCH"
]()
#: Contract 6.3 case 3, **the most dangerous spelling in this lane**. The
#: within-run rank is taken from an ATOMIC INTEGER counter instead of from
#: the ascending walk. The COUNT is order free, because integer addition is
#: associative; **the SLOT each position receives is not** -- it is arrival
#: order, which is exactly the defect the float atomic had, moved into the
#: index domain where it looks safe. **INERT on a fixture with no duplicate
#: ids, and INERT on a single-block launch**, where arrival order IS position
#: order -- so the gate must run more than one block.
comptime SAB_RANK_BY_ARRIVAL = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_RANK_BY_ARRIVAL"
]()
#: Contract 6.3 case 1. `emb.perm` is emitted with each run's ties in REVERSE
#: position order -- an order an unstable sort keyed on the id alone is
#: permitted to return. It is the shape `LINK_SAB_SORT_WEIGHT_ONLY` already
#: has in `hierarchy/ported/sparse/op/sort.mojo`. **INERT on a fixture with
#: no duplicate ids, and INERT on F-DUPSAME**, where the duplicates carry
#: bitwise equal rows.
comptime SAB_SORT_TIE_REVERSED = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_SORT_TIE_REVERSED"
]()
#: Contract section 8, DEVIATION 1311. A position carrying `padding_idx`
#: CONTRIBUTES instead of being dropped.
#:
#: **IT MUST MOVE `emb.counts`, `emb.run_begin` AND `emb.perm`, AND IT MUST
#: NOT MOVE `emb.dw`.** That is not a weakness of the arm, it is the clause:
#: contract section 8 says the drop-at-source and overwrite-afterwards
#: spellings are PROVABLY bit-equal in `dW`, because a position carrying
#: `padding_idx` can only ever contribute to row `padding_idx`, which
#: `emb_pad_row_kernel` overwrites. So this arm is the proof that the
#: equivalence holds, and **the card is the only instrument that can see it
#: at all** -- the loss lane's `L_IGNORED_ROW_NEG_ZERO` shape, at a second
#: site. A gate that compared only `emb.dw` would call it inert and delete
#: it.
#:
#: **INERT ENTIRELY on a fixture with no position carrying `padding_idx`**,
#: which is every fixture built from a `padding_idx`-free config.
comptime SAB_PAD_ROW_CONTRIBUTES = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_PAD_ROW_CONTRIBUTES"
]()
#: Contract section 8. Row `padding_idx` is filled with `-0.0`. **INERT on a
#: fixture with no `padding_idx`.**
comptime SAB_PAD_ROW_NEG_ZERO = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_PAD_ROW_NEG_ZERO"
]()
#: Contract section 4, seam E3. The accumulator is flushed only at the END
#: instead of after every add. **INERT on every fixture with no subnormal
#: INTERMEDIATE, and INERT ON APPLE ENTIRELY** -- `ftz` is bitwise a no-op on
#: an FTZ backend, so the pinned and unpinned spellings agree there. Gemm
#: 4.1's correction is the standing warning: Apple's bits not moving is NOT
#: evidence that a pin is unreached, and reach on Apple must be shown by an
#: oracle that REPORTS which arm the backend took, the way
#: `cluster/mojo_only/kmeans_identity_check.mojo::check_fused_contraction_pin`
#: does. This arm becomes a bit-level reach proof, with no edit, on the first
#: non-flushing backend.
comptime SAB_NO_FLUSH_ACC = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_NO_FLUSH_ACC"
]()
#: Contract section 4, seams G1 and G2, DEVIATION 1310. The gather copies raw
#: instead of flushing. **INERT on every fixture with no SUBNORMAL WEIGHT --
#: and NOT inert on Apple**, because a gather performs no arithmetic, so a
#: raw copy of a subnormal survives on every vendor including an FTZ one.
#: **That makes this the ONLY flush arm in this lane a single-column run can
#: see.** Fixture F-SUBW plants `0x00000001`.
comptime SAB_GATHER_NO_FLUSH = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_GATHER_NO_FLUSH"
]()
#: Contract section 8. An out-of-range id is CLAMPED into `[0, V)` instead of
#: refused. **INERT on a fixture with no out-of-range id**, which is every
#: ordinary fixture. A clamp turns a data bug into a wrong gradient on a REAL
#: vocabulary row and there is no stage at which that becomes visible.
comptime SAB_GATHER_CLAMP_OOR = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_GATHER_CLAMP_OOR"
]()
#: Contract 7.4, DEVIATION 1309. The `accumulate` path SEEDS FROM `+0.0` and
#: leaves the caller to add the two results, `dW = ftz(dW_prev + dW_micro)`,
#: instead of CARRYING the accumulator. **Visible only in clause (e)'s split
#: gate. INERT on any split that leaves every row's contributors on one side,
#: and on every exactly-representable fixture.** Fixture F-SPLIT carries both
#: halves, and the inert half is its own negative control.
#:
#: **NO KERNEL IN THIS FILE READS THIS SWITCH.** It is the one arm whose
#: wrong spelling lives in the CALLER, so the check file is what must branch
#: on it -- call the backward twice with `accumulate` false and add the two
#: `dW` buffers, instead of once with `accumulate` false and once with it
#: true. It is declared here anyway so that `emb_sabotage_name` can print it
#: and so that a build carrying it is labelled, which is the same reason
#: `gemm_identical.mojo` keeps its switch names in one place.
comptime SAB_ACCUM_BY_ADD = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_ACCUM_BY_ADD"
]()
#: Contract 7.4's price. The `accumulate` path REFILLS `dW` with `+0.0`
#: before folding, erasing the carry. **INERT on a single-microbatch gate**,
#: which is what a lane that never wrote clause (e) has.
comptime SAB_ACCUM_REFILLS = is_defined[
    "MOJOLEARN_EMB_SABOTAGE_ACCUM_REFILLS"
]()

comptime ANY_EMB_SABOTAGE = (
    SAB_FOLD_DESCENDING
    or SAB_FOLD_BALANCED_TREE
    or SAB_SEED_SEEDLESS
    or SAB_SINGLE_RUN_BYPASS
    or SAB_EMPTY_ROW_SKIPPED
    or SAB_EMPTY_ROW_NEG_ZERO
    or SAB_FOLD_READS_LAUNCH
    or SAB_RANK_BY_ARRIVAL
    or SAB_SORT_TIE_REVERSED
    or SAB_PAD_ROW_CONTRIBUTES
    or SAB_PAD_ROW_NEG_ZERO
    or SAB_NO_FLUSH_ACC
    or SAB_GATHER_NO_FLUSH
    or SAB_GATHER_CLAMP_OOR
    or SAB_ACCUM_BY_ADD
    or SAB_ACCUM_REFILLS
)


def emb_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner.

    A check MUST print this AND the numeric mode AND the resolved block size
    in its header, because a sabotage arm that silently failed to compile in
    looks exactly like a clause that is bit-inert, and `[[reached-but-inert]]`
    is the standing rule that those are not the same thing.
    """
    comptime if SAB_FOLD_DESCENDING:
        return String("FOLD_DESCENDING")
    comptime if SAB_FOLD_BALANCED_TREE:
        return String("FOLD_BALANCED_TREE")
    comptime if SAB_SEED_SEEDLESS:
        return String("SEED_SEEDLESS")
    comptime if SAB_SINGLE_RUN_BYPASS:
        return String("SINGLE_RUN_BYPASS")
    comptime if SAB_EMPTY_ROW_SKIPPED:
        return String("EMPTY_ROW_SKIPPED")
    comptime if SAB_EMPTY_ROW_NEG_ZERO:
        return String("EMPTY_ROW_NEG_ZERO")
    comptime if SAB_FOLD_READS_LAUNCH:
        return String("FOLD_READS_LAUNCH")
    comptime if SAB_RANK_BY_ARRIVAL:
        return String("RANK_BY_ARRIVAL")
    comptime if SAB_SORT_TIE_REVERSED:
        return String("SORT_TIE_REVERSED")
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        return String("PAD_ROW_CONTRIBUTES")
    comptime if SAB_PAD_ROW_NEG_ZERO:
        return String("PAD_ROW_NEG_ZERO")
    comptime if SAB_NO_FLUSH_ACC:
        return String("NO_FLUSH_ACC")
    comptime if SAB_GATHER_NO_FLUSH:
        return String("GATHER_NO_FLUSH")
    comptime if SAB_GATHER_CLAMP_OOR:
        return String("GATHER_CLAMP_OOR")
    comptime if SAB_ACCUM_BY_ADD:
        return String("ACCUM_BY_ADD")
    comptime if SAB_ACCUM_REFILLS:
        return String("ACCUM_REFILLS")
    return String("none")


# ===========================================================================
# THE ONE SCHEDULING ROW (DEVIATION 1320)
# ===========================================================================

#: SCHEDULING. What this lane would like a block to be, before the column's
#: cap and the identity floor are applied. 256 is `FLAT_TPB`'s value in
#: `gemm_identical.mojo` and `CE_TPB_WANT`'s in `training/mojo_only/loss.mojo`
#: and is chosen for the same non-reason: it is what fits, it has never been
#: measured here, and **do not read it as evidence that it is optimal
#: anywhere.** `lib_block_size`'s docstring makes exactly this disclaimer
#: about its own uniform rows.
comptime EMB_TPB_WANT = 256


def _emb_max_tpb[column: Int]() -> Int:
    """Threads per block, resolved the way
    `mojo_only/kernel_matrix.mojo::block_size_for` resolves its own.

    SCHEDULING, and the module docstring's three structural facts are why: it
    changes which thread does which work and never what is added to what.

    Two bounds, both from the kernel matrix and neither from a vendor branch:

      - **the identity floor.** Under `NUMERIC_IDENTICAL` the block is capped
        at `IDENTITY_FLOOR_BLOCK` so every column launches ONE geometry, the
        gate `block_size_for` grew on 2026-08-22 after an audit found
        IDENTITY_PATHS row 3 closed at the runtime REPORT and open at the
        comptime accessor the kernels actually compile against. **This lane
        does not strictly need it** -- no float crosses a thread boundary and
        every cross-thread operation is an integer one -- and it is applied
        anyway, because "this particular kernel does not care" is exactly the
        reasoning that put the hole in `block_size_for` in the first place.
      - **the vendor's dispatch cap**, `column_max_block_size`. Slack on all
        three founding columns (their caps are 1024), and NOT slack on the
        portable-baseline column, which guarantees only 128 invocations per
        workgroup and exposed the omission on its first run.

    `[[mojotrees-code-not-source-of-truth]]`: none of this has been measured.
    """
    comptime want = EMB_TPB_WANT
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime floored = (
        IDENTITY_FLOOR_BLOCK if identical
        and IDENTITY_FLOOR_BLOCK < want else want
    )
    comptime hard = column_max_block_size(column)
    return floored if floored < hard else hard


#: The resolved block. Every kernel below is launched at exactly this width.
comptime EMB_TPB = _emb_max_tpb[TARGET_COLUMN]()


def _grid_for(count: Int) -> Int:
    """Blocks for a flat `count`-element launch. Never zero, so a degenerate
    shape enqueues a kernel that returns rather than a launch with a zero
    grid dimension, which is a runtime error on some columns."""
    if count < 1:
        return 1
    return (count + EMB_TPB - 1) // EMB_TPB


# ===========================================================================
# K0: THE FORWARD GATHER (seams G1 and G2, contract section 4)
# ===========================================================================


def emb_gather_kernel(
    out_y: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    width_in: Int32,
    vocab_in: Int32,
):
    """`Y[t, j] = ftz(ftz(W[ids[t], j]))`. One thread owns one `(t, j)` cell.

    NO ARITHMETIC. Not a multiply, not an add, not a select. Contract 4.1.

    **THE FLUSH MOVES BITS HERE AND IT IS A KNOWING DEPARTURE FROM THE
    REFERENCE** (DEVIATION 1310). A copy performs no rounding, so a raw copy
    of a subnormal weight survives on EVERY vendor, Apple included, and the
    vendors would still AGREE. `ftz` therefore does not buy cross-vendor
    agreement at this seam -- that is already there -- it makes the embedding
    output obey the same denormal policy as every other stage on the card, so
    a subnormal cannot enter a transformer block through the one door that
    does no arithmetic. Applying it at G1 and again at G2 is bitwise the same
    as applying it once, and both are spelled because "the seam a kernel
    writes for another kernel to read" is the unit row 10's checklist is
    written in.

    **This is the only `ftz` seam in this lane that a single-column run can
    see move**, contract 9.3, which makes `SAB_GATHER_NO_FLUSH` this lane's
    only Apple-visible flush arm. Fixture F-SUBW plants a `0x00000001` weight.

    NO BOUNDS BRANCH ON THE ID ON THE NORMATIVE PATH. Contract section 8
    refuses an out-of-range id BY NAME on the host, before any launch, so a
    kernel-side clamp would be dead code that hides a data bug. The clamp
    exists only as `SAB_GATHER_CLAMP_OOR`, whose inert set is every fixture
    with no out-of-range id.

    `padding_idx` is NOT special here. `nn.Embedding`'s forward is a plain
    gather at every position, `padding_idx` included.
    """
    var n_positions = Int(n_positions_in)
    var width = Int(width_in)
    if width < 1:
        return
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n_positions * width:
        return
    var t = cell // width
    var j = cell - t * width
    var v = Int(ids.unsafe_load(t))
    comptime if SAB_GATHER_CLAMP_OOR:
        # SABOTAGE: clamp instead of refuse. INERT with no out-of-range id.
        var vocab = Int(vocab_in)
        if v < 0:
            v = 0
        if v >= vocab:
            v = vocab - 1
    var src = weight.unsafe_load(v * width + j)
    comptime if SAB_GATHER_NO_FLUSH:
        # SABOTAGE: the raw copy. INERT with no subnormal weight; NOT inert
        # on Apple, which is what makes it usable on one column.
        out_y.unsafe_store(cell, src)
        return
    out_y.unsafe_store(cell, ftz(ftz(src)))


# ===========================================================================
# R1: THE RUN COUNTS (contract 6.1, PLAN_SCAN)
# ===========================================================================


def emb_counts_kernel(
    counts: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    padding_idx_in: Int32,
):
    """`counts[v]` = the number of positions carrying `v`. One thread per `v`,
    walking `t` ASCENDING.

    **INTEGERS ONLY. Nothing here rounds and nothing here flushes**, so no
    block size, no grid shape and no vendor can reach the result. That is
    what lets this kernel's geometry be a free choice while contract 5.1's
    fold's is not.

    **NO ATOMIC.** The obvious spelling -- one thread per POSITION doing
    `atomicAdd(&counts[ids[t]], 1)` -- would give the same COUNTS, because
    integer addition is associative, and it is refused here for a reason that
    only bites two kernels later: it leads directly to computing the
    within-run RANK the same way, and **the rank an atomic hands out is
    arrival order**, which is exactly the defect the float atomic had, moved
    into the index domain where it looks safe. Contract 6.3 case 3, sabotage
    `SAB_RANK_BY_ARRIVAL`. `PLAN_SCAN` exists partly so that spelling never
    has to be reached for.

    THE COST, stated rather than hidden. This is `V * T` integer comparisons
    -- at the shipped Llama-3-8B shape (`V = 128256`, `T = 4096`) about 525
    million, over a 16 KB `ids` array that stays in cache on every column.
    `emb_perm_kernel` pays it a second time. **The run structure is computed
    ONCE over `T` and reused for all `d` columns**, which is what keeps this
    off the `V * T * d` cliff the one-hot GEMM of contract 5.2(d) falls off.
    It grows with `T`, and contract 6.1 states the bound: at `T = 100000`
    this is `1.3e10` and `PLAN_SORT` is the right plan instead. `PLAN_SORT`
    is not written.

    `padding_idx` DROPS AT THE SOURCE, DEVIATION 1311, so a position carrying
    it enters no run and row `padding_idx` keeps its `+0.0` fill.
    """
    var vocab = Int(vocab_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= vocab:
        return
    var n_positions = Int(n_positions_in)
    var pad = Int(padding_idx_in)
    var c = Int32(0)
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        # SABOTAGE: `padding_idx` positions are counted. INERT on a fixture
        # with no position carrying `padding_idx`.
        pad = EMB_NO_PADDING_IDX
    if v == pad:
        counts.unsafe_store(v, Int32(0))
        return
    for t in range(n_positions):
        if Int(ids.unsafe_load(t)) == v:
            c = c + Int32(1)
    counts.unsafe_store(v, c)


# ===========================================================================
# R2: THE RUN OFFSETS (contract 6.1)
# ===========================================================================


def emb_run_begin_kernel(
    run_begin: MutPointer[Int32, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
):
    """The exclusive prefix sum of `counts`, length `V + 1`.

    **THE FOLD SHAPE OF THIS SCAN IS FREE AND THAT IS A THEOREM.** Integer
    addition is exactly associative, so a serial scan, a Blelloch scan and a
    segmented scan give the same offsets bit for bit. That is the same
    argument `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.1 makes for
    its row maximum -- an execution plan may choose its own tree when the
    operation is exactly associative, and only then.

    **THIS SPELLING IS SINGLE-THREADED AND IT IS A SCHEDULING EMBARRASSMENT.**
    At `V = 128256` it is 128,256 dependent integer additions in ONE thread,
    launched at `grid_dim=1, block_dim=1`. It is here because it is obviously
    correct and because swapping it CANNOT MOVE A BIT;
    `gbdt/gpu_util/kernel/scan.mojo` is the replacement and contract OWED
    item 10 is the debt. Nothing about this is numerical.
    """
    var vocab = Int(vocab_in)
    var acc = Int32(0)
    for v in range(vocab):
        run_begin.unsafe_store(v, acc)
        acc = acc + counts.unsafe_load(v)
    run_begin.unsafe_store(vocab, acc)


# ===========================================================================
# R3: THE PERMUTATION (contract 6.1 and 6.3)
# ===========================================================================


def emb_perm_kernel(
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    n_positions_in: Int32,
    vocab_in: Int32,
    padding_idx_in: Int32,
):
    """`perm[run_begin[v] + r]` = the `r`-th position carrying `v`, ASCENDING
    `t`. One thread per `v`.

    **NO SORT, NO KEY, NO TIE CLASS, NO STABILITY QUESTION.** Each `v` owns
    its own region of `perm` and appends in `t` order, so the within-run
    ranks come out ascending BY CONSTRUCTION rather than by arrival. **The
    permutation is a pure function of `ids` and `V`**, which is contract 6.1,
    and it is the reason contract 6.2's whole determinism argument is about a
    plan that is not written rather than about this one.

    THE THREE WAYS TO GET THIS WRONG, contract 6.3, each with a sabotage.

      1. Keying on `id` alone with a sort that is NOT stable. That is
         `thrust::sort_by_key`'s exact defect, `sort.h:101`, which DEVIATION
         621 closed for the MST edges. `SAB_SORT_TIE_REVERSED` emits the
         reverse tie order an unstable sort is permitted to return, which is
         the shape `LINK_SAB_SORT_WEIGHT_ONLY` already has.
      2. Keying on `id` alone with a STABLE sort over an input that is not in
         position order. Stability preserves the order it was GIVEN, not
         ascending `t`. Unreachable from this kernel and reachable from
         `PLAN_SORT`, which is why contract 6.2(b) requires the two plans'
         `emb.perm` be compared rather than assumed equal.
      3. **Computing the rank with an ATOMIC INTEGER ADD on a per-bucket
         counter.** The COUNT is order free; **the SLOT is not.**
         `SAB_RANK_BY_ARRIVAL` is that spelling and it is **INERT on a
         fixture with no duplicate ids AND on a single-block launch**, where
         arrival order IS position order -- so the gate must run more than
         one block or the most dangerous arm in the lane looks inert.

    `emb.perm` IS A RECORDED STAGE, contract section 10, and that is the
    whole point of section 6 -- **a sort that is wrong is a WRONG ANSWER,
    detectable at `emb.perm` before it ever reaches a float**, rather than a
    silent redefinition of what "identical" means. A card that recorded only
    `emb.dw` would see a wrong tie order as a wrong gradient with no
    localization.
    """
    var vocab = Int(vocab_in)
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v >= vocab:
        return
    var n_positions = Int(n_positions_in)
    var pad = Int(padding_idx_in)
    comptime if SAB_PAD_ROW_CONTRIBUTES:
        pad = EMB_NO_PADDING_IDX
    if v == pad:
        return
    var w = Int(run_begin.unsafe_load(v))
    var hi = Int(run_begin.unsafe_load(v + 1))
    if w >= hi:
        return

    comptime if SAB_SORT_TIE_REVERSED:
        # SABOTAGE: the run is written with its ties in REVERSE position
        # order, which is an order an unstable id-keyed sort may return.
        # INERT with no duplicate ids and INERT on F-DUPSAME.
        var back = hi - 1
        for t in range(n_positions):
            if Int(ids.unsafe_load(t)) == v:
                perm.unsafe_store(back, Int32(t))
                back -= 1
        return

    comptime if SAB_RANK_BY_ARRIVAL:
        # SABOTAGE: contract 6.3 case 3, spelled without an atomic so that it
        # is reproducible enough to gate -- the rank is taken from the
        # position's own BLOCK INDEX rather than from the ascending walk,
        # which is what an arrival-ordered counter produces once more than
        # one block is in flight. INERT on a single-block launch and with no
        # duplicate ids.
        var nth = Int(block_dim.x)
        var r = 0
        var phase = 0
        while phase < 2:
            for t in range(n_positions):
                if Int(ids.unsafe_load(t)) == v:
                    var side = 1 if (t // nth) % 2 == 1 else 0
                    if side == phase:
                        perm.unsafe_store(w + r, Int32(t))
                        r += 1
            phase += 1
        return

    for t in range(n_positions):
        if Int(ids.unsafe_load(t)) == v:
            perm.unsafe_store(w, Int32(t))
            w += 1


# ===========================================================================
# E0: THE SEED (contract 5.5, 7.4 and 9.2)
# ===========================================================================


def emb_seed_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    cells_in: Int32,
):
    """`dW[v, j] = +0.0`, every one of the `V * d` cells. The `emb.dw_seed`
    stage on a non-accumulating call.

    **THE FILL IS NOT AN IMPLEMENTATION DETAIL AND IT MAY NOT BE SKIPPED.**
    Contract 5.5 -- the empty run's value is STATED, not derived, and an
    implementation must WRITE it rather than leave whatever was in the
    buffer. That is `gemm/IDENTICAL_FP32_CONTRACT.md` section 8's `k == 0`
    discipline and `training/IDENTICAL_LOSS_CONTRACT.md` 7.3's ignored-row
    discipline applied to a vocabulary row.

    **AT THE SHIPPED SHAPE THIS IS MOST OF THE OUTPUT.** At `V = 128256` and
    `T = 4096` at least 124,160 of the 128,256 rows are empty, and this
    kernel writes `V * d` = 525.3 M floats, 2.10 GB, against the 16.8 M
    floats of actual gradient -- a ratio of 31.3x. Contract 12.1's largest
    single fact is that **the atomic scatter pays exactly the same fill**
    (`at::zeros` then `atomicAdd`), so this is not a cost of the identity
    pin.

    **`+0.0` AND NOT `-0.0`.** `(+0) + (-0) = +0` in round-to-nearest on
    every backend, because it is IEEE-754 and not a codegen choice, so a
    `+0.0` seed is what makes a nonempty run of all-zero contributors return
    `+0.0` (contract 9.2(a)) and what gives contract 7.4's carry a value that
    cannot poison the chain. Sabotages `SAB_EMPTY_ROW_SKIPPED` -- **always
    inert unless the gate POISONS the buffer first**, because a fresh
    allocation may already be zero -- and `SAB_EMPTY_ROW_NEG_ZERO`.

    **IT IS NOT LAUNCHED AT ALL WHEN `accumulate` IS TRUE**, contract 7.4,
    and `SAB_ACCUM_REFILLS` is the arm that launches it anyway.
    """
    var cells = Int(cells_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= cells:
        return
    comptime if SAB_EMPTY_ROW_SKIPPED:
        # SABOTAGE: the store is skipped. ALWAYS INERT unless the gate
        # poisons the buffer first. The gate must poison.
        return
    comptime if SAB_EMPTY_ROW_NEG_ZERO:
        # SABOTAGE: `-0.0`. Never inert on `emb.dw`, and unlike the loss
        # lane's `L_IGNORED_ROW_NEG_ZERO` there is no downstream fold here to
        # launder it -- `dW` IS the output.
        dw.unsafe_store(i, Float32(-0.0))
        return
    dw.unsafe_store(i, Float32(0.0))


def emb_pad_row_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    width_in: Int32,
    padding_idx_in: Int32,
):
    """`dW[padding_idx, :] = +0.0`, STORED. Contract section 8, DEVIATION
    1311.

    **THIS IS BELT, NOT BRACES, AND IT IS SPELLED ANYWAY.** Positions
    carrying `padding_idx` are already dropped at the source by
    `emb_counts_kernel` and `emb_perm_kernel`, so run `padding_idx` is empty
    and its row already holds the `+0.0` fill. The two spellings reach the
    same bits, PROVABLY, because a position carrying `padding_idx` can only
    ever contribute to row `padding_idx`, which this kernel overwrites.

    The contract pins the DROP-AT-SOURCE spelling because under `PLAN_SORT`
    the two produce different `emb.perm` stages and the card records
    `emb.perm`. This kernel exists so that the `accumulate` path -- where
    the fill did not run -- still zeroes the row, which is what
    `nn.Embedding(padding_idx=p)` means. Sabotage `SAB_PAD_ROW_NEG_ZERO`,
    **inert on a fixture with no `padding_idx`.**
    """
    var width = Int(width_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= width:
        return
    var base = Int(padding_idx_in) * width
    comptime if SAB_PAD_ROW_NEG_ZERO:
        dw.unsafe_store(base + j, Float32(-0.0))
        return
    dw.unsafe_store(base + j, Float32(0.0))


# ===========================================================================
# E1-E4: THE FOLD (contract 5.1) -- THE ONE KERNEL THIS LANE IS ABOUT
# ===========================================================================


def emb_backward_kernel(
    dw: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin],
    perm: MutPointer[Int32, MutAnyOrigin],
    run_begin: MutPointer[Int32, MutAnyOrigin],
    vocab_in: Int32,
    width_in: Int32,
):
    """**THE CONTRACT'S FOLD**, contract 5.1. One thread owns one `(v, j)`
    cell, walks its own run ASCENDING and adds.

        acc = dW[v, j]                     the seed, already STORED by E0
        for r in [run_begin[v], run_begin[v+1])      ASCENDING
            acc = ftz( ftz(acc) + ftz(dY[perm[r] * d + j]) )
        dW[v, j] = ftz(acc)

    **NO FLOAT CROSSES A THREAD BOUNDARY.** No shared memory, no warp
    primitive, no atomic, no cross-block reduction. That is the construction
    `modeling_llama.mojo`, `modeling_mamba.mojo` and
    `training/mojo_only/optimizer.mojo` all state about themselves, and
    contract 0.2 is the argument that it survives a scatter-shaped gradient
    once the run structure is materialized. **It is the finding of this
    lane.**

    THE SIX CLAUSES, each separately falsifiable.

    1. **ASCENDING ABSOLUTE POSITION.** `perm` is ascending in `t` inside
       every run by `emb_perm_kernel`'s construction, so walking `r` upward
       walks `t` upward. `SAB_FOLD_DESCENDING` is **inert on every run of
       length <= 1 and on every run whose contributors are bitwise equal** --
       a permutation of a constant sequence is the same sequence, which is
       why fixture F-DUPSAME is a required NEGATIVE CONTROL and not a test.
    2. **SERIAL.** No sub-partition, no leaf, no tree. `R` contributors
       perform exactly `R` dependent additions. `SAB_FOLD_BALANCED_TREE` is
       **PROVABLY inert at every `R <= 3`** (contract 5.4: at R = 2 and R = 3
       the balanced tree IS the ascending chain, node for node) and first
       moves at `R = 4`. Contract 5.3 is why the tree is refused, in one
       line -- **a run length is DATA and a gemm `k` is a SHAPE.**
    3. **SEEDED**, and the seed arrives through `dW` itself rather than as a
       literal, which is what makes contract 7.4's carry the same code path
       as a fresh call. `SAB_SEED_SEEDLESS` and `SAB_SINGLE_RUN_BYPASS` are
       the two wrong spellings and **they share one inert mask** -- every
       input except a cell whose sole contributor is `-0.0`, where the
       seeded chain gives `0x00000000` and both arms give `0x80000000`. Both
       exist because a gate carrying only one would not know which clause it
       had proved.
    4. **ONE THREAD, ONE CELL.** The grid is `(V * d)` flat.
    5. **`R == 0` performs no addition and keeps the seed**, which E0 has
       already STORED. Not skipped -- already written.
    6. **`R == 1` performs ONE addition.** Deliberately NOT gemm 7.3's
       `P == 1` case: gemm's tree is SEEDLESS so there the rule and the
       optimization coincide, and this chain is SEEDED so they diverge, at
       exactly one input.

    THE FLUSHES ARE ROW 10's CHECKLIST UNIT. E1 flushes each `dY` as loaded,
    E2 flushes the accumulator as read, E3 flushes it after EVERY add and E4
    flushes the result as stored. **E3 is the expensive one and it is not
    optional** -- an accumulator that dips into the subnormal range mid-run
    is an INTERMEDIATE, and flushing only at the end makes Metal, which
    flushed on the spot, diverge from CUDA, which carried it, from that step
    onward. That is `gemm/IDENTICAL_FP32_CONTRACT.md` 5c verbatim, and it is
    the only arithmetic overhead this profile has, because there is no
    multiply-add to price it against. `SAB_NO_FLUSH_ACC` is the arm and it is
    **INERT ON APPLE ENTIRELY**, contract 9.3; gemm 4.1's correction is the
    standing warning that Apple's bits not moving is not evidence of unreach.

    THE ACCESS PATTERN IS THIS LANE'S WORST, and contract 12.2 names it
    rather than hiding it. Consecutive `r` at a fixed `j` read `dY` at stride
    `d`, which is a gather where the atomic scatter streams. **An execution
    plan that transposes the tile is free to**, because a transpose reaches
    no arithmetic -- what it may not change is which value is added to which,
    or the order.

    LOAD IMBALANCE, also named. A hot token's thread walks `R_max` while most
    threads walk zero. In the degenerate `R == T` fixture (F-HOT) the
    parallel width collapses from `V * d` to `d`; at `T = 4096` and
    `d = 4096` that is a 4096-deep dependent chain across 4096 threads.
    Bounded, correct and slow. Contract 12.2.

    **WHERE THIS MAY BE CHEAPER THAN THE ATOMIC, which is not obvious.** A
    token appearing 300 times makes 300 threads contend on one address in the
    atomic spelling and the hardware serializes them. Here ONE thread owns
    the address and performs 300 dependent adds with no contention, no
    retries and no cache-line ping-pong. The two are the same 300 steps and
    the atomic pays a lock protocol on top. **That is a PREDICTION. It has
    not been measured and contract 12.3 forbids quoting it as a result.**
    """
    var width = Int(width_in)
    if width < 1:
        return
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= vocab * width:
        return
    var v = cell // width
    var j = cell - v * width

    var lo = Int(run_begin.unsafe_load(v))
    var hi = Int(run_begin.unsafe_load(v + 1))
    if lo >= hi:
        # Clause 5. The empty run keeps its seed, which E0 STORED.
        return

    # Clause 3. The seed is read from `dW`, so the carry of contract 7.4 and
    # a fresh call are ONE code path and not two.
    var acc = dw.unsafe_load(cell)

    comptime if SAB_SEED_SEEDLESS:
        # SABOTAGE: gemm 9.2(b)'s seedless spelling. INERT except at a sole
        # `-0.0` contributor.
        acc = ftz(dy.unsafe_load(Int(perm.unsafe_load(lo)) * width + j))
        for r in range(lo + 1, hi):
            var t2 = Int(perm.unsafe_load(r))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(t2 * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_SINGLE_RUN_BYPASS:
        # SABOTAGE: "one contributor needs no adding". Same inert mask as
        # SAB_SEED_SEEDLESS and a different wrong reason.
        if hi - lo == 1:
            var t1 = Int(perm.unsafe_load(lo))
            dw.unsafe_store(cell, ftz(dy.unsafe_load(t1 * width + j)))
            return

    comptime if SAB_FOLD_DESCENDING:
        # SABOTAGE: the run walked DESCENDING. INERT at R <= 1 and on
        # bitwise-equal contributors.
        var rdx = hi - 1
        while rdx >= lo:
            var td = Int(perm.unsafe_load(rdx))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(td * width + j)))
            rdx -= 1
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_FOLD_READS_LAUNCH:
        # SABOTAGE: the walk starts at an offset derived from `block_dim.x`
        # and wraps, so the SAME data folded at two block sizes gives two
        # answers. Never inert on a run of length >= 2. It is the falsifier
        # for "no launch quantity reaches the arithmetic" and it reaches the
        # clause `gemm_identical.mojo`'s `SAB_LEAF_READS_LAUNCH` reaches one
        # layer down.
        var span = hi - lo
        var start = Int(block_dim.x) % span
        for qL in range(span):
            var rr = lo + (start + qL) % span
            var tr = Int(perm.unsafe_load(rr))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(tr * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    comptime if SAB_FOLD_BALANCED_TREE:
        # SABOTAGE: ONE LEVEL of gemm 7.2's adjacent pairing, folded into the
        # chain -- `acc += (a[2q] + a[2q+1])` for each adjacent pair, with an
        # odd tail added last. Contract 5.2(b)'s candidate, which the loss
        # lane's DEVIATION 1152 chose on its own axis and contract 5.3
        # refuses on this one.
        #
        # **WHY ONE LEVEL AND NOT THE WHOLE TREE, stated because a reader
        # will ask.** A full tree needs `R` floats of per-thread scratch, and
        # `[[stack_allocation is thread-local MEMORY, not registers]]` --
        # every `stack_allocation` in this tree is SHARED and a per-thread
        # slab of 32 floats at 256 threads is 32 KB, which is past the
        # portable baseline column's 16 KB guarantee. So the device arm is
        # the SCRATCH-FREE spelling and the FULL tree lives on the host as
        # `embedding_oracle.mojo::emb_fold_balanced_tree_diagnostic`, which
        # is what the check compares against.
        #
        # **THE TWO AGREE EXACTLY WHERE IT MATTERS.** At `R = 2, 3, 4` this
        # arm IS gemm's balanced tree, node for node -- one pairing level is
        # the whole tree at `R = 4` and the tree is the chain at `R = 2, 3`.
        # They diverge only at `R >= 8`, where the true tree does a second
        # pairing level and this does not. So the clause it falsifies is the
        # SERIAL one, it falsifies it at the same `R` the true tree does, and
        # **its inert mask is the same PROVABLE `R <= 3`** of contract 5.4.
        # A check must compare this arm against the HOST tree and record
        # where the two stop agreeing rather than assuming they do not.
        var span2 = hi - lo
        var pairs = span2 // 2
        for qT in range(pairs):
            var ta = Int(perm.unsafe_load(lo + 2 * qT))
            var tb = Int(perm.unsafe_load(lo + 2 * qT + 1))
            var pair_sum = ftz(
                ftz(dy.unsafe_load(ta * width + j))
                + ftz(dy.unsafe_load(tb * width + j))
            )
            acc = ftz(ftz(acc) + pair_sum)
        if span2 % 2 != 0:
            # The odd tail, added LAST. That is what makes this the tree at
            # `R = 3` and therefore provably inert there.
            var tt = Int(perm.unsafe_load(hi - 1))
            acc = ftz(ftz(acc) + ftz(dy.unsafe_load(tt * width + j)))
        dw.unsafe_store(cell, ftz(acc))
        return

    # THE PINNED PATH. Contract 5.1, and every other arm above returns.
    for r in range(lo, hi):
        var t = Int(perm.unsafe_load(r))
        var contribution = dy.unsafe_load(t * width + j)
        comptime if SAB_NO_FLUSH_ACC:
            # SABOTAGE: seam E3 dropped, the accumulator flushed only at the
            # end. INERT with no subnormal intermediate, and INERT ON APPLE
            # ENTIRELY.
            acc = acc + contribution
        else:
            acc = ftz(ftz(acc) + ftz(contribution))
    dw.unsafe_store(cell, ftz(acc))


# ===========================================================================
# THE LAUNCHERS
# ===========================================================================
# Both are the `_into` form: caller-owned buffers, caller-owned run scratch,
# ASYNCHRONOUS, nothing waits. A training step enqueues the loss backward,
# the `lm_head` backward and this back to back and synchronizes once, which
# is the shape the whole file exists for.


def emb_run_scratch_ints(vocab: Int, n_positions: Int) -> Int:
    """Integers of run scratch `identical_embedding_backward_into` needs.

    `counts` is `V`, `run_begin` is `V + 1`, `perm` is at most `T`. Never
    less than 1, so the buffers are always constructible.

    **SIZE THEM WITH THIS AND NOT WITH A GUESS.** `identical_gemm_into`'s own
    docstring records what the alternative cost the GEMM lane -- a 1-float
    workspace passed to a SPLITK dispatch still produced the right answer at
    `64 x 4` because the allocation had slack, and only at `64 x 64` did
    whole regions of the output come back `+0.0`.
    """
    var need = vocab + vocab + 1 + n_positions
    if need < 1:
        return 1
    return need


def emb_refuse_device_ids(
    ctx: DeviceContext,
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Contract section 8 and 9.1 ON THE DEVICE ENTRY POINTS, which is where
    they were missing. DEVIATION 1598.

    **THIS ONE IS NOT A NUMERICAL DEFECT. IT IS AN OUT-OF-BOUNDS READ.**
    `emb_gather_kernel` computes `weight.unsafe_load(v * width + j)` with NO
    bounds branch on the normative path -- the only bounds handling in the
    file lives inside `SAB_GATHER_CLAMP_OOR`, a SABOTAGE arm, so a build
    without that define has none at all. At `v == -1` the load addresses
    `weight - width + j`, which is BEFORE THE BUFFER. Found by
    `embedding_check.mojo` on 2026-08-25 (DEVIATION 1506), which measured
    the NaN half and DELIBERATELY DID NOT RUN the out-of-range half, on the
    grounds that performing an out-of-bounds read to demonstrate an
    out-of-bounds read is the bug rather than the test. That was the right
    call and this is the fix it named.

    **THE THIRD LANE WITH THE SAME SHAPE OF HOLE.** `loss.mojo` (DEVIATION
    1495) and `optimizer.mojo` (1496) both promised "refused by name before
    any recorded stage" in a contract, delivered it in the ORACLE, and
    omitted it from the device entry. Three for three. The pattern is that a
    refusal written once, in the reference implementation, reads as done.

    **IT CALLS THE ORACLE'S OWN `emb_refuse_ids`** rather than restating the
    range test, so both sides fail with the same name, the same position and
    the same value in the message -- which is the only thing that makes two
    refusals comparable. Contract section 8's own words are NOT CLAMPED, NOT
    WRAPPED, NOT SILENTLY DROPPED, and a clamp would turn a data bug into a
    wrong gradient on a real vocabulary row with no stage at which it becomes
    visible.

    **COST**: one download of `ids`, which is `n_positions` int32 and is the
    cheapest of the three lanes' refusals by a wide margin -- ids are one per
    token, not one per parameter. A device-side scan is OWED here as
    elsewhere and matters far less.
    """
    if n_positions <= 0:
        return
    var h = ctx.enqueue_create_host_buffer[DType.int32](n_positions)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=ids)
    ctx.synchronize()
    var lids = List[Int32]()
    for i in range(n_positions):
        lids.append(h.unsafe_ptr().unsafe_load(i))
    emb_refuse_ids(lids, cfg)
    _ = h


def identical_embedding_forward_into(
    ctx: DeviceContext,
    mut out_y: DeviceBuffer[DType.float32],
    mut weight: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Seams G1 and G2, enqueued. Nothing waits.

    EVERY BUFFER IS THE CALLER'S. `[[mojo-buffer-freed-at-last-use]]`: a
    `DeviceBuffer` is dead at its `.unsafe_ptr()`, so every one of these must
    outlive the caller's `ctx.synchronize()`.

    **THE HOST REFUSALS RUN BEFORE THIS, NOT INSIDE IT.** Contract section 8
    refuses an out-of-range id by name and section 9.1 refuses a nonfinite
    weight by bits, and both are
    `embedding_oracle.mojo`'s (`emb_refuse_ids`, `refuse_nonfinite`) because
    they are pure, host side and testable with no device -- which is
    `gemm_backward_a_call`'s discipline applied to a validation. A device
    that had to check would need a flag, a readback and a decision about what
    a half-written output means.

    `T == 0` and `d == 0` enqueue a kernel that returns, contract section 8.
    """
    # DEVIATION 1598: contract section 8 on the DEVICE path. The gather
    # has NO bounds branch outside a sabotage arm, so a negative id reads
    # BEFORE the weight buffer. FIRST statement in the body.
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)

    if cfg.width < 1 or n_positions < 1:
        return
    var cells = n_positions * cfg.width
    ctx.enqueue_function[emb_gather_kernel](
        out_y.unsafe_ptr(),
        weight.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.width),
        Int32(cfg.vocab),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )


def identical_embedding_backward_into(
    ctx: DeviceContext,
    mut dw: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut ids: DeviceBuffer[DType.int32],
    mut counts: DeviceBuffer[DType.int32],
    mut run_begin: DeviceBuffer[DType.int32],
    mut perm: DeviceBuffer[DType.int32],
    n_positions: Int,
    cfg: EmbConfig,
) raises:
    """Seams E0 through E4, enqueued. Nothing waits.

    THE ORDER OF THE ENQUEUES IS THE ORDER OF THE STAGES. Each kernel reads
    only buffers an earlier one wrote, and MAX runs one context in order, so
    no barrier is spelled here. A second stream would break that and would be
    a change to this function's contract rather than to its arithmetic.

    **`counts`, `run_begin` and `perm` ARE THE CALLER'S** and must be sized
    with `emb_run_scratch_ints` and kept alive past the caller's own
    `ctx.synchronize()`. They are `PLAN_SCAN`'s materialized run structure,
    contract 6.1, and they are also three RECORDED STAGES, contract section
    10, which is why they are caller-owned rather than hidden -- a check
    hashes them.

    **THE FILL IS SKIPPED WHEN `cfg.accumulate` IS TRUE**, contract 7.4 and
    DEVIATION 1309, and that single `if` is the whole of the microbatch
    clause. With the fill skipped, `emb_backward_kernel` seeds each cell's
    chain from the `dW` it was handed and CONTINUES it, which **reproduces
    the unsplit call BIT FOR BIT at EVERY split point** -- the first
    microbatch computes a prefix of the chain, the second continues it, and
    the resulting sequence of additions is the unsplit one term for term.
    `dW = ftz(dW_first + dW_second)` does NOT, in general, which is contract
    5.4's `R = 4` row again, and `SAB_ACCUM_BY_ADD` is that arm.

    Set beside `gemm/IDENTICAL_BACKWARD_PLAN.md` 3.2, where `dB` splits
    reproduce only at an ALIGNED split, one that is both a leaf boundary and
    a subtree boundary of v1's balanced tree -- **a chain has no boundaries
    to align to.** Do NOT cite that section's measurement as evidence that an
    accumulator must be a tree: its splits used TWO pieces, and over two
    pieces a serial running sum and a balanced tree are the SAME operation.
    That correction was made in that file on 2026-08-25.

    THE PRICE OF THE CARRY, contract 7.4. A caller that carries must (i)
    `+0.0`-fill `dW` exactly once before the FIRST microbatch, by calling
    this with `accumulate` false, (ii) NOT fill it again, and (iii) present
    microbatches in ASCENDING `t`. Out of order and contract 5.1 clause 1 is
    violated and the bits move. Sabotages `SAB_ACCUM_BY_ADD` and
    `SAB_ACCUM_REFILLS`.

    ROW INDEPENDENCE. Nothing in the fold reads `T` other than as the bound
    of `emb_counts_kernel`'s and `emb_perm_kernel`'s enumeration, so a caller
    may chunk the columns `j` and the vocabulary rows `v` any way it likes
    and the bits do not move. Only the run structure sees all `T` positions
    at once, and it is INTEGER.
    """
    # DEVIATION 1598: contract section 8 on the DEVICE path. The gather
    # has NO bounds branch outside a sabotage arm, so a negative id reads
    # BEFORE the weight buffer. FIRST statement in the body.
    emb_refuse_device_ids(ctx, ids, n_positions, cfg)

    if cfg.vocab < 1 or cfg.width < 1:
        return
    var cells = cfg.vocab * cfg.width

    # ---- E0, the seed. NOT LAUNCHED under `accumulate`, contract 7.4.
    var fill = not cfg.accumulate
    comptime if SAB_ACCUM_REFILLS:
        # SABOTAGE: refill even when carrying, erasing the carry. INERT on a
        # single-microbatch gate.
        fill = True
    if fill:
        ctx.enqueue_function[emb_seed_kernel](
            dw.unsafe_ptr(),
            Int32(cells),
            grid_dim=(_grid_for(cells), 1, 1),
            block_dim=(EMB_TPB, 1, 1),
        )

    if n_positions < 1:
        # Contract section 8. `T == 0` leaves every row at its seed, which is
        # `+0.0` on a fresh call and is already STORED. `padding_idx` still
        # gets its row, below.
        if cfg.has_padding():
            ctx.enqueue_function[emb_pad_row_kernel](
                dw.unsafe_ptr(),
                Int32(cfg.width),
                Int32(cfg.padding_idx),
                grid_dim=(_grid_for(cfg.width), 1, 1),
                block_dim=(EMB_TPB, 1, 1),
            )
        return

    # ---- R1. One thread per `v`, `V * T` integer comparisons. No atomic.
    ctx.enqueue_function[emb_counts_kernel](
        counts.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.vocab),
        Int32(cfg.padding_idx),
        grid_dim=(_grid_for(cfg.vocab), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    # ---- R2. One thread, `V` dependent integer adds. Contract OWED item 10:
    # this is a scheduling embarrassment and swapping it cannot move a bit.
    ctx.enqueue_function[emb_run_begin_kernel](
        run_begin.unsafe_ptr(),
        counts.unsafe_ptr(),
        Int32(cfg.vocab),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    # ---- R3. One thread per `v`, ascending `t`, no sort and no tie class.
    ctx.enqueue_function[emb_perm_kernel](
        perm.unsafe_ptr(),
        run_begin.unsafe_ptr(),
        ids.unsafe_ptr(),
        Int32(n_positions),
        Int32(cfg.vocab),
        Int32(cfg.padding_idx),
        grid_dim=(_grid_for(cfg.vocab), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    # ---- E1 through E4. One thread per `(v, j)`. No float crosses a thread
    # boundary here and there is nowhere in this launch that one could.
    ctx.enqueue_function[emb_backward_kernel](
        dw.unsafe_ptr(),
        dy.unsafe_ptr(),
        perm.unsafe_ptr(),
        run_begin.unsafe_ptr(),
        Int32(cfg.vocab),
        Int32(cfg.width),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(EMB_TPB, 1, 1),
    )

    # ---- `padding_idx`. Belt, contract section 8: the row is already `+0.0`
    # because the positions were dropped at the source, and the store is
    # spelled so that the `accumulate` path -- where the fill did not run --
    # still zeroes it, which is what `nn.Embedding(padding_idx=p)` means.
    if cfg.has_padding():
        ctx.enqueue_function[emb_pad_row_kernel](
            dw.unsafe_ptr(),
            Int32(cfg.width),
            Int32(cfg.padding_idx),
            grid_dim=(_grid_for(cfg.width), 1, 1),
            block_dim=(EMB_TPB, 1, 1),
        )
