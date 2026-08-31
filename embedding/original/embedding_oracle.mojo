# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host FP32 oracle of profile `mojolearn.identical.embedding.fp32.v1`.

**THIS FILE HAS NEVER BEEN COMPILED AND HAS NEVER BEEN EXECUTED.** No device
has run a single stage of it, no gate has ever failed against it, no sabotage
arm has ever been built, and every claim in its docstrings was derived on
paper or read out of source on 2026-08-25. Written by the embedding lane,
DEVIATIONS 1300-1339. The contract is
`embedding/IDENTICAL_EMBEDDING_CONTRACT.md` and this file is that contract in
code -- the two must be read together and every function below cites its
section.

WHAT IS OWED, and none of it is in this file
---------------------------------------------
  - `embedding/original/embedding_check.mojo` and
    `embedding/original/embedding_fixture.mojo` DO NOT EXIST. **Not one
    clause of the contract has been falsified by a sabotage.** Contract
    section 11 is a specification for gates, not a report of any, and
    contract OWED item 1 is the debt.
  - `PLAN_SORT` (contract 6.2) is specified and NOT WRITTEN. What is here
    instead is `emb_perm_by_total_order_key`, the HOST spelling of the same
    permutation, which exists so that the sort's determinism argument is
    CODE a check can run rather than prose a reader has to accept. Contract
    OWED item 2.
  - `refuse_nonfinite` is a FOURTH copy (DEVIATION 1313). The first three are
    `mamba/original/mamba_oracle.mojo:57`,
    `training/original/optimizer_oracle.mojo:162` and
    `training/original/loss_oracle.mojo:167`, and both training files
    already record the same debt. It belongs in `original/numerics.mojo`;
    that file is under concurrent edit by the numerics lane and the lift is
    theirs. Contract OWED item 3.
  - `merge_sort_u64_with_index` duplicates
    `hierarchy/derived/sparse/op/sort.mojo`'s. It is copied rather than
    imported because that module imports `max.gpu.host.DeviceBuffer` at
    module scope and a host-only oracle must build with no device present.
    Contract OWED item 4.
  - `core/identity_trace.mojo` recording is the check file's job and the
    check file does not exist, so nothing here emits a card.

NOT A PORT, and it replaces no upstream call. cuML, cuVS and RAFT contain no
embedding table -- they are classical-ML libraries -- so there is nothing in
this repository's mirror set to check against and this file is what stands in
for one. `torch.nn.Embedding` is the DESIGN reference and NOT a bit
reference. ATen's `embedding_dense_backward` could not be read; there is no
PyTorch checkout in `/Users/andrewhendel/CascadeProjects/upstream/`, which is
the same gap the transformer and loss lanes both record. **The ARITHMETIC
ORDER below is this repository's own** and is stated clause by clause in the
contract.

What CAN be said about the reference without reading it, because it decides a
clause: its accumulation buffer is `at::zeros` and it ADDS INTO IT, which is
the `+0.0` seed of contract 5.1, so the seed is the reference's behavior and
not an accident. And its order is the ARRIVAL ORDER OF ATOMICS, which is not
a spelling at all, so there is no upstream order for this file to mirror.
`COPY, DO NOT IMPROVE` has nothing here to copy, and that is stated so nobody
goes looking for it.

WHAT THIS FILE IS FOR
---------------------
1. **Be the definition.** The NORMATIVE answer of the profile is
   `emb_forward_oracle` and `emb_backward_oracle`. Not "close to"; the same
   bits, on Apple, NVIDIA and AMD, at every launch geometry and every
   execution plan.
2. **Be built from the DECLARED helpers.** Every seam is
   `original.numerics.ftz`, the actual helper and not a local copy, so this
   file cannot drift into an independent opinion about what IDENTICAL means.
   The consequence is that under `NUMERIC_FAST` the pin compiles away and
   **THIS FILE IS NOT THE CONTRACT** -- it is the FAST spelling of the same
   loops. Every check must print the mode it compiled in.
3. **Be the instrument that proves a fixture SEPARATES.** The fold shape and
   the permutation spelling are both PARAMETERS here, so "does this fixture
   distinguish the chain from the balanced tree" is a question this file
   answers rather than one the kernel is trusted about. That is what
   `emb_fold_balanced_tree_diagnostic` and `emb_perm_by_total_order_key` are
   for, and the contract's 11.2 fixture table was computed with them.
4. **Carry the ADVERSARIES beside the answer, so the choice stays
   falsifiable.** Contract 5.2 refused three candidates; two of them are
   spelled below as DIAGNOSTIC functions, clearly not normative, exactly as
   `gemm_oracle_serial` sits beside `gemm_oracle`.

THERE IS NO MULTIPLY IN THE NORMATIVE PATH (contract 4.1, DEVIATION 1317).
The backward is a pure sum and the forward is a pure copy, so
`identical_mul_add` occupies no seam of this profile and the whole multiply-
add policy of `gemm/IDENTICAL_FP32_CONTRACT.md` section 4 is satisfied
VACUOUSLY rather than followed. It is imported and used at exactly one place,
`emb_fold_via_fma_diagnostic`, whose only job is to make the equivalence
lemma of contract 4.1 a MEASUREMENT instead of an argument.

`[[mojo-buffer-freed-at-last-use]]`: nothing here touches a device.
`[[mojo-string-float-roundtrip]]`: nothing here prints; the check will, and
it must print hex bits beside every decimal.
`[[mojo-amp-plus-is-bitwise-and]]`: `pack_emb_key` uses `<<`, `|` and `&`
and never `&+`, which computes a bitwise AND with no compile error and has
produced wrong keys in this tree twice.
"""

from original.numerics import ftz, identical_mul_add


# ===========================================================================
# PROFILE CONSTANTS AND REFUSAL BOUNDS (contract section 3)
# ===========================================================================
# There is no constant here that reaches the ARITHMETIC, and contract 3.1 is
# why that is the clause rather than an absence. A serial ascending chain has
# no leaf size, no leaf count, no cap and no tree, so unlike
# `gemm/original/gemm_oracle.mojo`'s `CONTRACT_K_LEAF_MIN` and
# `CONTRACT_MAX_LEAVES` there is nothing below whose value a v2 could change.
# Both constants below are REFUSAL bounds.

#: The largest `T` this profile accepts. `2^30 - 1`, which keeps a position
#: inside the low 32 bits of `pack_emb_key`'s packed key with room to spare
#: and leaves the high half entirely to the id. Contract 3 and 6.2(a).
#:
#: A REFUSAL BOUND. It reaches no arithmetic; raising it changes which inputs
#: are accepted and moves no bit of any accepted one.
comptime EMB_MAX_POSITIONS = 1073741823

#: "no `padding_idx`". Any Int outside `[0, V)` works; this is the value
#: `nn.Embedding` uses for the absent case and is the only one this profile
#: ever passes. Contract section 8.
comptime EMB_NO_PADDING_IDX = -1

#: `+inf` by bits, because a COMPARE cannot be trusted to see an infinity on
#: a column that flushes compare operands (IDENTITY_PATHS row 49). Integer
#: operations do not flush anywhere.
comptime EMB_POS_INF_BITS = UInt32(0x7F800000)

#: The 32-bit mask `pack_emb_key` needs. `[[mojo-int-widening-sign-extends]]`:
#: an `Int32` widened to `UInt64` SIGN-EXTENDS, so a negative id becomes
#: `0xFFFFFFFF........` and sorts above every real token. This mask is not
#: decoration and the trap has cost this repository a run.
comptime EMB_KEY_MASK = UInt64(0xFFFFFFFF)


@fieldwise_init
struct EmbConfig(Copyable, Movable):
    """One embedding call's configuration. Every field is a HOST value known
    before any launch and none of them is data dependent.

    `accumulate` is the ONLY piece of state in this profile (contract section
    2 and 7.4). When it is true the backward SEEDS each cell's chain from the
    `dW` it was handed instead of from `+0.0`, which is what makes a
    microbatch split bit exact at EVERY split point rather than only at an
    aligned one. When it is false the backward fills `dW` with `+0.0` first.

    **A caller that carries must fill exactly once, must not fill again, and
    must present microbatches in ASCENDING `t`.** Contract 7.4's price.
    """

    var vocab: Int
    var width: Int
    var padding_idx: Int
    var accumulate: Bool

    @staticmethod
    def llama(vocab: Int, width: Int) -> Self:
        """A fresh gradient with no `padding_idx`, which is what
        `LlamaModel.embed_tokens` is configured as."""
        return Self(vocab, width, EMB_NO_PADDING_IDX, False)

    def has_padding(self) -> Bool:
        """Whether `padding_idx` names a real row. Contract section 8. The
        comparison is against HOST integers, so this is a configuration
        branch and never a data-dependent one."""
        return self.padding_idx >= 0 and self.padding_idx < self.vocab


# ===========================================================================
# THE REFUSALS (contract sections 8 and 9.1)
# ===========================================================================


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """IDENTITY_PATHS row 39: a NaN or an infinity in an input is REFUSED BY
    NAME before any recorded stage.

    **A FOURTH COPY. DEVIATION 1313.** The first three are
    `mamba/original/mamba_oracle.mojo:57`,
    `training/original/optimizer_oracle.mojo:162` and
    `training/original/loss_oracle.mojo:167`, and the last two both already
    record the same debt. All four must stay the same shape. The right home
    is `original/numerics.mojo` and the lift is the numerics lane's;
    contract OWED item 3.

    Why refuse rather than propagate. A computed NaN's PAYLOAD is vendor
    shaped -- row 39 measured `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA
    and `0xffc00000` on AMD for one IEEE answer -- so a certified stage may
    not contain one. Torch's behavior here (propagate) is a KNOWING
    DEPARTURE, contract section 13.

    **TESTED BY BITS, NOT BY COMPARES.** Metal flushes COMPARE operands (row
    49's measurement), so a compare-written test has one meaning on one
    column and another elsewhere. `(bits & 0x7FFFFFFF) > 0x7F800000` is a NaN
    of either sign and any payload; `== 0x7F800000` is an infinity of either
    sign.

    **THIS PROFILE HAS NO NONFINITE-INTERMEDIATE GAP**, which is a real
    difference from `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` section
    8, and contract 9.1 is the argument. A sum of finite terms can overflow
    to an infinity, deterministically and identically on every vendor because
    it is IEEE, and it CANNOT produce a NaN, because a NaN needs `inf - inf`
    or `0 * inf` and there is no subtraction and no multiplication anywhere
    in this profile. So refusing the inputs covers every NaN it can hold.
    """
    for i in range(len(values)):
        var au = rebind[UInt32](values[i].to_bits()) & UInt32(0x7FFFFFFF)
        if au > EMB_POS_INF_BITS:
            raise Error(
                String("embedding: NaN in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        if au == EMB_POS_INF_BITS:
            raise Error(
                String("embedding: infinity in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (contract 9.1)"
            )


def emb_refuse_shape(cfg: EmbConfig, n_positions: Int) raises:
    """Contract section 3's shape refusals and section 8's degenerate cases.

    `T == 0` and `d == 0` are LEGAL and produce stated values (contract 8);
    `V <= 0`, `d < 0` and `T < 0` are ERRORS. A negative shape is not a
    silent empty result, which is `gemm/IDENTICAL_FP32_CONTRACT.md` section
    8's last line.
    """
    if cfg.vocab <= 0:
        raise Error(
            String("embedding: V = ")
            + String(cfg.vocab)
            + " REFUSED; every id would be out of range (contract 8)"
        )
    if cfg.width < 0:
        raise Error(
            String("embedding: d = ") + String(cfg.width) + " REFUSED"
        )
    if n_positions < 0:
        raise Error(
            String("embedding: T = ") + String(n_positions) + " REFUSED"
        )
    if n_positions > EMB_MAX_POSITIONS:
        raise Error(
            String("embedding: T = ")
            + String(n_positions)
            + " exceeds EMB_MAX_POSITIONS = "
            + String(EMB_MAX_POSITIONS)
            + " (contract 3, the key packing's own bound)"
        )


def emb_refuse_ids(ids: List[Int32], cfg: EmbConfig) raises:
    """Contract section 8. An id below zero or at or past `V` is REFUSED BY
    NAME, with its position and its value in the message.

    **NOT CLAMPED, NOT WRAPPED, NOT SILENTLY DROPPED.** A clamp turns a data
    bug into a wrong gradient on a REAL vocabulary row, and there is no stage
    at which that becomes visible -- the clamped row's gradient is a
    perfectly ordinary float. Sabotage `EMB_GATHER_CLAMP_OOR`, whose inert
    set is every fixture with no out-of-range id, which is every ordinary
    fixture.

    **`padding_idx` GETS NO EXEMPTION HERE, and the version of this function
    that gave it one was wrong.** A real `padding_idx` is inside `[0, V)` by
    `EmbConfig.has_padding`'s own definition, so it passes the range check
    with nothing special done for it. Exempting "any id equal to
    `cfg.padding_idx`" would exempt `EMB_NO_PADDING_IDX`, which is `-1`, and
    an id of `-1` would then reach `emb_forward_oracle`'s gather as
    `w[-1 * d + j]` -- a read before the buffer, on the one path where
    `padding_idx` is NOT special (contract section 8, the forward gathers at
    every position including a padded one). The range check is the whole
    check and it is total.
    """
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v < 0 or v >= cfg.vocab:
            raise Error(
                String("embedding: id ")
                + String(v)
                + " at position "
                + String(t)
                + " is outside [0, "
                + String(cfg.vocab)
                + ") REFUSED (contract 8; never clamped)"
            )


# ===========================================================================
# THE RUN STRUCTURE (contract section 6, DEVIATIONS 1302 and 1322)
# ===========================================================================
# **THE SORT IS AN EXECUTION PLAN AND NOT THE SPECIFICATION.** Contract
# section 5.1 pins an ORDER over the contributing positions; it does not say
# how an implementation finds them. Any procedure that enumerates, for each
# `v`, exactly the positions with `ids[t] == v` in ASCENDING `t` produces the
# contract's bits, because it performs the same additions on the same values
# in the same order.
#
# The consequence that matters more than the analogy with gemm's named plans:
# **a sort that is wrong is a WRONG ANSWER, detectable against this oracle,
# rather than a silent redefinition of what "identical" means.** If the sort
# were the specification, an implementation-chosen tie order would be part of
# the contract and no oracle could catch it.
#
# TWO SPELLINGS LIVE HERE ON PURPOSE, the `gemm_oracle` / `gemm_oracle_serial`
# pattern:
#
#   `emb_perm_by_scan`               NORMATIVE.   PLAN_SCAN's order.
#   `emb_perm_by_total_order_key`    DIAGNOSTIC.  A stable merge sort on the
#                                    packed `(id, t)` key of contract 6.2(a).
#
# They must agree at every input, and a check that shows they do is the proof
# that contract 6.2(b)'s claim -- a stable sort by id over position-ordered
# input IS the total order -- is true rather than believed.


def emb_counts(ids: List[Int32], cfg: EmbConfig) -> List[Int32]:
    """`counts[v]`, the length of run `v`. Pass R1 of contract 6.1.

    A position whose id equals `padding_idx` is DROPPED AT THE SOURCE and
    enters no run (contract section 8, DEVIATION 1311). The alternative
    spelling -- fold everything, then overwrite row `padding_idx` -- reaches
    the same `dW` bits, provably, because a position carrying `padding_idx`
    can only ever contribute to row `padding_idx`. The contract pins the drop
    because the two produce different `emb.perm` stages and the card records
    `emb.perm`.

    INTEGERS ONLY. Nothing here rounds and nothing here flushes.
    """
    var counts = List[Int32](capacity=cfg.vocab)
    for _ in range(cfg.vocab):
        counts.append(Int32(0))
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v == cfg.padding_idx:
            continue
        if v < 0 or v >= cfg.vocab:
            continue  # `emb_refuse_ids` has already raised; belt only.
        counts[v] = counts[v] + Int32(1)
    return counts^


def emb_run_begin(counts: List[Int32]) -> List[Int32]:
    """The exclusive prefix sum of `counts`, length `V + 1`. Pass R2.

    **The fold shape of this scan is FREE and that is a THEOREM, not an
    indulgence.** Integer addition is exactly associative, so a serial scan,
    a Blelloch scan and a segmented scan give the same offsets bit for bit.
    That is the same argument
    `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.1 makes for its row
    maximum -- an execution plan may choose its own tree when the operation
    is exactly associative, and only then.

    Contract OWED item 10: the DEVICE spelling of this is a single-threaded
    serial scan over `V`, which at `V = 128256` is 128,256 dependent integer
    additions in one thread. That is a SCHEDULING embarrassment and not a
    numerical one, and `gbdt/gpu_util/kernel/scan.mojo` is the replacement.
    Swapping it cannot move a bit.
    """
    var begin = List[Int32](capacity=len(counts) + 1)
    var acc = Int32(0)
    for v in range(len(counts)):
        begin.append(acc)
        acc = acc + counts[v]
    begin.append(acc)
    return begin^


def emb_perm_by_scan(ids: List[Int32], cfg: EmbConfig) -> List[Int32]:
    """**THE NORMATIVE PERMUTATION.** Pass R3 of contract 6.1, `PLAN_SCAN`.

    For each `v` in ascending order, the positions with `ids[t] == v` in
    ASCENDING `t`, written into `[run_begin[v], run_begin[v+1])`.

    **NO SORT, NO KEY, NO TIE CLASS, NO STABILITY QUESTION.** Each `v` owns
    its own region and appends in `t` order, so the within-run ranks come out
    ascending BY CONSTRUCTION rather than by arrival. Contract 6.3 case 3 is
    the trap this avoids -- computing the rank with an atomic integer add
    makes the COUNT order free (integer addition is associative) and the SLOT
    each position receives arrival ordered, which is exactly the defect the
    float atomic had, moved into the index domain where it looks safe.

    THE COST, and contract 6.1 states it as a bound rather than hiding it.
    This is one pass per `v` over all `T` positions, `V * T` integer
    comparisons, and `emb_counts` is a second one. At the shipped Llama-3-8B
    shape (`V = 128256`, `T = 4096`) that is about 525 million comparisons
    per pass over a 16 KB `ids` array that stays in cache. **The run
    structure is computed ONCE over `T` and reused for all `d` columns**,
    which is what keeps this off the `V * T * d` cliff the one-hot GEMM of
    contract 5.2(d) falls off. It grows with `T`, so at `T = 100000` it is
    `1.3e10` and `PLAN_SORT` is the right plan instead.
    """
    var counts = emb_counts(ids, cfg)
    var begin = emb_run_begin(counts)
    var total = Int(begin[len(begin) - 1])
    var perm = List[Int32](capacity=total)
    for _ in range(total):
        perm.append(Int32(0))
    for v in range(cfg.vocab):
        # `padding_idx` DROPS AT THE SOURCE, DEVIATION 1311, and this skip is
        # LOAD-BEARING rather than an optimization. `emb_counts` already gave
        # row `padding_idx` a count of zero, so `begin[pad] == begin[pad+1]`;
        # without this line the loop below would still find every position
        # carrying `padding_idx` and write it at `begin[pad]`, which is the
        # first slot of the NEXT row's segment. `emb_perm_kernel` returns
        # early for the same reason and the two must agree.
        if v == cfg.padding_idx:
            continue
        var w = Int(begin[v])
        for t in range(len(ids)):
            if Int(ids[t]) == v:
                perm[w] = Int32(t)
                w += 1
    return perm^


def pack_emb_key(token_id: Int32, position: Int) -> UInt64:
    """Contract 6.2(a), DEVIATION 1303. The TOTAL order key `(id, t)`.

        key = (UInt64(id) & 0xFFFFFFFF) << 32 | (UInt64(t) & 0xFFFFFFFF)

    **No two positions share a key, because `t` is unique. Therefore any
    CORRECT sort returns the same permutation**, whatever its tie policy,
    whatever its block shape, whatever its vendor. That converts "the sort
    must be stable", a property of an implementation, into "the sort must be
    correct", a property anybody can check.

    It is DEVIATION 621's argument at a second site.
    `hierarchy/derived/sparse/op/sort.mojo` replaced an unstable
    `thrust::sort_by_key` on weight alone with a total order on
    `(weight_order_key, min(u,v), max(u,v))`, and `hierarchy/README.md` gives
    the consequence in one line -- "two distinct MST edges never tie under
    the triple, so the sorted list is a pure function of the edge set".

    TWO MOJO TRAPS LIVE IN THIS EXPRESSION AND BOTH HAVE COST THIS
    REPOSITORY A RUN.

      - `[[mojo-int-widening-sign-extends]]`. An `Int32` widened to `UInt64`
        SIGN-EXTENDS, so a negative id becomes `0xFFFFFFFF........` and sorts
        above every real token. `emb_refuse_ids` refuses negatives, but the
        refusal runs on the host and the packing may not, so the mask is
        spelled here and not assumed away.
      - `[[mojo-amp-plus-is-bitwise-and]]`. `x &+ k` computes `x & k` with no
        compile error, and it produced wrong hashes in this tree twice, the
        second time on 2026-08-24. Nothing in this function is written `&+`.
    """
    var hi = UInt64(Int(token_id)) & EMB_KEY_MASK
    var lo = UInt64(position) & EMB_KEY_MASK
    return (hi << 32) | lo


def merge_sort_u64_with_index(mut keys: List[UInt64], mut idx: List[Int32]):
    """Bottom-up merge sort of `keys` carrying `idx` along. Deterministic and
    STABLE, so a caller who packs a NON-total key still gets a defined
    (discovery) order among ties.

    **A COPY of `hierarchy/derived/sparse/op/sort.mojo::
    merge_sort_u64_with_index`**, contract OWED item 4. It is copied rather
    than imported because that module imports `max.gpu.host.DeviceBuffer` at
    module scope, and a host-only oracle that drags the GPU host module in is
    a host-only oracle that will not build without a device -- which is a
    real virtue in a reference and is the reason `gemm_oracle.mojo` has no
    device kernel either. The right home is a shared host utility and neither
    lane owns one.

    The comparison is `keys[j] < keys[i]` on the RIGHT operand first, which
    is what makes it stable -- an equal pair takes the LEFT element.
    """
    var n = len(keys)
    if n < 2:
        return
    var tk = List[UInt64](capacity=n)
    var ti = List[Int32](capacity=n)
    for _ in range(n):
        tk.append(UInt64(0))
        ti.append(Int32(0))
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[j] < keys[i]:
                    tk[k] = keys[j]
                    ti[k] = idx[j]
                    j += 1
                else:
                    tk[k] = keys[i]
                    ti[k] = idx[i]
                    i += 1
                k += 1
            while i < mid:
                tk[k] = keys[i]
                ti[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                tk[k] = keys[j]
                ti[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            keys[t] = tk[t]
            idx[t] = ti[t]
        width *= 2


def emb_perm_by_total_order_key(
    ids: List[Int32], cfg: EmbConfig
) -> List[Int32]:
    """**DIAGNOSTIC, NOT NORMATIVE.** The same permutation as
    `emb_perm_by_scan`, reached by sorting `pack_emb_key(id, t)`.

    This exists so that contract 6.2's determinism argument is CODE a check
    can run rather than prose a reader has to accept. Three separate claims
    become one assertion, `emb_perm_by_total_order_key == emb_perm_by_scan`
    at every fixture --

      1. the packed key is a TOTAL order, so the sorted list is a pure
         function of `ids` (contract 6.2(a));
      2. sorting on that key gives ascending `t` inside every run, which is
         what contract 5.1 clause 1 pins;
      3. therefore contract 6.2(b) holds -- a STABLE sort by `id` alone over
         a POSITION-ORDERED input is the same permutation, because stability
         supplies the low half of the key. That is
         `gbdt/gpu_util/kernel/radix_sort.mojo`'s own argument about
         CatBoost's `(bin || permutationPosition)`, quoted in contract
         6.2(b), and this function is how it stops being a quotation.

    Positions carrying `padding_idx` are dropped before the sort, exactly as
    `emb_perm_by_scan` drops them, so the two lists have the same length.

    **This is NOT `PLAN_SORT`.** `PLAN_SORT` is a DEVICE plan over
    `gbdt/gpu_util/kernel/radix_sort.mojo::launch_radix_sort_bins`, it is
    specified in contract 6.2 and it is NOT WRITTEN (contract OWED item 2).
    This is its host shadow and it proves the key, not the kernel.
    """
    var keys = List[UInt64]()
    var idx = List[Int32]()
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v == cfg.padding_idx:
            continue
        if v < 0 or v >= cfg.vocab:
            continue
        keys.append(pack_emb_key(ids[t], t))
        idx.append(Int32(t))
    merge_sort_u64_with_index(keys, idx)
    return idx^


# ===========================================================================
# THE ARITHMETIC (contract sections 4, 5 and 9)
# ===========================================================================


def emb_backward_cell(
    dy: List[Float32],
    perm: List[Int32],
    run_lo: Int,
    run_hi: Int,
    j: Int,
    width: Int,
    seed: Float32,
) -> Float32:
    """**THE CONTRACT'S FOLD.** One output cell, contract 5.1.

        acc = seed                                    (+0.0, or 7.4's carry)
        for r in [run_lo, run_hi)          ASCENDING sorted position, which
                                           IS ascending absolute t
            acc = ftz( ftz(acc) + ftz(dy[perm[r] * width + j]) )
        return ftz(acc)

    Six clauses live in those four lines and each is separately falsifiable.

    1. **ASCENDING absolute position.** `perm` is ascending in `t` inside
       every run by construction (`emb_perm_by_scan`), so walking `r` upward
       walks `t` upward. Not descending, not the order a sort happened to
       emit, not the order a block scheduler happened to arrive in. Sabotage
       `EMB_FOLD_DESCENDING`, **inert on every run of length <= 1 and on
       every run whose contributors are bitwise equal** -- a permutation of a
       constant sequence is the same sequence.
    2. **SERIAL.** No sub-partition, no leaf, no tree. `R` contributors
       perform exactly `R` dependent additions. Contract 5.3 is the argument
       and `emb_fold_balanced_tree_diagnostic` below is the refused
       alternative, spelled so the choice stays falsifiable.
    3. **SEEDED.** Contract 9.2. The `+0.0` seed is what makes the empty run
       `+0.0` with no special case, what makes an exactly-`+0.0` contributor
       inert, and what gives contract 7.4's carry something to start from.
       It also LAUNDERS a `-0.0`, which is a deliberate departure from
       `gemm/IDENTICAL_FP32_CONTRACT.md` 9.2(b)'s SEEDLESS tree and an
       agreement with the reference, whose buffer is `at::zeros`.
    4. **ONE THREAD OWNS ONE CELL.** No float crosses a thread boundary in
       the device spelling, which is the construction every identical kernel
       in this repository uses and which contract 0.2 says survives here.
    5. **`R == 0` returns the seed**, which is `+0.0` on a fresh call. A
       STATED value, contract 5.5, and it must be WRITTEN by the caller
       rather than skipped.
    6. **`R == 1` performs ONE addition and is NOT a bypass.** This is
       deliberately not gemm 7.3's `P == 1` case. Gemm's tree is SEEDLESS, so
       there the rule and the optimization coincide; this chain is SEEDED, so
       they DIVERGE, at exactly one input --

           sole contributor = 0x80000000 (-0.0)
               pinned    ftz( (+0.0) + (-0.0) ) = +0.0  ->  0x00000000
               bypassed  the contributor unchanged       ->  0x80000000

       Sabotage `EMB_SINGLE_RUN_BYPASS`, inert at every other input.

    THE FLUSHES ARE ROW 10's CHECKLIST UNIT. E1 flushes each `dy` as loaded,
    E2 flushes the accumulator as read, E3 flushes it after every add and E4
    flushes the result as stored. **E3 is the expensive one and it is not
    optional** -- an accumulator that dips into the subnormal range mid-run is
    an INTERMEDIATE, and flushing only at the end makes Metal (which flushed
    on the spot) and CUDA (which carried it) diverge from that step onward.
    That is `gemm/IDENTICAL_FP32_CONTRACT.md` 5c verbatim. E2 and E4 are
    bitwise redundant given E3, exactly as gemm's 5d and 5e are, and they are
    spelled anyway because the checklist's unit is the seam.

    **`ftz` is bitwise a no-op on an FTZ backend, so NONE of E1-E4 moves a
    bit on Apple**, and gemm 4.1's correction is the standing warning about
    what follows -- Apple's bits not moving is NOT evidence that a pin is
    unreached, and reach on Apple has to be shown by an oracle that REPORTS
    which arm the backend took.
    """
    var acc = seed
    for r in range(run_lo, run_hi):
        var t = Int(perm[r])
        acc = ftz(ftz(acc) + ftz(dy[t * width + j]))
    return ftz(acc)


def emb_fold_balanced_tree_diagnostic(
    contributions: List[Float32],
) -> Float32:
    """**DIAGNOSTIC AND REFUSED.** `gemm/IDENTICAL_FP32_CONTRACT.md` 7.2's
    fixed balanced tree over one run's contributions, adjacent pairing, odd
    tail carried, no seed and no padding.

    It is here for the reason `gemm_oracle_check.mojo` keeps
    `FOLD_SERIAL_ZERO_SEED` as an ADVERSARY -- **so that the choice contract
    5.3 made stays falsifiable** -- and for a second reason that is this
    lane's own: it is the instrument that computed contract 5.4's table, and
    that table is what makes `EMB_FOLD_BALANCED_TREE`'s inert set PROVABLE
    rather than observed.

    **THE TREE AND THE CHAIN FIRST DIFFER AT `R = 4`.** Contract 5.4:

        R = 0   chain +0.0            no tree             equal by 5.5
        R = 1   chain (+0)+a0         tree a0             NOT equal at -0.0
        R = 2   (a0+a1)               (a0+a1)             equal
        R = 3   ((a0+a1)+a2)          [a0+a1, carry a2] -> (a0+a1)+a2  equal
        R = 4   (((a0+a1)+a2)+a3)     (a0+a1)+(a2+a3)     NOT equal

    So a gate whose longest run is 3 reports `EMB_FOLD_BALANCED_TREE` as
    inert and somebody deletes it as a broken arm. Fixture F-TREE4 of
    contract 11.2 is the separator, by bits --

        {0x3F800000, 0x33800000, 0x33800000, 0x33800000}
            chain -> 0x3F800000   (1.0 + 2^-24 is the exact midpoint of
                                   1.0 and 1+2^-23; round-half-to-EVEN
                                   picks 1.0, three times over)
            tree  -> 0x3F800001   (level 1 pairs the three 2^-24 terms as
                                   [1.0, 2^-23]; 1.0 + 2^-23 is exact)

    WHY IT IS REFUSED, in one line, with contract 5.3 for the rest -- **a run
    length is DATA and a gemm `k` is a SHAPE**, so `P = f(R)` would make the
    arithmetic TOPOLOGY a function of the input, and an exactly-`+0.0`
    contributor from a padded or ignored position would stop being inert.
    """
    var n = len(contributions)
    if n == 0:
        return Float32(0.0)
    var current = contributions.copy()
    while len(current) > 1:
        var w = len(current)
        var pairs = w // 2
        var nxt = List[Float32]()
        for q in range(pairs):
            nxt.append(ftz(ftz(current[2 * q]) + ftz(current[2 * q + 1])))
        if w % 2 != 0:
            # THE CARRY. Bit for bit, no arithmetic, no padding.
            nxt.append(current[w - 1])
        current = nxt^
    return ftz(current[0])


def emb_fold_via_fma_diagnostic(
    contributions: List[Float32], seed: Float32
) -> Float32:
    """**DIAGNOSTIC.** The contract's own chain, spelled with
    `identical_mul_add(dy, 1.0, acc)` instead of a plain add.

    **It must agree with `emb_backward_cell` at EVERY input, and that is a
    LEMMA rather than a choice** (contract 4.1, DEVIATION 1317).
    `dy * 1.0` is exact for every Float32 including both zeros, so
    `fma(dy, 1.0, acc)` is ONE rounding of `dy + acc` -- the same one the
    plain add performs. That is
    `training/IDENTICAL_LOSS_CONTRACT.md` 5.3's ones-vector argument, which
    is how that lane routed three folds through gemm v1 with "no new
    arithmetic and no new fold to certify".

    **The contract pins the PLAIN ADD anyway**, because
    `gemm/IDENTICAL_FP32_CONTRACT.md` 4.2 already says a fold node is a plain
    add with nothing to fuse, and because a reader should not have to verify
    an exactness lemma to know what the arithmetic is. Nothing turns on the
    choice; this function is what turns "nothing turns on it" into an
    assertion.

    It is also the ONLY use of `identical_mul_add` in this file, and that is
    the point of DEVIATION 1317 -- there is no product anywhere on the
    normative path, so the whole multiply-add policy of gemm section 4 is
    satisfied VACUOUSLY rather than followed, and the trap that has bitten
    this repository repeatedly (contraction ACROSS expressions) has nothing
    here to contract.
    """
    var acc = seed
    for i in range(len(contributions)):
        acc = ftz(
            identical_mul_add(ftz(contributions[i]), Float32(1.0), ftz(acc))
        )
    return ftz(acc)


# ===========================================================================
# THE FORWARD (contract section 4, seams G1 and G2)
# ===========================================================================


def emb_forward_oracle(
    w: List[Float32], ids: List[Int32], cfg: EmbConfig
) raises -> List[Float32]:
    """**THE NORMATIVE FORWARD.** `Y[t, j] = ftz(ftz(W[ids[t], j]))`,
    row-major `[T, d]`.

    A GATHER. There is no arithmetic in it -- not a multiply, not an add, not
    a select -- and contract 4.1 is why that matters.

    **THE FLUSH AT G1 AND G2 MOVES BITS AND IS A KNOWING DEPARTURE**
    (DEVIATION 1310). A copy performs no rounding, so a raw copy of a
    subnormal weight survives on EVERY vendor, Apple included, and the
    vendors would still AGREE. `ftz` here therefore does not buy cross-vendor
    agreement -- that is already there -- it makes the embedding output obey
    the same denormal policy as every other stage on the card, so a subnormal
    cannot enter a transformer block through the one door that does no
    arithmetic. The reference flushes nothing in a gather.

    **That makes `EMB_GATHER_NO_FLUSH` this lane's ONLY Apple-visible flush
    arm.** Every other `ftz` seam here is bit-inert on an FTZ backend
    (contract 9.3), so on one column they cannot be seen to move anything;
    this one can, on a planted subnormal weight, fixture F-SUBW.

    Applying `ftz` at BOTH G1 and G2 is bitwise the same as applying it at
    either one. Both are spelled because "the seam a kernel writes for
    another kernel to read" is the unit row 10's checklist is written in.

    **`padding_idx` is NOT special in the forward.** `nn.Embedding`'s forward
    is a plain gather at every position, `padding_idx` included, and this
    profile follows it. Row `padding_idx` of `W` is ordinarily initialized to
    zero, which is the caller's business and not this contract's.

    `d == 0` and `T == 0` both produce an empty result and write nothing,
    contract section 8.
    """
    emb_refuse_shape(cfg, len(ids))
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("W"), w)
    # `y` and not `out`: `out` is a Mojo argument convention keyword
    # (`def __init__(out self)`) and is not a name to give a local.
    var y = List[Float32](capacity=len(ids) * cfg.width)
    for t in range(len(ids)):
        var v = Int(ids[t])
        var base = v * cfg.width
        for j in range(cfg.width):
            # G1 as loaded, G2 as stored. One of the two is redundant.
            y.append(ftz(ftz(w[base + j])))
    return y^


# ===========================================================================
# THE BACKWARD (contract sections 5, 7 and 9)
# ===========================================================================


def emb_backward_seed(
    cfg: EmbConfig, dw_prev: List[Float32]
) raises -> List[Float32]:
    """The `emb.dw_seed` stage, contract section 10.

    `accumulate == False`  ->  `+0.0` at every one of the `V * d` cells.
    `accumulate == True`   ->  `dw_prev`, flushed through seam E0.

    **THE FILL IS NOT AN IMPLEMENTATION DETAIL AND IT MAY NOT BE SKIPPED.**
    Contract 5.5 -- the empty run's value is STATED, not derived, and an
    implementation must write it rather than leave whatever was in the
    buffer. **At the shipped shape this is most of the output**: at
    `V = 128256` and `T = 4096` at least 124,160 of the 128,256 rows are
    empty. Sabotages `EMB_EMPTY_ROW_SKIPPED` (inert on any gate that does not
    POISON the buffer first, because a fresh allocation may already be zero)
    and `EMB_EMPTY_ROW_NEG_ZERO`.

    **`+0.0` and not `-0.0`.** `(+0) + (-0) = +0` in round-to-nearest on every
    backend, because it is IEEE-754 and not a codegen choice, so a `+0.0`
    seed is what makes a nonempty run of all-zero contributors return `+0.0`
    (contract 9.2(a)) and what makes contract 7.4's carry start from a value
    that cannot poison the chain.

    THE SEED IS WHY THIS STAGE IS ON THE CARD. `emb.dw_seed` gives
    `EMB_EMPTY_ROW_SKIPPED` and the `accumulate` path a stage of their own,
    which is transformer 5.1's "the card is the only instrument that can see
    this clause at all" at a third site after the loss lane's `ce.row`.
    """
    var cells = cfg.vocab * cfg.width
    if not cfg.accumulate:
        var fresh = List[Float32](capacity=cells)
        for _ in range(cells):
            fresh.append(Float32(0.0))
        return fresh^
    if len(dw_prev) != cells:
        raise Error(
            String("embedding: accumulate=True needs dW_prev of ")
            + String(cells)
            + " cells, got "
            + String(len(dw_prev))
            + " (contract 7.4)"
        )
    var carried = List[Float32](capacity=cells)
    for i in range(cells):
        # Seam E0. Bitwise a no-op on a value E4 already flushed; spelled
        # because the checklist's unit is the seam, and because a caller may
        # hand in a `dW` this profile did not write.
        carried.append(ftz(dw_prev[i]))
    return carried^


def emb_backward_oracle(
    dy: List[Float32],
    ids: List[Int32],
    cfg: EmbConfig,
    dw_prev: List[Float32],
) raises -> List[Float32]:
    """**THE NORMATIVE BACKWARD ANSWER of
    `mojolearn.identical.embedding.fp32.v1`.**

        dW[v, :] = sum over every t with ids[t] == v, ASCENDING t,
                   of dY[t, :], seeded +0.0, every seam flushed

    Row-major `[V, d]`. This is the value the device kernel must reproduce
    bit for bit, on Apple, NVIDIA and AMD, at every launch geometry and under
    every execution plan.

    WHAT IS AND IS NOT A FUNCTION OF, and the second half is the honest one.

    > `dW` is a pure function of the bits of `dY`, the bits of `ids`, `V`,
    > `d`, `padding_idx` and the seed -- and never of the block count, the
    > grid shape, the occupancy, the lane width, the vendor, the execution
    > plan, the order two runs were processed in, or the order two threads
    > arrived in.

    **It is NOT a function only of the multiset of contributing values**, and
    it is NOT invariant to which other sequences shared the launch. Contract
    7.2 is the full statement and it is this profile's one clause that offers
    strictly less than the transformer's and the loss lane's. If sequence `S`
    shares a launch with `S'` and `S'` also contains token `v`, then
    `dW[v, :]` sums both, and it SHOULD -- that is what the gradient of a
    shared weight means. A contract that claimed otherwise would be claiming
    that adding data does not change a gradient.

    WHAT IS INERT, contract 7.1, **with its one hole**. A contributor whose
    `dY` row is exactly `+0.0` -- which is what a padded position and an
    `ignore_index` position both produce -- does not move `dW[v, j]`, PROVIDED
    the accumulator is not `-0.0` at that step. That is what makes a padded
    batch give the same gradient as the unpadded one, and it is FALSE under a
    balanced tree, which is contract 5.3(ii) and the reason the fold is a
    chain.

    **The hole is real and reachable and this lane states it because two
    other contracts in this tree assert the theorem without it.** The `+0.0`
    seed forbids reaching `-0.0` by ADDITION; it does not forbid reaching it
    through `ftz` of a negative subnormal partial sum, seam E3. Fixture
    F-SUBACC of contract 11.2 is the counterexample, by bits --

        a0 = 0x80C00000 (-1.5 * 2^-126, normal, survives E1)
        a1 = 0x00800000 (+1.0 * 2^-126, normal, survives E1)
            acc = ftz(-0.5 * 2^-126) = ftz(-2^-127) = -0.0    0x80000000
        a2 = 0x00000000 (+0.0)
            acc = ftz((-0.0) + (+0.0)) = +0.0                 0x00000000

    -- an exactly-zero contributor has moved a bit. Contract OWED item 5
    carries the same correction to `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`
    7.1 and `training/IDENTICAL_LOSS_CONTRACT.md` 7.3, both of which write
    "and the seed forbids that". This lane does not edit those files.

    THE MICROBATCH CARRY, contract 7.4 and DEVIATION 1309. With
    `cfg.accumulate` true this seeds each cell's chain from `dw_prev` and
    continues it, which **reproduces the unsplit call BIT FOR BIT at EVERY
    split point** -- the first microbatch computes a prefix of the chain, the
    second continues it, and the resulting sequence of additions is the
    unsplit one term for term. `dW = ftz(dW_first + dW_second)` does NOT, in
    general, and that is contract 5.4's `R = 4` row again. Set beside
    `gemm/IDENTICAL_BACKWARD_PLAN.md` 3.2, where `dB` splits reproduce only
    at an ALIGNED split -- **a chain has no boundaries to align to**. Do NOT
    cite that section's measurement as evidence that an accumulator must be a
    tree; its splits used TWO pieces, and over two pieces a serial running
    sum and a balanced tree are the SAME operation. That correction was made
    in that file on 2026-08-25.

    ROW INDEPENDENCE. Nothing in the fold reads `T` other than as the bound
    of the enumeration, so a caller may chunk the columns `j` any way it
    likes, and may chunk the vocabulary rows `v` any way it likes, and the
    bits do not move. Only the RUN STRUCTURE sees all `T` positions at once,
    and it is integer.
    """
    emb_refuse_shape(cfg, len(ids))
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("dY"), dy)

    var dw = emb_backward_seed(cfg, dw_prev)

    # `dW[padding_idx, :] = +0.0`, STORED. Contract section 8, DEVIATION
    # 1311. BELT, NOT BRACES -- the positions are dropped at the source so
    # run `padding_idx` is empty and the fill already put `+0.0` there. It is
    # spelled because the `accumulate` path did NOT fill, and
    # `nn.Embedding(padding_idx=p)` means that row is zero on every call.
    #
    # It is written HERE and the device writes it AFTER the fold, and the two
    # are equivalent because the fold never touches row `padding_idx` -- its
    # run is empty, so `emb_backward_oracle`'s loop `continue`s on it and
    # `emb_backward_kernel` returns on it. Placing it before the `T == 0`
    # early return is what makes the degenerate shape agree too.
    if cfg.has_padding() and cfg.width > 0:
        var pad_base = cfg.padding_idx * cfg.width
        for j in range(cfg.width):
            dw[pad_base + j] = Float32(0.0)

    if cfg.width == 0 or len(ids) == 0:
        # Contract section 8. `T == 0` leaves every row at its seed, which is
        # `+0.0` on a fresh call -- STATED, and already WRITTEN above.
        return dw^

    var counts = emb_counts(ids, cfg)
    var begin = emb_run_begin(counts)
    var perm = emb_perm_by_scan(ids, cfg)

    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        if lo == hi:
            # Contract 5.5: the empty run keeps the seed, which the fill has
            # already STORED. Not skipped -- already written.
            continue
        var base = v * cfg.width
        for j in range(cfg.width):
            # The seed is read into a local FIRST rather than passed as
            # `dw[base + j]` while `dw` is also the assignment target. Read
            # then write, two statements, so there is no question about what
            # the argument saw.
            var seed = dw[base + j]
            dw[base + j] = emb_backward_cell(
                dy, perm, lo, hi, j, cfg.width, seed
            )
    return dw^


# ===========================================================================
# THE STAGES (contract section 10)
# ===========================================================================


struct EmbStages(Movable):
    """Every recorded stage of one embedding call, in the card's order.

    **THREE OF THE NINE ARE INTEGER STAGES AND THAT IS THE WHOLE OF CONTRACT
    SECTION 6.** `emb.perm` is where a divergent sort becomes visible BEFORE
    it reaches a float. A card that recorded only `emb.dw` would see a wrong
    tie order as a wrong gradient with no localization, and a card that
    recorded only floats could not carry `emb.perm` at all.

    `weight` and `fwd` are empty on a backward-only call and `dy`, `counts`,
    `run_begin`, `perm`, `dw_seed` and `dw` are empty on a forward-only one.
    A check that hashes an empty list records a zero-length stage, which the
    differ must treat as "absent" rather than as "agreeing with an absent
    one" -- `IdentityTrace.record_device`'s own length hazard, pointed the
    other way.

    `ids`, `weight` and `dy` are INPUTS and are recorded anyway, for the
    reason the transformer contract records `rope.inv_freq` -- an input built
    the wrong way is a silent divergence that no computed stage localizes.
    """

    var ids: List[Int32]          # [T]       as given
    var weight: List[Float32]     # [V, d]    forward only, as given
    var fwd: List[Float32]        # [T, d]    G1, G2
    var dy: List[Float32]         # [T, d]    backward only, as given
    var counts: List[Int32]       # [V]       R1
    var run_begin: List[Int32]    # [V + 1]   R2
    var perm: List[Int32]         # [T]       R3
    var dw_seed: List[Float32]    # [V, d]    the fill, or the carried dW
    var dw: List[Float32]         # [V, d]    E0 through E4

    def __init__(out self):
        self.ids = List[Int32]()
        self.weight = List[Float32]()
        self.fwd = List[Float32]()
        self.dy = List[Float32]()
        self.counts = List[Int32]()
        self.run_begin = List[Int32]()
        self.perm = List[Int32]()
        self.dw_seed = List[Float32]()
        self.dw = List[Float32]()


def emb_forward_stages(
    w: List[Float32], ids: List[Int32], cfg: EmbConfig
) raises -> EmbStages:
    """The forward's three card stages. Contract section 10."""
    var st = EmbStages()
    st.ids = ids.copy()
    st.weight = w.copy()
    st.fwd = emb_forward_oracle(w, ids, cfg)
    return st^


def emb_backward_stages(
    dy: List[Float32],
    ids: List[Int32],
    cfg: EmbConfig,
    dw_prev: List[Float32],
) raises -> EmbStages:
    """The backward's six card stages, in card order. Contract section 10.

    The run structure is recomputed here rather than threaded out of
    `emb_backward_oracle`, so that the two spellings of the answer -- the one
    a check compares against the device and the one it records -- cannot come
    to two opinions about the permutation. Cheap on the host and the point of
    an oracle.
    """
    var st = EmbStages()
    st.ids = ids.copy()
    st.dy = dy.copy()
    st.counts = emb_counts(ids, cfg)
    st.run_begin = emb_run_begin(st.counts)
    st.perm = emb_perm_by_scan(ids, cfg)
    st.dw_seed = emb_backward_seed(cfg, dw_prev)
    st.dw = emb_backward_oracle(dy, ids, cfg, dw_prev)
    return st^


def emb_max_run_length(counts: List[Int32]) -> Int:
    """`R_max`, the length of the longest run.

    **A FIXTURE INSTRUMENT, and contract 11.2 requires it be asserted rather
    than hoped for.** `EMB_FOLD_BALANCED_TREE` is PROVABLY inert at every
    `R <= 3` (contract 5.4), and `R_max` is a property of the DATA, not of
    the shape -- so a gate that plants duplicates without checking how long
    the longest run came out can report a working sabotage as a broken arm.
    A check must assert `emb_max_run_length >= 4` on the fixtures that are
    supposed to separate the fold, and assert the INERT MASK on the ones that
    are not.

    The two negative-control fixtures of contract 11.2 are exactly the two
    this function is for. **F-NODUP** (all ids distinct) has `R_max <= 1` and
    every order clause and every sort clause is inert on it. **F-DUPSAME**
    (duplicates present, all carrying bitwise equal `dY` rows) has
    `R_max >= 2` and the order clauses are STILL inert on it, because a
    permutation of a constant sequence is the same sequence. Neither fixture
    can see the accumulation order at all, and both are REQUIRED, with their
    inert masks asserted.
    """
    var best = 0
    for v in range(len(counts)):
        var c = Int(counts[v])
        if c > best:
            best = c
    return best
