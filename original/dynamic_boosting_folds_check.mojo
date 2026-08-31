# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate for `gbdt/methods/dynamic_boosting_folds.mojo`.

HOST ONLY. No GPU, no fixture, no tolerance. `CreateFolds`
(`catboost/cuda/methods/dynamic_boosting.h:189-223`) is integer arithmetic
over `sampleCount`, `min_fold_size`, `fold_len_multiplier` and the sample
grouping, so its answer is a CLOSED FORM and the gate is that closed form,
written here from the recurrence rather than copied from the port.

Run:

    pixi run mojo run -I . original/dynamic_boosting_folds_check.mojo

THE INDEPENDENT FORMS THIS FILE CHECKS AGAINST
----------------------------------------------
For the default grouping (`TWithoutQueriesGrouping`, one document per group,
which is what every pointwise loss gets) `NextQueryOffsetForLine(line)` is
`min(line + 1, n)`. So with `m0` the first estimate size, the eval right edges
obey

    R_0 = min(trunc(m0 * g), n) + 1, clamped to n
    R_k = min(trunc(R_{k-1} * g), n) + 1, clamped to n

and at `g == 2` that recurrence has a CLOSED FORM with no recurrence left in
it:

    R_k = min(2^(k+1) * (m0 + 1) - 1, n)
    fold count = the smallest j >= 1 with 2^j * (m0 + 1) - 1 >= n

which is what `closed_form_rights_g2` computes -- a formula in `k`, not a
replay of their loop. F3 checks the port against it fold for fold, and F4
pins four series as HAND-COMPUTED LITERALS so that a bug shared by the port
and the formula still has something to fail against.

WHY THE `+ 1` IS THE WHOLE GAME. Every boundary `CreateFolds` computes goes
through `NextQueryOffsetForLine`, so with no groups every fold edge is one
past the geometric value. `2 * (m0 + 1) - 1` rather than `2 * m0` is not a
rounding artefact; it is that call. Drop it and the entire series moves. S1
below does exactly that.

GATES -- each is a distinct MECHANISM, not a distinct assertion:

  F1  `MinEstimationSize`, all three arms, against a literal table. Includes
      the 499/500 CLIFF (1 -> 10) and the `folds >= maxFolds` arm, which at
      the default `min_fold_size = 100` needs n > 13,107,200 and is reached
      here both there and cheaply at `min_fold_size = 1`.
  F2  THE PLAIN ARM. One fold, both slices [0, n), for every n, every growth
      rate, every `min_fold_size` and both groupings -- and identical across
      all of them, because the arm returns before any of them is read. This
      is the arm every fit in this tree takes today.
  F3  ORDERED at g = 2.0, no groups, against the closed form above, over
      n = 4..600 one at a time (the 500 cliff sits in there) plus a strided
      sweep to 20,000 and four large values.
  F4  Five HAND-COMPUTED series as literals: (n=1000, g=2), (n=1000, g=1.5),
      (n=1000, g=3), and the 499-vs-500 pair that straddles the cliff.
  F5  The INVARIANTS, over every configuration F3 and F6 build: prefixes are
      nested, eval slices tile [m0, n) with no gap and no overlap, right
      edges strictly increase, the last one is exactly n.
  F6  THE GROUPED ARM. Boundaries must snap FORWARD to a group offset, and
      the check refuses to pass if the grouping turned out inert -- a
      grouping whose every boundary already sat on a group edge would verify
      nothing about snapping, which is how uniform fixtures pass broken
      permutation code in this repository.
  F7  THE MULTI-DEVICE ARM (`:194-198`), skipped at one device and therefore
      unchecked by everything else. Both sides of the `devCount > 1` switch,
      per `PORTING_RULES` rule 8.
  F8  The live refusals: too few groups (on both sides of the device switch),
      growth rate <= 1.0, an empty group, an all-size-1 grouping.
  F9  `FoldCount` / `FoldBits`, and the POWER-OF-TWO STRIPE they feed. Read
      through the EXISTING `PointwisePartOffsetsHelper` rather than a second
      copy of its arithmetic, and gated to actually DIVERGE from the
      histogram offset at a real fold count.
  F10 `TFoldAndPermutationStorage`'s shape: one row per learn permutation,
      one entry per fold of that permutation, plus one `Estimation` off the
      grid.
  F11 THE `learnPermutationCount - 1` MODULUS. At the default
      `permutation_count = 4` the structure search never sees permutation 2,
      and that is transcribed, not corrected.
  F12 `IntLog2` at the powers of two, where a float `ceil(log2(x))` is most
      likely to land a hair off, plus the full identity over [1, 1 << 20].

SABOTAGE, AND WHICH GATE EACH ONE MOVED
---------------------------------------
Ten sabotages, one per MECHANISM, each applied to the LIBRARY, run, and
reverted. All ten reddened the run; none was a no-op.

  S1   drop the `+ 1` from `TWithoutQueriesGrouping::NextQueryOffsetForLine`
       -> F3, F5. Also the measured case for DEVIATION 112: without the
          guard this HANGS at `g = 1.05` instead of failing. With it, the
          guard raises at 502 passes over 499 samples.
  S2   `MinEstimationSize`'s cliff 500 -> 100
       -> F1 (499 gives 9, not 1), F3, F5.
  S3   the `folds >= maxFolds` arm as `folds > maxFolds`
       -> F1 only (n = 13,107,201 gives 100, not 51). The RARE arm, and the
          only gate that reaches it.
  S4   drop the `docCount / 50` half of the final `Min`
       -> F1, F3.
  S5   the Plain arm learns on `min_estimation` instead of everything
       -> F2 only.
  S6   the growth loop steps from `min_estimation` instead of the previous
       right edge -> F3 by DEVIATION 112's guard (9 passes over 6 samples);
       another configuration that would hang without it.
  S7   the grouped `NextQueryOffsetForLine` ignores groups
       -> F6 only, 11 gates inside it. F3, F5 and everything ungrouped stay
          green, which is the point: reach is per branch.
  S8   the `devCount > 1` arm fires at `> 2`
       -> F7, F5.
  S9   `learnPermutationId`'s modulus loses the `- 1`
       -> F11 only ("structure search reached permutation 2").
  S10  `FoldCount` is `folds.size()` instead of `2 * folds.size()`
       -> F9 only.

WHAT IS NOT GATED, AND WHY
--------------------------
`CB_ENSURE(minEstimationSize, "Error: min learn size should be positive")`
(`:201`) is UNREACHABLE. It runs after the group-count ensure, so `n >= 4`;
`MinEstimationSize` returns at least 1 on every arm; and
`NextQueryOffsetForLine(>= 1)` is at least 2 under both groupings. The
`devCount > 1` arm only ever raises it further. It is transcribed because it
is theirs; there is no input that fires it.

The `static_cast<ui32>` wrap described in DEVIATION 111 is likewise not
gated: reaching it needs `sampleCount * growthRate >= 2^32` with
`growthRate > 1`, which their own `ui32 sampleCount` cannot supply.
"""

from gbdt.ctrs.ctr_bins_builder import int_log2
from gbdt.methods.dynamic_boosting_folds import (
    EBoostingType,
    IQueriesGrouping,
    TFold,
    TFoldAndPermutationStorage,
    create_folds,
    estimation_permutation,
    fold_bits_for_folds,
    fold_count_for_folds,
    learn_permutation_count,
    learn_permutation_id,
    min_estimation_size,
)
from gbdt.methods.kernel.split_properties_helpers import (
    PointwisePartOffsetsHelper,
)
from std.math import floor


# =========================================================================
# INDEPENDENT ORACLES.  Nothing below calls the file under test.
# =========================================================================


def ceil_log2_by_doubling(x: Int) -> Int:
    """`ceil(log2(x))` for `x >= 1`, computed by DOUBLING rather than by bit
    length or by a float log. Third independent spelling of `NCB::IntLog2`.
    """
    var bits = 0
    var v = 1
    while v < x:
        v *= 2
        bits += 1
    return bits


def oracle_min_estimation_size(n: Int, m: Int) -> Int:
    """`MinEstimationSize` from their source text, with every sub-expression
    spelled differently: ceiling division as `-((-n) // d)` rather than
    `(n + d - 1) // d`, `ceil(log2)` by doubling rather than by bit length,
    and the `Min` written out as a branch.
    """
    if n < 500:
        return 1
    var per_fold = -((-n) // m)
    if ceil_log2_by_doubling(per_fold) >= 18:
        return -((-n) // 262144)
    var fiftieth = n // 50
    if fiftieth < m:
        return fiftieth
    return m


def prefix_boundaries(group_sizes: List[Int]) -> List[Int]:
    """The document index each group STARTS at, followed by `n`. Everything
    the grouped oracle needs, and no group ids at all.
    """
    var bounds = List[Int]()
    var at = 0
    for i in range(len(group_sizes)):
        bounds.append(at)
        at += group_sizes[i]
    bounds.append(at)
    return bounds^


def oracle_next_offset(bounds: List[Int], line: Int, n: Int) -> Int:
    """The smallest group start STRICTLY GREATER than `line`, or `n`.

    Their grouped implementation looks the group id up in `QueryIds` and then
    indexes `QueryOffsets[gid + 1]`; this scans the boundary list for the
    first entry past `line`. Same answer, no shared indirection.
    """
    for i in range(len(bounds)):
        if bounds[i] > line:
            if bounds[i] > n:
                return n
            return bounds[i]
    return n


def oracle_ungrouped_next_offset(line: Int, n: Int) -> Int:
    """`TWithoutQueriesGrouping::NextQueryOffsetForLine`, one document per
    group."""
    if line + 1 < n:
        return line + 1
    return n


def oracle_rights_ungrouped(
    n: Int, m0: Int, g: Float64
) raises -> List[Int]:
    """The eval right edges for the default grouping, at any `g > 1`.

    `floor()` is used explicitly rather than leaning on `Int(Float64)`'s
    truncation, so the port and the oracle do not share the rounding.
    """
    var rights = List[Int]()
    var r = m0
    var guard = 0
    while True:
        guard += 1
        if guard > n + 4:
            raise Error("oracle did not terminate")
        var stepped = Int(floor(Float64(r) * g))
        if stepped > n:
            stepped = n
        var nxt = oracle_ungrouped_next_offset(stepped, n)
        rights.append(nxt)
        if nxt >= n:
            break
        r = nxt
    return rights^


def oracle_rights_grouped(
    n: Int, m0: Int, g: Float64, bounds: List[Int]
) raises -> List[Int]:
    """The same series with the boundaries snapped forward to group starts."""
    var rights = List[Int]()
    var r = m0
    var guard = 0
    while True:
        guard += 1
        if guard > n + 4:
            raise Error("oracle did not terminate")
        var stepped = Int(floor(Float64(r) * g))
        if stepped > n:
            stepped = n
        var nxt = oracle_next_offset(bounds, stepped, n)
        rights.append(nxt)
        if nxt >= n:
            break
        r = nxt
    return rights^


def closed_form_rights_g2(n: Int, m0: Int) -> List[Int]:
    """THE CLOSED FORM, `g == 2.0`, default grouping. No recurrence.

        R_k = min(2^(k+1) * (m0 + 1) - 1, n),  k = 0, 1, 2, ...

    Derivation: below the clamp, `R_{k+1} = 2 * R_k + 1`, whose fixed shape
    is `R_k + 1 = 2^(k+1) * (m0 + 1)`. The clamp is exact rather than
    approximate because `min(trunc(2 r), n) + 1` is `n` the moment `2 r >= n`
    and is `2 r + 1 <= n` otherwise.
    """
    var rights = List[Int]()
    var k = 1
    while True:
        var unclamped = (1 << k) * (m0 + 1) - 1
        var r = min(unclamped, n)
        rights.append(r)
        if r >= n:
            break
        k += 1
    return rights^


def closed_form_fold_count_g2(n: Int, m0: Int) -> Int:
    """The smallest `j >= 1` with `2^j * (m0 + 1) - 1 >= n`, i.e.
    `ceil(log2((n + 1) / (m0 + 1)))` floored at 1. Counted, not derived from
    the list above, so a bug in the list cannot hide in the count.
    """
    var j = 1
    while (1 << j) * (m0 + 1) - 1 < n:
        j += 1
    return j


# =========================================================================
# reporting
# =========================================================================


def rights_of(folds: List[TFold]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(folds)):
        out.append(folds[i].quality_evaluate_samples.right)
    return out^


def show(xs: List[Int]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ", "
        s += String(xs[i])
    return s + "]"


def same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def check_invariants(
    folds: List[TFold], n: Int, m0: Int, label: String
) raises -> Int:
    """F5, applied to every fold list any other gate builds."""
    var bad = 0
    if len(folds) == 0:
        print("FAIL F5[" + label + "]: no folds at all")
        return 1
    if folds[0].estimate_samples.left != 0:
        print("FAIL F5[" + label + "]: first estimate does not start at 0")
        bad += 1
    if folds[0].estimate_samples.right != m0:
        print(
            "FAIL F5[" + label + "]: first estimate right =",
            folds[0].estimate_samples.right,
            "expected m0 =",
            m0,
        )
        bad += 1
    for k in range(len(folds)):
        var est = folds[k].estimate_samples
        var ev = folds[k].quality_evaluate_samples
        if est.left != 0:
            print("FAIL F5[" + label + "]: fold", k, "estimate is not a prefix")
            bad += 1
        if ev.left != est.right:
            print(
                "FAIL F5[" + label + "]: fold",
                k,
                "eval starts at",
                ev.left,
                "but estimate ends at",
                est.right,
                "-- ordered boosting requires them adjacent and disjoint",
            )
            bad += 1
        if ev.right <= ev.left:
            print("FAIL F5[" + label + "]: fold", k, "eval slice is empty")
            bad += 1
        if k > 0:
            if est.right != folds[k - 1].quality_evaluate_samples.right:
                print(
                    "FAIL F5[" + label + "]: fold",
                    k,
                    "does not learn on everything fold",
                    k - 1,
                    "was scored on",
                )
                bad += 1
            if ev.right <= folds[k - 1].quality_evaluate_samples.right:
                print("FAIL F5[" + label + "]: right edges not increasing")
                bad += 1
        if ev.right > n:
            print("FAIL F5[" + label + "]: fold", k, "runs past n")
            bad += 1
    if folds[len(folds) - 1].quality_evaluate_samples.right != n:
        print(
            "FAIL F5[" + label + "]: last eval right =",
            folds[len(folds) - 1].quality_evaluate_samples.right,
            "expected",
            n,
        )
        bad += 1
    return bad


def main() raises:
    var failures = 0

    # -------------------------------------------------------------- F12
    # IntLog2 first, because F1 depends on it being exact.
    var f12 = 0
    for x in range(1, (1 << 20) + 1):
        if int_log2(x) != ceil_log2_by_doubling(x):
            print("FAIL F12: int_log2(", x, ") =", int_log2(x))
            f12 += 1
            if f12 > 4:
                break
    var pow2_probe: List[Int] = [
        1,
        2,
        1024,
        65536,
        131071,
        131072,
        131073,
        262144,
    ]
    var pow2_want: List[Int] = [0, 1, 10, 16, 17, 17, 18, 18]
    for i in range(len(pow2_probe)):
        if int_log2(pow2_probe[i]) != pow2_want[i]:
            print(
                "FAIL F12: int_log2(",
                pow2_probe[i],
                ") =",
                int_log2(pow2_probe[i]),
                "expected",
                pow2_want[i],
            )
            f12 += 1
    failures += f12
    if f12 == 0:
        print(
            "  ok   F12 -- IntLog2 exact over [1, 1<<20] and at the four"
            " powers of two MinEstimationSize's threshold lands on"
        )

    # --------------------------------------------------------------- F1
    # MinEstimationSize, LITERAL expected values, hand-derived from
    # dynamic_boosting.h:177-185.
    var f1 = 0
    var f1_n: List[Int] = [
        1,
        4,
        499,
        500,
        749,
        999,
        1000,
        4949,
        4999,
        5000,
        5001,
        1000000,
        13107200,
        13107201,
        13369344,
    ]
    var f1_m: List[Int] = [
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
    var f1_want: List[Int] = [
        1,
        1,
        1,
        10,
        14,
        19,
        20,
        98,
        99,
        100,
        100,
        100,
        100,
        51,
        51,
    ]
    for i in range(len(f1_n)):
        var got = min_estimation_size(f1_n[i], f1_m[i])
        if got != f1_want[i]:
            print(
                "FAIL F1: min_estimation_size(",
                f1_n[i],
                ",",
                f1_m[i],
                ") =",
                got,
                "expected",
                f1_want[i],
            )
            f1 += 1
    # The rare `folds >= maxFolds` arm, reached cheaply at min_fold_size = 1.
    # ceil(200000 / 1) needs 18 bits, so the arm fires and the answer is
    # ceil(200000 / 262144) == 1 rather than min(1, 4000) == 1 -- which are
    # the same number, so that alone would be INERT.  min_fold_size = 2 at
    # n = 300000 separates them: the arm gives ceil(300000/262144) == 2 and
    # the fallthrough would give min(2, 6000) == 2.  Still equal.  The arm is
    # only distinguishable where ceil(n / 2^18) EXCEEDS min_fold_size, which
    # needs n > min_fold_size * 262144 while ceil(n / min_fold_size) > 2^17,
    # i.e. min_fold_size < 2.  So: min_fold_size = 1, n = 600000 gives
    # ceil(600000 / 262144) == 3, and the fallthrough would give 1.
    var rare_got = min_estimation_size(600000, 1)
    if rare_got != 3:
        print("FAIL F1: maxFolds arm at (600000, 1) =", rare_got, "expected 3")
        f1 += 1
    if min_estimation_size(131072, 1) != 1:
        print("FAIL F1: (131072, 1) should still take the Min arm")
        f1 += 1
    # ceil(131073/1) needs 18 bits -> arm fires -> ceil(131073/262144) == 1
    if min_estimation_size(131073, 1) != 1:
        print("FAIL F1: (131073, 1) =", min_estimation_size(131073, 1))
        f1 += 1
    # A broad sweep against the independently spelled oracle.
    var f1_ms: List[Int] = [1, 2, 7, 100, 1000, 100000]
    for mi in range(len(f1_ms)):
        var m = f1_ms[mi]
        for n in range(1, 6000):
            if min_estimation_size(n, m) != oracle_min_estimation_size(n, m):
                print(
                    "FAIL F1 sweep: n =",
                    n,
                    "m =",
                    m,
                    "got",
                    min_estimation_size(n, m),
                    "want",
                    oracle_min_estimation_size(n, m),
                )
                f1 += 1
                break
        var big: List[Int] = [
            131071,
            131072,
            131073,
            600000,
            13107199,
            13107200,
            13107201,
            33554432,
        ]
        for bi in range(len(big)):
            var n2 = big[bi]
            if min_estimation_size(n2, m) != oracle_min_estimation_size(n2, m):
                print(
                    "FAIL F1 big: n =",
                    n2,
                    "m =",
                    m,
                    "got",
                    min_estimation_size(n2, m),
                    "want",
                    oracle_min_estimation_size(n2, m),
                )
                f1 += 1
    failures += f1
    if f1 == 0:
        print(
            "  ok   F1 -- MinEstimationSize: the 499->500 cliff (1 -> 10),"
            " the docCount/50 taper, the 13,107,200 maxFolds edge"
            " (100 -> 51), and the arm at min_fold_size = 1"
        )

    # --------------------------------------------------------------- F2
    # THE PLAIN ARM.  One fold, both slices [0, n), independent of growth
    # rate, min_fold_size and grouping -- because the arm returns before any
    # of them is read.  This is the arm every fit in this tree takes today.
    var f2 = 0
    var plain_n: List[Int] = [4, 5, 99, 100, 499, 500, 501, 5000, 1000000]
    var plain_g: List[Float64] = [1.0000001, 1.5, 2.0, 7.0]
    var plain_m: List[Int] = [1, 100, 10000]
    for ni in range(len(plain_n)):
        var n = plain_n[ni]
        for gi in range(len(plain_g)):
            for mi in range(len(plain_m)):
                for dev in range(1, 3):
                    if n < 4 * dev:
                        continue
                    var grouping = IQueriesGrouping.without_queries(n)
                    var pf = create_folds(
                        n,
                        plain_g[gi],
                        grouping,
                        EBoostingType.Plain,
                        plain_m[mi],
                        dev,
                    )
                    if len(pf) != 1:
                        print(
                            "FAIL F2: Plain at n =",
                            n,
                            "gave",
                            len(pf),
                            "folds",
                        )
                        f2 += 1
                        continue
                    if (
                        pf[0].estimate_samples.left != 0
                        or pf[0].estimate_samples.right != n
                        or pf[0].quality_evaluate_samples.left != 0
                        or pf[0].quality_evaluate_samples.right != n
                    ):
                        print(
                            "FAIL F2: Plain at n =",
                            n,
                            "slices",
                            pf[0].estimate_samples.left,
                            pf[0].estimate_samples.right,
                            pf[0].quality_evaluate_samples.left,
                            pf[0].quality_evaluate_samples.right,
                        )
                        f2 += 1
    # Plain with a real grouping is the same answer: the arm never reads it.
    var plain_sizes = List[Int]()
    for _i in range(200):
        plain_sizes.append(5)
    var plain_u32 = List[UInt32]()
    for i in range(len(plain_sizes)):
        plain_u32.append(UInt32(plain_sizes[i]))
    var plain_grouped = IQueriesGrouping.queries(plain_u32^)
    var pg = create_folds(
        1000, 2.0, plain_grouped, EBoostingType.Plain, 100, 1
    )
    if len(pg) != 1 or pg[0].quality_evaluate_samples.right != 1000:
        print("FAIL F2: Plain with a grouping did not collapse to one fold")
        f2 += 1
    # Plain still refuses what the ensures refuse -- the arm sits AFTER them.
    var plain_refused = False
    try:
        var g3 = IQueriesGrouping.without_queries(3)
        _ = create_folds(3, 2.0, g3, EBoostingType.Plain, 100, 1)
    except:
        plain_refused = True
    if not plain_refused:
        print("FAIL F2: Plain at n = 3 should still hit the group-count ensure")
        f2 += 1
    var plain_refused_g = False
    try:
        var g1k = IQueriesGrouping.without_queries(1000)
        _ = create_folds(1000, 1.0, g1k, EBoostingType.Plain, 100, 1)
    except:
        plain_refused_g = True
    if not plain_refused_g:
        print("FAIL F2: Plain at growth_rate = 1.0 should still be refused")
        f2 += 1
    failures += f2
    if f2 == 0:
        print(
            "  ok   F2 -- Plain collapses to ONE fold covering [0, n) at 9"
            " sizes x 4 growth rates x 3 min_fold_sizes x 2 device counts,"
            " and still honours both ensures"
        )

    # --------------------------------------------------------------- F3
    # ORDERED, g = 2.0, default grouping, against the CLOSED FORM.
    var f3 = 0
    var f5 = 0
    var f3_cases = 0
    var f3_ns = List[Int]()
    for n in range(4, 601):
        f3_ns.append(n)
    var n2 = 601
    while n2 < 20000:
        f3_ns.append(n2)
        n2 += 7
    var f3_big: List[Int] = [20000, 100000, 999983, 1000000]
    for i in range(len(f3_big)):
        f3_ns.append(f3_big[i])
    for i in range(len(f3_ns)):
        var n = f3_ns[i]
        var grouping = IQueriesGrouping.without_queries(n)
        var folds = create_folds(
            n, 2.0, grouping, EBoostingType.Ordered, 100, 1
        )
        var m0 = min(oracle_min_estimation_size(n, 100) + 1, n)
        var want = closed_form_rights_g2(n, m0)
        var got = rights_of(folds)
        f3_cases += 1
        if not same(got, want):
            if f3 < 5:
                print(
                    "FAIL F3: n =",
                    n,
                    "got",
                    show(got),
                    "closed form",
                    show(want),
                )
            f3 += 1
        if len(folds) != closed_form_fold_count_g2(n, m0):
            if f3 < 5:
                print(
                    "FAIL F3: n =",
                    n,
                    "fold count",
                    len(folds),
                    "closed form",
                    closed_form_fold_count_g2(n, m0),
                )
            f3 += 1
        f5 += check_invariants(folds, n, m0, String("g2 n=") + String(n))
    # other growth rates against the general oracle, including one barely
    # above 1.0 -- the configuration DEVIATION 112's guard is sized for.
    var f3_gs: List[Float64] = [1.0000001, 1.05, 1.3333333, 1.5, 2.0, 3.0, 9.5]
    var f3_gn: List[Int] = [499, 500, 501, 1000, 5000]
    var max_folds_seen = 0
    for gi in range(len(f3_gs)):
        for ni in range(len(f3_gn)):
            var n = f3_gn[ni]
            var grouping = IQueriesGrouping.without_queries(n)
            var folds = create_folds(
                n, f3_gs[gi], grouping, EBoostingType.Ordered, 100, 1
            )
            var m0 = min(oracle_min_estimation_size(n, 100) + 1, n)
            var want = oracle_rights_ungrouped(n, m0, f3_gs[gi])
            var got = rights_of(folds)
            f3_cases += 1
            max_folds_seen = max(max_folds_seen, len(folds))
            if not same(got, want):
                if f3 < 5:
                    print(
                        "FAIL F3: n =",
                        n,
                        "g =",
                        f3_gs[gi],
                        "got",
                        show(got),
                        "oracle",
                        show(want),
                    )
                f3 += 1
            f5 += check_invariants(
                folds, n, m0, String("g n=") + String(n)
            )
    failures += f3
    if f3 == 0:
        print(
            "  ok   F3 --",
            f3_cases,
            "fold series match the closed form / the general oracle fold for"
            " fold; deepest series",
            max_folds_seen,
            "folds",
        )

    # --------------------------------------------------------------- F4
    # HAND-COMPUTED literals.  Derived on paper from dynamic_boosting.h:
    #   n = 1000: MinEstimationSize = min(100, 20) = 20, m0 = 21
    #     g=2.0  21 -> 42+1=43 -> 86+1=87 -> 175 -> 351 -> 703 -> clamp 1000
    #     g=1.5  21 -> 31+1=32 -> 48+1=49 -> 74 -> 112 -> 169 -> 254 -> 382
    #                -> 574 -> 862 -> clamp 1000
    #     g=3.0  21 -> 63+1=64 -> 192+1=193 -> 580 -> clamp 1000
    #   n = 499:  docCount < 500 so MinEstimationSize = 1, m0 = 2
    #     g=2.0  2 -> 5 -> 11 -> 23 -> 47 -> 95 -> 191 -> 383 -> clamp 499
    #   n = 500:  MinEstimationSize = min(100, 10) = 10, m0 = 11
    #     g=2.0  11 -> 23 -> 47 -> 95 -> 191 -> 383 -> clamp 500
    var f4 = 0
    var lit_1000_g2: List[Int] = [43, 87, 175, 351, 703, 1000]
    var lit_1000_g15: List[Int] = [
        32,
        49,
        74,
        112,
        169,
        254,
        382,
        574,
        862,
        1000,
    ]
    var lit_1000_g3: List[Int] = [64, 193, 580, 1000]
    var lit_499_g2: List[Int] = [5, 11, 23, 47, 95, 191, 383, 499]
    var lit_500_g2: List[Int] = [23, 47, 95, 191, 383, 500]
    var g1000 = IQueriesGrouping.without_queries(1000)
    var g499 = IQueriesGrouping.without_queries(499)
    var g500 = IQueriesGrouping.without_queries(500)
    var lit_got_a = rights_of(
        create_folds(1000, 2.0, g1000, EBoostingType.Ordered, 100, 1)
    )
    var lit_got_b = rights_of(
        create_folds(1000, 1.5, g1000, EBoostingType.Ordered, 100, 1)
    )
    var lit_got_c = rights_of(
        create_folds(1000, 3.0, g1000, EBoostingType.Ordered, 100, 1)
    )
    var lit_got_d = rights_of(
        create_folds(499, 2.0, g499, EBoostingType.Ordered, 100, 1)
    )
    var lit_got_e = rights_of(
        create_folds(500, 2.0, g500, EBoostingType.Ordered, 100, 1)
    )
    if not same(lit_got_a, lit_1000_g2):
        print("FAIL F4: n=1000 g=2.0", show(lit_got_a))
        f4 += 1
    if not same(lit_got_b, lit_1000_g15):
        print("FAIL F4: n=1000 g=1.5", show(lit_got_b))
        f4 += 1
    if not same(lit_got_c, lit_1000_g3):
        print("FAIL F4: n=1000 g=3.0", show(lit_got_c))
        f4 += 1
    if not same(lit_got_d, lit_499_g2):
        print("FAIL F4: n=499 g=2.0", show(lit_got_d))
        f4 += 1
    if not same(lit_got_e, lit_500_g2):
        print("FAIL F4: n=500 g=2.0", show(lit_got_e))
        f4 += 1
    # the CLIFF itself, stated as a gate rather than left implicit:
    # one more document changes the FIRST estimate size by 9 and removes two
    # whole folds.
    var cliff_lo = create_folds(499, 2.0, g499, EBoostingType.Ordered, 100, 1)
    var cliff_hi = create_folds(500, 2.0, g500, EBoostingType.Ordered, 100, 1)
    if cliff_lo[0].estimate_samples.right != 2:
        print("FAIL F4: n=499 first estimate is not 2")
        f4 += 1
    if cliff_hi[0].estimate_samples.right != 11:
        print("FAIL F4: n=500 first estimate is not 11")
        f4 += 1
    if len(cliff_lo) != 8 or len(cliff_hi) != 6:
        print(
            "FAIL F4: cliff fold counts",
            len(cliff_lo),
            len(cliff_hi),
            "expected 8 and 6",
        )
        f4 += 1
    failures += f4
    if f4 == 0:
        print(
            "  ok   F4 -- five hand-computed series match, and the 499/500"
            " cliff moves the first estimate 2 -> 11 and the fold count"
            " 8 -> 6"
        )

    # --------------------------------------------------------------- F6
    # THE GROUPED ARM.  Boundaries snap FORWARD to a group offset.
    #
    # The grouping is deliberately NOT uniform: 500 groups of 1 followed by
    # 100 groups of 5.  Uniform group sizes would make every boundary land on
    # a group edge by construction and the gate would verify nothing about
    # snapping -- the failure mode this repository has hit five times.  The
    # anti-inert assertion below refuses to pass unless at least one boundary
    # actually MOVED relative to the ungrouped answer.
    var f6 = 0
    var sizes_a = List[Int]()
    for _i in range(500):
        sizes_a.append(1)
    for _i in range(100):
        sizes_a.append(5)
    # a second shape: one enormous group late, which should SWALLOW a fold
    var sizes_b = List[Int]()
    for _i in range(600):
        sizes_b.append(1)
    sizes_b.append(400)
    var shapes = List[List[Int]]()
    shapes.append(sizes_a.copy())
    shapes.append(sizes_b.copy())
    var moved_total = 0
    for si in range(len(shapes)):
        var sizes = shapes[si].copy()
        var bounds = prefix_boundaries(sizes)
        var n = bounds[len(bounds) - 1]
        var u32 = List[UInt32]()
        for i in range(len(sizes)):
            u32.append(UInt32(sizes[i]))
        var grouping = IQueriesGrouping.queries(u32^)
        var gs: List[Float64] = [1.5, 2.0, 3.0]
        for gi in range(len(gs)):
            var folds = create_folds(
                n, gs[gi], grouping, EBoostingType.Ordered, 100, 1
            )
            var m0 = oracle_next_offset(
                bounds, oracle_min_estimation_size(n, 100), n
            )
            var want = oracle_rights_grouped(n, m0, gs[gi], bounds)
            var got = rights_of(folds)
            if not same(got, want):
                print(
                    "FAIL F6: shape",
                    si,
                    "g =",
                    gs[gi],
                    "got",
                    show(got),
                    "oracle",
                    show(want),
                )
                f6 += 1
            # every boundary is a group start or n
            for k in range(len(got)):
                var on_edge = got[k] == n
                for bi in range(len(bounds)):
                    if bounds[bi] == got[k]:
                        on_edge = True
                if not on_edge:
                    print(
                        "FAIL F6: shape",
                        si,
                        "boundary",
                        got[k],
                        "splits a group",
                    )
                    f6 += 1
            # anti-inert: it must DIFFER from the ungrouped answer somewhere
            var flat = IQueriesGrouping.without_queries(n)
            var flat_folds = create_folds(
                n, gs[gi], flat, EBoostingType.Ordered, 100, 1
            )
            if not same(rights_of(flat_folds), got):
                moved_total += 1
            f5 += check_invariants(
                folds, n, m0, String("grouped ") + String(si)
            )
    if moved_total == 0:
        print(
            "FAIL F6: the grouping was INERT -- every boundary already sat on"
            " a group edge, so nothing here verified that",
            "NextQueryOffsetForLine snaps forward at all",
        )
        f6 += 1
    # and the size-1 prefix must NOT move: snapping is per boundary, not a
    # global shift.
    var u32a = List[UInt32]()
    for i in range(len(sizes_a)):
        u32a.append(UInt32(sizes_a[i]))
    var grouping_a = IQueriesGrouping.queries(u32a^)
    var ga = rights_of(
        create_folds(1000, 2.0, grouping_a, EBoostingType.Ordered, 100, 1)
    )
    var expect_a: List[Int] = [43, 87, 175, 351, 705, 1000]
    if not same(ga, expect_a):
        print("FAIL F6: 500x1 then 100x5 at g=2 gave", show(ga))
        f6 += 1
    failures += f6
    if f6 == 0:
        print(
            "  ok   F6 -- grouped boundaries snap forward to group starts,"
            " never split a group,",
            moved_total,
            "of 6 configurations actually moved, and the size-1 prefix of the"
            " series is untouched (703 -> 705 at one boundary only)",
        )

    # --------------------------------------------------------------- F7
    # THE MULTI-DEVICE ARM, both sides of the switch.
    var f7 = 0
    var g1000b = IQueriesGrouping.without_queries(1000)
    var dev1 = rights_of(
        create_folds(1000, 2.0, g1000b, EBoostingType.Ordered, 100, 1)
    )
    var dev2 = rights_of(
        create_folds(1000, 2.0, g1000b, EBoostingType.Ordered, 100, 2)
    )
    # devCount = 2: minEstimationSize = max(21, GetQueryOffset(min(32, 500)))
    #             = max(21, 32) = 32; then 32 -> 64+1=65 -> 131 -> 263 -> 527
    #             -> clamp 1000
    var dev2_want: List[Int] = [65, 131, 263, 527, 1000]
    if not same(dev1, lit_1000_g2):
        print("FAIL F7: devCount = 1 must skip the arm entirely")
        f7 += 1
    if not same(dev2, dev2_want):
        print("FAIL F7: devCount = 2 gave", show(dev2))
        f7 += 1
    var dev2_folds = create_folds(
        1000, 2.0, g1000b, EBoostingType.Ordered, 100, 2
    )
    if dev2_folds[0].estimate_samples.right != 32:
        print(
            "FAIL F7: devCount = 2 first estimate =",
            dev2_folds[0].estimate_samples.right,
            "expected 32",
        )
        f7 += 1
    # devCount = 4 raises it further: min(64, 500) -> 64
    var dev4_folds = create_folds(
        1000, 2.0, g1000b, EBoostingType.Ordered, 100, 4
    )
    if dev4_folds[0].estimate_samples.right != 64:
        print(
            "FAIL F7: devCount = 4 first estimate =",
            dev4_folds[0].estimate_samples.right,
            "expected 64",
        )
        f7 += 1
    # the queryCount/2 half of the Min bites on a small pool: at n = 40 and
    # devCount = 2, min(32, 20) = 20 -> offset 20, above the ungrouped 2.
    var g40 = IQueriesGrouping.without_queries(40)
    var dev2_small = create_folds(
        40, 2.0, g40, EBoostingType.Ordered, 100, 2
    )
    if dev2_small[0].estimate_samples.right != 20:
        print(
            "FAIL F7: n = 40 devCount = 2 first estimate =",
            dev2_small[0].estimate_samples.right,
            "expected 20 (the queryCount/2 half of the Min)",
        )
        f7 += 1
    f5 += check_invariants(dev2_folds, 1000, 32, String("dev2"))
    f5 += check_invariants(dev4_folds, 1000, 64, String("dev4"))
    failures += f7
    if f7 == 0:
        print(
            "  ok   F7 -- devCount 1 skips the arm; 2 and 4 raise the first"
            " estimate to 32 and 64; the queryCount/2 half of the Min bites"
            " at n = 40"
        )

    failures += f5
    if f5 == 0:
        print(
            "  ok   F5 -- nesting, adjacency, tiling of [m0, n), strict"
            " growth and a last edge of exactly n hold on every series above"
        )

    # --------------------------------------------------------------- F8
    var f8 = 0
    var r1 = False
    try:
        var g3 = IQueriesGrouping.without_queries(3)
        _ = create_folds(3, 2.0, g3, EBoostingType.Ordered, 100, 1)
    except:
        r1 = True
    if not r1:
        print("FAIL F8: n = 3 at devCount 1 should raise (3 < 4)")
        f8 += 1
    var r2 = True
    try:
        var g4 = IQueriesGrouping.without_queries(4)
        _ = create_folds(4, 2.0, g4, EBoostingType.Ordered, 100, 1)
    except:
        r2 = False
    if not r2:
        print("FAIL F8: n = 4 at devCount 1 should NOT raise")
        f8 += 1
    var r3 = False
    try:
        var g7 = IQueriesGrouping.without_queries(7)
        _ = create_folds(7, 2.0, g7, EBoostingType.Ordered, 100, 2)
    except:
        r3 = True
    if not r3:
        print("FAIL F8: n = 7 at devCount 2 should raise (7 < 8)")
        f8 += 1
    var r4 = False
    try:
        var gk = IQueriesGrouping.without_queries(1000)
        _ = create_folds(1000, 1.0, gk, EBoostingType.Ordered, 100, 1)
    except:
        r4 = True
    if not r4:
        print("FAIL F8: growth_rate = 1.0 should raise")
        f8 += 1
    var r5 = False
    try:
        var gk2 = IQueriesGrouping.without_queries(1000)
        _ = create_folds(1000, 0.5, gk2, EBoostingType.Ordered, 100, 1)
    except:
        r5 = True
    if not r5:
        print("FAIL F8: growth_rate = 0.5 should raise")
        f8 += 1
    var r6 = False
    try:
        var ones = List[UInt32]()
        for _i in range(10):
            ones.append(UInt32(1))
        _ = IQueriesGrouping.queries(ones^)
    except:
        r6 = True
    if not r6:
        print("FAIL F8: an all-size-1 grouping should raise")
        f8 += 1
    var r7 = False
    try:
        var withzero: List[UInt32] = [UInt32(3), UInt32(0), UInt32(2)]
        _ = IQueriesGrouping.queries(withzero^)
    except:
        r7 = True
    if not r7:
        print("FAIL F8: an empty group should raise")
        f8 += 1
    failures += f8
    if f8 == 0:
        print(
            "  ok   F8 -- refuses n < 4*devCount on both sides of the device"
            " switch, refuses growth_rate <= 1.0, refuses empty and"
            " all-size-1 groupings; n = 4 is accepted"
        )

    # --------------------------------------------------------------- F9
    # FoldCount / FoldBits and the power-of-two stripe.
    var f9 = 0
    var diverged = 0
    var checked_counts = 0
    for i in range(len(f3_gn)):
        var n = f3_gn[i]
        var grouping = IQueriesGrouping.without_queries(n)
        for gi in range(len(f3_gs)):
            var folds = create_folds(
                n, f3_gs[gi], grouping, EBoostingType.Ordered, 100, 1
            )
            var fc = fold_count_for_folds(len(folds), EBoostingType.Ordered)
            var fb = fold_bits_for_folds(len(folds), EBoostingType.Ordered)
            checked_counts += 1
            if fc != 2 * len(folds):
                print("FAIL F9: FoldCount", fc, "for", len(folds), "folds")
                f9 += 1
            if fb != ceil_log2_by_doubling(fc):
                print("FAIL F9: FoldBits", fb, "for FoldCount", fc)
                f9 += 1
            if (1 << fb) < fc:
                print(
                    "FAIL F9: stripe 1 <<",
                    fb,
                    "cannot address",
                    fc,
                    "folds -- the fold id would collide with a depth bit",
                )
                f9 += 1
            # read the EXISTING helper rather than a second copy of it
            var helper = PointwisePartOffsetsHelper(UInt32(fc))
            var hist = helper.histogram_offset(UInt32(1), UInt32(0))
            var part = helper.data_partition_offset(UInt32(1), UInt32(0))
            if Int(hist) != fc:
                print("FAIL F9: histogram offset", hist, "!=", fc)
                f9 += 1
            if Int(part) != (1 << fb):
                print("FAIL F9: data partition offset", part, "!= 1 <<", fb)
                f9 += 1
            if Int(part) != Int(hist):
                diverged += 1
    if diverged == 0:
        print(
            "FAIL F9: the two offsets never diverged, so nothing here checked"
            " that the fold axis is STRIPED rather than packed -- every fold"
            " count tried happened to be a power of two"
        )
        f9 += 1
    # Plain: one part, zero fold bits, matching the hard-coded 0 in
    # pointwise_optimization_subsets.cpp:13.
    if fold_count_for_folds(1, EBoostingType.Plain) != 1:
        print("FAIL F9: Plain FoldCount is not 1")
        f9 += 1
    if fold_bits_for_folds(1, EBoostingType.Plain) != 0:
        print("FAIL F9: Plain FoldBits is not 0")
        f9 += 1
    if fold_bits_for_folds(37, EBoostingType.Plain) != 0:
        print("FAIL F9: Plain FoldBits must ignore the fold list size")
        f9 += 1
    # a worked case: n = 1000, g = 2 -> 6 folds -> FoldCount 12, FoldBits 4,
    # stripe 16.  12 is not a power of two, so the two offsets differ by 4
    # for every part index.
    var w = create_folds(1000, 2.0, g1000b, EBoostingType.Ordered, 100, 1)
    if fold_count_for_folds(len(w), EBoostingType.Ordered) != 12:
        print("FAIL F9: n=1000 g=2 FoldCount is not 12")
        f9 += 1
    if fold_bits_for_folds(len(w), EBoostingType.Ordered) != 4:
        print("FAIL F9: n=1000 g=2 FoldBits is not 4")
        f9 += 1
    failures += f9
    if f9 == 0:
        print(
            "  ok   F9 -- FoldCount = 2 * folds and FoldBits = IntLog2 over",
            checked_counts,
            "configurations; the striped and packed offsets diverge in",
            diverged,
            "of them; Plain gives 1 and 0",
        )

    # -------------------------------------------------------------- F10
    var f10 = 0
    var perm_count = 4
    var lpc = learn_permutation_count(perm_count)
    var grid = List[List[Int]]()
    var per_perm_folds = List[Int]()
    for _p in range(lpc):
        var gp = IQueriesGrouping.without_queries(1000)
        var fp = create_folds(1000, 2.0, gp, EBoostingType.Ordered, 100, 1)
        per_perm_folds.append(len(fp))
        var row = List[Int]()
        for k in range(len(fp)):
            row.append(fp[k].quality_evaluate_samples.right)
        grid.append(row^)
    var storage = TFoldAndPermutationStorage[Int](grid^, -1)
    if storage.permutation_count() != lpc:
        print("FAIL F10: storage has", storage.permutation_count(), "rows")
        f10 += 1
    for p in range(lpc):
        if storage.fold_count_for_permutation(p) != per_perm_folds[p]:
            print("FAIL F10: row", p, "has the wrong fold count")
            f10 += 1
    if storage.get(0, 0) != 43 or storage.get(2, 5) != 1000:
        print("FAIL F10: get() returned", storage.get(0, 0), storage.get(2, 5))
        f10 += 1
    if storage.estimation != -1:
        print("FAIL F10: Estimation is not off the grid")
        f10 += 1
    # NOT GATED, and said out loud: their `Get` is `FoldData.at(p).at(f)`,
    # two bounds-CHECKED accesses that throw. Mojo's `List.__getitem__`
    # ABORTS the process instead of raising, so an out-of-range `get` cannot
    # be caught and cannot be asserted on from inside this check. The bound
    # is still enforced; it just terminates rather than unwinds.
    failures += f10
    if f10 == 0:
        print(
            "  ok   F10 -- storage is",
            lpc,
            "permutations x 6 folds plus one Estimation off the grid",
        )

    # -------------------------------------------------------------- F11
    # THE `learnPermutationCount - 1` MODULUS.
    var f11 = 0
    if estimation_permutation(4) != 3:
        print("FAIL F11: estimationPermutation(4) is not 3")
        f11 += 1
    if learn_permutation_count(4) != 3:
        print("FAIL F11: learnPermutationCount(4) is not 3")
        f11 += 1
    if learn_permutation_count(1) != 1 or estimation_permutation(1) != 0:
        print("FAIL F11: the permutationCount = 1 fallback is wrong")
        f11 += 1
    if learn_permutation_count(2) != 1:
        print("FAIL F11: learnPermutationCount(2) is not 1")
        f11 += 1
    var seen_0 = False
    var seen_1 = False
    for r in range(0, 1000):
        var id = learn_permutation_id(r, 3)
        if id == 0:
            seen_0 = True
        elif id == 1:
            seen_1 = True
        else:
            print("FAIL F11: structure search reached permutation", id)
            f11 += 1
            break
    if not (seen_0 and seen_1):
        print("FAIL F11: the modulus did not reach both 0 and 1")
        f11 += 1
    for r in range(0, 50):
        if learn_permutation_id(r, 1) != 0:
            print("FAIL F11: the learnPermutationCount = 1 arm is not 0")
            f11 += 1
            break
    failures += f11
    if f11 == 0:
        print(
            "  ok   F11 -- at permutation_count = 4 the fold grid has 3 rows"
            " and structure search reaches only rows 0 and 1; row 2 has folds"
            " and cursors and is never searched. Transcribed, not corrected."
        )

    if failures != 0:
        raise Error(String(failures) + " gate(s) failed")
    print(
        "dynamic boosting folds: F1-F12 pass -- CreateFolds, MinEstimationSize,"
        " both groupings, both device arms, both boosting types"
    )
