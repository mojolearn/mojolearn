# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The pointwise family's odds and ends: pick the better split, name it, and
four bit-interleave helpers.

PORT OF `catboost/cuda/methods/helpers.{h,cpp}` at CatBoost `54a8143a`.
Transliterated. Do not improve.

`helpers.h` is 104 lines and `helpers.cpp` is 173, and between them they hold
five unrelated jobs. Sorted by what they need to run:

| theirs | line | ported here | what it needs |
|---|---|---|---|
| `TakeBest(a, b)` | `helpers.h:11` | `take_best` | nothing |
| `TakeBest(a, b, c)` | `helpers.h:19` | `take_best3` | nothing |
| `ReverseBits` | `helpers.h:53` | `reverse_bits` | nothing |
| `GetOddBits` | `helpers.h:64` | `get_odd_bits` | nothing |
| `GetEvenBits` | `helpers.h:77` | `get_even_bits` | nothing |
| `MergeBits` | `helpers.h:90` | `merge_bits` | nothing |
| `ToSplit` | `helpers.cpp:157` | `to_split` | the manager's `IsCat` and one cap |
| `SplitConditionToString` | `helpers.cpp:70` | `split_condition_to_string` | borders + nan mode |
| `SplitConditionToString` (+`ESplitValue`) | `helpers.cpp:106` | `split_condition_to_string_value` | borders + nan mode |
| `PrintBestScore` | `helpers.cpp:142` | `print_best_score` | the above |
| `HasPermutationDependentSplit` | `helpers.cpp:60` | `has_permutation_dependent_split` | `IsCtr`/`IsPermutationDependent` |
| `GetBinsForModel` | `helpers.cpp:3` | **NOT PORTED** | `TScopedCacheHolder`, `TTreeUpdater`, `TFeatureParallelDataSet` |
| `CacheBinsForModel` | `helpers.cpp:45` | **NOT PORTED** | the same three |

WHY THE LAST TWO ARE NOT HERE, spelled out so nobody reads their absence as
an oversight. `GetBinsForModel` builds a document->leaf bin array for a
finished `TObliviousTreeStructure` by replaying every split through
`TTreeUpdater` (`gpu_data/oblivious_tree_bin_builder.h`) and MEMOISING the
result in a `TScopedCacheHolder` keyed on whether any split is
permutation-dependent. All three of those types are unported, and its only
callers are the FEATURE-PARALLEL learner and the feature-parallel leaves
estimator (`feature_parallel_pointwise_oblivious_tree.h:43`,
`add_oblivious_tree_model_feature_parallel.cpp:12` and `:31`,
`leaves_estimation/oblivious_tree_leaves_estimator.h:143`) -- none of which
this repository has. Porting them now would be writing a cache for a caller
that does not exist, which is the defect `PORTING_RULES.md` rule 3 names.
They belong with rung 2 of `PORTING.md` 91 E.

WHAT "TAKES THE MANAGER'S ANSWERS AS ARGUMENTS" MEANS, and why it is not a
redesign. `TBinarizedFeaturesManager` is unported. Every function above that
names it reads exactly one or two facts out of it -- `IsCat(featureId)`,
`GetBinCount(featureId)`, `GetBorders(featureId)`, `GetNanMode(featureId)`,
`IsCtr(featureId)`, `IsPermutationDependent(ctr)` -- and then does arithmetic
on the answer. Those reads are unwrapped into parameters here, so the
arithmetic below is theirs line for line and the lookup is the caller's. When
a features manager lands, it supplies the arguments and nothing in this file
changes.

======================== DEVIATION 99 ========================
THREE DEPARTURES, ALL IN THIS FILE, NONE ARITHMETIC.

1. **`GetBinsForModel` and `CacheBinsForModel` are absent**, for the reason
   above: their three dependencies are unported and their four call sites are
   all in the feature-parallel learner, which is unported too. Recorded here
   rather than silently skipped.

2. **The manager is a parameter list.** `ToSplit`, both
   `SplitConditionToString` overloads, `PrintBestScore` and
   `HasPermutationDependentSplit` take `const TBinarizedFeaturesManager&`
   upstream. There is no such type here, so each takes the values it would
   have read. One consequence is real and is NOT hidden: their `ToSplit`
   opens with

       if (manager.IsFeatureBundle(props.FeatureId)) {
           return manager.TranslateFeatureBundleSplitToBinarySplit(...);
       }

   (`helpers.cpp:159-161`). Feature bundles are unported, so `to_split` has
   NO bundle arm. A caller that ever has bundles must add it; until then the
   branch is unreachable rather than wrong, and `is_feature_bundle` is
   accepted as an argument purely so a caller cannot forget to raise.

3. **Float-to-text is Mojo's, not C++ `ostream`'s.** `messageBuilder <<
   borders[binIdx]` formats a `float` with `std::ostream`'s default six
   significant digits; `String(Float32)` here prints Mojo's shortest
   round-tripping form. `0.1f` prints as `0.1` both ways and `1.0f/3` prints
   as `0.333333` there and `0.33333334` here. This affects LOG TEXT ONLY --
   no split, score, bin or model value is derived from these strings, and
   `MEMORY.md`'s "String(float) does not round-trip" rule is why nothing
   numeric ever will be. `checks/pointwise_subsets_check.mojo` gates the
   STRUCTURE of every message (which comparator, which border index, which
   nan arm) and deliberately does not gate the digits.
==============================================================
"""

from gbdt.methods.greedy_subsets_searcher.points_subsets import (
    TBestSplitProperties,
)
from gbdt.models.oblivious_model import (
    BIN_SPLIT_TAKE_BIN,
    BIN_SPLIT_TAKE_GREATER,
    TBinarySplit,
    TObliviousTreeStructure,
)
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    NAN_MODE_MAX,
    NAN_MODE_MIN,
)


# --- ESplitValue (`cuda/data/feature.h:26-29`) ---------------------------
#
# Their enumerator ORDER, so `GetSplitValue` (`feature.h:31-33`,
# `value == ESplitValue::Zero ? 0 : 1`) is the identity on these numbers.

comptime SPLIT_VALUE_ZERO = 0
comptime SPLIT_VALUE_ONE = 1


def best_split_properties_less(
    first: TBestSplitProperties, second: TBestSplitProperties
) -> Bool:
    """`TBestSplitProperties::operator<` (`gpu_data/gpu_structures.h:80-93`).

    Transcribed here and not added to `TBestSplitProperties` itself because
    `gbdt/methods/greedy_subsets_searcher/` belongs to another lane this
    round. It is `TakeBest`'s only dependency and `TakeBest` is this file's,
    so it lives with its caller until the struct can grow a `__lt__`.

    THREE THINGS ABOUT IT ARE EASY TO GET WRONG AND ALL THREE ARE GATED.

    **It orders by `Gain`, not by `Score`.** Their `Score` field is carried
    for logging (`gbdt/methods/greedy_subsets_searcher/kernel/compute_scores
    .mojo` already records this about the device-side argmax) and never
    decides anything. A comparator that reads `Score` picks a different
    split whenever the two disagree.

    **`FeatureId` breaks ties as a `ui32`, and the undefined sentinel is
    `(ui32)-1`.** Ours is an `Int32` holding `-1`, which as a SIGNED number is
    the smallest value there is and as their UNSIGNED one is the largest. So
    the cast below is load-bearing: without it an undefined candidate wins
    every gain tie instead of losing every one of them, and `TakeBest` over a
    level where nothing scored returns the sentinel rather than the first
    real split. Their `Gain` also defaults to `+inf`, so the tie is reached
    only when both sides are undefined -- but `PORTING_RULES.md` 0c is the
    rule that says port the branch, not the reachability argument.

    **`BinId` is the last resort and it is `<`, not `<=`.** Equal records are
    therefore NOT less than each other, and since `TakeBest` is written
    `first < second ? first : second`, a full tie falls through to the
    SECOND argument. This docstring said "first" until
    `checks/pointwise_subsets_check.mojo` was run against it and the
    named tie case failed; the code was right and the sentence was wrong.
    """
    if first.gain < second.gain:
        return True
    elif first.gain == second.gain:
        # `ui32 FeatureId`, theirs. See the note above.
        var f1 = UInt32(first.feature_id)
        var f2 = UInt32(second.feature_id)
        if f1 < f2:
            return True
        elif f1 == f2:
            return UInt32(first.bin_id) < UInt32(second.bin_id)
        else:
            return False
    else:
        return False


def take_best(
    first: TBestSplitProperties, second: TBestSplitProperties
) -> TBestSplitProperties:
    """`TakeBest` (`helpers.h:11-17`).

        return first < second ? first : second;

    A MINIMUM under their comparator, which orders by ascending `Gain`, and
    `Gain` is negated at the source so a better split has the smaller one.

    **ON A FULL TIE IT KEEPS `second`, NOT `first`.** The ternary is
    `first < second ? first : second` and the comparator is strict, so equal
    records fall through to the else. Their level loop folds it LEFT over
    every scorer -- `bestSplitProp = TakeBest(bestSplitProp, calcer->
    ReadOptimalSplit())`
    (`oblivious_tree_doc_parallel_structure_searcher.cpp:115` and `:119`) --
    so `second` is the NEW candidate and a candidate that ties the incumbent
    on gain, feature id AND bin id REPLACES it. `pointwise_scores_calcer.h
    :100` folds the other way round (`TakeBest(helper.second->
    ReadOptimalSplit(), best)`), where the same rule keeps the INCUMBENT.
    Both are theirs; neither is tidied here.
    """
    return first if best_split_properties_less(first, second) else second


def take_best3(
    first: TBestSplitProperties,
    second: TBestSplitProperties,
    third: TBestSplitProperties,
) -> TBestSplitProperties:
    """`TakeBest(first, second, third)` (`helpers.h:19-23`).

        return TakeBest(TakeBest(first, second), third);

    LEFT-ASSOCIATED, and that is not decoration: with a strict `<` and a
    ternary that falls through to `second`, the left fold keeps the LAST of
    three equal candidates and a right-associated version would keep the
    middle one. `pairwise_structure_searcher.cpp:113` is the caller.
    """
    return take_best(take_best(first, second), third)


def to_split(
    feature_id: Int32,
    bin_id: Int32,
    defined: Bool,
    is_feature_bundle: Bool,
    is_cat: Bool,
    bin_count: Int32,
    border_count: Int32,
) raises -> TBinarySplit:
    """`ToSplit` (`helpers.cpp:157-173`), with the manager unwrapped.

    Theirs, branch for branch:

        CB_ENSURE(props.Defined(), "Need best split properties");
        if (manager.IsFeatureBundle(props.FeatureId)) { ...translate... }
        TBinarySplit bestSplit;
        bestSplit.FeatureId = props.FeatureId;
        bestSplit.BinIdx = props.BinId;
        if (manager.IsCat(props.FeatureId)) {
            bestSplit.SplitType = EBinSplitType::TakeBin;
            bestSplit.BinIdx = Min<ui32>(manager.GetBinCount(...), BinIdx);
        } else {
            bestSplit.SplitType = EBinSplitType::TakeGreater;
            bestSplit.BinIdx = Min<ui32>(manager.GetBorders(...).size() - 1,
                                         bestSplit.BinIdx);
        }

    THE TWO CAPS ARE NOT THE SAME EXPRESSION AND THE ASYMMETRY IS THEIRS.
    The categorical arm clamps to `GetBinCount(featureId)` and the float arm
    clamps to `GetBorders(featureId).size() - 1`. So a cat feature of `k`
    bins admits `BinIdx == k`, one past its own bin ids, while a float
    feature of `b` borders stops at `b - 1`. Their comment says why the clamp
    exists at all -- "Float arithmetic could generate empty bin splits for
    ctrs" -- and says nothing about why the two arms differ. It is
    transcribed rather than reconciled; `border_count` is `GetBorders().size()`
    here and the `- 1` is applied below, so a reviewer can diff the line.

    `is_feature_bundle` raises rather than translating, per DEVIATION 99.
    """
    # `CB_ENSURE(props.Defined(), "Need best split properties")`
    if not defined:
        raise Error("Need best split properties")
    if is_feature_bundle:
        raise Error(
            "ToSplit: feature bundles are not ported. Theirs calls"
            " TranslateFeatureBundleSplitToBinarySplit (helpers.cpp:159-161);"
            " see DEVIATION 99"
        )
    if is_cat:
        var capped = bin_id
        if bin_count < capped:
            capped = bin_count
        return TBinarySplit(feature_id, capped, Int32(BIN_SPLIT_TAKE_BIN))
    else:
        var limit = border_count - 1
        var capped = bin_id
        if limit < capped:
            capped = limit
        return TBinarySplit(feature_id, capped, Int32(BIN_SPLIT_TAKE_GREATER))


def _border_text(borders: List[Float32], at: Int) raises -> String:
    """`featuresManager.GetBorders(id)[binIdx]` rendered. See DEVIATION 99.3
    for why the digits are not gated; the INDEX is, because picking the wrong
    border is the failure that matters."""
    if at < 0 or at >= len(borders):
        raise Error(
            String("border index ") + String(at) + " is outside the "
            + String(len(borders))
            + " borders this feature has; their `GetBorders(id)[BinIdx]` would"
            " read out of bounds here"
        )
    return String(borders[at])


def split_condition_to_string(
    split: TBinarySplit, borders: List[Float32], nan_mode: Int
) raises -> String:
    """`SplitConditionToString(manager, split)` (`helpers.cpp:70-104`).

    Their three nan arms, in their order:

        Forbidden : ">" << borders[BinIdx]
        Min       : BinIdx > 0 ? ">" << borders[BinIdx - 1]
                               : "== -inf (nan)"
        Max       : BinIdx < borders.size() ? ">" << borders[BinIdx]
                                            : "== +inf (nan)"
                    with CB_ENSURE(BinIdx == borders.size()) on the else

    **`Min` SHIFTS THE BORDER INDEX DOWN BY ONE AND `Max` DOES NOT.** That is
    the whole content of the function and it is easy to paraphrase away.
    Under `ENanMode::Min` the NaN bucket is bin 0, so every real border sits
    one bin higher than its index; under `Max` the NaN bucket is the last
    one, so the real borders keep their indices and only the overflow bin is
    special. `Forbidden` has no NaN bucket at all.

    The `TakeBin` short-circuit comes first and reads no borders
    (`helpers.cpp:73-74`).
    """
    if split.split_type == Int32(BIN_SPLIT_TAKE_BIN):
        return String("TakeBin")

    var bin_idx = Int(split.bin_idx)
    if nan_mode == NAN_MODE_FORBIDDEN:
        return String(">") + _border_text(borders, bin_idx)
    elif nan_mode == NAN_MODE_MIN:
        if bin_idx > 0:
            return String(">") + _border_text(borders, bin_idx - 1)
        else:
            return String("== -inf (nan)")
    else:
        # `CB_ENSURE(nanMode == ENanMode::Max, "Unexpected nan mode")`
        if nan_mode != NAN_MODE_MAX:
            raise Error("Unexpected nan mode")
        if bin_idx < len(borders):
            return String(">") + _border_text(borders, bin_idx)
        else:
            # `CB_ENSURE(split.BinIdx == borders.size(), "Bin index is too
            #  large")`
            if bin_idx != len(borders):
                raise Error("Bin index is too large")
            return String("== +inf (nan)")


def split_condition_to_string_value(
    split: TBinarySplit, borders: List[Float32], nan_mode: Int, value: Int
) raises -> String:
    """`SplitConditionToString(manager, split, ESplitValue)`
    (`helpers.cpp:106-140`), the arm that names the side you did NOT take.

        const bool inverse = value == ESplitValue::Zero;

    and then the same four decisions with every comparator flipped:

        TakeBin     -> "SkipBin"
        ">"         -> "<="
        "== -inf"   -> "!= -inf"
        "== +inf"   -> "!= +inf"

    Called once per level of a leaf's path when their greedy searcher logs a
    non-symmetric tree (`greedy_subsets_searcher/greedy_search_helper.cpp
    :564`), which is why the ZERO direction needs a spelling at all.

    Note what does NOT flip: the border INDEX arithmetic. `Min` still reads
    `BinIdx - 1` and `Max` still bounds on `borders.size()`. Only the
    operator changes.
    """
    if value != SPLIT_VALUE_ZERO and value != SPLIT_VALUE_ONE:
        raise Error(
            String("ESplitValue is Zero(0) or One(1), not ") + String(value)
        )
    var inverse = value == SPLIT_VALUE_ZERO

    if split.split_type == Int32(BIN_SPLIT_TAKE_BIN):
        return String("SkipBin") if inverse else String("TakeBin")

    var cmp = String("<=") if inverse else String(">")
    var eq = String("!=") if inverse else String("==")
    var bin_idx = Int(split.bin_idx)

    if nan_mode == NAN_MODE_FORBIDDEN:
        return cmp + _border_text(borders, bin_idx)
    elif nan_mode == NAN_MODE_MIN:
        if bin_idx > 0:
            return cmp + _border_text(borders, bin_idx - 1)
        else:
            return eq + String(" -inf (nan)")
    else:
        if nan_mode != NAN_MODE_MAX:
            raise Error("Unexpected nan mode")
        if bin_idx < len(borders):
            return cmp + _border_text(borders, bin_idx)
        else:
            if bin_idx != len(borders):
                raise Error("Bin index is too large")
            return eq + String(" +inf (nan)")


def best_score_message(
    best_split: TBinarySplit,
    borders: List[Float32],
    nan_mode: Int,
    score: Float64,
    depth: UInt32,
) raises -> String:
    """The `logEntry` `PrintBestScore` builds (`helpers.cpp:142-155`), before
    it reaches `CATBOOST_INFO_LOG`.

        "Best split for depth " << depth << ": " << FeatureId << " / "
        << BinIdx << " (" << splitTypeMessage << ") with score " << score

    Separated from the printing so a check can compare the text without
    capturing stdout. Their CTR tail --

        if (featuresManager.IsCtr(bestSplit.FeatureId)) {
            logEntry << " tensor : " << GetCtr(...).FeatureTensor
                     << "  (ctr type " << ...Configuration.Type << ")";
        }

    -- is NOT appended here: it needs `TBinarizedFeaturesManager::GetCtr` and
    a `TFeatureTensor` printer, both unported. DEVIATION 99.2.
    """
    return (
        String("Best split for depth ")
        + String(depth)
        + ": "
        + String(best_split.feature_id)
        + " / "
        + String(best_split.bin_idx)
        + " ("
        + split_condition_to_string(best_split, borders, nan_mode)
        + ")"
        + " with score "
        + String(score)
    )


def print_best_score(
    best_split: TBinarySplit,
    borders: List[Float32],
    nan_mode: Int,
    score: Float64,
    depth: UInt32,
) raises:
    """`PrintBestScore` (`helpers.cpp:142-155`). `CATBOOST_INFO_LOG` is
    `print` here; this repository has no log-level machinery and inventing
    one would be a third category of file (`PORTING_RULES.md` 0b-ii)."""
    print(best_score_message(best_split, borders, nan_mode, score, depth))


def has_permutation_dependent_split(
    structure: TObliviousTreeStructure,
    is_ctr: List[Bool],
    is_permutation_dependent: List[Bool],
) raises -> Bool:
    """`HasPermutationDependentSplit` (`helpers.cpp:60-68`).

        for (const auto& split : structure.Splits) {
            if (featuresManager.IsCtr(split.FeatureId)) {
                auto ctr = featuresManager.GetCtr(split.FeatureId);
                if (featuresManager.IsPermutationDependent(ctr)) return true;
            }
        }
        return false;

    THE TWO PREDICATES ARE NESTED, NOT AND-ED, and that is why they arrive as
    two separate lists here rather than one. `IsPermutationDependent` takes a
    `TCtr`, so it is only defined once `IsCtr` has said there is one; a
    feature that is not a CTR has no answer to the second question at all.
    Flattening the pair into a single `is_permutation_dependent[featureId]`
    would work today and would quietly invent a value for every plain float
    column.

    `is_permutation_dependent` is indexed by FEATURE id, standing in for
    `IsPermutationDependent(GetCtr(featureId))` -- the caller does the
    `GetCtr` because `TCtr` is where the feature tensor lives and this file
    does not need it.

    Its two callers are `GetBinsForModel` and `CacheBinsForModel`
    (`helpers.cpp:6` and `:50`), both of which pick a cache SCOPE with the
    answer: permutation-dependent structures cache per permutation,
    independent ones cache once. Neither is ported (DEVIATION 99.1), so this
    predicate is currently unreached in-tree, and it is here because the
    third caller, `binarizations_manager.cpp:138`, shows the same question
    being asked of a CTR's own tensor one level down.
    """
    for i in range(len(structure.splits)):
        var fid = Int(structure.splits[i].feature_id)
        if fid < 0 or fid >= len(is_ctr):
            raise Error(
                String("split ") + String(i) + " names feature " + String(fid)
                + ", outside the " + String(len(is_ctr))
                + " features the caller described"
            )
        if is_ctr[fid]:
            if fid >= len(is_permutation_dependent):
                raise Error(
                    String("feature ") + String(fid)
                    + " is a CTR but the caller gave no"
                    " IsPermutationDependent answer for it"
                )
            if is_permutation_dependent[fid]:
                return True
    return False


# =========================================================================
# The bit helpers (`helpers.h:53-98`).
#
# WHO CALLS THEM UPSTREAM, checked rather than assumed. `ReverseBits(u,
# nBits)` has no caller in `catboost/cuda/` at all -- the `ReverseBits` at
# `private/libs/data_types/groupid.h:14` is a DIFFERENT one-argument function
# and `algo/ut/monotonic_constraints_ut.cpp:66` is a CPU unit test. `GetOddBits`
# and `GetEvenBits` are called only by `methods/ut/test_pairwise_tree_searcher
# .cpp:152-153`. `MergeBits` has no caller either; the `MergeBits` in
# `ctrs/ctr_kernels.h:170` is an unrelated device kernel of the same name.
#
# So all four are ported for completeness of the assigned file and NONE has a
# caller here or there. `PORTING_RULES.md` rule 3 says an unported file is
# visible and a mis-ported one is not -- these are transcribed and gated
# against an independent host oracle for exactly that reason, and their lack
# of a caller is stated here rather than discovered later.
#
# What they mean, which their code does not say: `GetEvenBits` extracts bits
# 0, 2, 4, ... and packs them down; `GetOddBits` does the same for bits 1, 3,
# 5, ...; `MergeBits(x, y)` is the inverse, interleaving x into the even
# positions and y into the odd ones. That is a Morton code over a pair of
# leaf coordinates, which is what the PAIRWISE searcher's test uses them for.
# Both extractors stop after EIGHT output bits (their last term is
# `(mask & 16384) >> 7`), so they are 16-bit-in / 8-bit-out and are NOT
# general; `MergeBits` likewise consumes only the low 8 bits of each input.
# Transcribed as written, ceiling included.
# =========================================================================


def reverse_bits(u: Int, n_bits: Int) -> UInt32:
    """`ReverseBits(int u, int nBits)` (`helpers.h:53-62`).

    The standard five-step butterfly reversal of a full 32-bit word followed
    by `v >>= (32 - nBits)`, which drops the reversed word down so the
    `nBits` low bits of the input come back in the `nBits` low bits of the
    output.

    `n_bits == 0` is a 32-place shift of a `ui32` in their code, which is
    UNDEFINED BEHAVIOUR in C++ and is a defined 0 in Mojo. No caller passes
    it. Left as the shift rather than special-cased, because a guard here
    would be an answer they do not have.
    """
    var v = UInt32(u)
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1)
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2)
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4)
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8)
    v = (v >> 16) | (v << 16)
    v >>= UInt32(32 - n_bits)
    return v


def get_odd_bits(val: Int) -> Int:
    """`GetOddBits(int val)` (`helpers.h:64-76`), all eight terms.

        int mask = (val & 0xAAAAAAAA) >> 1;
        r |= (mask & 1);        r |= (mask & 4) >> 1;
        r |= (mask & 16) >> 2;  r |= (mask & 64) >> 3;
        r |= (mask & 256) >> 4; r |= (mask & 1024) >> 5;
        r |= (mask & 4096) >> 6; r |= (mask & 16384) >> 7;

    `0xAAAAAAAA` does not fit an `int`, so their `val & 0xAAAAAAAA` promotes
    to unsigned and the `>> 1` is logical. `UInt32` here says the same thing
    without depending on a promotion rule.
    """
    var mask = (UInt32(val) & 0xAAAAAAAA) >> 1
    var r = UInt32(0)
    r |= mask & 1
    r |= (mask & 4) >> 1
    r |= (mask & 16) >> 2
    r |= (mask & 64) >> 3
    r |= (mask & 256) >> 4
    r |= (mask & 1024) >> 5
    r |= (mask & 4096) >> 6
    r |= (mask & 16384) >> 7
    return Int(r)


def get_even_bits(val: Int) -> Int:
    """`GetEvenBits(int val)` (`helpers.h:77-89`), all eight terms.

    Their variable is `c` and not `r`, and there is no `>> 1` on the mask
    because the even bits are already in place.
    """
    var mask = UInt32(val) & 0x55555555
    var c = UInt32(0)
    c |= mask & 1
    c |= (mask & 4) >> 1
    c |= (mask & 16) >> 2
    c |= (mask & 64) >> 3
    c |= (mask & 256) >> 4
    c |= (mask & 1024) >> 5
    c |= (mask & 4096) >> 6
    c |= (mask & 16384) >> 7
    return Int(c)


def merge_bits(x: Int, y: Int) -> Int:
    """`MergeBits(int x, int y)` (`helpers.h:90-101`), all eight terms.

        res |= (x & 1) | ((y & 1) << 1);
        res |= ((x & 2) << 1) | ((y & 2) << 2);
        ...
        res |= ((x & 128) << 7) | ((y & 128) << 8);

    `x` lands in the even positions and `y` in the odd ones, so
    `GetEvenBits(MergeBits(x, y)) == x & 255` and
    `GetOddBits(MergeBits(x, y)) == y & 255`. That round trip is what
    `checks/pointwise_subsets_check.mojo` gates, over the whole 8-bit
    square, because it ties the three functions to each other rather than to
    a transcription of the same constants.
    """
    var a = UInt32(x)
    var b = UInt32(y)
    var res = UInt32(0)
    res |= (a & 1) | ((b & 1) << 1)
    res |= ((a & 2) << 1) | ((b & 2) << 2)
    res |= ((a & 4) << 2) | ((b & 4) << 3)
    res |= ((a & 8) << 3) | ((b & 8) << 4)
    res |= ((a & 16) << 4) | ((b & 16) << 5)
    res |= ((a & 32) << 5) | ((b & 32) << 6)
    res |= ((a & 64) << 6) | ((b & 64) << 7)
    res |= ((a & 128) << 7) | ((b & 128) << 8)
    return Int(res)
