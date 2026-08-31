# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Borders for a column that may contain NaN, and the bin a NaN lands in.

PORT OF `CalcQuantization` (`catboost/libs/data/quantization.cpp:300-346`)
and the NaN half of the apply-time quantizer
(`libs/model/cpu/quantization.h:368-409`) at CatBoost `54a8143a`.
Transliterated. Do not improve.

## The whole mechanism is a SENTINEL BORDER, and nothing downstream knows

There is no NaN branch in their histogram kernels, their scorer, their
partitioner or their GPU evaluator. NaN is handled entirely in
quantization, and everything after it sees an ordinary bin index. That is
what makes this cheap to port and it is worth stating plainly, because the
opposite -- a NaN flag threaded through the tree -- is what one would
otherwise build.

Their `CalcQuantization` (`:321-346`), in full:

    int nonNanValuesBorderCount = binarizationOptions.BorderCount;
    if (hasNans) {
        *nanMode = binarizationOptions.NanMode;
        --nonNanValuesBorderCount;              // ONE BORDER IS SPENT
    } else {
        *nanMode = ENanMode::Forbidden;
    }
    ... BestSplit(values, nonNanValuesBorderCount) ...
    if (*nanMode == ENanMode::Min) {
        Borders.insert(Borders.begin(), numeric_limits<float>::lowest());
    } else if (*nanMode == ENanMode::Max) {
        Borders.push_back(numeric_limits<float>::max());
    }

**A COLUMN WITH NaNs GETS ONE FEWER REAL BORDER**, and that is not a
rounding detail: `border_count` is a budget for the whole column, and the
NaN bin comes out of it rather than being added to it. A port that inserted
the sentinel WITHOUT the decrement would hand the histogram one more bin
than the caller asked for, which changes the grid policy a feature lands in
(`grid_policy.mojo:83` is a step function) and therefore which kernel reads
it.

## Why a sentinel works, and why apply substitutes infinities instead

Quantization is `sum over borders of (value > border)`. NaN compares false
against everything, so a NaN naturally scores 0 -- which IS the Min answer,
and is why `Min` needs no comparison change at all. `Max` does: NaN would
still score 0 and land at the bottom.

Their evaluator solves both with one substitution rather than two paths
(`cpu/quantization.h:385-408`): replace NaN with `-infinity` for `AsFalse`
and `+infinity` for `AsTrue`, then run the ordinary comparison.

    -inf > lowest()  is false  ->  bin 0                       (Min)
    +inf > max()     is true   ->  bin len(borders)            (Max)

This port takes the substitution for BOTH sides -- learn and apply -- so the
one comparison kernel it already has is untouched, and the learn and apply
paths agree by construction rather than by two implementations that have to
be kept in step.
"""

from std.math import inf

from gbdt.grid_creator.binarization import best_split
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    NAN_MODE_MAX,
    NAN_MODE_MIN,
)

#: `TFloatFeature::ENanValueTreatment` (`libs/model/features.h:51-55`), what
#: a MODEL records so its apply path can reproduce the learn-time bin.
comptime NAN_TREATMENT_AS_IS = 0
comptime NAN_TREATMENT_AS_FALSE = 1
comptime NAN_TREATMENT_AS_TRUE = 2


def has_nans(values: List[Float32]) -> Bool:
    """Their `denseData->GetData()->Find(isnan)`
    (`quantized_features_info.cpp:210-211`)."""
    for i in range(len(values)):
        if values[i] != values[i]:
            return True
    return False


def compute_nan_mode(values: List[Float32], nan_mode_option: Int) -> Int:
    """`ComputeNanMode` (`quantized_features_info.cpp:202-239`).

    A column with no NaN is `Forbidden` no matter what was asked, which is
    what keeps the border budget intact for every column that does not need
    a NaN bin.
    """
    if nan_mode_option == NAN_MODE_FORBIDDEN:
        return NAN_MODE_FORBIDDEN
    if has_nans(values):
        return nan_mode_option
    return NAN_MODE_FORBIDDEN


def nan_value_treatment(nan_mode: Int) -> Int:
    """`CreateFloatFeatures` (`private/libs/algo/helpers.cpp:56-64`):

        Min -> AsFalse, HasNans = true
        Max -> AsTrue,  HasNans = true
        (otherwise the default, AsIs, HasNans = false)
    """
    if nan_mode == NAN_MODE_MIN:
        return NAN_TREATMENT_AS_FALSE
    if nan_mode == NAN_MODE_MAX:
        return NAN_TREATMENT_AS_TRUE
    return NAN_TREATMENT_AS_IS


def nan_substitution(treatment: Int) -> Float32:
    """The value their evaluator puts in a NaN's place
    (`cpu/quantization.h:385-408`). Only meaningful for the two treatments
    that substitute; `AsIs` never reaches a substitution because a NaN
    under it is an error."""
    if treatment == NAN_TREATMENT_AS_TRUE:
        return Float32(inf[DType.float32]())
    return Float32(-inf[DType.float32]())


def substitute_nans(mut values: List[Float32], treatment: Int):
    """`BinarizeFloats<UseNanSubstitution=true>`'s prologue, in place."""
    if treatment == NAN_TREATMENT_AS_IS:
        return
    var sub = nan_substitution(treatment)
    for i in range(len(values)):
        if values[i] != values[i]:
            values[i] = sub


def calc_quantization(
    var values: List[Float32], border_count: Int, nan_mode_option: Int
) raises -> Tuple[List[Float32], Int]:
    """`CalcQuantization` (`quantization.cpp:300-346`).

    Returns the borders WITH the sentinel already in them, and the mode the
    column RESOLVED to. `values` is consumed, as their `featureValues` is.
    """
    var nan_mode = compute_nan_mode(values, nan_mode_option)

    var non_nan_border_count = border_count
    if nan_mode != NAN_MODE_FORBIDDEN:
        non_nan_border_count -= 1

    var borders = List[Float32]()
    if non_nan_border_count > 0:
        # `BestSplit` already drops NaNs -- their `filterNans`
        borders = best_split(values^, non_nan_border_count)

    if nan_mode == NAN_MODE_MIN:
        # `numeric_limits<float>::lowest()`, which is -FLT_MAX and NOT
        # -infinity: the substituted `-inf` must compare FALSE against it,
        # and `-inf > -inf` is false while `-inf > -FLT_MAX` is also false,
        # so either would work here -- but a real value equal to -FLT_MAX
        # must land ABOVE the NaN bin, and only the finite bound gives it
        # that. Theirs is the finite one.
        var with_nan = List[Float32]()
        with_nan.append(Float32(-3.4028234663852886e38))
        for i in range(len(borders)):
            with_nan.append(borders[i])
        borders = with_nan^
    elif nan_mode == NAN_MODE_MAX:
        borders.append(Float32(3.4028234663852886e38))

    return (borders^, nan_mode)
