"""The loss and its parameters, and the leaf estimator that follows from it.

PORT OF `catboost/private/libs/options/loss_description.{h,cpp}` at CatBoost
`54a8143a` -- the loss function and its parameter accessors. Transliterated.
Do not improve.

**The leaf-estimation defaults are NOT here.** `GetEstimationMethodDefaults`
and `SetLeavesEstimationDefault` are `catboost_options.cpp:30-360`, so they
are in `gbdt/options/catboost_options.mojo`. They were written here first,
next to the parameters they read, and that was wrong twice over: wrong
mirror address, and an import pointing back at `catboost_options` where
their own graph runs one way only.

## Why this file exists at all, and why it is not optional

Until 2026-08-21 a loss was a `String` compared against three spellings in
`gbdt/train.mojo`, because all three of RMSE, Logloss and CrossEntropy take
their leaf values from the SAME estimator with only the iteration count
differing. That stops being true the moment a fourth loss arrives:

    Quantile, MAE, MAPE, LogLinQuantile   Der2 is identically zero
    Lq                                    Newton or Gradient, DEPENDING ON q
    Expectile, Tweedie, Huber, Poisson    Newton, but 5 / 20 / 1 / 10 iters

A Newton step on a loss whose second derivative is zero divides the gradient
by `lambda` alone. It does not crash and it does not look wrong; it fits a
different model. **The objective's derivative formula is only half of a
loss.** The other half is this table, and CatBoost keeps it in its options
layer rather than in its kernels for the same reason.

## The three mandatory parameters, and the one that is silently dropped

`Lq` needs `q`, `Huber` needs `delta`, `Tweedie` needs `variance_power`, and
`Expectile` needs `alpha`. All four are `CB_ENSURE`d by their own file
(`catboost_options.cpp:126`, `:82`, `:222`; `loss_description.cpp:202`,
`:247`) and all four raise here.

`variance_power` then reaches their CPU (`algo/tensor_search_helpers.cpp:308`)
and NOT their GPU -- see DEVIATION 62 in
`gbdt/targets/kernel/pointwise_targets.mojo`, which is where the decision to
thread it anyway is argued and priced.
"""

#: NOTHING IS IMPORTED FROM `catboost_options.mojo` HERE, AND THAT IS
#: STRUCTURAL. Their dependency runs ONE WAY: `catboost_options.h` includes
#: `loss_description.h`, and `loss_description.cpp` includes nothing of
#: theirs back. The leaf-estimation defaults were briefly written into this
#: file, which both put them at the wrong mirror address and pointed the
#: import the wrong way; they are in `catboost_options.mojo` now, where
#: `SetLeavesEstimationDefault` is in their tree, and this file stayed a
#: leaf of the graph.
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_EXPECTILE,
    OBJECTIVE_HUBER,
    OBJECTIVE_LOGLINQUANTILE,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_LQ,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_POISSON,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
    OBJECTIVE_TWEEDIE,
    objective_from_name,
    objective_name,
)

#: `GetDefaultTargetBorder()`, the Logloss threshold
#: (`loss_description.cpp:90-93`).
comptime DEFAULT_TARGET_BORDER = Float32(0.5)

#: `GetAlpha`'s fallback (`loss_description.cpp:95-97`).
comptime DEFAULT_ALPHA = Float32(0.5)

#: `MAE` is `Quantile` at one half, set by their `Init` rather than by the
#: user (`pointwise_target_impl.h:272-275`).
comptime MAE_ALPHA = Float32(0.5)


@fieldwise_init
struct TLossDescription(Copyable, Movable):
    """Their `TLossDescription`: a loss function plus its parameter map.

    Theirs holds `TMap<TString, TString>` and parses on each accessor.
    Ours holds the four parameters TYPED, with a `has_` flag each, because
    a decimal round trip through `String` is a known defect on this
    toolchain (`String(Float32)` returns a one-ULP-wrong value for 0.46%
    of float32) and a loss parameter is exactly the kind of number that
    must not move. The `has_` flags are their `GetLossParamsMap().contains`
    tests, one for one.
    """

    var loss_function: Int
    var has_alpha: Bool
    var alpha_param: Float32
    var has_q: Bool
    var q_param: Float32
    var has_delta: Bool
    var delta_param: Float32
    var has_variance_power: Bool
    var variance_power_param: Float32
    var has_border: Bool
    var border_param: Float32

    def name(self) -> String:
        return objective_name(self.loss_function)

    def get_alpha(self) -> Float32:
        """`GetAlpha` (`loss_description.cpp:95-102`): 0.5 when unset."""
        return self.alpha_param if self.has_alpha else DEFAULT_ALPHA

    def get_lq_param(self) raises -> Float32:
        """`GetLqParam`; `q` is mandatory (`catboost_options.cpp:82`)."""
        if not self.has_q:
            raise Error("Param q is mandatory for Lq loss")
        return self.q_param

    def get_huber_param(self) raises -> Float32:
        """`GetHuberParam` (`loss_description.cpp:202-207`)."""
        if not self.has_delta:
            raise Error("For Huber delta parameter is mandatory")
        return self.delta_param

    def get_tweedie_param(self) raises -> Float32:
        """`GetTweedieParam` (`loss_description.cpp:247-254`)."""
        if not self.has_variance_power:
            raise Error(
                "For Tweedie variance_power parameter is mandatory"
            )
        return self.variance_power_param

    def get_logloss_border(self) -> Float32:
        """`GetLogLossBorder` (`loss_description.cpp:90-93`)."""
        return (
            self.border_param if self.has_border
            else DEFAULT_TARGET_BORDER
        )

    def kernel_alpha(self) raises -> Float32:
        """THE ONE FLOAT their target kernel receives (`:451`).

        `TPointwiseTargetsImpl::Init` (`pointwise_target_impl.h:259-299`)
        decides what it means per loss and stores it in the SAME member,
        and `Approximate` hands that member to the kernel
        (`:151-166`, `:346-356`). Reproduced case for case, with the one
        recorded departure: Tweedie's slot carries `variance_power` here
        and carries their unset `Alpha` (zero) there.
        """
        var loss = self.loss_function
        if loss == OBJECTIVE_MAE:
            # `Alpha = 0.5;` (`:272-275`) -- NOT the user's alpha
            return MAE_ALPHA
        if loss == OBJECTIVE_LQ:
            return self.get_lq_param()
        if loss == OBJECTIVE_HUBER:
            return self.get_huber_param()
        if loss == OBJECTIVE_TWEEDIE:
            return self.get_tweedie_param()  # DEVIATION 62
        if (
            loss == OBJECTIVE_QUANTILE
            or loss == OBJECTIVE_LOGLINQUANTILE
            or loss == OBJECTIVE_EXPECTILE
        ):
            return self.get_alpha()
        # RMSE, MAPE, Poisson, Logloss, CrossEntropy: their `Init` sets no
        # Alpha, so the kernel receives the member's declared 0 (`:364`).
        return Float32(0.0)

    def validate(self) raises:
        """Their `Init` switch's `CB_ENSURE`s, gathered.

        `Expectile`'s alpha check lives in `GetEstimationMethodDefaults`
        (`catboost_options.cpp:126`) rather than in `Init`; the other three
        live in both places. Running them here means a bad configuration
        raises before a single buffer is allocated, which is where theirs
        raises too.
        """
        var loss = self.loss_function
        if loss == OBJECTIVE_LQ:
            _ = self.get_lq_param()
        elif loss == OBJECTIVE_HUBER:
            _ = self.get_huber_param()
        elif loss == OBJECTIVE_TWEEDIE:
            _ = self.get_tweedie_param()
        elif loss == OBJECTIVE_EXPECTILE:
            if not self.has_alpha:
                raise Error("Param alpha is mandatory for expectile loss")


def make_loss_description(
    name: String,
    alpha: Float32 = Float32(-1.0),
    q: Float32 = Float32(-1.0),
    delta: Float32 = Float32(-1.0),
    variance_power: Float32 = Float32(-1.0),
    border: Float32 = Float32(-1.0),
) raises -> TLossDescription:
    """Build a description from their spellings.

    A parameter left at its sentinel is UNSET, which is their "the map does
    not contain this key". The sentinel is negative because all four
    parameters are positive in every configuration CatBoost accepts:
    `alpha` is a quantile level in (0, 1), `q >= 0`, `delta > 0`, and
    `variance_power` is in (1, 2) -- their own metric registry rejects
    anything else. A caller that wants a negative parameter has left the
    range their kernels were written for.
    """
    var d = TLossDescription(
        objective_from_name(name),
        alpha >= Float32(0.0), alpha,
        q >= Float32(0.0), q,
        delta >= Float32(0.0), delta,
        variance_power >= Float32(0.0), variance_power,
        border >= Float32(0.0), border,
    )
    d.validate()
    return d^
