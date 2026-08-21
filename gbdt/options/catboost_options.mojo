"""Training options, under CatBoost's own names.

MIRRORS `catboost/private/libs/options/`, principally
`catboost_options.cpp`, `boosting_options.cpp`, `oblivious_tree_options.cpp`,
`bootstrap_options.h` and `data_processing_options.cpp`. Their spellings are
kept exactly, including the ones this port does not honor yet, because a name
that differs from CatBoost's is a name somebody has to translate every time
they read their docs against our source.

**Every option carries an `honored` note.** An option that exists and is
ignored is worse than one that is absent: absent fails loudly, ignored fails
silently, and this repository has already spent a day on machinery that was
present and unreachable.

## A constructor default is not always the shipped default

Three of the values below do NOT come from the option's own constructor, and
reading only `oblivious_tree_options.cpp` gets them wrong:

- `max_leaves` is constructed at 31 (`oblivious_tree_options.cpp:24`) and
  then OVERWRITTEN to `1 << depth` for every policy but Lossguide
  (`catboost_options.cpp:993-1001`), which also refuses any user value that
  is not `1 << depth`. The literal 31 never reaches a symmetric tree.
- `border_count` is 128 on GPU and 254 on CPU
  (`data_processing_options.cpp:14-19`). This is a GPU port, so it is 128.
- `leaf_estimation_method` is constructed at `Gradient`
  (`oblivious_tree_options.cpp:14`) and then set per LOSS; RMSE gets
  `Newton` (`catboost_options.cpp:59-64`, `:304-306`).

So each entry below cites the line that actually decides the value, which is
not always the line that names the option.
"""


# THEIR INCLUDE, IN THEIR DIRECTION: `catboost_options.h` includes
# `loss_description.h` and `loss_description.cpp` includes nothing of
# theirs back, so this edge points one way and the graph stays acyclic.
from gbdt.options.loss_description import TLossDescription
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_CROSSENTROPY,
    OBJECTIVE_EXPECTILE,
    OBJECTIVE_HUBER,
    OBJECTIVE_LOGLINQUANTILE,
    OBJECTIVE_LOGLOSS,
    OBJECTIVE_LQ,
    OBJECTIVE_MAE,
    OBJECTIVE_MAPE,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_POISSON,
    OBJECTIVE_QUANTILE,
    OBJECTIVE_RMSE,
    OBJECTIVE_TWEEDIE,
)
from gbdt.ctrs.ctr import (
    CTR_BORDERS,
    CTR_BUCKETS,
    CTR_COUNTER,
    CTR_FEATURE_FREQ,
    TCtrConfig,
    TPrior,
    ctr_type_name,
    get_default_priors,
    is_supported_ctr_type_gpu,
    need_target_classifier,
)
from gbdt.ctrs.ctr_binarization import (
    BORDER_SELECTION_MEDIAN,
    BORDER_SELECTION_MIN_ENTROPY,
    BORDER_SELECTION_UNIFORM,
    TBinarizationOptions,
    border_selection_name,
)


# --- grow_policy -----------------------------------------------------------
comptime GROW_SYMMETRIC = 0
comptime GROW_DEPTHWISE = 1
comptime GROW_LOSSGUIDE = 2


def grow_policy_name(p: Int) -> String:
    if p == GROW_SYMMETRIC:
        return String("SymmetricTree")
    if p == GROW_DEPTHWISE:
        return String("Depthwise")
    return String("Lossguide")


# --- score_function --------------------------------------------------------
#
# `enum class EScoreFunction` -- `private/libs/options/enums.h:72-80`. THEIR
# ORDER AND THEIR VALUES, so a number read out of one of their configs means
# the same thing here.
#
# The shipped default is COSINE, not L2:
# `ScoreFunction("score_function", EScoreFunction::Cosine)` --
# `private/libs/options/oblivious_tree_options.cpp:22`. This port scored
# every level with L2 and had no option at all, so a run configured as stock
# CatBoost picked a DIFFERENT SPLIT AT EVERY LEVEL. Cosine is not a rescaled
# L2 -- it normalizes by `sqrt(sum(w * mu^2))`, so it ranks candidates by the
# ANGLE between the gradient and the step rather than by the raw drop.

comptime SCORE_FUNCTION_SOLAR_L2 = 0
comptime SCORE_FUNCTION_COSINE = 1
comptime SCORE_FUNCTION_NEWTON_L2 = 2
comptime SCORE_FUNCTION_NEWTON_COSINE = 3
comptime SCORE_FUNCTION_LOO_L2 = 4
comptime SCORE_FUNCTION_SAT_L2 = 5
comptime SCORE_FUNCTION_L2 = 6


def score_function_name(s: Int) -> String:
    if s == SCORE_FUNCTION_SOLAR_L2:
        return String("SolarL2")
    if s == SCORE_FUNCTION_COSINE:
        return String("Cosine")
    if s == SCORE_FUNCTION_NEWTON_L2:
        return String("NewtonL2")
    if s == SCORE_FUNCTION_NEWTON_COSINE:
        return String("NewtonCosine")
    if s == SCORE_FUNCTION_LOO_L2:
        return String("LOOL2")
    if s == SCORE_FUNCTION_SAT_L2:
        return String("SatL2")
    return String("L2")


# --- determinism -----------------------------------------------------------
#
# NO CATBOOST COUNTERPART. They ship one GPU backend and accept whatever
# their float atomics do, so the question does not arise for them. It arises
# here because this port targets Metal, CUDA and HIP from one source.
#
# THREE LEVELS, A LADDER RATHER THAN TWO SWITCHES, because across-device
# determinism strictly implies within-device determinism and two independent
# booleans would let a caller ask for the combination that cannot exist.

comptime DETERMINISM_OFF = 0
"""No guarantee. Every row of the kernel matrix is read from the device's own
column and histograms flush through whatever the vendor's fastest path is.
On NVIDIA and AMD that is a float atomic, so the last bits move between two
runs of the SAME fit on the SAME device."""

comptime DETERMINISM_DEVICE = 1
"""Same fit, same device, same model, every run. The flush becomes a
fixed-point integer accumulator, whose addition is associative, so the answer
does not depend on which block lands first. Scheduling rows stay on the
device's column, so this costs only the flush.

This level FORCES the integer accumulator on every backend, Apple included.
It is not free anywhere. Metal does have a working float atomic add -- 1024
threads each adding 1.0 through `Atomic.fetch_add` returns exactly 1024.0 on
this machine -- so the float flush that CatBoost uses
(`hist_half_byte.cu:45-51`) is available here too, and choosing the integer
path is choosing to give that up. What it costs is UNRECORDED: no interleaved
measurement of the two flushes exists yet, and a number without one is not
allowed in this tree."""

comptime DETERMINISM_CROSS_DEVICE = 2
"""Same fit, ANY supported GPU, same model. Adds the numeric rows that differ
between vendors to what `DEVICE` already pins: the replication factor and the
reduction width move to `COLUMN_BIT_IDENTICAL`, so the reduction tree has one
shape everywhere.

Apple's lane width already equals the pinned 32 and its shared-memory budget
IS the safe column, so the bit-identical column and the apple column
coincide, and this level adds nothing to `DEVICE` there. NVIDIA gives up a
768-thread block for 512, and AMD additionally gives up its 64-wide
wavefront. On top of whatever `DEVICE` already costs, which is unrecorded."""


def determinism_name(d: Int) -> String:
    if d == DETERMINISM_OFF:
        return String("off")
    if d == DETERMINISM_DEVICE:
        return String("device")
    return String("cross_device")


# --- bootstrap_type --------------------------------------------------------
#
# `enum class EBootstrapType` -- `private/libs/options/enums.h:92-98`. THEIR
# ORDER AND THEIR VALUES.
#
# The shipped default is BAYESIAN at `bagging_temperature` 1.0 --
# `bootstrap_options.h:15-18`. It is not a sampling refinement that can be
# skipped: `TWeakObjective::StochasticDer` runs `BootstrapAndFilter` BEFORE
# the derivatives are computed (`weak_objective_impl.h:31-36`), so every row's
# weight is multiplied by a fresh random draw once per tree, and the
# histogram, the split scores and the leaf values are all computed from the
# reweighted target. Stock CatBoost therefore grows a DIFFERENT tree from the
# one this port grows even when every other option matches.

comptime BOOTSTRAP_POISSON = 0
comptime BOOTSTRAP_BAYESIAN = 1
comptime BOOTSTRAP_BERNOULLI = 2
comptime BOOTSTRAP_MVS = 3
comptime BOOTSTRAP_NO = 4


def bootstrap_type_name(b: Int) -> String:
    if b == BOOTSTRAP_POISSON:
        return String("Poisson")
    if b == BOOTSTRAP_BAYESIAN:
        return String("Bayesian")
    if b == BOOTSTRAP_BERNOULLI:
        return String("Bernoulli")
    if b == BOOTSTRAP_MVS:
        return String("MVS")
    return String("No")


# --- leaf_estimation_method ------------------------------------------------
#
# `enum class ELeavesEstimation` -- `private/libs/options/enums.h:64-70`.
# THEIR ORDER AND THEIR VALUES.

comptime LEAF_ESTIMATION_GRADIENT = 0
comptime LEAF_ESTIMATION_NEWTON = 1
comptime LEAF_ESTIMATION_EXACT = 2
comptime LEAF_ESTIMATION_SIMPLE = 3


def leaf_estimation_method_name(m: Int) -> String:
    if m == LEAF_ESTIMATION_GRADIENT:
        return String("Gradient")
    if m == LEAF_ESTIMATION_NEWTON:
        return String("Newton")
    if m == LEAF_ESTIMATION_EXACT:
        return String("Exact")
    return String("Simple")


@fieldwise_init
struct CatBoostOptions(Copyable, Movable):
    """Their option names, their defaults, and what this port honors."""

    var depth: Int
    """`depth`. CatBoost's default is 6. HONORED: `run_tree`'s `max_depth`."""

    var grow_policy: Int
    """`grow_policy`. Default SymmetricTree. HONORED for SymmetricTree only;
    Depthwise and Lossguide are not ported and `check()` refuses them rather
    than silently growing a symmetric tree under another name."""

    var max_leaves: Int
    """`max_leaves`. **Default `1 << depth`, which is 64 at the default depth
    of 6, NOT the 31 the option is constructed with.** For every policy but
    Lossguide, CatBoost overwrites the constructed 31 with `1 << MaxDepth` and
    refuses any user value that differs (`catboost_options.cpp:993-1001`,
    error text "max_leaves option works only with lossguide tree growing").

    HONORED as a terminating condition in the same sense theirs is:
    `ShouldTerminate` stops the tree at `leafCount >= Options.MaxLeaves`
    (`greedy_search_helper.cpp:678-683`), and at `1 << depth` that bound is
    reached exactly when the depth bound is. `check()` enforces the equality
    so that a caller who sets 31 by hand gets CatBoost's refusal rather than a
    tree silently truncated to depth 5."""

    var min_data_in_leaf: Int
    """`min_data_in_leaf`. Default 1.

    **CatBoost IGNORES this option under SymmetricTree.** `IsTerminalLeaf`
    guards the size test with `Options.Policy != EGrowPolicy::SymmetricTree`
    (`greedy_search_helper.cpp:691-694`), so the only policy this port grows
    is the one where their leaf-size test never runs. Our score kernel has no
    minimum-count test either, which for SymmetricTree is agreement and not a
    gap.

    `check()` still refuses anything but 1. That is STRICTER than CatBoost,
    which accepts any value here and discards it, and it is kept so the option
    cannot become silently live the day Lossguide lands."""

    var l2_leaf_reg: Float32
    """`l2_leaf_reg`. Default 3.0 (`oblivious_tree_options.cpp:15`, and
    `GetEstimationMethodDefaults` returns the same 3.0 for RMSE,
    `catboost_options.cpp:38`). HONORED: `run_tree_layout` takes it and passes
    it to both the score kernel's `lambda_l2` and the leaf estimator's `l2`.
    The probe entry points in the same file still hardcode 1.0; they are
    checks, not the training path.

    A zero is not a zero: CatBoost substitutes `1e-20` for it after the
    defaults are resolved (`catboost_options.cpp:357-359`), so that the leaf
    denominator can never be the raw leaf weight. `l2_leaf_reg_effective()`
    below is that substitution and is what a caller should pass down."""

    var score_function: Int
    """`score_function`. **Default Cosine**, which is CatBoost's shipped
    default (`oblivious_tree_options.cpp:22`), NOT L2. HONORED for Cosine,
    NewtonCosine, L2 and NewtonL2: the score kernel selects the calcer at
    comptime. SolarL2, SatL2 and LOOL2 are not ported and `check()` refuses
    them."""

    var model_size_reg: Float32
    """`model_size_reg`. Default 0.5 (`oblivious_tree_options.cpp:28`). It
    feeds `UpdateFeatureWeightsForBestSplits`, which the score kernel then
    multiplies into every gain (`compute_scores.cu:136-137`).

    NOT HONORED, and **NO LONGER A NO-OP, as of the day `train()` gained a
    categorical path.** The paragraph that stood here said the function
    fills the weight vector with 1.0 and RETURNS EARLY when the CTR count is
    zero, so that nothing diverged until CTRs landed. The first half is
    still true and the conclusion is now false, so it is deleted rather than
    annotated: with CTR columns present, `GetCtrsCount() != 0` and the early
    return at `update_feature_weights.cpp:20-22` is not taken. Every CTR
    column that is not yet USED then gets

        pow(1 + maxCtrUniqueValues / maxUniqueValues, -modelSizeReg)

    (`:27-44`) instead of 1.0, and the score kernel multiplies that into
    every gain (`compute_scores.cu:136-137`). So a CTR fit here scores CTR
    candidates HIGHER than stock CatBoost does, by exactly that factor, and
    that is a live divergence rather than a dormant one. `check()` still
    refuses any value but 0.5, which no longer makes the option safe -- it
    only keeps it from being TWO divergences."""

    var border_count: Int
    """`border_count`. **Default 128, not 254.** The value is task-type
    dependent: `type == ETaskType::GPU ? 128 : 254`
    (`data_processing_options.cpp:14-19`), and this is the GPU port. 254 is
    CatBoost's CPU default and was ours by mistake, which is a quantization
    twice as fine as stock CatBoost on this backend.

    NOT HONORED: quantization is not ported, so the port consumes fold counts
    a caller has already produced. This is the option that would drive them,
    and the number it would drive them to is 128."""

    var leaf_estimation_method: Int
    """`leaf_estimation_method`. **Default Newton for RMSE.** The option is
    constructed at `Gradient` (`oblivious_tree_options.cpp:14`) and then
    replaced per loss: `GetEstimationMethodDefaults` returns `Newton` for
    RMSE (`catboost_options.cpp:59-64`) and `SetLeavesEstimationDefault`
    installs it (`catboost_options.cpp:304-306`).

    HONORED at Newton. It matters that this is Newton rather than Simple,
    because `NeedEstimation()` is `LeavesEstimationMethod != Simple`
    (`greedy_subsets_searcher.h:67-69`): under the shipped default CatBoost
    RE-ESTIMATES every leaf after the structure search rather than keeping the
    value the search produced. See `leaves_estimation.mojo` for why the two
    land on the same number for MSE at one iteration.

    Gradient is refused even though it also coincides for MSE, because it
    coincides only for MSE and a silent agreement is not a port."""

    var leaf_estimation_iterations: Int
    """`leaf_estimation_iterations`. Default 1 for RMSE
    (`catboost_options.cpp:61`, then `:316-321`). HONORED only at 1: the leaf
    estimator is the single Newton step, which is the branch their walker
    takes at `Iterations == 1` before any backtracking exists
    (`descent_helpers.cpp:149-154`). `check()` refuses anything larger.

    `leaf_estimation_backtracking` (default `AnyImprovement`,
    `oblivious_tree_options.cpp:21`) is deliberately absent rather than
    present-and-ignored: at one iteration their walker returns before a step
    estimator is ever constructed, so the option has no effect to honor."""

    var random_strength: Float32
    """`random_strength`. **CatBoost's default is 1.0 and OURS IS 0.0**
    (`oblivious_tree_options.cpp:17`), one of only three places this port's
    default differs from theirs; the others are `bootstrap_type` and
    `determinism`.

    Not a preference: no score noise is applied here, so 1.0 would be a
    default that `check()` refuses, and defaults that fail their own
    validation are how a library ships an unusable out-of-the-box
    configuration. 0.0 is the value that describes what actually happens."""

    var rsm: Float32
    """`rsm`, feature sampling rate. Default 1.0. NOT HONORED: every feature
    is scored every level. Refused below 1.0."""

    # --- boosting_options.cpp, which had NO representative here at all ------

    var learning_rate: Float32
    """`learning_rate`. **Default 0.03** (`boosting_options.cpp:10`).

    `doc_parallel_boosting.fit` defaulted to 0.3, a tenfold larger step than
    stock CatBoost, which changes every prediction of every fit that did not
    pass one explicitly. HONORED: it is their `step`, applied by
    `iterationModel.Rescale(step)` (`doc_parallel_boosting.h:389-391`) and
    folded into `add_model_value_kernel` here.

    CatBoost also RETUNES this from the data when the user set neither it nor
    the three leaf-estimation options (`options_helper.cpp:269-288`, a fitted
    curve in `iterationCount` and `learnObjectCount`). That retune is not
    ported: it needs the row count and the loss at option-resolution time, and
    substituting a guess for a fitted curve is exactly the silent deviation
    this file exists to prevent. 0.03 is the value their curve backs off to."""

    var iterations: Int
    """`iterations`. Default 1000 (`boosting_options.cpp:13`). HONORED as
    `fit`'s `n_estimators`, which is a required argument there rather than a
    defaulted one, so nothing can run on an unstated tree budget."""

    var boost_from_average: Bool
    """`boost_from_average`. Default false (`boosting_options.cpp:17`).
    HONORED at false, which is what the port does: `fit` seeds the cursor with
    zeros, matching their `StartingPoint`-less branch
    (`doc_parallel_boosting.h:180-186`). True would seed it with
    `CalcOptimumConstApprox` and is refused."""

    # --- bootstrap_options.h, and this one changes the tree -----------------

    var bootstrap_type: Int
    """`bootstrap` / `type`. **CatBoost's default is Bayesian and OURS IS No**,
    the third and largest place this port's default differs from theirs
    (`bootstrap_options.h:18`).

    Not a preference and not a tuning choice. `BootstrapAndFilter` runs once
    per tree BEFORE any derivative is computed
    (`weak_objective_impl.h:31-36`), multiplying every row's weight by a fresh
    random draw, so under stock defaults the histogram, the split scores and
    the leaf values are all computed from a reweighted target. Nothing in this
    port samples, so `No` is the value that describes what actually happens,
    and `check()` refuses the rest rather than accepting a name it discards.

    The consequence to state plainly: **a shipped-defaults run of this port is
    not a shipped-defaults run of CatBoost**, and any comparison between them
    has to say so."""

    var bagging_temperature: Float32
    """`bagging_temperature`. Default 1.0 (`bootstrap_options.h:16`). Belongs
    to Bayesian bootstrap only. NOT HONORED, and inert while `bootstrap_type`
    is `No`. Carried at their value so the struct is theirs field for field."""

    var subsample: Float32
    """`subsample`. Default 0.66 (`bootstrap_options.h:15`). Belongs to
    Bernoulli, Poisson and MVS. NOT HONORED, and inert while `bootstrap_type`
    is `No`. Their own validator refuses it beside Bayesian
    (`bootstrap_options.cpp:15-19`), so at their default it is unreachable
    too."""

    var determinism: Int
    """NO CATBOOST COUNTERPART. See the three constants above. Default
    `DETERMINISM_DEVICE`, because a library that returns a different model on
    a rerun should have to be asked for that.

    What that default COSTS is unrecorded. It forces the fixed-point integer
    accumulator on every backend, including Apple, where a working float
    `atomicAdd` has since been probed and where the `FAST` arm is being moved
    onto it. No interleaved measurement of the two flushes exists yet, so this
    docstring names the cost rather than pricing it."""

    def l2_leaf_reg_effective(self) -> Float32:
        """`l2_leaf_reg`, with CatBoost's zero substitution applied.

            if (treeConfig.L2Reg == 0.0f) { treeConfig.L2Reg = 1e-20f; }

        -- `catboost_options.cpp:357-359`. They perform this once while
        resolving defaults, so every consumer downstream of that point sees
        the substituted value; here it is a call because the options struct is
        immutable at the point the training path reads it.
        """
        if self.l2_leaf_reg == Float32(0.0):
            return Float32(1e-20)
        return self.l2_leaf_reg

    @staticmethod
    def default() -> Self:
        """CatBoost's defaults, with THREE deliberate departures.

        `random_strength` is 0.0 rather than 1.0 and `bootstrap_type` is `No`
        rather than `Bayesian`, because neither feature is ported and each
        would fail `check()` at CatBoost's value. `determinism` has no
        CatBoost counterpart at all and defaults to `device`.

        Everything else is theirs, at the line that DECIDES the value rather
        than the line that names it: depth 6, SymmetricTree, max_leaves
        `1 << depth` = 64 (`catboost_options.cpp:993-1001`), min_data_in_leaf
        1, l2_leaf_reg 3.0, score_function Cosine, model_size_reg 0.5,
        border_count 128 for GPU (`data_processing_options.cpp:14-19`),
        leaf_estimation_method Newton for RMSE
        (`catboost_options.cpp:59-64`), leaf_estimation_iterations 1, rsm 1.0,
        learning_rate 0.03 (`boosting_options.cpp:10`), iterations 1000,
        boost_from_average false, bagging_temperature 1.0, subsample 0.66.
        """
        return Self(
            6, GROW_SYMMETRIC, 1 << 6, 1, 3.0, SCORE_FUNCTION_COSINE, 0.5,
            128, LEAF_ESTIMATION_NEWTON, 1, 0.0, 1.0,
            0.03, 1000, False,
            BOOTSTRAP_NO, 1.0, 0.66,
            DETERMINISM_DEVICE,
        )

    def check(self) raises:
        """Refuse what is not honored, by name.

        The rule this enforces: an option that is present and ignored is
        worse than one that is absent, because absent fails loudly and
        ignored fails silently.
        """
        if self.grow_policy != GROW_SYMMETRIC:
            raise Error(
                "grow_policy="
                + grow_policy_name(self.grow_policy)
                + " is not ported; only SymmetricTree is implemented, and"
                " growing a symmetric tree under another policy's name would"
                " be a silently wrong model"
            )
        if self.depth < 1 or self.depth > 16:
            raise Error("depth must be in [1, 16]; got " + String(self.depth))
        # `CB_ENSURE(MaxLeaves == 1u << MaxDepth, "max_leaves option works
        # only with lossguide tree growing")` -- `catboost_options.cpp:993`.
        # Theirs is a hard refusal for every policy but Lossguide, and the
        # value is not merely cosmetic: `ShouldTerminate` stops the tree at
        # `leafCount >= MaxLeaves` (`greedy_search_helper.cpp:678-683`), so
        # the option's constructed 31 would cap a depth-6 symmetric tree at
        # 32 leaves, which is depth 5.
        if self.max_leaves != (1 << self.depth):
            raise Error(
                "max_leaves option works only with lossguide tree growing;"
                " for SymmetricTree CatBoost requires max_leaves == 1 <<"
                " depth, which is "
                + String(1 << self.depth)
                + " at depth "
                + String(self.depth)
                + "; got "
                + String(self.max_leaves)
            )
        if self.leaf_estimation_method != LEAF_ESTIMATION_NEWTON:
            raise Error(
                "leaf_estimation_method="
                + leaf_estimation_method_name(self.leaf_estimation_method)
                + " is not ported; only Newton has an implementation, and"
                " Newton is CatBoost's own default for RMSE"
            )
        if self.leaf_estimation_iterations != 1:
            raise Error(
                "leaf_estimation_iterations="
                + String(self.leaf_estimation_iterations)
                + " is not ported; the leaf estimator is a single Newton"
                " step and the descent loop is not written"
            )
        if self.random_strength != 0.0:
            raise Error(
                "random_strength is not ported; no score noise is applied,"
                " so set it to 0.0 rather than believing scores were"
                " randomized"
            )
        if self.rsm != 1.0:
            raise Error(
                "rsm is not ported; every feature is scored at every level,"
                " so set it to 1.0"
            )
        if (
            self.score_function != SCORE_FUNCTION_COSINE
            and self.score_function != SCORE_FUNCTION_NEWTON_COSINE
            and self.score_function != SCORE_FUNCTION_L2
            and self.score_function != SCORE_FUNCTION_NEWTON_L2
        ):
            raise Error(
                "score_function="
                + score_function_name(self.score_function)
                + " is not ported; only Cosine, NewtonCosine, L2 and"
                " NewtonL2 have a calcer in the score kernel"
            )
        if self.model_size_reg != 0.5:
            raise Error(
                "model_size_reg is not ported; UpdateFeatureWeightsForBest"
                "Splits is not written, so every feature weight is 1.0 and"
                " any value but the default 0.5 would be silently discarded."
                " NOTE that at 0.5 this is no longer a no-op either once a"
                " fit has CTR columns -- see the option's docstring"
            )
        if self.min_data_in_leaf != 1:
            raise Error(
                "min_data_in_leaf is not ported; CatBoost itself ignores it"
                " under SymmetricTree (greedy_search_helper.cpp:691-694) and"
                " the score kernel here applies no minimum-count test either,"
                " so it is refused rather than accepted and discarded"
            )
        # The three boosting options. `learning_rate` and `iterations` are
        # honored, `boost_from_average` is honored only at CatBoost's own
        # default of false (`boosting_options.cpp:17`).
        if self.learning_rate <= 0.0:
            raise Error(
                "learning_rate must be positive; got "
                + String(self.learning_rate)
            )
        if self.iterations < 1:
            raise Error(
                "Iterations count should be positive; got "
                + String(self.iterations)
            )
        if self.boost_from_average:
            raise Error(
                "boost_from_average is not ported; the cursor is seeded with"
                " zeros, and seeding it with CalcOptimumConstApprox"
                " (doc_parallel_boosting.h:145-156) is a different model, not"
                " a different starting guess"
            )
        # Bootstrap. Theirs defaults to Bayesian at temperature 1.0 and
        # reweights every row once per tree BEFORE the derivatives are taken
        # (`weak_objective_impl.h:31-36`), so accepting the name and skipping
        # the sampling would change every split silently.
        if self.bootstrap_type != BOOTSTRAP_NO:
            raise Error(
                "bootstrap_type="
                + bootstrap_type_name(self.bootstrap_type)
                + " is not ported; no row sampling happens anywhere in this"
                " tree, so set it to No rather than believing the target was"
                " reweighted. CatBoost's own default is Bayesian, which means"
                " a defaults run here is NOT a defaults run of CatBoost"
            )
        if self.bagging_temperature < 0.0:
            raise Error("Bagging temperature should be >= 0")
        if self.subsample <= 0.0 or self.subsample > 1.0:
            raise Error("Subsample should be in (0,1]")
        if self.determinism < DETERMINISM_OFF or (
            self.determinism > DETERMINISM_CROSS_DEVICE
        ):
            raise Error("determinism must be off, device or cross_device")


# ==========================================================================
# CATEGORICAL FEATURES AND CTRs
# ==========================================================================
#
# MIRRORS `private/libs/options/cat_feature_options.{h,cpp}` and the two
# functions of `catboost_options.cpp` that resolve their defaults:
# `CreateDefaultCounter` (`:392-415`) and `SetCtrDefaults` (`:429-478`).
#
# THE ONE THING TO READ IF YOU READ NOTHING ELSE. Their GPU `simple_ctr`
# default is TWO descriptions and the first of them has THREE priors, so a
# categorical feature becomes FOUR numeric columns:
#
#     Borders      priors {0,1},{0.5,1},{1,1}   ctr_binarization Uniform 15
#     FeatureFreq  prior  {0.0,1}               ctr_binarization MinEntropy 15
#
# Both binarizations are real defaults on real code paths and they are
# DIFFERENT. `Uniform, 15` is the two-argument `TCtrDescription`
# constructor's default (`cat_feature_options.cpp:167-170`), which is what
# the Borders description is built with; `MinEntropy, 15` is passed
# explicitly by `CreateDefaultCounter` for a SimpleCtr projection and
# re-applied by `SetDefaultBinarizationsIfNeeded` to FeatureFreq
# descriptions ONLY (`:418-427`, the guard is
# `description.Type.Get() == ECtrType::FeatureFreq`). Reading either as
# "the CTR binarizer" is wrong, and this repository has now made that
# mistake in both directions.


# --- EProjectionType ------------------------------------------------------
#
# The argument `CreateDefaultCounter` switches on, and the only thing that
# separates the SIMPLE-ctr FeatureFreq grid (MinEntropy) from the TREE-ctr
# one (Median).

comptime PROJECTION_TREE_CTR = 0
comptime PROJECTION_SIMPLE_CTR = 1


# --- ECounterCalc ---------------------------------------------------------
#
# Decides WHICH FeatureFreq calcer runs. `SkipTest` is the default
# (`cat_feature_options.cpp:233`) and sends the work to
# `TWeightedBinFreqCalcer`; `Full` sends it to `TCtrBinBuilder`'s pure-freq
# arm. Both are implemented in `gbdt/ctrs/`, and they agree exactly on a fit
# with no test pool.

comptime COUNTER_CALC_FULL = 0
comptime COUNTER_CALC_SKIP_TEST = 1


def counter_calc_method_name(m: Int) -> String:
    if m == COUNTER_CALC_FULL:
        return String("Full")
    return String("SkipTest")


@fieldwise_init
struct TCtrDescription(Copyable, Movable):
    """`NCatboostOptions::TCtrDescription` (`cat_feature_options.cpp:139-176`).

    FOUR constructors chain in their source and the chain is where the
    defaults come from, so it is worth spelling out which one supplies what:

        (type, priors, ctrBinarization, targetBinarization)  the base
        (type, priors, ctrBinarization) -> base with
            targetBinarization = TBinarizationOptions(MinEntropy, 1)
        (type, priors) -> the above with
            ctrBinarization = TBinarizationOptions(Uniform, 15)
        (type) -> (type, {})

    `PriorEstimation` is `EPriorEstimation::No` in every one of them and is
    not carried here: the GPU refuses anything else for every ctr type but
    Borders (`catboost_options.cpp:524-533`), and no estimator is ported.
    """

    var ctr_type: Int
    var priors: List[TPrior]
    var ctr_binarization: TBinarizationOptions
    var target_binarization: TBinarizationOptions

    @staticmethod
    def with_priors(ctr_type: Int) raises -> Self:
        """Their two-argument constructor `TCtrDescription(type,
        GetDefaultPriors(type))` (`cat_feature_options.cpp:167-170`), which
        is exactly how `SetCtrDefaults` builds the Borders description --
        and therefore where `Uniform, 15` enters."""
        return Self(
            ctr_type,
            get_default_priors(ctr_type),
            TBinarizationOptions(BORDER_SELECTION_UNIFORM, 15),
            TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, 1),
        )


def create_default_counter(projection_type: Int) raises -> TCtrDescription:
    """`TCatBoostOptions::CreateDefaultCounter`
    (`catboost_options.cpp:392-415`), the GPU branch.

    Their CPU branch returns `TCtrDescription(Counter,
    GetDefaultPriors(Counter))`, which is the CPU's spelling of the same
    counts under a different normalization -- and the reason a local
    CatBoost CPU arm can only be an information-matched comparison for our
    FeatureFreq columns, never a bitwise oracle. This is the GPU port, so
    this returns the GPU branch and `Counter` never appears.
    """
    var border_selection_type: Int
    if projection_type == PROJECTION_TREE_CTR:
        border_selection_type = BORDER_SELECTION_MEDIAN
    elif projection_type == PROJECTION_SIMPLE_CTR:
        border_selection_type = BORDER_SELECTION_MIN_ENTROPY
    else:
        raise Error("Unknown projection type " + String(projection_type))
    return TCtrDescription(
        CTR_FEATURE_FREQ,
        get_default_priors(CTR_FEATURE_FREQ),
        TBinarizationOptions(border_selection_type, 15),
        TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, 1),
    )


struct TCatFeatureParams(Copyable, Movable):
    """`NCatboostOptions::TCatFeatureParams` (`cat_feature_options.cpp:226-239`)
    with the fields this port can act on.

    `PerFeatureCtrs`, `StoreAllSimpleCtrs`, `CtrLeafCountLimit` and
    `CtrHistoryUnit` are deliberately ABSENT rather than present and
    ignored, which is this file's standing rule. Each would need machinery
    that does not exist: per-feature descriptions need a feature id map, the
    leaf-count limit bounds a tree-ctr cache that is not built, and
    `CtrHistoryUnit::Group` needs the four groupwise kernels
    `gbdt/ctrs/kernel/ctr_calcers.mojo` documents as unported.
    """

    var simple_ctrs: List[TCtrDescription]
    """`simple_ctrs`. Default on GPU is Borders (three priors) plus
    `CreateDefaultCounter(SimpleCtr)` (`catboost_options.cpp:449-452`)."""

    var combination_ctrs: List[TCtrDescription]
    """`combinations_ctrs`. Same pair with `CreateDefaultCounter(TreeCtr)`.
    NOT HONORED: feature combinations are not ported, and `check()` refuses
    a `max_ctr_complexity` above 1 rather than accepting the descriptions
    and computing nothing from them."""

    var target_binarization: TBinarizationOptions
    """`target_binarization`, whose plain-options spelling is
    `ctr_target_border_count` (`plain_options_helper.cpp:421`). Default
    `(MinEntropy, 1)` (`cat_feature_options.cpp:230`).

    HONORED: `gbdt/ctrs/ctr_binarization.build_target_borders` is one call
    into the MinEntropy DP that `pixi run check-minentropy` already gates
    against CatBoost's own borders.

    ONE grid for the whole fit. The GPU refuses a per-CTR override outright
    (`catboost_options.cpp:505`), which is why `TCtrDescription` carries a
    `target_binarization` field that this struct's value overrides."""

    var max_tensor_complexity: Int
    """`max_ctr_complexity`. CatBoost's default is 4
    (`cat_feature_options.cpp:231`). **OURS IS 1**, because feature
    combinations are not ported; `check()` refuses anything larger rather
    than silently computing simple CTRs under a name that promises
    combinations of up to four features."""

    var one_hot_max_size: Int
    """`one_hot_max_size`. Default 2 on GPU (`cat_feature_options.cpp:232`).
    HONORED: `train()` takes a per-feature `one_hot` flag list, and a
    feature at or below this cardinality gets equality splits instead of
    CTRs -- their dispatch, where a one-hot feature never gets a CTR."""

    var counter_calc_method: Int
    """`counter_calc_method`. Default `SkipTest`
    (`cat_feature_options.cpp:233`). HONORED, and both sides are
    implemented (see the constants above); with no test pool they agree
    exactly, which is what makes the default safe to exercise on either
    arm."""

    def __init__(
        out self,
        var simple_ctrs: List[TCtrDescription],
        var combination_ctrs: List[TCtrDescription],
        target_binarization: TBinarizationOptions,
        max_tensor_complexity: Int,
        one_hot_max_size: Int,
        counter_calc_method: Int,
    ):
        self.simple_ctrs = simple_ctrs^
        self.combination_ctrs = combination_ctrs^
        self.target_binarization = target_binarization
        self.max_tensor_complexity = max_tensor_complexity
        self.one_hot_max_size = one_hot_max_size
        self.counter_calc_method = counter_calc_method

    @staticmethod
    def default() raises -> Self:
        """`SetCtrDefaults`'s `default:` branch
        (`catboost_options.cpp:449-452`) for the GPU task type, plus their
        constructor's values for the rest.

        The loss switch above it sends PairLogit and PairLogitPairwise to a
        counter-only pair; neither is a ported loss, so the `default:`
        branch is the only one reachable here.

        ONE DEPARTURE, the same shape as `bootstrap_type`'s:
        `max_tensor_complexity` is 1 rather than CatBoost's 4, because
        combinations are not ported and 4 is a value `check()` would refuse.
        A default that fails its own validation is how a library ships an
        unusable out-of-the-box configuration.
        """
        var simple = List[TCtrDescription]()
        simple.append(TCtrDescription.with_priors(CTR_BORDERS))
        simple.append(create_default_counter(PROJECTION_SIMPLE_CTR))

        var combos = List[TCtrDescription]()
        combos.append(TCtrDescription.with_priors(CTR_BORDERS))
        combos.append(create_default_counter(PROJECTION_TREE_CTR))

        return Self(
            simple^,
            combos^,
            TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, 1),
            1,
            2,
            COUNTER_CALC_SKIP_TEST,
        )

    @staticmethod
    def feature_freq_only() raises -> Self:
        """`default()` WITH THE Borders DESCRIPTION REMOVED. An OPT-IN
        surface, not the fallback: `train()` falls back to `default()`,
        which is CatBoost's own GPU `simple_ctr`, and the two sentences
        that used to stand here saying otherwise are deleted rather than
        annotated.

        The first said this port had no permutation machinery, so the
        ordered statistic would have to run in ROW order -- a different and
        much worse estimator rather than a slower one.
        `gbdt/data/permutation.mojo` ported `TDataPermutation` on
        2026-08-21 (`PORTING.md` 55). The second said a `Borders` model
        could not carry its apply-time CTR tables, so the fallback stayed
        here to avoid shipping a default that trains and cannot score;
        `build_ctr_tables` grew the target-class histogram arm and that is
        no longer true either.

        What this configuration IS for: a fit restricted to FREQUENCY
        information, which is the arm the AMAZON quality row compares
        against CatBoost's CPU `Counter`, and the permutation-independent
        half of the port on its own. It is one column per categorical
        feature instead of four.

        The caveat that survives: this port builds ONE set of CTR columns
        where their loop builds `permutation_count` of them, so a `Borders`
        fit here carries more of the ordered statistic's noise than theirs.
        That is deviation 55a -- a quality difference on the same
        estimator, not a different one.

        Both surfaces are exercised: `mojo_only/ctr_device_check.mojo`,
        `mojo_only/ctr_apply_check.mojo` and `mojo_only/ctr_train_check.mojo`
        each run `default()` and this one side by side, because a switch
        with one side unexercised is an unchecked branch
        (`PORTING_RULES.md` 8).
        """
        var simple = List[TCtrDescription]()
        simple.append(create_default_counter(PROJECTION_SIMPLE_CTR))
        var combos = List[TCtrDescription]()
        combos.append(create_default_counter(PROJECTION_TREE_CTR))
        return Self(
            simple^,
            combos^,
            TBinarizationOptions(BORDER_SELECTION_MIN_ENTROPY, 1),
            1,
            2,
            COUNTER_CALC_SKIP_TEST,
        )

    def simple_ctr_configs(self) raises -> List[TCtrConfig]:
        """`TBinarizedFeaturesManager::CreateCtrConfigsFromDescription`
        (`cuda/data/binarizations_manager.cpp:394-433`), for the simple-ctr
        descriptions and ONE cat feature.

        **This is the function that turns descriptions into COLUMNS**, and
        the fan-out lives in two places at once:

        * every prior gets its own config (`:395`, the outer loop);
        * `Buckets` and `Borders` additionally get one config per target
          bin, `numBins` being `TargetBorders.size() + 1` for Buckets and
          `TargetBorders.size()` for Borders (`:415-419`), with the
          0-class skipped for BINARY Buckets (`:422-424`).

        At the GPU default -- one target border -- Borders emits ParamId 0
        only, so the count is 3 priors x 1 bin + 1 FeatureFreq = FOUR
        columns per categorical feature.

        `CreateCtrConfigsFromDescription`'s first act is to `continue` on
        any target-needing type when `!HasTargetBinarization()`
        (`:397-399`); that is `need_target_classifier` here, and the guard
        is why a fit with no target grid produces frequency columns only
        rather than raising.
        """
        var target_border_count = self.target_binarization.border_count
        var out = List[TCtrConfig]()
        for d in range(len(self.simple_ctrs)):
            ref desc = self.simple_ctrs[d]
            if need_target_classifier(desc.ctr_type) and (
                target_border_count == 0
            ):
                continue
            for p in range(len(desc.priors)):
                var prior = desc.priors[p]
                # `if (defaultConfig.Prior.size() == 1) Prior.push_back(1)`
                # (`:407-409`) -- our TPrior is always a pair, so their
                # one-element case cannot arise.
                if desc.ctr_type == CTR_BUCKETS or (
                    desc.ctr_type == CTR_BORDERS
                ):
                    var num_bins: Int
                    if desc.ctr_type == CTR_BUCKETS:
                        num_bins = target_border_count + 1
                    else:
                        num_bins = target_border_count
                    for i in range(num_bins):
                        # "don't calc 0-class ctr for binary
                        # classification, it's unneeded" (`:422-424`)
                        if (
                            i == 0
                            and num_bins == 2
                            and desc.ctr_type == CTR_BUCKETS
                        ):
                            continue
                        out.append(TCtrConfig(desc.ctr_type, prior, i, d))
                else:
                    out.append(TCtrConfig(desc.ctr_type, prior, 0, d))
        return out^

    def ctr_binarization_for(
        self, config: TCtrConfig
    ) -> TBinarizationOptions:
        """The grid a given output column is quantized with.

        `TCtrConfig::CtrBinarizationConfigId` indexes the manager's list of
        distinct binarization descriptions (`GetOrCreateCtrBinarizationId`,
        `binarizations_manager.cpp:435-446`); here it indexes the
        description that minted the config, which is the same mapping with
        no deduplication step. Dedup saves memory in their manager and
        decides nothing.
        """
        return self.simple_ctrs[
            config.ctr_binarization_config_id
        ].ctr_binarization

    def check(self) raises:
        """Refuse what is not honored, by name -- the same rule the rest of
        this file follows."""
        if self.max_tensor_complexity != 1:
            raise Error(
                "max_ctr_complexity="
                + String(self.max_tensor_complexity)
                + " is not ported; feature combinations (tree CTRs) need"
                " their own tree_ctr_datasets_visitor machinery, so set it"
                " to 1 rather than believing combinations were built."
                " CatBoost's own default is 4, which means a defaults run"
                " here is NOT a defaults run of CatBoost"
            )
        if self.one_hot_max_size < 0:
            raise Error("one_hot_max_size must be non-negative")
        if self.counter_calc_method != COUNTER_CALC_SKIP_TEST and (
            self.counter_calc_method != COUNTER_CALC_FULL
        ):
            raise Error("counter_calc_method must be SkipTest or Full")
        if (
            self.target_binarization.border_selection_type
            != BORDER_SELECTION_MIN_ENTROPY
        ):
            raise Error(
                "target_binarization border_type="
                + border_selection_name(
                    self.target_binarization.border_selection_type
                )
                + " is not ported; CatBoost's default is MinEntropy with one"
                " border (cat_feature_options.cpp:230) and the GPU takes its"
                " count from ctr_target_border_count, a per-CTR override"
                " being refused outright (catboost_options.cpp:505)"
            )
        if self.target_binarization.border_count < 1:
            raise Error(
                "ctr_target_border_count must be at least 1; got "
                + String(self.target_binarization.border_count)
            )
        for i in range(len(self.simple_ctrs)):
            ref d = self.simple_ctrs[i]
            # `CB_ENSURE(IsSupportedCtrType(ETaskType::GPU, ctrType))`
            # (`catboost_options.cpp:503`)
            if not is_supported_ctr_type_gpu(d.ctr_type):
                var extra = String("")
                if d.ctr_type == CTR_COUNTER:
                    extra = String(
                        "; Counter is CatBoost's CPU spelling of the same"
                        " counts and is not a GPU ctr type at all"
                        " (restrictions.h:33-43) -- use FeatureFreq"
                    )
                raise Error(
                    "Ctr type "
                    + ctr_type_name(d.ctr_type)
                    + " is not implemented on GPU yet"
                    + extra
                )
            if d.ctr_type != CTR_BORDERS and d.ctr_type != CTR_FEATURE_FREQ:
                raise Error(
                    "simple_ctr="
                    + ctr_type_name(d.ctr_type)
                    + " is not ported; only Borders and FeatureFreq have a"
                    " calcer in gbdt/ctrs/"
                )
            if len(d.priors) == 0:
                raise Error("Provide at least one prior for CTR")
            # Their `ValidateCtr` WARNS rather than refuses here, and the
            # warning is worth carrying verbatim because it names both
            # grids (`catboost_options.cpp:537-539`).
            if (
                d.ctr_type == CTR_FEATURE_FREQ
                and d.ctr_binarization.border_selection_type
                == BORDER_SELECTION_UNIFORM
            ):
                print(
                    "warning: Uniform ctr binarization for featureFreq ctr"
                    " is not good choice. Use MinEntropy for simpleCtrs and"
                    " Median for combinations-ctrs instead"
                )


# =========================================================================
# THE LEAF-ESTIMATION DEFAULTS.
#
# PORT OF `GetEstimationMethodDefaults` (`catboost_options.cpp:30-271`) and
# `TCatBoostOptions::SetLeavesEstimationDefault` (`:273-360`).
#
# THEY LIVE HERE BECAUSE THEY LIVE THERE. Both were written into
# `gbdt/options/loss_description.mojo` first, beside the loss parameters
# they read, and that was a mirror violation: `loss_description.cpp` holds
# the PARAMETER ACCESSORS and nothing else, and these two are
# `catboost_options.cpp`'s. PORTING_RULES 4 -- their paths are our paths --
# is not a filing preference, it is what makes a reviewer able to open one
# of their files beside one of ours and diff branch for branch.
# =========================================================================


@fieldwise_init
struct TLeavesEstimationDefaults(Copyable, Movable):
    """The tuple their helper returns (`catboost_options.cpp:30-271`)."""

    var newton_iterations: Int
    var gradient_iterations: Int
    var estimation_method: Int
    var l2_reg: Float32


def get_estimation_method_defaults(
    loss: TLossDescription,
) raises -> TLeavesEstimationDefaults:
    """`GetEstimationMethodDefaults(ETaskType::GPU, loss)`.

    Transcribed branch for branch from their switch, GPU arm taken at every
    `taskType` test, for the twelve objectives this port trains. Their
    remaining cases are ranking, multi-target and user-defined losses that
    do not reach this file; each is `NOT PORTED` rather than defaulted,
    which is why the tail raises instead of falling through to the initial
    values.
    """
    # Their four initializers (`:34-37`) are `1, 1, Newton, 3.0`. Every
    # branch below assigns all four, because the twelve objectives this
    # port trains are exactly the ones whose cases are complete in their
    # switch; declaring without initializing keeps that fact checkable by
    # the compiler instead of hiding a missed assignment behind a default.
    var newton: Int
    var gradient: Int
    var method: Int
    var l2 = Float32(3.0)
    var f = loss.loss_function

    if f == OBJECTIVE_RMSE:
        # `:59-64`
        method = LEAF_ESTIMATION_NEWTON
        newton = 1
        gradient = 1
    elif f == OBJECTIVE_LQ:
        # `:82-93`: the method depends on the PARAMETER, not the loss
        method = LEAF_ESTIMATION_NEWTON
        if loss.get_lq_param() < Float32(2.0):
            method = LEAF_ESTIMATION_GRADIENT
        newton = 1
        gradient = 1
    elif (
        f == OBJECTIVE_MAE
        or f == OBJECTIVE_MAPE
        or f == OBJECTIVE_QUANTILE
        or f == OBJECTIVE_LOGLINQUANTILE
    ):
        # `:113-124`
        method = LEAF_ESTIMATION_GRADIENT
        newton = 1
        gradient = 1
    elif f == OBJECTIVE_EXPECTILE:
        # `:125-132`
        if not loss.has_alpha:
            raise Error("Param alpha is mandatory for expectile loss")
        newton = 5
        gradient = 10
        method = LEAF_ESTIMATION_NEWTON
    elif f == OBJECTIVE_POISSON:
        # `:151-156`
        method = LEAF_ESTIMATION_NEWTON
        newton = 10
        gradient = 1
    elif f == OBJECTIVE_LOGLOSS or f == OBJECTIVE_CROSSENTROPY:
        # `:157-164`
        newton = 10
        gradient = 40
        method = LEAF_ESTIMATION_NEWTON
    elif f == OBJECTIVE_HUBER:
        # `:187-192`
        method = LEAF_ESTIMATION_NEWTON
        newton = 1
        gradient = 1
    elif f == OBJECTIVE_MULTICLASS:
        # `:106-112`. NEWTON AT ONE ITERATION, which is what makes the
        # blocked Cholesky worth having: one solve per leaf, once per tree.
        method = LEAF_ESTIMATION_NEWTON
        newton = 1
        gradient = 10
    elif f == OBJECTIVE_TWEEDIE:
        # `:221-231`. THE GPU ARM: twenty iterations, where their CPU
        # takes one. We are a GPU, so twenty.
        _ = loss.get_tweedie_param()
        method = LEAF_ESTIMATION_NEWTON
        newton = 20
        gradient = 20
    else:
        raise Error(
            "no leaf-estimation default for loss '" + loss.name() + "'"
        )

    return TLeavesEstimationDefaults(newton, gradient, method, l2)


def use_exact_leaves(loss: TLossDescription) -> Bool:
    """Their `useExact` (`catboost_options.cpp:289-295`), GPU + Plain arm.

        const bool useExact = EqualToOneOf(loss, MAE, MAPE, RMSPE,
                Quantile, GroupQuantile, MultiQuantile)
            && SystemOptions->IsSingleHost()
            && ((TaskType == GPU && BoostingType == Plain) || ...)

    Both trailing conjuncts are constants here. `IsSingleHost` is true --
    this port has one device and no distributed mode. `BoostingType` is
    Plain -- `gbdt/methods/doc_parallel_boosting.mojo` IS their plain
    doc-parallel loop and ordered boosting is NOT PORTED, so there is no
    configuration in this tree where the test could go the other way.

    NOTE WHAT IS NOT ON THE LIST: `LogLinQuantile`. It shares the Gradient
    default with the other quantile losses and does NOT get upgraded to
    Exact, because their exact estimator solves a weighted quantile of the
    TARGETS and LogLinQuantile's residual is `target - exp(prediction)`.
    Reading their list as "the quantile family" instead of as the six
    literal names would put a wrong estimator on it.

    A reader comparing against CatBoost's own GPU defaults should know that
    their GPU picks ORDERED boosting by default for these losses
    (`catboost_options.cpp:802-807`), under which their own `useExact` is
    false and Gradient stands. Their CPU -- the arm this port is measured
    against, because their GPU does not run on this machine -- takes the
    Exact branch (`:293`), so Exact is what the comparison needs.
    """
    var f = loss.loss_function
    return (
        f == OBJECTIVE_MAE
        or f == OBJECTIVE_MAPE
        or f == OBJECTIVE_QUANTILE
    )


@fieldwise_init
struct TResolvedLeavesEstimation(Copyable, Movable):
    """What the boosting loop actually needs: a method and a count."""

    var method: Int
    var iterations: Int
    var l2_reg: Float32


def set_leaves_estimation_default(
    loss: TLossDescription,
    method_override: Int = -1,
    iterations_override: Int = -1,
    l2_override: Float32 = Float32(-1.0),
) raises -> TResolvedLeavesEstimation:
    """`TCatBoostOptions::SetLeavesEstimationDefault` (`:273-360`).

    Their `TOption` carries "is set" with the value, so the body reads
    `NotSet()` / `SetDefault()`; ours takes explicit overrides with a
    sentinel meaning unset, which is the same two states.

    The order is theirs and it matters: `useExact` REPLACES the method the
    switch chose and resets both iteration counts to one (`:296-300`)
    BEFORE the override is consulted, so an explicit
    `leaf_estimation_iterations` still wins over the exact default.
    """
    var d = get_estimation_method_defaults(loss)
    var method = d.estimation_method
    var newton = d.newton_iterations
    var gradient = d.gradient_iterations

    if use_exact_leaves(loss):
        # `:296-300`
        method = LEAF_ESTIMATION_EXACT
        newton = 1
        gradient = 1

    var l2 = d.l2_reg
    if l2_override >= Float32(0.0):
        l2 = l2_override

    if method_override >= 0:
        method = method_override

    var iterations: Int
    if method == LEAF_ESTIMATION_NEWTON:
        iterations = newton
    elif method == LEAF_ESTIMATION_GRADIENT:
        iterations = gradient
    else:
        # `:326-330`: Exact and Simple are one iteration, always
        iterations = 1

    if iterations_override >= 0:
        iterations = iterations_override

    if method == LEAF_ESTIMATION_SIMPLE and iterations != 1:
        # `:338-341`
        raise Error(
            "Leaves estimation iterations can't be greater, than 1 for"
            " Simple leaf-estimation mode"
        )

    return TResolvedLeavesEstimation(method, iterations, l2)
