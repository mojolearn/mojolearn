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

    NOT HONORED, and an exact no-op at present: that function fills the
    weight vector with 1.0 and RETURNS EARLY when the CTR count is zero
    (`update_feature_weights.cpp:14-22`), and CTRs are not ported. The score
    kernel takes the weight buffer anyway so that this stops being a
    divergence the day CTRs land. `check()` refuses any other value, because
    any other value would be silently discarded."""

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
                "model_size_reg is not ported; feature weights are all 1.0"
                " because CTRs are not ported, so any value but the default"
                " 0.5 would be silently discarded"
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
