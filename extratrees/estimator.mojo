"""The callable surface: sklearn's parameter names, and refusal by name.

**Why this file exists.** `ported/randomforest/randomforest.mojo` already fits
a forest, but its arguments are cuML's: a `DecisionTreeParams` whose
`max_features` is a RATIO, an `n_trees`, a seed. A caller arriving from
scikit-learn holds `max_features='sqrt'`, `max_depth=None`,
`min_weight_fraction_leaf=0.0` and `monotonic_cst=None`, and has no way to know
which of those this port honours, which it silently ignores, and which do not
exist. That gap is what DEVIATION 154 recorded as a debt against whoever wrote
this layer, and this file is that layer.

Nothing here is a port. `ported/` mirrors cuML and is governed by COPY, DO NOT
IMPROVE; this file is host-side policy neither cuML nor sklearn has a
counterpart for, in the same category as `mojo_only/`. It follows
`cluster/estimator.mojo` and `neighbors/estimator.mojo`, which are the first
files of this kind in the tree.

THE ONE RULE IT ENFORCES
------------------------
**Every sklearn parameter is either honoured or REFUSED BY NAME.** None is
accepted and ignored. `gbdt/`'s `check()` refuses every unported CatBoost
option the same way and for the same reason: an option that is silently
dropped is indistinguishable, from the caller's side, from an option that
works, and the user finds out from a model that is quietly not what they asked
for.

`params_check` and `estimator_check` between them enumerate BOTH sides of every
switch in here, which rule 8 requires of a parameter that selects behaviour.

THE TWO PARAMETERS THAT NEEDED A DECISION
------------------------------------------
1. **`max_depth=None`.** sklearn grows until every leaf is pure. This port has
   no unlimited: cuML's own `validity_check` asserts `max_depth >= 0`
   (`decisiontree.cu:29`), and their documented `-1` default cannot survive it.
   Refusing `None` would reject sklearn's own default, and silently capping is
   the thing this file exists to prevent. So `None` is accepted and mapped to
   `ceil_log2(n_rows) + DEPTH_SLACK`, **and the result reports whether the cap
   actually bound.** A caller who hits it is told, rather than handed a
   shallower forest that looks fine.
2. **`max_features`.** sklearn accepts `'sqrt'`, `'log2'`, `None`, a float
   fraction or an int count, and resolves them to an integer count
   (`max(1, int(np.sqrt(n_features)))` and friends). This port takes a RATIO,
   which `n_sampled_cols_for` then truncates back to an integer
   (`builder.cuh:222`). Round-tripping an integer through a ratio and a
   truncation is exactly the kind of arithmetic that lands one short, so the
   count is resolved FIRST, in integers, and the ratio is nudged to the middle
   of its truncation bucket. `resolve_max_features` returns the COUNT so a
   check can compare against sklearn's rule directly rather than against a
   ratio.

THE DEVICE OPTION, AND WHY IT IS A SECOND ENTRY POINT
------------------------------------------------------
This file used to offer no device path at all, and the reason it gave was
DEVIATION 183: the device could not compute the float gain, so
`min_impurity_decrease` was ACCEPTED AND INERT there. A parameter honoured on
one path and ignored on the other is the exact thing this file exists to
prevent, so the device path could not be offered while that was true.

**183 is closed.** The gain is now formed on the host from the exact integers
the score pass already produced, `split_not_valid` applies unchanged, and
`device_tree_check` asserts the device tree equals the host tree at the DEFAULT
gate as well as with the gate disabled. So `min_impurity_decrease` is honoured
on both arms, every other parameter was already path-independent -- every one
of them is resolved by `resolve` BEFORE either trainer is reached -- and the
option can be offered.

**DEVIATION BLOCK -- DEVIATION 187. THE DEVICE IS AN ENTRY POINT, NOT A FIELD
ON `ExtraTreesConfig`.** A `device=True` field would be the obvious shape and
it is wrong twice over. `ExtraTreesConfig` is documented as *sklearn's
constructor, with its names and its defaults*, and sklearn has no such
parameter -- putting one there makes the struct a mix of sklearn's surface and
ours, and the next reader cannot tell which is which by looking. And a flag
cannot carry a `DeviceContext`: the caller must supply one, so the device arm
needs an argument the host arm does not have, which is a signature difference
no boolean can hide. `fit_extra_trees_classifier_device(ctx, ...)` is that
signature.

**The refusals cannot drift between the arms, structurally rather than by
discipline.** Both `fit_extra_trees_classifier` and its `_device` twin call
`classifier_plan`, which is the ONLY place the criterion check and `resolve`
(and therefore `refuse_unported`) are invoked, and both call `depth_cap_bound`
for the report. Neither arm has a line of policy of its own. A refusal added to
`refuse_unported` tomorrow reaches both without either being touched, and there
is no way to add one to a single arm without first splitting that helper.

**Two refusals fire on the device arm and not on the host arm**, and they are
the device trainer's own, raised BY NAME out of `train_classification_device`:
more classes than the score kernel's comptime shared-memory width admits
(DEVIATION 172), and more rows than the published `Int64` Gini numerator is
exact for (DEVIATION 175). Neither bound is restated here, because restating a
bound is how a copy drifts from its constant; they propagate out of the first
tree, before any kernel is enqueued. `device_forest_check` uses the first of
them as its REACH proof that this file's device arm really reaches the device
trainer -- the two arms produce identical forests, so no OUTPUT can tell them
apart, and only a device-exclusive refusal can.

**Regression has the same pair of arms.** DEVIATION 188 refused the device
regressor BY NAME while `train_regression_device` did not exist; deviation 189
gave the device reduction an exact MSE key and deviation 206 brought the
regression device path level with classification, so the refusal's reason is
gone and 188 is CLOSED. `fit_extra_trees_regressor_device(ctx, ...)` follows
187's shape exactly: both regressor arms call `regressor_plan` and nothing
else before their trainer, so a refusal cannot reach one arm and miss the
other. The one thing the device arm does that the host arm does not is
QUANTIZE the labels (deviation 135): the device trainer consumes fixed-point
integers, so this file derives the scale from the whole label vector's
magnitude sum -- the same derivation `device_regression_check` uses -- and the
leaf values it returns differ from the host arm's by at most one quantization
step, which is 135's ruling and not a defect.
"""

from extratrees.mojo_only.fixed_point import ceil_log2, choose_scale, quantize
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_GINI,
    CRITERION_MSE,
    DecisionTreeParams,
    validity_check,
)
from extratrees.ported.randomforest.randomforest import (
    Forest,
    fit_classification,
    fit_classification_device,
    fit_regression,
    fit_regression_device,
)
from max.gpu.host import DeviceContext


comptime DEPTH_SLACK: Int32 = 16
"""How far past `ceil_log2(n_rows)` an unlimited-depth request is allowed.

A balanced tree over `n` rows is `log2(n)` deep; an unbalanced one can be
`n - 1`. The slack covers the unbalanced tail of any realistic fit without
allowing a pathological one to allocate for millions of levels, and the fit
REPORTS when the cap bound so the number is never load-bearing in silence.
"""

comptime MAX_FEATURES_SQRT: Int = -1
comptime MAX_FEATURES_LOG2: Int = -2
comptime MAX_FEATURES_ALL: Int = -3
"""Sentinels for sklearn's string forms. A positive value is an explicit count;
a value in `(0, 1]` passed to `max_features_fraction` is a fraction."""


def resolve_max_features(
    spec: Int, fraction: Float64, n_features: Int
) raises -> Int:
    """sklearn's `max_features` resolution, as an integer COUNT.

    From `sklearn/ensemble/_forest.py` and `sklearn/tree/_classes.py`:

        'sqrt'  -> max(1, int(np.sqrt(n_features)))
        'log2'  -> max(1, int(np.log2(n_features)))
        None    -> n_features
        float f -> max(1, int(f * n_features))
        int   k -> k

    `int()` truncates in every one of those, which is why this is written in
    integers throughout: `int(np.sqrt(9))` is 3, and a float pipeline that
    computed 2.9999997 would give 2.
    """
    if n_features < 1:
        raise Error("n_features must be >= 1; got " + String(n_features))
    if spec == MAX_FEATURES_ALL:
        return n_features
    if spec == MAX_FEATURES_SQRT:
        # integer isqrt: the largest k with k*k <= n
        var k = 1
        while (k + 1) * (k + 1) <= n_features:
            k += 1
        return k if k >= 1 else 1
    if spec == MAX_FEATURES_LOG2:
        # int(log2(n)) is floor(log2(n)) for n >= 1
        var k = 0
        var acc = 1
        while acc * 2 <= n_features:
            acc *= 2
            k += 1
        return k if k >= 1 else 1
    if spec > 0:
        if spec > n_features:
            raise Error(
                "max_features="
                + String(spec)
                + " exceeds n_features="
                + String(n_features)
            )
        return spec
    if spec == 0:
        # sklearn's fraction form arrives here.
        if not (fraction > 0.0 and fraction <= 1.0):
            raise Error(
                "max_features as a fraction must be in (0, 1]; got "
                + String(fraction)
            )
        var k = Int(fraction * Float64(n_features))
        return k if k >= 1 else 1
    raise Error("unknown max_features spec " + String(spec))


def count_to_ratio(count: Int, n_features: Int) -> Float32:
    """The count as the ratio `n_sampled_cols_for` will truncate back.

    `builder.cuh:222` computes `Int32(max_features * n_cols)`, so the ratio
    must land in the MIDDLE of the bucket that truncates to `count`, not on its
    lower edge: `count / n_features` can round below and truncate to
    `count - 1`.

    CLAMPED AT 1.0, and the clamp is not cosmetic. At `count == n_features` the
    mid-bucket value is `(n + 0.5) / n`, which is greater than one, and
    `validity_check` refuses `max_features > 1.0` because cuML does
    (`decisiontree.cu:33-35`). `estimator_check` caught exactly that on the
    `max_features=None` path -- the most common configuration there is. The
    clamp is safe because `Int32(1.0 * n) == n` exactly: 1.0 is representable
    and the product is an integer-valued float below 2^24 for any column count
    this port can hold.
    """
    var ratio = (Float32(count) + 0.5) / Float32(n_features)
    return 1.0 if ratio > 1.0 else ratio


@fieldwise_init
struct ExtraTreesConfig(ImplicitlyCopyable, Movable):
    """sklearn's constructor, with its names and its defaults.

    Defaults are `ExtraTreesClassifier`'s. `for_regression()` switches the two
    that differ (`criterion`, `max_features`).
    """

    var n_estimators: Int32
    var criterion: Int32
    var max_depth: Int32
    """-1 means sklearn's `None`. See DEPTH_SLACK."""
    var min_samples_split: Int32
    var min_samples_leaf: Int32
    var min_weight_fraction_leaf: Float64
    var max_features_spec: Int
    var max_features_fraction: Float64
    var max_leaf_nodes: Int32
    """-1 means sklearn's `None`."""
    var min_impurity_decrease: Float32
    var bootstrap: Bool
    var oob_score: Bool
    var random_state: UInt64
    var warm_start: Bool
    var ccp_alpha: Float64
    var has_class_weight: Bool
    var has_monotonic_cst: Bool
    var max_samples_set: Bool

    def __init__(out self):
        """`ExtraTreesClassifier()`'s defaults, name for name."""
        self.n_estimators = 100
        self.criterion = CRITERION_GINI
        self.max_depth = -1
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features_spec = MAX_FEATURES_SQRT
        self.max_features_fraction = 0.0
        self.max_leaf_nodes = -1
        self.min_impurity_decrease = 0.0
        self.bootstrap = False
        self.oob_score = False
        self.random_state = 0
        self.warm_start = False
        self.ccp_alpha = 0.0
        self.has_class_weight = False
        self.has_monotonic_cst = False
        self.max_samples_set = False

    def for_regression(self) -> Self:
        """`ExtraTreesRegressor()`'s defaults: the same, except `criterion`
        is `squared_error` and `max_features` is `1.0` (all of them)."""
        var out = self.copy()
        out.criterion = CRITERION_MSE
        out.max_features_spec = MAX_FEATURES_ALL
        return out^


def refuse_unported(config: ExtraTreesConfig) raises:
    """Every sklearn parameter this port does NOT honour, refused BY NAME.

    Each entry names the deviation or the `UNPORTED.tsv` row that explains it,
    so a caller gets a reason rather than a wall.
    """
    if config.min_weight_fraction_leaf != 0.0:
        raise Error(
            "min_weight_fraction_leaf is not ported: it is a fraction of the"
            " total SAMPLE WEIGHT, and sample_weight is unported (see"
            " UNPORTED.tsv and DEVIATION 144, whose exact integer comparison"
            " assumes unweighted counts). Only the default 0.0 is accepted."
        )
    if config.has_monotonic_cst:
        raise Error(
            "monotonic_cst is not ported: sklearn enforces it as a fourth"
            " rejection branch in the split search (_splitter.pyx:679-689)"
            " and cuML has no counterpart at all. DEVIATION 154."
        )
    if config.has_class_weight:
        raise Error(
            "class_weight is not ported: it becomes a per-sample weight, and"
            " sample_weight is unported. See UNPORTED.tsv."
        )
    if config.bootstrap:
        raise Error(
            "bootstrap=True is not ported: sklearn's ExtraTrees defaults to"
            " False and cuML's with-replacement row sampler"
            " (randomforest.cuh:64-67) has no caller here. See UNPORTED.tsv."
        )
    if config.oob_score:
        raise Error(
            "oob_score=True requires bootstrap=True, which is not ported;"
            " with bootstrap=False every tree sees every row and there is no"
            " out-of-bag set to score."
        )
    if config.max_samples_set:
        raise Error(
            "max_samples applies only when bootstrap=True, which is not"
            " ported. sklearn itself raises for this combination."
        )
    if config.warm_start:
        raise Error(
            "warm_start=True is not ported: there is no incremental fit here,"
            " and accepting it would silently refit from scratch."
        )
    if config.ccp_alpha != 0.0:
        raise Error(
            "ccp_alpha is not ported: cost-complexity pruning is a"
            " post-processing pass over a fitted tree that neither cuML nor"
            " this port implements. Only the default 0.0 is accepted."
        )
    if config.max_leaf_nodes != -1:
        raise Error(
            "max_leaf_nodes is not ported under sklearn's semantics."
            " sklearn's is a BEST-FIRST growth limit that changes the ORDER"
            " nodes are expanded in; cuML's max_leaves is a soft cap on a"
            " breadth-first frontier (builder.cuh:86-88) and stops pushing"
            " work once the counter is reached. They are different"
            " algorithms, so accepting the name would be accepting a"
            " different model. Use cuML's max_leaves through"
            " DecisionTreeParams if that is what you want."
        )
    if config.n_estimators < 1:
        raise Error(
            "n_estimators must be >= 1; got " + String(config.n_estimators)
        )


@fieldwise_init
struct FitPlan(ImplicitlyCopyable, Movable):
    """What `resolve` decided, reported rather than hidden."""

    var params: DecisionTreeParams
    var n_trees: Int32
    var max_features_count: Int
    var depth_was_unlimited: Bool
    """True when the caller passed sklearn's `max_depth=None` and this plan
    substituted a cap. `depth_cap_bound` after the fit says whether it
    mattered."""


def resolve(
    config: ExtraTreesConfig, n_rows: Int, n_features: Int
) raises -> FitPlan:
    """sklearn's configuration into cuML's, with every refusal applied first."""
    refuse_unported(config)

    var count = resolve_max_features(
        config.max_features_spec, config.max_features_fraction, n_features
    )

    var params = DecisionTreeParams()
    params.min_samples_split = config.min_samples_split
    params.min_samples_leaf = config.min_samples_leaf
    params.min_impurity_decrease = config.min_impurity_decrease
    params.split_criterion = config.criterion
    params.max_features = count_to_ratio(count, n_features)
    params.max_leaves = -1

    var unlimited = config.max_depth < 0
    if unlimited:
        params.max_depth = Int32(ceil_log2(n_rows if n_rows > 1 else 2)) + DEPTH_SLACK
    else:
        params.max_depth = config.max_depth

    validity_check(params)
    return FitPlan(params, config.n_estimators, count, unlimited)


@fieldwise_init
struct FitResult(Movable):
    """A fitted forest, and what the fit had to decide."""

    var forest: Forest
    var plan: FitPlan
    var depth_cap_bound: Bool
    """True when some tree reached `params.max_depth` exactly. On an unlimited
    request that means the substituted cap CHANGED the model, and the caller is
    told rather than handed a quietly shallower forest."""


def classifier_plan(
    config: ExtraTreesConfig, n_rows: Int32, n_features: Int32
) raises -> FitPlan:
    """Every check and every refusal a classifier fit applies, in ONE place.

    Both `fit_extra_trees_classifier` and `fit_extra_trees_classifier_device`
    call this and nothing else before their trainer. That is DEVIATION 187's
    guarantee made structural: there is no line of policy in either arm, so a
    refusal cannot reach one and miss the other, and adding one to a single arm
    would require splitting this function first.
    """
    if config.criterion != CRITERION_GINI:
        raise Error(
            "the classification criterion must be gini; entropy and the"
            " regression criteria are refused by validity_check"
        )
    return resolve(config, Int(n_rows), Int(n_features))


def depth_cap_bound(forest: Forest, plan: FitPlan) -> Bool:
    """Whether any tree reached `params.max_depth` exactly.

    Shared by both arms for the same reason `classifier_plan` is: on an
    unlimited-depth request this is the number that says whether the
    substituted cap CHANGED the model, and an arm that computed it differently
    would report a different model without fitting a different one.
    """
    var bound = False
    for t in range(len(forest.trees)):
        if forest.trees[t].depth_counter >= plan.params.max_depth:
            bound = True
    return bound


def fit_extra_trees_classifier(
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """`ExtraTreesClassifier.fit`, with sklearn's names honoured or refused."""
    var plan = classifier_plan(config, n_rows, n_features)
    var forest = fit_classification(
        x_col_major,
        labels,
        n_rows,
        n_features,
        n_classes,
        plan.params,
        plan.n_trees,
        config.random_state,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def fit_extra_trees_classifier_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    labels: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    n_classes: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """`ExtraTreesClassifier.fit` with the split search on the GPU.

    DEVIATION BLOCK -- DEVIATION 187, stated in full in the module docstring.
    The signature is `fit_extra_trees_classifier`'s plus a `DeviceContext`, and
    the body is the same three lines with `fit_classification_device` in place
    of `fit_classification`. Every sklearn parameter is resolved and refused by
    `classifier_plan` before either trainer is reached, so the two arms cannot
    honour different sets.

    **The forest it returns is the SAME forest the host arm returns**, tree for
    tree and node for node, because deviation 183 closed the last difference
    between the device and host trees. `device_forest_check` asserts that
    against this entry point, not only against the raw
    `fit_classification_device`.

    **What is NOT the same, and it is the device trainer's own refusals rather
    than this file's:** `train_classification_device` refuses a class count
    above the score kernel's comptime shared width (DEVIATION 172) and a row
    count above the exact Int64 Gini bound (DEVIATION 175). Both raise out of
    the first tree, before any kernel is enqueued, and both propagate from here
    unchanged. A configuration this arm refuses and the host arm accepts is
    therefore possible, is named when it happens, and is what
    `device_forest_check` uses to prove this arm reaches the device at all.
    """
    var plan = classifier_plan(config, n_rows, n_features)
    var forest = fit_classification_device(
        ctx,
        x_col_major,
        labels,
        n_rows,
        n_features,
        n_classes,
        plan.params,
        plan.n_trees,
        config.random_state,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def regressor_plan(
    config: ExtraTreesConfig, n_rows: Int32, n_features: Int32
) raises -> FitPlan:
    """`classifier_plan`'s regression twin, for DEVIATION 187's reason.

    Both `fit_extra_trees_regressor` and `fit_extra_trees_regressor_device`
    call this and nothing else before their trainer, so neither arm has a line
    of policy of its own and a refusal cannot reach one and miss the other.
    """
    if config.criterion != CRITERION_MSE:
        raise Error(
            "the regression criterion must be squared_error; the others are"
            " refused by validity_check"
        )
    return resolve(config, Int(n_rows), Int(n_features))


def quantize_labels(
    y: List[Float32], n_rows: Int32
) raises -> Tuple[List[Int32], Float64]:
    """The label vector in deviation 135's fixed point, and its scale.

    `choose_scale` takes the sum of magnitudes over the WHOLE label vector,
    because any node's rows are a subset of it -- 135's bound, which makes
    accumulator overflow impossible rather than unlikely. The same derivation
    `device_regression_check` uses, kept here so a caller of the device arm
    does not have to know deviation 135 exists.
    """
    var mag = Float64(0.0)
    for r in range(Int(n_rows)):
        var v = Float64(y[r])
        mag += v if v >= 0.0 else -v
    var scale = choose_scale(mag, Int(n_rows))
    var q = List[Int32]()
    for r in range(Int(n_rows)):
        q.append(Int32(quantize(Float64(y[r]), scale)))
    return (q^, scale)


def fit_extra_trees_regressor(
    x_col_major: List[Float32],
    y: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """`ExtraTreesRegressor.fit`, same contract."""
    var plan = regressor_plan(config, n_rows, n_features)
    var forest = fit_regression(
        x_col_major,
        y,
        n_rows,
        n_features,
        plan.params,
        plan.n_trees,
        config.random_state,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)


def fit_extra_trees_regressor_device(
    ctx: DeviceContext,
    x_col_major: List[Float32],
    y: List[Float32],
    n_rows: Int32,
    n_features: Int32,
    config: ExtraTreesConfig,
) raises -> FitResult:
    """`ExtraTreesRegressor.fit` with the split search on the GPU.

    DEVIATION BLOCK -- DEVIATION 188, CLOSED. This function refused BY NAME
    while `train_regression_device` did not exist: the device reduction had
    nothing exact to rank MSE candidates by. Deviation 189 published cuML's
    own MSE gain as an exact `Int64` key and deviation 206 brought the
    regression device path level with classification, so the refusal's reason
    is gone and keeping it would be the stale-switch defect the Borders
    default already taught this repository once.

    The shape is DEVIATION 187's, unchanged: the signature is
    `fit_extra_trees_regressor`'s plus a `DeviceContext`, both arms call
    `regressor_plan` and nothing else before their trainer, and the dataset is
    uploaded once for the forest (`fit_regression_device`, deviation 184's
    split applied to regression by 206).

    **What is NOT identical to the host arm, and it is deviation 135's ruling
    rather than a defect:** the device trainer consumes labels QUANTIZED to
    fixed point, so `quantize_labels` derives the scale here, the tree
    STRUCTURE is bit-identical to the host arm's (the split decision is made
    on integer sums on both sides, deviations 135 and 189), and the LEAF
    VALUES differ by at most one quantization step -- they are means of
    quantized labels where the host's are `Float64` means.
    `device_regression_check` asserts all three against THIS entry point, and
    uses the leaf-value difference as the REACH proof that this arm fits on
    the device at all: an arm that silently served the host fit would return
    bit-equal leaves, and the check requires at least one to differ.
    """
    var plan = regressor_plan(config, n_rows, n_features)
    var ql = quantize_labels(y, n_rows)
    var forest = fit_regression_device(
        ctx,
        x_col_major,
        ql[0],
        ql[1],
        n_rows,
        n_features,
        plan.params,
        plan.n_trees,
        config.random_state,
    )
    var bound = depth_cap_bound(forest, plan)
    return FitResult(forest^, plan, bound)
