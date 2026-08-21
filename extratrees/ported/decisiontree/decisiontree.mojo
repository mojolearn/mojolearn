"""Tree parameters, their defaults, and the check that refuses the rest.

A PORT of cuML's `cpp/include/cuml/tree/decisiontree.hpp` (the
`DecisionTreeParams` struct and `set_tree_params`'s default arguments),
`cpp/include/cuml/tree/algo_helper.h` (the `CRITERION` enum) and
`cpp/src/decisiontree/decisiontree.cu:27-45` (`validity_check`), pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`.

WHY THE VALIDATION IS PART OF THE PORT AND NOT PAPERWORK
---------------------------------------------------------
`gbdt/` learned this the expensive way: an option that is silently accepted and
silently ignored is indistinguishable, from the caller's side, from an option
that works. So every option is either implemented or REFUSED BY NAME. Two
options in cuML's struct are refused here, and each says why below.

ONE THING THEIR OWN HEADER GETS WRONG ABOUT THEIR OWN CODE
-----------------------------------------------------------
`decisiontree.hpp:31` documents `max_depth` as "Unlimited (e.g., until leaves
are pure), If `-1`", and their default in `set_tree_params` is `-1`. Their
`validity_check` then asserts `params.max_depth >= 0`
(`decisiontree.cu:29`), so the documented default cannot survive validation --
their Python layer substitutes a concrete depth before the C++ ever sees it.
Recorded because it is exactly the kind of "the docs describe the intent, the
branches are the algorithm" gap `PORTING_RULES.md` rule 3 is about, and because
a port that trusted the header would ship an unlimited-depth default that their
own code rejects. **This port takes the BRANCH, not the comment**: `max_depth`
must be `>= 0`.
"""


comptime CRITERION_GINI = 0
comptime CRITERION_ENTROPY = 1
comptime CRITERION_MSE = 2
comptime CRITERION_MAE = 3
comptime CRITERION_POISSON = 4
comptime CRITERION_GAMMA = 5
comptime CRITERION_INVERSE_GAUSSIAN = 6
comptime CRITERION_END = 7
"""`algo_helper.h:20-29`, in their order, so the integers match theirs.

`CRITERION_END` is their "not specified" sentinel and their default
(`decisiontree.hpp:93`): it means GINI for classification and MSE for
regression, resolved by the caller who knows which it is.
"""


# ==========================================================================
# DEVIATION BLOCK 138 -- `max_n_bins` is refused, not defaulted
#
# THEIRS: `DecisionTreeParams::max_n_bins` (`decisiontree.hpp:41`), default
#   128, validated to `(0, 1024]` (`decisiontree.cu:36-38`). It is the size of
#   the quantile set their split search scans.
# OURS: refused by name. There are no quantiles and no bins in this directory;
#   `quantiles.cuh` is in `UNPORTED.tsv` as deliberate, because deleting it is
#   the point of this formulation.
# WHY REFUSE RATHER THAN IGNORE: a caller who passes `max_n_bins=1024`
#   expecting a finer search would get a tree that ignored the request, and
#   nothing would say so. `gbdt/`'s `check()` refuses every unported CatBoost
#   option by name for the same reason.
# PRICE: a caller porting a cuML configuration across has to delete the line.
#   That is the intended cost -- it is the one line that says the two learners
#   are different algorithms.
# ==========================================================================


@fieldwise_init
struct DecisionTreeParams(ImplicitlyCopyable, Movable):
    """`decisiontree.hpp:29-67`, field for field, minus `max_n_bins`.

    Defaults are the ones in `set_tree_params`'s signature
    (`decisiontree.hpp:86-95`), EXCEPT `max_depth`: see the module docstring.
    """

    var max_depth: Int32
    """Maximum tree depth. Their default is -1, which their own
    `validity_check` rejects; see the module docstring."""

    var max_leaves: Int32
    """Maximum leaf nodes per tree, a SOFT constraint in their code
    (`builder.cuh:86-88` stops pushing work items once the counter is reached,
    it does not prune). Unlimited if -1. Their default: -1."""

    var max_features: Float32
    """Ratio of columns to consider per node split. Their default: 1.0.

    sklearn's ExtraTrees defaults differ and are what this lane targets:
    `sqrt` for classification, `1.0` for regression. `sqrt` is not expressible
    as a ratio without knowing the column count, so the CALLER resolves it and
    passes the ratio -- which is also what cuML's Python layer does."""

    var min_samples_leaf: Int32
    """Minimum rows in each leaf. Their default: 1. sklearn's default is also
    1 (`ExtraTreesClassifier.__init__`)."""

    var min_samples_split: Int32
    """Minimum rows needed to split an internal node. Their default: 2, and
    their `validity_check` requires >= 2. sklearn's default is also 2."""

    var split_criterion: Int32
    """Their default: `CRITERION_END`, meaning "resolve from the task"."""

    var min_impurity_decrease: Float32
    """Their default: 0.0. Note `split_not_valid` rejects a split whose gain is
    `<=` this, so at their default a zero-gain split is rejected."""

    var max_batch_size: Int32
    """Maximum nodes processed in one frontier batch. Their default: 4096.
    This is the width of the breadth-first frontier and it is a SCHEDULING
    parameter: it must not change the tree. That property is checkable and is
    checked."""

    def __init__(out self):
        """`set_tree_params`'s default arguments (`decisiontree.hpp:86-95`),
        with `max_depth` at 16 rather than their -1 -- their own
        `validity_check` rejects -1 and their Python layer never passes it."""
        self.max_depth = 16
        self.max_leaves = -1
        self.max_features = 1.0
        self.min_samples_leaf = 1
        self.min_samples_split = 2
        self.split_criterion = CRITERION_END
        self.min_impurity_decrease = 0.0
        self.max_batch_size = 4096


def validity_check(params: DecisionTreeParams) raises:
    """`decisiontree.cu:27-45`, transcribed assertion for assertion, plus the
    refusals this port owes its caller.

    Their assertions are kept in their order and with their bounds so that a
    configuration cuML rejects is rejected here for the same stated reason.
    """
    # --- theirs, in their order ------------------------------------------
    if params.max_depth < 0:
        raise Error("Invalid max depth " + String(params.max_depth))
    if not (params.max_leaves == -1 or params.max_leaves > 0):
        raise Error("Invalid max leaves " + String(params.max_leaves))
    if not (params.max_features > 0.0 and params.max_features <= 1.0):
        raise Error(
            "max_features value "
            + String(params.max_features)
            + " outside permitted (0, 1] range"
        )
    # `decisiontree.cu:36-38` validate max_n_bins here. See DEVIATION 138.
    if params.split_criterion == CRITERION_MAE:
        raise Error("MAE not supported.")
    if params.min_samples_leaf < 1:
        raise Error(
            "Invalid value for min_samples_leaf "
            + String(params.min_samples_leaf)
            + ". Should be >= 1."
        )
    if params.min_samples_split < 2:
        raise Error(
            "Invalid value for min_samples_split: "
            + String(params.min_samples_split)
            + ". Should be >= 2."
        )

    # --- ours: refuse what is not ported, BY NAME ------------------------
    # cuML supports four regression criteria beyond MSE (`algo_helper.h:20-29`;
    # the kernels exist as `poisson-*.cu`, `gamma-*.cu`,
    # `inverse_gaussian-*.cu`). None is ported. sklearn's ExtraTrees has its
    # own list (`friedman_mse`, `absolute_error`, `poisson`) and none of those
    # is ported either. A criterion that is silently downgraded to MSE would
    # train a model the caller did not ask for.
    if params.split_criterion == CRITERION_POISSON:
        raise Error("split_criterion=POISSON is not ported in extratrees/")
    if params.split_criterion == CRITERION_GAMMA:
        raise Error("split_criterion=GAMMA is not ported in extratrees/")
    if params.split_criterion == CRITERION_INVERSE_GAUSSIAN:
        raise Error(
            "split_criterion=INVERSE_GAUSSIAN is not ported in extratrees/"
        )
    if params.split_criterion == CRITERION_ENTROPY:
        raise Error(
            "split_criterion=ENTROPY is not ported in extratrees/ yet; use"
            " GINI"
        )
    if params.split_criterion > CRITERION_END or params.split_criterion < 0:
        raise Error(
            "Unknown split criterion " + String(params.split_criterion)
        )

    if params.max_batch_size < 1:
        raise Error(
            "Invalid max_batch_size "
            + String(params.max_batch_size)
            + ". Should be >= 1."
        )
