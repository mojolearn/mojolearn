# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One decision tree: its parameters, its flat storage, and the walk that
turns a row into a prediction.

MIRRORS, at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), read-only at
`~/CascadeProjects/upstream/cuml-v26.08.00`:

  * `cpp/include/cuml/tree/algo_helper.h`   -- `CRITERION`
  * `cpp/include/cuml/tree/decisiontree.hpp` -- `DecisionTreeParams`,
    `set_tree_params`, `TreeMetaDataNode`
  * `cpp/src/decisiontree/decisiontree.cu`  -- `validity_check`,
    `set_tree_params` body, the text/JSON dumps
  * `cpp/src/decisiontree/decisiontree.cuh` -- `class DecisionTree`:
    `fit`, `predict`, `predict_all`, `predict_one`

## THE WALK, and the two things that silently differ between forests

`predict_one` is 14 lines (`decisiontree.cuh:370-389`) and every one of
them is a decision somebody could get wrong from memory:

    std::size_t idx = 0;
    auto n          = tree.sparsetree[idx];
    while (!n.IsLeaf()) {
      if (row[n.ColumnId()] <= n.QueryValue()) {
        idx = n.LeftChildId();
      } else {
        idx = n.RightChildId();
      }
      n = tree.sparsetree[idx];
    }
    for (int i = 0; i < num_outputs; i++) {
      preds_out[i] += tree.vector_leaf[idx * num_outputs + i];
    }

  * **The comparison is `<=` and it goes LEFT** (`decisiontree.cuh:379`).
    Corroborated at a second, independent site: their treelite export
    writes every split as `tl::Operator::kLE` with `default_left = true`
    (`decisiontree.cuh:203-204`), so the traversal and the exported
    model agree. Not taken from memory; taken from those two lines.
  * **A row landing EXACTLY on `quesval` therefore goes LEFT.** This is
    the single most common silent difference between two tree
    implementations and it is checked per row in
    `mojo_only/predict_check.mojo`.
  * **The root is index 0** and the loop tests the leaf flag BEFORE
    reading a threshold, so a one-node tree (`sparsetree = [leaf]`)
    returns that leaf's value without ever touching `row`.
  * **The accumulation is `+=`, not `=`** (`decisiontree.cuh:387`).
    `predict_one` ADDS the leaf vector into whatever the caller passed.
    That is not an optimization -- it is the mechanism the forest uses
    to sum trees: `RandomForest::predict` (`randomforest.cuh:402-413`)
    zero-initializes one `row_prediction` per ROW and then calls this
    function once per tree into the same buffer.
  * **The leaf value lives in `vector_leaf`, indexed by NODE id**, at
    `idx * num_outputs + i` (`decisiontree.cuh:387`) -- so `vector_leaf`
    is sized `sparsetree.size() * num_outputs` and the slots belonging
    to internal nodes are never read.

`predict_all` (`decisiontree.cuh:357-368`) walks rows at
`&rows[row_id * n_cols]`, i.e. the prediction input is **ROW-MAJOR**,
unlike the training input which is column-major by default
(`randomforest.hpp:181-186`).

## FIT: their dispatch, recorded even though the body is not ported

`DecisionTree::fit` (`decisiontree.cuh:234-333`) does exactly two things
before handing off to `Builder<ObjectiveT>::train()`:

  1. `CRITERION_END` is resolved to GINI for an integer LabelT and MSE
     otherwise (`decisiontree.cuh:251-256`). That is where the header's
     "default CRITERION_END, i.e., GINI for classification or MSE for
     regression" (`decisiontree.hpp:75-76`) actually happens.
  2. It selects one of four `Builder` instantiations on
     (classification vs regression) x (weighted vs unweighted)
     (`decisiontree.cuh:259-332`). `sample_weight != nullptr` picks the
     weighted arm -- and `RowSampler::tree_sample_weight()` returns
     nullptr whenever bootstrapping is on (`randomforest.cuh:167`), so
     at their default `bootstrap=True` the UNWEIGHTED arm is what runs.

`Builder` is `batched_levelalgo/builder.mojo`, which another lane is
writing. `fit` here is a declared call site whose body raises by name;
see `DecisionTreeParams.check_fit_supported()`.

================= DEVIATION BLOCK (whole file) =================
DEVIATION 118. Four departures, priced.

(a) `TreeMetaDataNode<T, L>` (`decisiontree.hpp:92-101`) carries `L` and
never uses it. Read its seven members: `int treeid`, `int
depth_counter`, `int leaf_counter`, `double train_time`,
`std::vector<T> vector_leaf`, `std::vector<SparseTreeNode<T, L>>
sparsetree`, `int num_outputs`. `L` reaches only the `SparseTreeNode`
template argument, and `SparseTreeNode`'s own `LabelT` is itself
phantom (DEVIATION 116b). So `L` is phantom twice over and
`TreeMetaDataNode[dtype]` here takes one parameter.
PRICE: zero -- a phantom type parameter has no runtime meaning. Their
four typedefs (`TreeClassifierF/D`, `TreeRegressorF/D`,
`decisiontree.hpp:133-136`) collapse to two distinct Mojo types instead
of four names, which is a loss of two SPELLINGS and of no behaviour.
The label type is NOT phantom one level up, at
`RandomForestMetaData<T, L>`, where it is the type `predict` writes;
it is carried there.

(b) `get_tree_text` / `get_tree_json` / `get_node_text` /
`get_node_json` / `to_string_high_precision`
(`decisiontree.cuh:69-152`, `decisiontree.cu:85-97`) are NOT PORTED
YET and `get_tree_json` below raises by name rather than emitting a
tree.
WHY, and this is a priced refusal rather than laziness: their dumps
exist to be DIFFED, and their whole value is
`to_string_high_precision` (`decisiontree.cuh:69-82`), which sets
`std::setprecision(std::numeric_limits<T>::max_digits10)` precisely so
a threshold round-trips exactly. Mojo's `String(Float32)` does NOT
round-trip: 0.46% of float32 values come back one ULP wrong and
`String(Float32(1.4e-45))` is `"0.0"` (this repository's own measured
scar). A JSON dump emitted through it would be a diff surface that
disagrees with their file on roughly one threshold in two hundred and
blames the tree for it. The fix is known -- write
`<decimal>/<hex bits>` and compare the hex -- but that is a FORMAT
CHANGE to their output, which is a bigger decision than this lane
owns. PRICE: no human-readable tree dump until then; the machine-
readable diff surface (`SparseTreeNode.__eq__`, `flatnode.h:59-64`,
plus `vector_leaf` compared as raw bits) is unaffected and is what an
oracle comparison against their wheel would use anyway.

(c) `TreeMetaDataNode::train_time` (`decisiontree.hpp:97`, a `double`
holding milliseconds) is CARRIED as a Float64 field so the struct
matches theirs field for field, and is NEVER WRITTEN and NEVER READ by
this port. `get_tree_summary_text` below therefore omits the
" Tree Fitting - Overall time --> N milliseconds" line their version
prints (`decisiontree.cu:80-81`). Timing is out of scope this round by
instruction; a field that would report a duration is left at its
initial value rather than filled with a number nobody measured.
PRICE: their summary string is two lines and ours is one. Recorded so
that a later lane restoring the line knows it was removed on purpose
and not lost.

(d) `double` resolution, per site, for this file. There is exactly one
`double` in the ported surface -- `train_time` -- and it is host-side
and inert per (c), so it stays Float64. `DecisionTreeParams` has no
`double`: `max_features` and `min_impurity_decrease` are `float`
(`decisiontree.hpp:33, 54`) and are Float32 here, which matters
because `max_features` is truncated to an integer feature count
downstream (see `randomforest.mojo`, `compute_max_features`).
=================================================================
"""

from ensemble.flatnode import SparseTreeNode
from mojo_only.numerics import ftz


def _ftz_feature[dt: DType, //](x: Scalar[dt]) -> Scalar[dt]:
    """DEVIATION 1942, row 10: the host predict walk reads X on the CPU,
    which honors denormals on every vendor, while training read the
    device copy that `fit_forest` flushed. Flushing the feature here
    makes `row[col] <= quesval` compare the same value the tree was
    grown on. `ftz` is a comptime no-op under FAST."""
    comptime if dt == DType.float32:
        return ftz(x.cast[DType.float32]()).cast[dt]()
    else:
        return x


# ---------------------------------------------------------------------------
# CRITERION -- `algo_helper.h:10-18`
# ---------------------------------------------------------------------------
# A C++ unscoped enum with no explicit values, so the enumerators are
# 0..7 in declaration order. The integer values are load-bearing: their
# Python layer maps the STRINGS "0".."7" onto them
# (`randomforest_common.pyx:104-120`) and `validity_check` refuses MAE by
# its literal value 3 (`decisiontree.cu:28`), not by its name.
comptime GINI: Int = 0
comptime ENTROPY: Int = 1
comptime MSE: Int = 2
comptime MAE: Int = 3
comptime POISSON: Int = 4
comptime GAMMA: Int = 5
comptime INVERSE_GAUSSIAN: Int = 6
comptime CRITERION_END: Int = 7


def criterion_name(c: Int) -> String:
    """Their enumerator spellings, for error messages that name names."""
    if c == GINI:
        return "GINI"
    if c == ENTROPY:
        return "ENTROPY"
    if c == MSE:
        return "MSE"
    if c == MAE:
        return "MAE"
    if c == POISSON:
        return "POISSON"
    if c == GAMMA:
        return "GAMMA"
    if c == INVERSE_GAUSSIAN:
        return "INVERSE_GAUSSIAN"
    if c == CRITERION_END:
        return "CRITERION_END"
    return String("<unknown CRITERION ") + String(c) + ">"


# ---------------------------------------------------------------------------
# DecisionTreeParams -- `decisiontree.hpp:20-61`
# ---------------------------------------------------------------------------


@fieldwise_init
struct DecisionTreeParams(ImplicitlyCopyable, Movable):
    """`ML::DT::DecisionTreeParams`, `decisiontree.hpp:20-61`.

    Fields are in THEIR declaration order, with THEIR names. Only one of
    them has an in-struct default in their header
    (`min_impurity_decrease = 0.0f`, `decisiontree.hpp:54`); the rest are
    uninitialized in C++ and are only ever populated through
    `set_tree_params`, whose signature defaults
    (`decisiontree.hpp:81-90`) are the real C++ defaults and are
    reproduced there.

    THE C++ DEFAULT IS NOT ALWAYS THE SHIPPED DEFAULT. The values a user
    of `cuml.ensemble.RandomForest*` actually gets come from the Python
    estimator, and two of them disagree with this header. The full table
    lives in `randomforest.mojo`; the two disagreements are `max_depth`
    (C++ -1, Python `None` -> INT32_MAX) and `max_features`
    (C++ 1.0f, Python `'sqrt'` for the classifier / `1.0` for the
    regressor).
    """

    # `decisiontree.hpp:25` -- Maximum tree depth. INT32_MAX for unlimited.
    var max_depth: Int32
    # `decisiontree.hpp:29` -- Max leaf nodes per tree. Soft. -1 = unlimited.
    var max_leaves: Int32
    # `decisiontree.hpp:33` -- RATIO of columns considered per node split.
    var max_features: Float32
    # `decisiontree.hpp:37` -- max bins used by the split algorithm per feature.
    var max_n_bins: Int32
    # `decisiontree.hpp:41` -- min rows in each leaf node.
    var min_samples_leaf: Int32
    # `decisiontree.hpp:45` -- min rows needed to split an internal node.
    var min_samples_split: Int32
    # `decisiontree.hpp:49` -- GINI/ENTROPY classification, MSE etc regression.
    var split_criterion: Int
    # `decisiontree.hpp:54` -- the one field with an in-struct default, 0.0f.
    var min_impurity_decrease: Float32
    # `decisiontree.hpp:60` -- max nodes processed in one batch (batched algo).
    var max_batch_size: Int32

    def validity_check(self) raises:
        """`ML::DT::validity_check`, `decisiontree.cu:17-35`.

        Their asserts, transcribed in their order and with their
        messages. Note `params.split_criterion != 3` at `:28` -- they
        compare the literal, and the literal is MAE.
        """
        if not (self.max_depth >= 0):
            raise Error("Invalid max depth " + String(self.max_depth))
        if not ((self.max_leaves == -1) or (self.max_leaves > 0)):
            raise Error("Invalid max leaves " + String(self.max_leaves))
        if not ((self.max_features > 0) and (self.max_features <= 1.0)):
            raise Error(
                "max_features value "
                + String(self.max_features)
                + " outside permitted (0, 1] range"
            )
        if not (self.max_n_bins > 0):
            raise Error("Invalid max_n_bins " + String(self.max_n_bins))
        if not (self.max_n_bins <= 1024):
            raise Error("max_n_bins should not be larger than 1024")
        if self.split_criterion == MAE:
            raise Error("MAE not supported.")
        if not (self.min_samples_leaf >= 1):
            raise Error(
                "Invalid value for min_samples_leaf "
                + String(self.min_samples_leaf)
                + ". Should be >= 1."
            )
        if not (self.min_samples_split >= 2):
            raise Error(
                "Invalid value for min_samples_split: "
                + String(self.min_samples_split)
                + ". Should be >= 2."
            )

    def check(self) raises:
        """Refuse what is not honored, by name.

        The rule this repository enforces: an option that is present and
        ignored is worse than one that is absent, because absent fails
        loudly and ignored fails silently.

        What this method refuses is what a caller can set TODAY and get a
        wrong answer from. Training parameters are a separate question
        with a separate answer -- `fit` is not ported, so it refuses ALL
        of them, by name, in `check_fit_supported()` below rather than
        letting one of them look honored.
        """
        # Their own refusal, kept: MAE is enumerated (`algo_helper.h:14`)
        # and rejected (`decisiontree.cu:28`). Their Python layer raises
        # NotImplementedError for it too
        # (`randomforest_common.pyx:147-151`).
        if self.split_criterion == MAE:
            raise Error(
                "split_criterion=MAE is not supported by cuML either --"
                " `validity_check` refuses it at decisiontree.cu:28 and"
                " randomforest_common.pyx:147 raises NotImplementedError."
                " There is no upstream MAE path to port."
            )
        if self.split_criterion < GINI or self.split_criterion > CRITERION_END:
            raise Error(
                "split_criterion="
                + String(self.split_criterion)
                + " is not a CRITERION; algo_helper.h:10-18 enumerates 0"
                " (GINI) through 7 (CRITERION_END). Their Python layer"
                " silently warns and substitutes CRITERION_END here"
                " (randomforest_common.pyx:142-146); this port refuses"
                " instead, because a substituted criterion is a"
                " different model under the same name."
            )
        self.validity_check()

    def check_fit_supported(self) raises:
        """Every training field now HAS a consumer. Nothing is refused here.

        THIS METHOD USED TO RAISE, and the sentence it raised with --
        "DecisionTree.fit is NOT PORTED YET ... the only thing that reads
        them is Builder<ObjectiveT>::train(), which does not exist yet" --
        is deleted rather than annotated, because it is false. All nine
        `DecisionTreeParams` fields are read:

          max_depth, max_leaves, min_samples_split  -> `NodeQueue::
            IsExpandable` and `Push` (`builder.cuh:82-88`, `:101`)
          max_features        -> `n_sampled_cols_for` (`builder.cuh:240`)
                                 and the round schedule (`:437-455`)
          max_n_bins          -> `computeQuantiles` and every histogram
          min_samples_leaf    -> `GainPerSplit` (`objectives.cuh:128-130`)
          split_criterion     -> `GainPerSplit`'s switch, all six reachable
                                 criteria exercised by `criteria_check`
          min_impurity_decrease -> `Gain` (`objectives.cuh:173`, `:374`)
          max_batch_size      -> `NodeQueue::Pop` (`builder.cuh:70-78`)

        The three that go through the objective -- min_samples_leaf,
        split_criterion, min_impurity_decrease -- reach it only because
        `Builder` CONSTRUCTS the objective from `params`, at their
        `builder.cuh:592-596` call site. While it took a pre-built
        objective from the caller instead, those three rows were false:
        nothing read them from `params` and nothing checked that the
        caller's objective agreed. `criteria_check` arm F is what holds
        them now, and it passes no objective at all.

        It is kept as a method rather than deleted so that a future
        unported field has somewhere to be refused BY NAME. That is the
        rule it exists for: an option present and ignored is worse than one
        absent, because absent fails loudly.
        """
        pass


def set_tree_params(
    mut params: DecisionTreeParams,
    cfg_max_depth: Int32 = -1,
    cfg_max_leaves: Int32 = -1,
    cfg_max_features: Float32 = 1.0,
    cfg_max_n_bins: Int32 = 128,
    cfg_min_samples_leaf: Int32 = 1,
    cfg_min_samples_split: Int32 = 2,
    cfg_min_impurity_decrease: Float32 = 0.0,
    cfg_split_criterion: Int = CRITERION_END,
    cfg_max_batch_size: Int32 = 4096,
) raises:
    """`ML::DT::set_tree_params`, declared `decisiontree.hpp:81-90`,
    defined `decisiontree.cu:51-72`.

    THE DEFAULTS ABOVE ARE THEIR C++ DEFAULTS, taken from the
    declaration (`decisiontree.hpp:82-90`) -- the definition repeats the
    signature without them, as C++ requires. They are NOT all what a
    Python user gets; see `randomforest.mojo` for the full table.

    Their argument ORDER puts `cfg_min_impurity_decrease` before
    `cfg_split_criterion` (`decisiontree.hpp:88-89`) while the STRUCT
    declares `split_criterion` before `min_impurity_decrease`
    (`decisiontree.hpp:49, 54`). Both orders are kept as they are; the
    mismatch is theirs and is exactly the kind of thing a positional
    call gets wrong, which is why every call site in this port names its
    arguments.
    """
    params.max_depth = cfg_max_depth
    params.max_leaves = cfg_max_leaves
    params.max_features = cfg_max_features
    params.max_n_bins = cfg_max_n_bins
    params.min_samples_leaf = cfg_min_samples_leaf
    params.min_samples_split = cfg_min_samples_split
    params.split_criterion = cfg_split_criterion
    params.min_impurity_decrease = cfg_min_impurity_decrease
    params.max_batch_size = cfg_max_batch_size
    params.validity_check()


def default_tree_params() raises -> DecisionTreeParams:
    """Not in their header: C++ gets these from the default arguments of
    `set_tree_params` on a stack-declared struct
    (`randomforest.cu:566-576`). Mojo has no uninitialized struct, so the
    same defaults are spelled once here."""
    var p = DecisionTreeParams(
        max_depth=-1,
        max_leaves=-1,
        max_features=1.0,
        max_n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        split_criterion=CRITERION_END,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
    )
    set_tree_params(p)
    return p


# ---------------------------------------------------------------------------
# TreeMetaDataNode -- `decisiontree.hpp:92-101`
# ---------------------------------------------------------------------------


@fieldwise_init
struct TreeMetaDataNode[dtype: DType](Copyable, Movable):
    """`ML::DT::TreeMetaDataNode<T, L>`, `decisiontree.hpp:92-101`.

    Their `L` is phantom; see DEVIATION 118a. Their seven members in
    their order.

    `vector_leaf` is indexed BY NODE, `idx * num_outputs + k`
    (`decisiontree.cuh:387`), so it is sized
    `len(sparsetree) * num_outputs` and the slots belonging to internal
    nodes are dead storage that their builder also allocates.
    """

    # `decisiontree.hpp:94`
    var treeid: Int32
    # `decisiontree.hpp:95`
    var depth_counter: Int32
    # `decisiontree.hpp:96`
    var leaf_counter: Int32
    # `decisiontree.hpp:97` -- double, milliseconds. NEVER WRITTEN HERE.
    # DEVIATION 118c: timing is out of scope this round.
    var train_time: Float64
    # `decisiontree.hpp:98`
    var vector_leaf: List[Scalar[Self.dtype]]
    # `decisiontree.hpp:99`
    var sparsetree: List[SparseTreeNode[Self.dtype]]
    # `decisiontree.hpp:100`
    var num_outputs: Int32


def get_tree_summary_text[
    dtype: DType
](tree: TreeMetaDataNode[dtype]) -> String:
    """`ML::DT::get_tree_summary_text`, `decisiontree.cu:74-83`.

    Their second line reports `train_time` in milliseconds and is
    omitted; DEVIATION 118c.
    """
    return (
        String(" Decision Tree depth --> ")
        + String(tree.depth_counter)
        + " and n_leaves --> "
        + String(tree.leaf_counter)
        + "\n"
    )


def get_tree_json[dtype: DType](tree: TreeMetaDataNode[dtype]) raises -> String:
    """`ML::DT::get_tree_json`, `decisiontree.cu:92-97`. NOT PORTED YET.

    Raises by name rather than emitting a tree whose thresholds would
    not round-trip. DEVIATION 118b has the reason and the price.
    """
    _ = tree.num_outputs
    raise Error(
        "get_tree_json / get_tree_text / get_node_json / get_node_text /"
        " to_string_high_precision (decisiontree.cuh:69-152,"
        " decisiontree.cu:85-97) are NOT PORTED YET. Their whole value is"
        " to_string_high_precision's max_digits10 round-trip"
        " (decisiontree.cuh:76-78), and Mojo's String(Float32) does not"
        " round-trip -- so a dump written through it would disagree with"
        " their file on roughly one threshold in two hundred and blame"
        " the tree. Compare trees through SparseTreeNode.__eq__"
        " (flatnode.h:59-64) and raw vector_leaf bits instead."
    )


# ---------------------------------------------------------------------------
# class DecisionTree -- `decisiontree.cuh:232-391`
# ---------------------------------------------------------------------------


struct DecisionTree:
    """`ML::DT::DecisionTree`, `decisiontree.cuh:232-391`.

    Their class holds no state -- it is a namespace of static members --
    and this one holds none either. It exists so `DecisionTree.predict`
    greps against their tree.
    """

    @staticmethod
    def fit[dtype: DType](params: DecisionTreeParams) raises:
        """`DecisionTree::fit`, `decisiontree.cuh:234-333`. NOT PORTED.

        Their body does two things: resolve `CRITERION_END` to GINI/MSE
        (`:251-256`) and dispatch to one of four `Builder<ObjectiveT>`
        instantiations (`:259-332`). BOTH now live elsewhere in this port
        -- the resolution in `Builder`'s constructor, the dispatch in the
        `O` a caller picks -- so this single-tree entry point has nothing
        left of its own and no callers.

        IT RAISES RATHER THAN RETURNING. This docstring used to say it
        raised while the body was `params.check_fit_supported()`, which is
        `pass`: it accepted every call and silently did nothing. A door
        that is not wired has to say so, because a caller cannot tell a
        successful no-op from a successful fit.
        """
        raise Error(
            "DecisionTree.fit is not ported: fit a one-tree forest through"
            " ensemble.randomforest.fit_forest instead. Their"
            " decisiontree.cuh:251-256 criterion resolution lives in"
            " Builder's constructor and their :259-332 objective dispatch"
            " is the type parameter O."
        )

    @staticmethod
    def predict[
        dtype: DType
    ](
        tree: TreeMetaDataNode[dtype],
        rows: List[Scalar[dtype]],
        n_rows: Int,
        n_cols: Int,
        mut predictions: List[Scalar[dtype]],
        num_outputs: Int,
        rows_offset: Int = 0,
        preds_offset: Int = 0,
    ) raises:
        """`DecisionTree::predict`, `decisiontree.cuh:335-355`.

        `rows_offset` / `preds_offset` are not extra parameters: they are
        the POINTER ARITHMETIC their caller does. `RandomForest::predict`
        calls this as `predict(..., &h_input[row_id * row_size], 1,
        n_cols, row_prediction.data(), ...)` (`randomforest.cuh:405-412`),
        i.e. it hands in a pointer into the middle of the row buffer.
        Mojo has no interior pointer into a `List`, so the offset their
        `&` produced is passed alongside. Default 0 reproduces their
        signature exactly.

        Their two asserts are kept in their order and with their
        meanings; the host/device one becomes a bounds fact because
        there are no device pointers on this path at all (DEVIATION 119).
        """
        # `decisiontree.cuh:350-352`
        if len(tree.sparsetree) == 0:
            raise Error(
                "Cannot predict w/ empty tree, tree size "
                + String(len(tree.sparsetree))
            )
        # Not theirs: their `is_host_ptr` assert (`decisiontree.cuh:346`)
        # has no counterpart, so the equivalent structural mistake --
        # a short buffer -- is caught instead.
        if len(rows) < rows_offset + n_rows * n_cols:
            raise Error(
                "rows holds "
                + String(len(rows))
                + " values but rows_offset + n_rows * n_cols is "
                + String(rows_offset + n_rows * n_cols)
            )
        if len(predictions) < preds_offset + n_rows * num_outputs:
            raise Error(
                "predictions holds "
                + String(len(predictions))
                + " values but preds_offset + n_rows * num_outputs is "
                + String(preds_offset + n_rows * num_outputs)
            )
        DecisionTree.predict_all(
            tree,
            rows,
            n_rows,
            n_cols,
            predictions,
            num_outputs,
            rows_offset,
            preds_offset,
        )

    @staticmethod
    def predict_all[
        dtype: DType
    ](
        tree: TreeMetaDataNode[dtype],
        rows: List[Scalar[dtype]],
        n_rows: Int,
        n_cols: Int,
        mut preds: List[Scalar[dtype]],
        num_outputs: Int,
        rows_offset: Int = 0,
        preds_offset: Int = 0,
    ) raises:
        """`DecisionTree::predict_all`, `decisiontree.cuh:357-368`.

        `&rows[row_id * n_cols]` -- the inference input is ROW-MAJOR.
        `preds + row_id * num_outputs` -- one output block per row.
        The two offsets carry their caller's `&` (see `predict` above).
        """
        for row_id in range(n_rows):
            DecisionTree.predict_one(
                rows,
                rows_offset + row_id * n_cols,
                tree,
                preds,
                preds_offset + row_id * num_outputs,
                num_outputs,
            )

    @staticmethod
    def predict_one[
        dtype: DType
    ](
        rows: List[Scalar[dtype]],
        row_offset: Int,
        tree: TreeMetaDataNode[dtype],
        mut preds_out: List[Scalar[dtype]],
        preds_offset: Int,
        num_outputs: Int,
    ) raises:
        """`DecisionTree::predict_one`, `decisiontree.cuh:370-389`.

        Their pointer arguments become (list, offset) pairs; the offsets
        are the pointer arithmetic their caller does at
        `decisiontree.cuh:366`.

        THE WALK, line for line:
          * start at node 0 (`:376-377`)
          * loop while NOT a leaf, leaf being `left_child_id == -1`
            (`:378`, `flatnode.h:58`)
          * `row[ColumnId()] <= QueryValue()` goes LEFT (`:379-380`);
            equality goes LEFT
          * otherwise RIGHT, which is `LeftChildId() + 1`
            (`:382`, `flatnode.h:45`)
          * ADD the leaf's `num_outputs` values into the caller's buffer
            (`:386-388`) -- `+=`, so the forest can sum trees here
        """
        var idx: Int = 0
        var n = tree.sparsetree[idx]
        while not n.IsLeaf():
            if (
                _ftz_feature(rows[row_offset + Int(n.ColumnId())])
                <= n.QueryValue()
            ):  # `decisiontree.cuh:379`, DEVIATION 1942 flush
                idx = Int(n.LeftChildId())
            else:
                idx = Int(n.RightChildId())
            n = tree.sparsetree[idx]
        for i in range(num_outputs):
            preds_out[preds_offset + i] += tree.vector_leaf[
                idx * num_outputs + i
            ]
