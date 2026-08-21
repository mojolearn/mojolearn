"""The split record and its total order.

A PORT of cuML `cpp/src/decisiontree/batched-levelalgo/split.cuh`, pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`. Fields are theirs
(`split.cuh:38-45`), the default construction is theirs (`split.cuh:47-52`) and
`update` is their branch chain transcribed in their order (`split.cuh:76-90`).

WHY A cuML FILE IS THE UPSTREAM FOR A PAPER PORT
------------------------------------------------
The split RULE in this directory comes from Geurts, Ernst & Wehenkel 2006 by
way of scikit-learn's `RandomSplitter`, because no GPU library ships that
formulation. The split RECORD does not need inventing: cuML already wrote the
struct a GPU tree builder reduces over, and `PORTING_RULES.md` 0c is a list of
eleven times this project invented something that a competitor's file already
answered better. So the record, the tie-break and the validity test are ports.

`warpReduce` and `evalBestSplit` (`split.cuh:92-152`) are NOT in this file yet.
They are the device-side reduction of this struct and belong with the kernels;
they are listed in `UNPORTED.tsv` until the score pass exists to call them.
Their tie-break is `update`, which IS here, so the order they impose is already
pinned down and checkable on the host.
"""


# ==========================================================================
# DEVIATION BLOCK 133 -- the tie-break is a total order, sklearn's is not
#
# THEIRS (sklearn, `_splitter.pyx:690`): `if current_proxy_improvement >
#   best_proxy_improvement` -- strictly greater, so a tie is resolved by
#   whichever candidate the SEQUENTIAL draw loop reached first. That is a
#   statement about loop order, not an order on the candidates.
# OURS: `Split.update` below, cuML's `split.cuh:78-90` branch for branch --
#   greater metric wins; equal metric, greater colid wins; equal colid,
#   greater quesval wins.
# WHY: this builder evaluates candidates in parallel and reduces them in an
#   unspecified order, so "first" does not exist (DEVIATION 130). A total
#   order over the candidate's own fields is the only tie-break that survives
#   an unspecified reduction order, and cuML already wrote it.
# PRICE: on an exact tie we take a different feature than sklearn would. Ties
#   are what the duplicate-feature analytic fixture exists to produce, so the
#   behaviour is pinned by a check rather than left to chance.
# ==========================================================================


@fieldwise_init
struct Split(ImplicitlyCopyable, Movable):
    """All info pertaining to splitting a node. `split.cuh:32-45`."""

    var quesval: Float32
    """Threshold to compare in this node. Theirs is `DataT quesval`."""

    var colid: Int32
    """Feature index. Theirs is `IdxT colid`."""

    var best_metric_val: Float32
    """Best info gain on this node. Theirs is `DataT best_metric_val`."""

    var n_left: Int32
    """Number of samples in the left child. Theirs is `int nLeft`."""

    comptime Min = Float32.MIN_FINITE
    """`split.cuh:36`: `-std::numeric_limits<DataT>::max()`.

    NOT negative infinity. Mojo spells the negative of the largest finite
    float `MIN_FINITE`; it is the same bit pattern their expression produces.
    """

    def __init__(out self):
        """`split.cuh:54-59`, the default constructor."""
        self.quesval = Self.Min
        self.best_metric_val = Self.Min
        self.colid = -1
        self.n_left = 0

    def update(mut self, other: Self) -> Bool:
        """Updates the current split if the input gain is better.

        `split.cuh:76-90`, transcribed branch for branch. Returns whether the
        update happened, as theirs does.
        """
        var update_result = False
        if other.best_metric_val > self.best_metric_val:
            update_result = True
        elif other.best_metric_val == self.best_metric_val:
            if other.colid > self.colid:
                update_result = True
            elif other.colid == self.colid:
                if other.quesval > self.quesval:
                    update_result = True
        if update_result:
            self = other
        return update_result

    def is_valid(self) -> Bool:
        """Whether a candidate was ever recorded here. `split.cuh:145` tests
        `this->colid != -1` before touching the shared best split."""
        return self.colid != -1
