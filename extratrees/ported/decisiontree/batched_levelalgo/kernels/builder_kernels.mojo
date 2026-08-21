"""The work-item structs the builder and its kernels share.

A PORT of the host-visible half of cuML
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh`, pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`:

| ours | theirs |
|---|---|
| `InstanceRange` | `builder_kernels.cuh:34-37` |
| `NodeWorkItem` | `builder_kernels.cuh:39-43` |
| `WorkloadInfo` | `builder_kernels.cuh:49-57` |
| `split_not_valid` | `builder_kernels.cuh:59-67` |

NOT IN THIS FILE, DELIBERATELY, AND WHERE THEY WENT
---------------------------------------------------
- `fnv1a32` (`builder_kernels.cuh:92-105`) and the `PCGenerator` it keys are in
  `extratrees/mojo_only/pcg_rng.mojo`, with RAFT as the upstream for the
  generator itself. They are a RAFT primitive plus four lines of cuML glue, and
  this tree keeps RAFT primitives in `mojo_only/` the way `cluster/` does.
- `excess_sample_with_replacement_kernel` (`:152`) and `algo_L_sample_kernel`
  (`:246`), the per-node feature samplers, are listed in `UNPORTED.tsv` until
  the builder has a frontier to sample for.
- `lower_bound` (`:110-125`) is a QUANTILE lookup. There are no quantiles in
  this directory and there never will be; see `UNPORTED.tsv` on
  `quantiles.cuh`.

WHAT `WorkloadInfo` IS FOR, because the name does not say it
------------------------------------------------------------
It is cuML's answer to the ragged-frontier problem: a batch of nodes has wildly
different row counts, so instead of one block per node they flatten the batch
into a list of blocks and give each block a row telling it which node it serves
and which slice of that node's rows it owns (`builder.cuh:378-393` fills it,
`builder_kernels_impl.cuh:236-241` reads it). A "large" node is one needing
more than one block, and only large nodes get a global-memory scratch slot.
This lane keeps that shape: it is the piece that makes a breadth-first frontier
efficient, and it is theirs, not ours.
"""

from extratrees.ported.decisiontree.batched_levelalgo.split import Split


@fieldwise_init
struct InstanceRange(ImplicitlyCopyable, Movable):
    """The range of instances belonging to a node, as a range in
    `Dataset.row_ids`. `builder_kernels.cuh:34-37`."""

    var begin: Int32
    var count: Int32


@fieldwise_init
struct NodeWorkItem(ImplicitlyCopyable, Movable):
    """One node awaiting a split. `builder_kernels.cuh:39-43`."""

    var idx: Int32
    """Index of the work item in the tree. Theirs is `size_t idx`."""

    var depth: Int32
    var instances: InstanceRange


@fieldwise_init
struct WorkloadInfo(ImplicitlyCopyable, Movable):
    """Which node a threadblock serves, and which slice of it.
    `builder_kernels.cuh:49-57`."""

    var nodeid: Int32
    """Node in the batch this threadblock works on."""

    var large_nodeid: Int32
    """Counts only LARGE nodes -- those needing more than one block along the
    x dimension, and therefore a global-memory scratch slot."""

    var offset_blockid: Int32
    """This block's offset among all blocks working on this node."""

    var num_blocks: Int32
    """Total blocks working on this node."""


def split_not_valid(
    split: Split,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    num_rows: Int32,
) -> Bool:
    """`builder_kernels.cuh:59-67`, transcribed.

    Theirs: `split.best_metric_val <= min_impurity_decrease ||
    split.nLeft < min_samples_leaf ||
    (IdxT(num_rows) - split.nLeft) < min_samples_leaf`.

    Note the FIRST clause is `<=`, so a split whose gain exactly equals
    `min_impurity_decrease` is rejected, and with their default of 0 a
    zero-gain split is rejected too. sklearn's test is the other way round --
    `min_impurity_decrease` is checked as `>=` against a differently scaled
    quantity in `_tree.pyx` -- which is why this lane reports the gain in BOTH
    forms rather than assuming they are the same number.
    """
    return (
        split.best_metric_val <= min_impurity_decrease
        or split.n_left < min_samples_leaf
        or (num_rows - split.n_left) < min_samples_leaf
    )
