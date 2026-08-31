# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""sklearn's "keep drawing while every draw was constant" clause.

==========================================================================
DEVIATION BLOCK -- DEVIATION 205. THIS CLOSES DEVIATION 151.

151 recorded that we stop splitting a node when every one of its
`max_features` sampled columns is constant, while sklearn keeps drawing,
and it PRICED that difference without fixing it. The price was measured
again on 2026-08-21 against covtype at 581,012 rows and it is not small:
at `max_features=5` scikit-learn reaches 0.660-0.716 train accuracy and
we reach 0.645-0.677, and we build 3-4.7x FEWER nodes (103,166 against
28,882 at depth 16). covtype is 44 binary one-hot columns out of 54, so
a narrow sample is constant often, and every time it is we stop and they
do not.

WHAT SKLEARN ACTUALLY DOES, `_splitter.pyx:573-577`::

    while (f_i > n_total_constants and
            (n_visited_features < max_features or
             n_visited_features <= n_found_constants + n_drawn_constants)):

Decoded, with `V` the number of draws so far and `C` the number of them
that were constant: `V - C` is the count of NON-constant features
actually evaluated. The loop runs while there is still a feature that is
not already known constant AND (the budget is unspent OR nothing
non-constant has been evaluated yet). So:

  * `max_features` is a budget on TOTAL DRAWS, and a constant feature
    SPENDS one of them -- it is not "keep drawing until you have
    max_features usable columns";
  * but the second clause overrides the exhausted budget for exactly as
    long as every draw has been constant. Once one non-constant feature
    is evaluated, `V - C >= 1` forever, and the loop ends the moment the
    budget is gone.

So in the regime this deviation is about -- all `k` drawn columns
constant -- sklearn draws on, one at a time, without replacement, and
stops at the FIRST non-constant feature, having evaluated exactly ONE.

AND THAT MAKES THE RULE CHEAP TO PORT EXACTLY. The remaining features
are drawn in a uniformly random order, so "the first non-constant one in
that order" is UNIFORMLY DISTRIBUTED over the node's non-constant
columns. This module draws that uniform choice directly. It is the same
distribution, not an approximation of it, and it costs one RNG draw
instead of a sequential loop no GPU wants to run.

The columns already drawn need no exclusion: they were all constant, so
they are not in the set being drawn from.

WHAT IS *NOT* PORTED, AND IT IS NOT NEEDED FOR THIS. sklearn also
carries a node's discovered-constant set DOWN to its children through
`ParentInfo.n_constant_features` (`_splitter.pyx:723-734`), so a child
never re-tests a feature an ancestor proved constant. That is a
COST optimisation on their sequential loop -- it changes which features
are skipped, not which non-constant feature is finally chosen, because
the skipped ones are constant at the child too and could never have been
selected. Our per-node scan tests every column afresh, which is more
work and the same answer. Recorded here rather than left to be noticed.

THE HOST AND THE DEVICE MUST AGREE, so the choice lives in ONE function.
`rescue_pick` below is called by the host trainer and by
`rescue_select_kernel`; neither has a rule of its own, which is what
keeps `device_tree_check` able to compare the two paths cell for cell.
==========================================================================
"""

from extratrees.mojo_only.pcg_rng import (
    PCGenerator,
    SplitKey,
    key_for,
    uniform_int_u32,
)


comptime RESCUE_FEATURE_SALT: UInt32 = 0xFFFFFFFF
"""The `feature_id` slot the rescue's own draw occupies in `key_for`.

`key_for(seed, tree_id, node_id, feature_id)` keys every threshold draw on the
column it belongs to, so the rescue's choice needs a slot no column can
occupy. `0xFFFFFFFF` is not a column index on any dataset this port accepts:
`n_cols` is an `Int32`, so the largest legal column id is `0x7FFFFFFE`. Using
a real column's slot would tie the choice of column to that column's threshold
draw, which is a correlation nobody asked for and nobody would find.
"""


def rescue_key(seed: UInt64, tree_id: Int32, node_id: UInt32) -> SplitKey:
    """The key for this node's rescue draw."""
    return key_for(
        seed, UInt32(Int(tree_id)), node_id, RESCUE_FEATURE_SALT
    )


def rescue_pick(key: SplitKey, n_nonconstant: Int) raises -> Int:
    """Which of the node's non-constant columns, counted in ASCENDING column
    order, sklearn's loop would have landed on.

    Returns an index into the node's non-constant columns, in `[0,
    n_nonconstant)`. The caller supplies the count and maps the index back to a
    column; the ORDER it counts in is part of the contract, because host and
    device must map the same index to the same column.

    `uniform_int_u32` is RAFT's own Lemire draw -- the same function every
    other integer draw in this lane uses -- so the rescue introduces no new
    generator.
    """
    if n_nonconstant < 1:
        raise Error(
            "rescue_pick needs at least one non-constant column; a node with"
            " none is a leaf and must not reach here"
        )
    var gen = PCGenerator(key.seed, key.subsequence, UInt64(0))
    return Int(uniform_int_u32(gen, UInt32(0), UInt32(n_nonconstant)))
