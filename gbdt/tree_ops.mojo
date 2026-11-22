# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tree-builder op table of the executor (lane t1-tree-builder-ops,
2026-11-08; plan EXECUTOR_T1_TREE_OPS (git history), plan
plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) section 4 rows "GBDT rounds"
and "RandomForest / ExtraTrees / DecisionTree levels").

A boosting round becomes one `LoopStep` of `autograd/program_loop_device.mojo`
built from these ops, in the order of plan EXECUTOR_T1_TREE_OPS (git history)
section 3. Each op is a typed enqueue-only
function in the module that owns its kernel (the op body IS the existing
kernel, so the words do not change); this table is the one registry of what
the ops are called, which buffers they write and whether they are behind the
loop's gate. Lookup is one `comptime for` over the codes, never an `elif`
chain per op (the step-op table of autograd/block_step_ops.mojo).

The gate rule. Word 0 of `gate` is the loop's done word (`LoopRegisters.words`
inside a device loop, `core.step_gate.step_gate_open_word` outside). An op
that writes CARRIED state (the RNG seeds, the learn and held-out cursors, the
loss history, the detector) is gated: once the loop is done it writes
nothing, so rounds enqueued past the stopping one leave the state as the
stopping round left it. An op that writes only ROUND-LOCAL scratch (the
subsets, histograms, winner slots, partition buffers) is not gated: a round
past the stop recomputes scratch nobody reads. A status op writes a failure
word that `core.step_gate.step_fail_joins_loop` turns into a stop.
"""

comptime TREE_OP_BOOTSTRAP = 0
comptime TREE_OP_SEARCH = 1
comptime TREE_OP_GATE_WALK = 2
comptime TREE_OP_TREE_RECORDS = 3
comptime TREE_OP_LEAF_BINS = 4
comptime TREE_OP_PARTITION = 5
comptime TREE_OP_APPLY = 6
comptime TREE_OP_HELDOUT_LOSS = 7
comptime TREE_OP_OD_STEP = 8
# lane t1b-tree-leaf-and-node-ops (2026-11-09): the leaf estimate and its
# rerun snapshot (GBDT), and the forest level loop's batches and finish
# (RandomForest / DecisionTree, ExtraTrees). Codes are table rows, not the
# round's order: plan EXECUTOR_T1_TREE_OPS (git history) section 3 is the order.
comptime TREE_OP_LEAF_ESTIMATE = 9
comptime TREE_OP_SEED_SNAPSHOT = 10
comptime TREE_OP_FOREST_BATCHES = 11
comptime TREE_OP_FOREST_FINISH = 12
# lane t3-gbdt-greedy-on-device-loop (2026-11-09): the greedy SymmetricTree
# searcher (the default arm) as an enqueue-only op, and its learn cursor add
# on the searcher's own partition (EXECUTOR_T1_TREE_OPS.md section 8).
comptime TREE_OP_GREEDY_SEARCH = 13
comptime TREE_OP_SEARCHER_APPLY = 14
comptime TREE_OP_FIRST = TREE_OP_BOOTSTRAP
comptime TREE_OP_LAST = TREE_OP_SEARCHER_APPLY
comptime N_TREE_OPS = TREE_OP_LAST - TREE_OP_FIRST + 1


struct TreeOpInfo(ImplicitlyCopyable, Movable):
    """One row of the tree-op table."""

    var code: Int
    #: the op's name (loop timers, traces, refusals)
    var name: StaticString
    #: the enqueue function and its module
    var entry: StaticString
    #: True: writes carried state and is behind the gate
    var gated: Bool
    #: True: writes a failure word for `step_fail_joins_loop`
    var status: Bool
    #: True: the op is the round's criterion (sets the done word itself)
    var criterion: Bool

    def __init__(
        out self, code: Int, name: StaticString, entry: StaticString, gated: Bool, status: Bool,
        criterion: Bool,
    ):
        self.code = code
        self.name = name
        self.entry = entry
        self.gated = gated
        self.status = status
        self.criterion = criterion

