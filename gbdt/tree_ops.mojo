# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The tree-builder op table of the executor (lane t1-tree-builder-ops,
2026-11-08; docs/plans/EXECUTOR_T1_TREE_OPS.md, plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 4 rows "GBDT rounds"
and "RandomForest / ExtraTrees / DecisionTree levels").

A boosting round becomes one `LoopStep` of `autograd/program_loop_device.mojo`
built from these ops, in the order of docs/plans/EXECUTOR_T1_TREE_OPS.md
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
# round's order: docs/plans/EXECUTOR_T1_TREE_OPS.md section 3 is the order.
comptime TREE_OP_LEAF_ESTIMATE = 9
comptime TREE_OP_SEED_SNAPSHOT = 10
comptime TREE_OP_FOREST_BATCHES = 11
comptime TREE_OP_FOREST_FINISH = 12
comptime TREE_OP_FIRST = TREE_OP_BOOTSTRAP
comptime TREE_OP_LAST = TREE_OP_FOREST_FINISH
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


def _tree_info[CODE: Int]() -> TreeOpInfo:
    comptime if CODE == TREE_OP_BOOTSTRAP:
        return TreeOpInfo(CODE, "tree_bootstrap", "gbdt.gpu_util.kernel.bootstrap.tree_op_bootstrap", True, False, False)
    elif CODE == TREE_OP_SEARCH:
        return TreeOpInfo(
            CODE, "tree_search",
            "gbdt.methods.oblivious_tree_doc_parallel_structure_searcher.fit_oblivious_tree_structure_traced(enqueue_only=True)",
            False, True, False,
        )
    elif CODE == TREE_OP_GATE_WALK:
        return TreeOpInfo(CODE, "tree_gate_walk", "gbdt.methods.kernel.tree_gate_walk.tree_op_gate_walk", True, True, False)
    elif CODE == TREE_OP_TREE_RECORDS:
        return TreeOpInfo(CODE, "tree_records", "gbdt.methods.kernel.tree_gate_walk.tree_op_tree_records", True, False, False)
    elif CODE == TREE_OP_LEAF_BINS:
        return TreeOpInfo(CODE, "tree_leaf_bins", "gbdt.methods.kernel.tree_gate_walk.tree_op_leaf_bins", True, False, False)
    elif CODE == TREE_OP_PARTITION:
        return TreeOpInfo(
            CODE, "tree_partition", "gbdt.methods.leaves_estimation.doc_parallel_leaves_estimator.tree_op_partition",
            False, True, False,
        )
    elif CODE == TREE_OP_APPLY:
        return TreeOpInfo(CODE, "tree_apply", "gbdt.methods.kernel.tree_gate_walk.tree_op_apply", True, False, False)
    elif CODE == TREE_OP_HELDOUT_LOSS:
        return TreeOpInfo(
            CODE, "tree_heldout_loss", "gbdt.methods.doc_parallel_boosting.tree_op_heldout_loss", True, False, False,
        )
    elif CODE == TREE_OP_OD_STEP:
        return TreeOpInfo(
            CODE, "tree_od_step", "gbdt.overfitting_detector.od_device.od_op_step", True, False, True,
        )
    elif CODE == TREE_OP_LEAF_ESTIMATE:
        # round-local walk; its two leaf outputs and its status word are
        # written only while the gate is open
        return TreeOpInfo(
            CODE, "tree_leaf_estimate", "gbdt.methods.doc_parallel_boosting.tree_op_leaf_estimate", False, True, False,
        )
    elif CODE == TREE_OP_SEED_SNAPSHOT:
        return TreeOpInfo(
            CODE, "tree_seed_snapshot", "gbdt.gpu_util.kernel.bootstrap.tree_op_seed_snapshot", True, False, False,
        )
    elif CODE == TREE_OP_FOREST_BATCHES:
        # the node append (`launch_loop_push`; ExtraTrees `enqueue_push`) and
        # the queue / node-count bookkeeping in the device header; inert once
        # the queue is empty (the forest's own gate), the overflow word is
        # its status and queue-empty its criterion, read every LOOP_K batches
        return TreeOpInfo(
            CODE, "forest_batches",
            "ensemble.decisiontree.batched_levelalgo.builder.Builder._enqueue_loop_batches | extratrees.impl.decisiontree.batched_levelalgo.builder._et_run_depthwise (cycle body)",
            True, True, True,
        )
    else:
        return TreeOpInfo(
            CODE, "forest_finish",
            "ensemble.decisiontree.batched_levelalgo.builder.Builder._finish_tree_device(defer_finish=True) + collect_finished",
            False, False, False,
        )


def tree_op_info(code: Int) raises -> TreeOpInfo:
    """The table row of tree op `code`."""
    comptime for i in range(N_TREE_OPS):  # driver-loop: over the tree-op table (thirteen rows, compile time)
        comptime CODE = TREE_OP_FIRST + i
        if code == CODE:
            return _tree_info[CODE]()
    raise Error("tree ops: " + String(code) + " is not a tree op")


def tree_op_name(code: Int) raises -> String:
    """The op's name, for `loop_lap` timers and refusals."""
    return String(tree_op_info(code).name)
