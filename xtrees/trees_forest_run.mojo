# SPDX-License-Identifier: Apache-2.0
"""RUN of the forest door (bindings/trees_public_forest.mojo): the fit and
predict request values and the decisions that read only typed values (the
export size checks, the verify verdict, the export selection, the fit
notice bits). The door's UNPACK admits X, y and the estimator's fields;
its PACK wraps the exported model words and issues the warnings this
module returns as codes. No Python import: `tools/python_free_dirs.sh`
covers this directory.

Lane doorsplit-trees (2026-11-11): moved from the door. Each function is a
per-call scalar decision over one fit's metadata; there is no independent
work to split across cores.
"""
from std.os import getenv
from core.entry_table import EntryRef
from core.forest_calls import ForestExport, xf_fit
from core.forest_fit_request import ForestFitArgs
from ensemble.nan_refusal import RF_NONFINITE_REFUSAL
from ensemble.rf_notices import (
    RF_NOTICE_ALL, RF_NOTICE_BINS_CLAMPED, RF_NOTICE_MAX_SAMPLES_IGNORED, RF_NOTICE_OOB_PARTIAL,
)

comptime FOREST_MODEL_FIELDS = 5


struct ForestFit(Copyable, Movable):
    """The fit an operation launches, as the image's native request words
    (XF_FIT): classifier or regressor, whether the image may read X
    row-major (the plain and weighted ET fits and the plain RF fits; the
    fit reads it row-major when X arrives C-ordered), and the trailing
    words (row weights, a shard start, the auxiliary outputs, the largest
    weight). Lane calltable-forest: this replaced the compiled entry name
    and its trailing Python argument list."""
    var classifier: Bool
    var rowmajor_ok: Bool
    var weights: Int
    var tree_start: Int
    var oob: Int
    var score: Int
    var importance: Int
    var weight_max: Float64

    def __init__(out self, classifier: Bool, rowmajor_ok: Bool = False, weights: Int = 0, tree_start: Int = 0,
                 oob: Int = 0, score: Int = 0, importance: Int = 0, weight_max: Float64 = 0.0):
        self.classifier = classifier
        self.rowmajor_ok = rowmajor_ok
        self.weights = weights
        self.tree_start = tree_start
        self.oob = oob
        self.score = score
        self.importance = importance
        self.weight_max = weight_max


struct ForestHandle(Movable):
    """A forest snapshot a predict launches on: its handle, and whether the
    resident table keeps it (pinned for the call) or the call owns it alone
    (released when the call ends through the image's native release
    `release_op` (XF_RESIDENT_RELEASE / XF_STRICT_RELEASE): a model that
    takes no weak reference, or a snapshot over the table's byte cap). `key`
    is the forest entry resolved once for the operation (lane
    entry-table-zero-alloc)."""
    var handle: Int
    var kept: Bool
    var key: EntryRef
    var release_op: Int

    def __init__(out self, handle: Int, kept: Bool, key: EntryRef, release_op: Int):
        self.handle = handle
        self.kept = kept
        self.key = key.copy()
        self.release_op = release_op


def forest_fit_core(key: EntryRef, classifier: Bool, row_major: Bool, args: ForestFitArgs,
                    device_scan: Bool) raises -> Int:
    """THE forest fit (lane forest-typed-fit, 2026-11-12): one typed request
    (`ForestFitArgs`: X and y addresses, the params views, the row weights,
    the tree-ID start and the auxiliary outputs) launched in the image `key`
    names (XF_FIT). Returns the export handle the caller's PACK copies and
    releases. The public forest door (bindings/trees_public_forest.mojo
    `_launch`) and the tree-range shard worker (`forest_shard_fit`, called
    by bindings/parallel_shard_native.mojo) both call this; neither builds a
    model object for it. An image that scans X on the device reports a
    non-finite cell inside its own error text; it is raised as the one
    refusal the doors match on. One synchronous launch per call: the
    parallelism is inside the image's fit."""
    try:
        return xf_fit(key, classifier, row_major, args)
    except error:
        if device_scan and RF_NONFINITE_REFUSAL in String(error):
            raise Error(RF_NONFINITE_REFUSAL)
        raise error^


def forest_export_selection() raises -> String:
    """`MOJOLEARN_FOREST_EXPORT`: 'into' (default) or 'verify'."""
    # TOMBSTONE: the MOJOLEARN_FOREST_EXPORT=legacy arm and its plain rf_/et_*_fit entries (dead native entries: no release, identity or CI path set it) deleted 2026-10-26 by f515bbcb9; code recoverable at a88745022.
    # Restore: git apply experiments/removed/dead_entries_trees.patch
    var value = getenv("MOJOLEARN_FOREST_EXPORT")
    var selection = value if value else String("into")
    if selection != "into" and selection != "verify":
        raise Error("MOJOLEARN_FOREST_EXPORT must be into or verify")
    return selection


def forest_export_counts(sizes: ForestExport) raises -> List[Int]:
    """The five model arrays' element counts of one export (offsets, colid,
    quesval, left_child, leaves), after the size and metadata checks."""
    var trees = sizes.trees
    var nodes = sizes.nodes
    var outputs = sizes.outputs
    if not (1 <= trees and trees < 2147483647 and nodes >= trees and outputs >= 1
            and nodes <= 2147483647 // outputs):
        raise Error("native forest export exceeds supported counts")
    if len(sizes.meta) < 2 or sizes.meta[0] != trees or sizes.meta[1] != outputs:
        raise Error("native forest export metadata disagrees with counts")
    var counts: List[Int] = [trees + 1, nodes, nodes, nodes, nodes * outputs]
    return counts^


def forest_export_dtype(field: Int) -> String:
    """The flat dtype of model array `field` (`_offsets` .. `_leaves`)."""
    if field == 2 or field == 4:
        return "<f4"
    return "<i4"


def forest_export_verdict_check(verdict: Int) raises:
    """XF_EXPORT_VERIFY's verdict: 0 agrees, -1 legacy metadata, k > 0 the
    k-th model array differs."""
    var names: List[String] = ["_offsets", "_colid", "_quesval", "_left_child", "_leaves"]
    if verdict == -1:
        raise Error("RuntimeError: forest export legacy metadata mismatch")
    if verdict > 0:
        raise Error("RuntimeError: forest export byte mismatch: " + names[verdict - 1])


def forest_fit_notice_codes(mask: Int) raises -> List[Int]:
    """The fit's notices (the third metadata word, an
    `ensemble/rf_notices.mojo` bit mask) as notice codes in the order the
    fit reached them; PACK issues each as a UserWarning."""
    if mask < 0 or (mask & ~RF_NOTICE_ALL) != 0:
        raise Error("forest fit returned unknown notice bits " + String(mask))
    var out = List[Int]()
    var codes: List[Int] = [RF_NOTICE_BINS_CLAMPED, RF_NOTICE_MAX_SAMPLES_IGNORED, RF_NOTICE_OOB_PARTIAL]
    for i in range(len(codes)):  # small-loop(codes): the three fit notice bits
        if (mask & codes[i]) != 0:
            out.append(codes[i])
    return out^


def trees_refuse(name: String, why: String) raises:
    """`randomforest._refuse`: a parameter refused by name."""
    raise Error("NotImplementedError: " + name + " is not implemented: " + why
                + " Refused by name rather than accepted and ignored.")


def trees_check_features(cols: Int, fitted: Int) raises:
    """A predict's feature-count refusal against the fitted width."""
    if cols != fitted:
        raise Error("X has " + String(cols) + " features, fit saw " + String(fitted))
