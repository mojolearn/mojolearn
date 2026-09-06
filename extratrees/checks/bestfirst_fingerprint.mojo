# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The BEFORE side of DEVIATION 466's default-bits gate.

Prints two 64-bit digests of one DEFAULT ExtraTrees forest -- every node
field and every leaf bit, folded as exact bits -- one for the HOST arm and
one for the DEVICE arm. It imports NOTHING that DEVIATION 466 added, so it
runs unmodified on the tree before that change and on the tree after it, and
the numbers must be equal on both sides.

BOTH ARMS, and the DEVICE one is the one that matters. The host arm gained
only a dispatch at the top of `train_classification`; the device driver's
LEVEL CYCLE was edited in place, with the depth-wise code moved inside an
`else`. A host-only fingerprint would be the weaker half of the claim.

    # on the PRE-patch tree
    pixi run mojo run -I . extratrees/checks/bestfirst_fingerprint.mojo
    # paste the number into DEFAULT_FINGERPRINT in bestfirst_check.mojo

The fixture, the config and the seed are printed beside the number so a
mismatch cannot be blamed on a drifted input. This file is a PROBE, not a
check: it asserts nothing and has no PASS line. `bestfirst_check.mojo` is
where the number becomes a gate.

NO DURATION IS TAKEN HERE.
"""

from std.sys.info import has_accelerator
from max.gpu.host import DeviceContext

from extratrees.checks.fixtures import (
    Dataset as FixtureDataset,
    hashed_classification,
)
from extratrees.estimator import (
    ExtraTreesConfig,
    MAX_FEATURES_ALL,
    fit_extra_trees_classifier,
    fit_extra_trees_classifier_device,
)
from extratrees.impl.randomforest.randomforest import Forest


comptime FP_SEED: UInt64 = 0xB3E5F1
comptime FP_ROWS: Int = 512
comptime FP_COLS: Int = 6
comptime FP_CLASSES: Int = 3
comptime FP_TREES: Int32 = 6
comptime FP_DEPTH: Int32 = 7


def column_major(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32](
        length=fixture.n_rows * fixture.n_cols, fill=Float32(0.0)
    )
    for r in range(fixture.n_rows):
        for c in range(fixture.n_cols):
            out[c * fixture.n_rows + r] = fixture.value(r, c)
    return out^


def float_labels(fixture: FixtureDataset) -> List[Float32]:
    var out = List[Float32]()
    for r in range(fixture.n_rows):
        out.append(Float32(Int(fixture.label[r])))
    return out^


def mix64(h_in: UInt64, v: UInt64) -> UInt64:
    var h = h_in ^ v
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB
    return h ^ (h >> 31)


def forest_fingerprint(forest: Forest) -> UInt64:
    """MUST STAY BYTE-FOR-BYTE THE SAME FUNCTION as the copy in
    `bestfirst_check.mojo`. It is duplicated on purpose: this file has to
    compile against the PRE-patch tree, so it cannot import from anything
    the patch touched."""
    var h = UInt64(0x243F6A8885A308D3)
    h = mix64(h, UInt64(len(forest.trees)))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        h = mix64(h, UInt64(Int(tree.treeid)))
        h = mix64(h, UInt64(Int(tree.depth_counter)))
        h = mix64(h, UInt64(Int(tree.leaf_counter)))
        h = mix64(h, UInt64(Int(tree.num_outputs)))
        h = mix64(h, UInt64(tree.num_nodes()))
        for i in range(tree.num_nodes()):
            ref n = tree.sparsetree[i]
            # The three integer fields are masked to 32 bits rather than
            # widened: `colid` and `left_child_id` are -1 on a leaf, and a
            # sign-extended -1 would fold in 32 bits of 1s that say nothing.
            # (Mojo int widening sign-extends -- the traps register.)
            h = mix64(h, UInt64(Int(n.colid) & 0xFFFFFFFF))
            h = mix64(h, UInt64(n.quesval.to_bits[DType.uint32]()))
            h = mix64(h, UInt64(n.best_metric_val.to_bits[DType.uint32]()))
            h = mix64(h, UInt64(Int(n.left_child_id) & 0xFFFFFFFF))
            h = mix64(h, UInt64(Int(n.instance_count) & 0xFFFFFFFF))
        h = mix64(h, UInt64(len(tree.vector_leaf)))
        for i in range(len(tree.vector_leaf)):
            h = mix64(h, UInt64(tree.vector_leaf[i].to_bits[DType.uint32]()))
    return h


def default_config() -> ExtraTreesConfig:
    """MUST STAY THE SAME CONFIG as `bestfirst_check.mojo`'s
    `bf_config(-1)`. Only fields that exist on both sides of the patch are
    set, and `max_leaf_nodes` is deliberately left at its default."""
    var c = ExtraTreesConfig()
    c.n_estimators = FP_TREES
    c.max_depth = FP_DEPTH
    c.max_features_spec = MAX_FEATURES_ALL
    c.random_state = FP_SEED
    return c^


def main() raises:
    comptime assert has_accelerator(), "the device fingerprint needs a GPU"
    var ctx = DeviceContext()
    var clf = hashed_classification(FP_SEED, FP_ROWS, FP_COLS, FP_CLASSES)
    var xc = column_major(clf)
    var lab = float_labels(clf)
    var fit = fit_extra_trees_classifier(
        xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        default_config(),
    )
    var dfit = fit_extra_trees_classifier_device(
        ctx, xc, lab, Int32(FP_ROWS), Int32(FP_COLS), Int32(FP_CLASSES),
        default_config(),
    )
    print("device", ctx.name())
    print(
        "fixture hashed_classification(seed=", FP_SEED, "rows=", FP_ROWS,
        "cols=", FP_COLS, "classes=", FP_CLASSES, ")",
    )
    print(
        "config n_estimators=", FP_TREES, "max_depth=", FP_DEPTH,
        "max_features=all random_state=", FP_SEED,
    )
    var total = 0
    for t in range(len(fit.forest.trees)):
        total += fit.forest.trees[t].num_nodes()
    print("nodes", total, "trees", len(fit.forest.trees))
    print("DEFAULT_FINGERPRINT_HOST", forest_fingerprint(fit.forest))
    print("DEFAULT_FINGERPRINT_DEVICE", forest_fingerprint(dfit.forest))
