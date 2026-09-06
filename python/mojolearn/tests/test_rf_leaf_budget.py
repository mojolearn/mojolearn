# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the leaf budget on `mojolearn.RandomForest*`.

Written 2026-09-01, the day DEVIATION 408 closed. The model for this file is
`python/mojolearn/tests/test_svr_surface.py` and it follows the same house
rules: every arm runs, every verdict prints, and a bitwise question is only
ASKED under `identical`.

WHAT THIS CLOSES
----------------
`python/mojolearn/randomforest.py` used to map sklearn's `max_leaf_nodes`
straight onto cuML's `max_leaves`:

    max_leaves=(-1 if max_leaf_nodes is None else int(max_leaf_nodes))

Those are not the same parameter. sklearn's `max_leaf_nodes=k` is a GROWTH
ORDER: `tree/_classes.py:446-447` reads "Use BestFirst if max_leaf_nodes
given; use DepthFirst otherwise" and constructs `BestFirstTreeBuilder`,
which expands the frontier node with the largest impurity improvement until
exactly k leaves exist. cuML's `max_leaves` is a CAP on a LEVEL-ORDER
grower that reorders nothing: `NodeQueue::Pop` takes from the FRONT of a
FIFO and `Push` appends to the BACK (`builder.cuh:70-78`, `:117`,
transcribed in `ensemble/decisiontree/batched_levelalgo/builder.mojo`), and
the budget is spent by whichever nodes that order reaches first
(`IsExpandable` at `:82-88`, and the `break` inside `Push` at `:101`). Same
k, different tree, no error and no warning: a caller who asked for a leaf
budget got a different algorithm and could not tell.

WHAT THIS FILE WOULD HAVE CAUGHT, EXACTLY
-----------------------------------------
Run against the code as it stood on 2026-08-31, the REFUSAL arm goes RED
and this program exits 1. `RandomForestClassifier(max_leaf_nodes=64)`
constructed without complaint on that code, so `Report.raises` prints

    FAIL max_leaf_nodes is refused by name on the classifier
         -- IS INERT: the call was ACCEPTED

for both estimator classes and for both an int and a numpy int. Nothing
else in the file needs to move for the verdict: the defect was that a
refusal did not exist, and the arm that fires it is the arm that finds it.

The LEVEL_ORDER arm is the other half of the same statement and it is
GREEN in both eras, deliberately. It measures what the grower actually does
-- nodes are appended in breadth-first order, so within one tree the node
index is non-decreasing in depth -- and that is the property that makes the
old alias WRONG rather than merely undocumented. A best-first builder
appends in improvement order and would fail this arm, which is why the two
parameters cannot be spelled alike.

WHY A GREEN RESULT WAS NOT ENOUGH
---------------------------------
THE ALIAS HAD A CERTIFIED GREEN CELL STANDING ON IT FOR MONTHS.
`tools/e2_matrix_fit.py` defines `rf_clf_maxleaf` as
`rf("clf", max_leaf_nodes=64, ...)`, and every recorded round scores it
IDENTICAL (388 stages) on every column -- `bench/results/e1/
e2_verdicts_round1.md`, `e2_verdicts_round2.md` and the dated
`e3_verdicts_trees.md` files. One line above it in the same tables sits its
extratrees twin, `et_clf_maxleaf`, scored REFUSED=, because the ET port
refused the parameter by name until 2026-09-01. A reader of those tables
would conclude that RF had the capability and ET did not. The truth was the
reverse.

IDENTICAL there is a cross-vendor and cross-mode statement about OUR OWN
output. It certifies that the aliased answer is REPRODUCIBLE. It is not a
comparison against sklearn and it cannot see that the answer is to a
different question, and a reproducible wrong answer is exactly the kind
that survives. That is the concrete reason this file exists and the reason
it asks about the CONTRACT (which parameter means what) rather than about
the bits.

THE ARMS, AND WHAT EACH ONE CAN SEE
-----------------------------------
    PROVENANCE   which binary answered and which tier this run reports
                 about, read back off the binding rather than off the
                 environment variable that asked for it
    REFUSAL      `max_leaf_nodes` is refused by name on both classes, with
                 a message that names best-first growth, names `max_leaves`
                 as cuML's different knob and names the surface where
                 sklearn's semantics do exist. THE ARM THAT GOES RED ON THE
                 OLD BEHAVIOUR
    VALIDATION   `max_leaves` accepts cuML's sentinel and a positive count
                 and refuses everything else HERE, so the caller never
                 meets `DecisionTreeParams.check()`'s "Invalid max leaves"
    LEVEL_ORDER  the grower is level-order: node index is non-decreasing in
                 depth within every tree, capped and uncapped alike. The
                 positive evidence that this is not a best-first builder
    BUDGET       `max_leaves=k` REACHES the builder and cuts the forest --
                 every tree has at most k leaves, at least one has exactly
                 k, and the same fit without the cap has more. Reach is
                 shown by flipping the knob and watching the tree move, not
                 by reading the parameter back out of `_cfg`
    CONTRAST     REPORT ONLY: what `ExtraTreesClassifier(max_leaf_nodes=k)`
                 returns for the same k, which is sklearn's semantics --
                 exactly k leaves per tree. Skipped, not failed, if that
                 surface or its binding is mid-flight

WHAT IS ASSERTED AND WHAT IS REPORTED
-------------------------------------
Every arm here asks a STRUCTURAL question -- a refusal, an exception type,
a leaf count, a depth ordering -- and none of them is a bitwise question,
so all of them are asserted in every tier. That is deliberate:
`[[fast-is-not-identical]]` forbids asking a FAST arm a bitwise question,
and the contract this file gates is not a numeric one. The byte-level
difference between the capped and uncapped forests is REPORTED and never
asserted, because under FAST the bits are allowed to move.

WHAT THIS DOES NOT PROVE
------------------------
- **It is not a comparison against sklearn.** Nothing here fits an sklearn
  forest. It could not: `ensemble/` splits over at most `n_bins` per-feature
  quantiles where sklearn searches exact thresholds, so the trees would
  differ for a second reason and the arm would prove nothing. What is
  asserted is which SEMANTICS this surface claims, not that our best-first
  matches theirs -- this surface has no best-first at all, which is the
  point.
- **It is one machine and one vendor.** No cross-vendor claim is made or
  needed; the level-order property is a property of the host control plane
  in `builder.mojo`, which is the same source on every column.
- **It does not gate the builder.** `ensemble/checks/builder_check.mojo`
  does that, arm C, with the `max_leaves` budget sabotaged in the shipped
  code path.

OPEN, AND OWED TO ANOTHER LANE
------------------------------
`tools/e2_matrix_fit.py:451` still spells the cell
`rf("clf", max_leaf_nodes=64, ...)` and will now RAISE instead of fitting.
The one-line fix is `max_leaves=64`, which sets the identical slot-5 value
the alias used to set, so the cell's own hash does not move; it is outside
this lane's editable set and is recorded in the handoff rather than applied
here.

HOW TO RUN IT
-------------
    # the extension, if it is not already built (fast is the default tier)
    bash bindings/build_rf.sh

    # the gate
    cd python && python3 -m mojolearn.tests.test_rf_leaf_budget

ENVIRONMENT
    MOJOLEARN_NUMERIC_MODE   fast (default), deterministic or identical
"""

import os
import sys

import numpy as np

from mojolearn import RandomForestClassifier, RandomForestRegressor
from mojolearn import _backend


# ===========================================================================
# REPORTING
# ===========================================================================
# EVERY ARM RUNS AND EVERY VERDICT IS PRINTED BEFORE THE PROCESS EXITS
# NON-ZERO, the same rule `test_svr_surface` and `test_linalg_identity`
# follow: stopping at the first failure shows one arm's opinion when the
# useful evidence is WHICH arms a defect reaches and which it walks past.


class Report(object):
    def __init__(self):
        self.rows = []
        self.notes = []

    def ok(self, arm, what):
        self.rows.append((True, arm, what))

    def bad(self, arm, what):
        self.rows.append((False, arm, what))

    def note(self, text):
        self.notes.append(text)

    def check(self, arm, cond, what, detail=""):
        if cond:
            self.ok(arm, what)
        else:
            self.bad(arm, what + ((" -- " + detail) if detail else ""))
        return bool(cond)

    def report_only(self, arm, what, detail=""):
        """Something this file MEASURES and does not judge. Printed with its
        value and never counted as a failure."""
        self.rows.append((True, arm, "[REPORT, not asserted] %s%s"
                          % (what, (" -- " + detail) if detail else "")))

    def raises(self, arm, exc_type, needles, what, fn, *a, **kw):
        """A refusal that never fires is not a refusal.

        `needles` is every phrase the message must carry, because the point
        of this particular refusal is WHAT IT SAYS: a bare
        NotImplementedError would leave the caller with no way to reach the
        capability they asked for, and this refusal's whole job is to name
        cuML's different knob and the surface that has sklearn's.
        """
        try:
            fn(*a, **kw)
        except exc_type as exc:
            missing = [n for n in needles if n not in str(exc)]
            if not missing:
                self.ok(arm, what)
                return True
            self.bad(arm, "%s -- raised %s but the message does not name %s:"
                     " %s" % (what, exc_type.__name__,
                              ", ".join(repr(m) for m in missing), exc))
            return False
        except Exception as exc:  # noqa: BLE001 - the wrong exception is data
            self.bad(arm, "%s -- raised %s, want %s: %s"
                     % (what, type(exc).__name__, exc_type.__name__, exc))
            return False
        self.bad(arm, "%s -- IS INERT: the call was ACCEPTED" % what)
        return False

    @property
    def failures(self):
        return [r for r in self.rows if not r[0]]

    def render(self, out):
        arm = None
        for ok, a, what in self.rows:
            if a != arm:
                out.write("\n  %s\n" % a)
                arm = a
            out.write("    %s %s\n" % ("pass" if ok else "FAIL", what))
        for n in self.notes:
            out.write("\n%s\n" % n)
        out.write("\n  %d checks, %d failed\n"
                  % (len(self.rows), len(self.failures)))


class GateAbort(Exception):
    """A condition under which no verdict about the leaf budget can be
    reached at all, as distinct from a failing check. A failure says the
    surface is wrong; an abort says the gate did not run."""


# ===========================================================================
# THE FIXTURE
# ===========================================================================
# HASHED, NOT `np.random`. The BUDGET arm compares two fits of the SAME data
# at two settings of one knob, so a reseeded generator would make the
# comparison about the data instead of about the knob. Same construction as
# `test_svr_surface.hashed_unit`: 21 bits, so the numerator is exact in a
# float32 mantissa, over a power of two, so the division is exact and
# nothing rounds upstream of the thing being measured.

_MASK64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def hashed_unit(count, salt):
    """`count` values in [-1, 1), a pure function of `(count, salt)`."""
    idx = np.arange(count, dtype=np.uint64) + np.uint64(1)
    z = (idx * np.uint64(_GOLDEN) + np.uint64(salt & _MASK64))
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(_MIX_A)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(_MIX_B)
    z = z ^ (z >> np.uint64(31))
    q = (z >> np.uint64(43)).astype(np.int64)
    return ((q - 1048576).astype(np.float32) / np.float32(1048576.0))


#: The fixture's shape. `N` and `K` are distinct and neither is a power of
#: two, so a transposed matrix cannot land on a self-consistent call.
FIX_N = 2000
FIX_K = 6

#: The forest. Small on purpose -- this file asks a contract question and a
#: bigger fit would answer it no better while costing the box a window.
#: `max_depth=8` is what makes the uncapped tree grow well past `BUDGET_K`,
#: which is the precondition the BUDGET arm needs to mean anything.
FIT_TREES = 4
FIT_DEPTH = 8
FIT_SEED = 7

#: The cap. Small enough to bite well inside `max_depth=8` and large enough
#: that a tree reaching it has passed through several levels.
BUDGET_K = 12


def fixture():
    """`(X, y_clf, y_reg)`, a pure function of nothing but this file."""
    x = hashed_unit(FIX_N * FIX_K, 0x2C4B).reshape(FIX_N, FIX_K)
    w = np.array([0.9, -0.6, 0.35, -1.1, 0.45, 0.2], dtype=np.float32)
    signal = (x @ w).astype(np.float32)
    noise = hashed_unit(FIX_N, 0x7A19) * np.float32(0.25)
    y_reg = (signal + noise).astype(np.float32)
    y_clf = (y_reg > np.float32(0.0)).astype(np.int32)
    return x, y_clf, y_reg


# ===========================================================================
# READING A FITTED FOREST
# ===========================================================================
# The five arrays `_fit_arrays` keeps are the model, laid out by
# `bindings/_mojolearn_rf.mojo::_forest_out`: `_offsets` is a prefix of
# length n_trees + 1, and within one tree the node index is TREE-LOCAL --
# `_rebuild_trees` on the predict side re-slices by the same offsets. A node
# is a LEAF iff its left child is -1 (`cuml cpp/include/cuml/tree/
# flatnode.h:58`, and it is a flag nowhere), and the right child is
# `left + 1` (`:45`), which is why nothing here needs a right-child array.


def tree_slices(model):
    """`[(left_child_of_tree_t)]`, one int32 array per tree."""
    off = np.asarray(model._offsets, dtype=np.int64)
    lc = np.asarray(model._left_child, dtype=np.int64)
    return [lc[off[t]:off[t + 1]] for t in range(int(model._n_trees))]


def leaf_count(left_child):
    """Leaves in one tree. Their `leaf_counter` in another spelling: it
    starts at 1 and rises by one per SPLIT (`builder.cuh:111`), and a split
    turns one leaf into an internal node plus two leaves, so the two counts
    agree by construction and disagreeing would itself be a finding."""
    return int(np.count_nonzero(left_child < 0))


def node_depths(left_child):
    """Depth of every node, root at 0, or `None` if the tree is malformed.

    Walked rather than stored, because the depth is not in the model: the
    model is a flat node array and the depth is a property of the SHAPE,
    which is what the LEVEL_ORDER arm is asking about.
    """
    n = int(left_child.shape[0])
    depth = [-1] * n
    if n == 0:
        return depth
    depth[0] = 0
    stack = [0]
    while stack:
        i = stack.pop()
        left = int(left_child[i])
        if left < 0:
            continue
        right = left + 1
        if left >= n or right >= n or depth[left] >= 0 or depth[right] >= 0:
            return None  # not a tree: a re-entered or out-of-range child
        depth[left] = depth[i] + 1
        depth[right] = depth[i] + 1
        stack.append(left)
        stack.append(right)
    if any(d < 0 for d in depth):
        return None  # unreachable nodes
    return depth


# ===========================================================================
# ARM: PROVENANCE
# ===========================================================================


def arm_provenance(rep):
    """Which binary answered, and therefore which tier this run is about.

    Read back off the BINDING the estimator holds, never off the
    environment variable that asked for it. `_mojolearn_rf` exports no
    `rf_numeric_mode`, so the tier is resolved the way every estimator
    resolves it, through `NumericModeMixin.numeric_mode_used`, which reads
    the loaded module's own path rather than a string that was passed in.
    """
    arm = "PROVENANCE"
    asked = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower()
    try:
        est = RandomForestClassifier()
        ext = est._bind("_mojolearn_rf")
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        raise GateAbort("the _mojolearn_rf extension did not load: %s" % exc)
    rep.check(arm, hasattr(ext, "rf_classifier_fit"),
              "the loaded binary exports rf_classifier_fit")
    rep.check(arm, hasattr(ext, "rf_regressor_fit"),
              "the loaded binary exports rf_regressor_fit")
    compiled = est.numeric_mode_used()
    rep.check(arm, compiled in ("fast", "identical", "deterministic"),
              "the binding reports a tier it knows", str(compiled))
    rep.note("  binary tier   %s\n  env asked for %s\n  vendor        %s"
             % (compiled, asked, _backend.read_vendor(ext)))
    return compiled


# ===========================================================================
# ARM: REFUSAL -- THE ARM THAT GOES RED ON THE OLD BEHAVIOUR
# ===========================================================================


def arm_refusal(rep):
    """`max_leaf_nodes` is refused by name, on both classes, with a message
    that leaves the caller somewhere to go.

    On the code before DEVIATION 408 every row in this arm reads
    "IS INERT: the call was ACCEPTED", because the constructor took the
    value and wrote it into slot 5 as though it were cuML's `max_leaves`.

    The message needles are asserted, not just the exception. A refusal
    whose message did not name `max_leaves` and did not name the extratrees
    surface would be a second silent failure in a politer form: the caller
    would learn that they cannot have what they asked for and not that the
    library has it one import away.
    """
    arm = "REFUSAL"
    # Matched VERBATIM against `randomforest.py::_MAX_LEAF_NODES_WHY`,
    # capitalization included. A case-insensitive needle would pass on a
    # message that had been reworded into something vaguer.
    needles = ("max_leaf_nodes", "BEST-FIRST GROWTH", "max_leaves",
               "ExtraTreesClassifier", "LEVEL-ORDER")
    for name, cls in (("classifier", RandomForestClassifier),
                      ("regressor", RandomForestRegressor)):
        rep.raises(arm, NotImplementedError, needles,
                   "max_leaf_nodes=64 is refused by name on the %s" % name,
                   cls, max_leaf_nodes=64)
        rep.raises(arm, NotImplementedError, ("max_leaf_nodes",),
                   "a numpy int max_leaf_nodes is refused on the %s too"
                   " (the guard tests for None, not for a Python int)"
                   % name,
                   cls, max_leaf_nodes=np.int64(64))
        rep.raises(arm, NotImplementedError, ("max_leaf_nodes",),
                   "max_leaf_nodes=2, the smallest value sklearn accepts,"
                   " is refused on the %s" % name,
                   cls, max_leaf_nodes=2)
    # And the default is not refused: a guard that fired on None would
    # break every caller and would pass the three rows above.
    for name, cls in (("classifier", RandomForestClassifier),
                      ("regressor", RandomForestRegressor)):
        try:
            cls(max_leaf_nodes=None)
            rep.ok(arm, "max_leaf_nodes=None, the default, still constructs"
                        " the %s" % name)
        except Exception as exc:  # noqa: BLE001 - a refusal here is the bug
            rep.bad(arm, "max_leaf_nodes=None was refused on the %s: %s"
                    % (name, exc))


# ===========================================================================
# ARM: VALIDATION
# ===========================================================================


def arm_validation(rep):
    """`max_leaves` takes cuML's own sentinel and nothing loose.

    The values are refused on THIS side of the boundary so the message can
    name the sentinel; `DecisionTreeParams.check()`
    (`ensemble/decisiontree/decisiontree.mojo:257-258`) would refuse them
    too, with "Invalid max leaves" and no way to know what to pass instead.
    Both refusals existing is the point -- this one is not load-bearing for
    correctness and is load-bearing for the caller.
    """
    arm = "VALIDATION"
    for val, why in ((0, "zero"), (-2, "a sentinel that is not theirs"),
                     (-100, "a large negative")):
        rep.raises(arm, ValueError, ("max_leaves",),
                   "max_leaves=%d (%s) is refused" % (val, why),
                   RandomForestClassifier, max_leaves=val)
    rep.raises(arm, ValueError, ("max_leaves",),
               "max_leaves=8.0, a float, is refused rather than truncated",
               RandomForestClassifier, max_leaves=8.0)
    rep.raises(arm, ValueError, ("max_leaves",),
               "max_leaves=True is refused: a bool is an int in Python and"
               " would otherwise become a one-leaf budget",
               RandomForestClassifier, max_leaves=True)
    for val, why in ((-1, "cuML's sentinel for unlimited"),
                     (None, "None, accepted as a spelling of -1"),
                     (1, "the smallest cap"),
                     (np.int64(32), "a numpy int")):
        try:
            est = RandomForestClassifier(max_leaves=val)
            got = est._cfg["max_leaves"]
            want = -1 if val is None else int(val)
            rep.check(arm, got == want,
                      "max_leaves=%r (%s) reaches slot 5 as %d"
                      % (val, why, want), "slot 5 holds %r" % (got,))
        except Exception as exc:  # noqa: BLE001 - a refusal here is the bug
            rep.bad(arm, "max_leaves=%r (%s) was refused: %s"
                    % (val, why, exc))


# ===========================================================================
# ARM: LEVEL_ORDER
# ===========================================================================


def arm_level_order(rep, fits):
    """The grower is LEVEL-ORDER, which is why the alias was wrong.

    `NodeQueue` is a FIFO: `Pop` takes from the front and `Push` appends to
    the back (`builder.cuh:70-78`, `:117`), and a child's id is
    `tree->sparsetree.size()` read at the moment it is appended
    (`:105-110`). So nodes are appended in breadth-first order and, within
    one tree, THE NODE INDEX IS NON-DECREASING IN DEPTH. That is a
    signature, not a coincidence: a best-first builder appends in
    improvement order, so its node index says nothing about depth, and this
    arm would go red on one.

    This is the arm that makes the REFUSAL arm mean something. Without it,
    "max_leaf_nodes is refused" is a claim about a wrapper; with it, the
    claim is that the thing on the other side of the wrapper is a different
    algorithm from the one the caller named.
    """
    arm = "LEVEL_ORDER"
    for label, model in fits:
        for t, lc in enumerate(tree_slices(model)):
            depths = node_depths(lc)
            if depths is None:
                rep.bad(arm, "%s tree %d: the node array is not a tree"
                        % (label, t))
                continue
            monotone = all(depths[i] <= depths[i + 1]
                           for i in range(len(depths) - 1))
            rep.check(arm, monotone,
                      "%s tree %d: node index is non-decreasing in depth"
                      " (%d nodes, depth %d)"
                      % (label, t, len(depths), max(depths)),
                      "" if monotone else "first inversion at index %d"
                      % next(i for i in range(len(depths) - 1)
                             if depths[i] > depths[i + 1]))


# ===========================================================================
# ARM: BUDGET
# ===========================================================================


def arm_budget(rep, capped, uncapped):
    """`max_leaves=k` REACHES the builder and cuts the forest.

    Reach is shown by flipping the knob and watching the tree move, not by
    reading the value back out of `_cfg` -- `[[verify-reach-not-output]]`.
    Three things are asserted and each one can fail on its own:

      * every tree has AT MOST k leaves. Their guard is
        `leaf_counter >= max_leaves` tested before each split
        (`builder.cuh:101`) and `leaf_counter` only ever rises by one, so
        the bound is exact and not approximate.
      * at least one tree has EXACTLY k. A cap that never binds would pass
        the bound above while proving nothing, which is
        `[[reached-but-inert]]`.
      * the same fit WITHOUT the cap has more than k leaves in every tree.
        This is the precondition for the other two: if the uncapped forest
        were already under the budget then the cap could be a no-op and
        every row here would still be green.
    """
    arm = "BUDGET"
    cap_counts = [leaf_count(lc) for lc in tree_slices(capped)]
    free_counts = [leaf_count(lc) for lc in tree_slices(uncapped)]
    rep.note("  leaves per tree, max_leaves=%d   %s\n"
             "  leaves per tree, max_leaves=-1   %s"
             % (BUDGET_K, cap_counts, free_counts))
    rep.check(arm, all(c <= BUDGET_K for c in cap_counts),
              "every capped tree has at most %d leaves" % BUDGET_K,
              str(cap_counts))
    rep.check(arm, any(c == BUDGET_K for c in cap_counts),
              "at least one capped tree has exactly %d leaves, so the cap"
              " BINDS rather than sitting above the tree" % BUDGET_K,
              str(cap_counts))
    rep.check(arm, all(c > BUDGET_K for c in free_counts),
              "every uncapped tree grows past %d leaves, so the comparison"
              " above is about the cap and not about the data" % BUDGET_K,
              str(free_counts))
    rep.check(arm, sum(cap_counts) < sum(free_counts),
              "the capped forest is strictly smaller than the uncapped one")
    # REPORTED, never asserted: under FAST the bits are allowed to move, so
    # "the two models differ byte for byte" is not a question this tier can
    # be asked. It is printed because a reader of a green run should see
    # that the two fits are not the same object.
    same = (np.ascontiguousarray(capped._quesval).tobytes()
            == np.ascontiguousarray(uncapped._quesval).tobytes())
    rep.report_only(arm, "the two forests' thresholds differ byte for byte",
                    "IDENTICAL BYTES" if same else "differ")


# ===========================================================================
# ARM: CONTRAST
# ===========================================================================


def arm_contrast(rep):
    """REPORT ONLY: what sklearn's semantics look like, one import away.

    `ExtraTreesClassifier(max_leaf_nodes=k)` is best-first growth to
    exactly k leaves (`extratrees/`, DEVIATION BLOCKS 466 to 469). Printing
    its leaf counts beside the BUDGET arm's is the clearest statement of
    what the two parameters mean and why one may not be spelled as the
    other.

    NOTHING HERE IS ASSERTED and every failure is swallowed with its
    reason. That surface and its binding belong to another lane and may be
    mid-flight; a gate on `randomforest.py` that went red because
    `extratrees.py` was being edited would be a gate about the wrong file.
    """
    arm = "CONTRAST"
    try:
        from mojolearn import ExtraTreesClassifier
    except Exception as exc:  # noqa: BLE001 - report, never fail
        rep.report_only(arm, "ExtraTreesClassifier did not import",
                        "%s: %s" % (type(exc).__name__, exc))
        return
    x, y_clf, _ = fixture()
    try:
        et = ExtraTreesClassifier(
            n_estimators=FIT_TREES, max_depth=FIT_DEPTH,
            random_state=FIT_SEED, max_leaf_nodes=BUDGET_K,
        )
        et.fit(x, y_clf)
    except Exception as exc:  # noqa: BLE001 - report, never fail
        rep.report_only(arm, "the extratrees best-first fit did not run",
                        "%s: %s" % (type(exc).__name__, exc))
        return
    counts = [leaf_count(lc) for lc in tree_slices(et)]
    rep.report_only(
        arm,
        "ExtraTreesClassifier(max_leaf_nodes=%d), sklearn's semantics,"
        " leaves per tree" % BUDGET_K, str(counts))
    rep.report_only(
        arm,
        "every extratrees tree has exactly %d leaves" % BUDGET_K,
        "yes" if all(c == BUDGET_K for c in counts) else "NO: %s" % counts)


# ===========================================================================
# THE RUN
# ===========================================================================


def fit_forests():
    """The two forests every structural arm reads: the same data, the same
    seed, the same everything, with `max_leaves` the only difference."""
    x, y_clf, _ = fixture()
    base = dict(n_estimators=FIT_TREES, max_depth=FIT_DEPTH,
                random_state=FIT_SEED)
    try:
        capped = RandomForestClassifier(max_leaves=BUDGET_K, **base)
        capped.fit(x, y_clf)
        uncapped = RandomForestClassifier(max_leaves=-1, **base)
        uncapped.fit(x, y_clf)
    except Exception as exc:  # noqa: BLE001 - no verdict without a fit
        raise GateAbort("the %dx%d classifier fit did not run: %s: %s"
                        % (FIX_N, FIX_K, type(exc).__name__, exc))
    return capped, uncapped


def main(out=sys.stdout):
    out.write("test_rf_leaf_budget: the leaf budget on the RandomForest\n"
              "surface. sklearn's max_leaf_nodes is best-first growth;\n"
              "cuML's max_leaves caps a level-order grower. This file\n"
              "gates that the two are not aliased (DEVIATION 408).\n")
    rep = Report()
    aborted = []
    try:
        mode = arm_provenance(rep)
    except GateAbort as exc:
        out.write("\nCANNOT START: %s\n" % exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - the whole run depends on this
        out.write("\nCANNOT START: %s: %s\n" % (type(exc).__name__, exc))
        return 2

    # The two constructor arms need no GPU and run even if the fit cannot,
    # because they are the arms that carry the verdict on the defect.
    arms = [("REFUSAL", lambda: arm_refusal(rep)),
            ("VALIDATION", lambda: arm_validation(rep))]
    fits = None
    try:
        capped, uncapped = fit_forests()
        fits = (("capped", capped), ("uncapped", uncapped))
        arms.append(("LEVEL_ORDER", lambda: arm_level_order(rep, fits)))
        arms.append(("BUDGET", lambda: arm_budget(rep, capped, uncapped)))
    except GateAbort as exc:
        aborted.append(("LEVEL_ORDER + BUDGET", str(exc)))
    arms.append(("CONTRAST", lambda: arm_contrast(rep)))

    for name, fn in arms:
        try:
            fn()
        except GateAbort as exc:
            aborted.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001 - report, then exit non-zero
            aborted.append((name, "%s: %s" % (type(exc).__name__, exc)))

    rep.render(out)
    for name, why in aborted:
        out.write("\n  %s ARM DID NOT RUN\n" % name)
        for line in why.splitlines():
            out.write("    %s\n" % line)

    out.write("\n")
    if rep.failures or aborted:
        out.write("test_rf_leaf_budget: RED. %d checks failed, %d arms did"
                  " not run.\n" % (len(rep.failures), len(aborted)))
        return 1
    out.write(
        "test_rf_leaf_budget: GREEN in the %s tier. max_leaf_nodes is\n"
        "refused by name on both estimators with a message that names\n"
        "best-first growth, cuML's max_leaves and the surface that has\n"
        "sklearn's semantics; max_leaves takes cuML's sentinel and cuts\n"
        "the forest to its budget; and the grower it caps appends nodes in\n"
        "breadth-first order, which is the property that made the old\n"
        "alias a wrong answer rather than a synonym. Nothing here is a\n"
        "comparison against sklearn and nothing here is a bitwise claim.\n"
        % mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
