#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""AN INDEPENDENT ORACLE FOR THE `cross-val-folds` LANE (2026-09-16).

    pixi run check-cross-val-folds-oracle
    python3 tools/cross_val_folds_oracle_check.py [--sabotage-expected]

WHY THIS FILE EXISTS. The oracle/applicability audit found four lanes a
release record runs whose passing cell is a hash compared
only against a previous hash of the same code. `cross-val-folds` is one of
them, and like `bpe-trainer` its lane body is pure Python integer work, so
every column of a record computes the same bytes by construction and a
cross-column diff adds nothing. It has no second implementation in this tree
and no CPU-versus-GPU pair.

WHAT THE LANE ALREADY HAS, AND WHY IT IS NOT ENOUGH. The lane hashes a
`partition` part that is 1 when every fold is nonempty, train and test are
disjoint, train is the complement, and the test blocks hold every row exactly
once. That is a real in-cell property check, and the audit's `_same_bytes`
derivation does not see it because the lane HASHES it instead of raising.
It is still not an oracle for the thing that can go wrong, and the lane's own
docstring says why: under
`MOJOLEARN_FOLD_ORDER_SABOTAGE=1` every one of those invariants still holds
and every fold keeps its size, because rotating the row-to-fold assignment by
one is still a partition. `partition` stays 1. The assignment is wrong and
nothing in the cell can tell.

THE TWO ORACLES HERE, AND WHY BOTH

  (1) A COMBINATORIAL INVARIANT ARM, which raises instead of hashing and adds
      the checks the cell does not make: fold sizes differing by at most one,
      per-class counts per fold differing by at most one on the stratified
      branch, and reproducibility of the assignment.

  (2) AN INDEPENDENT RECOMPUTATION ARM, which is the one that can catch a
      wrong assignment. It rebuilds both branches from their definition by a
      different route from ours:

      ours (`python/mojolearn/model_selection.py::_default_folds`)
        the stratified branch never constructs the sorted encoded label
        vector. It counts each residue modulo `n_splits` in closed form,
        `first = (fold - offset) % n_splits` and
        `count = 1 + (len(rows) - 1 - first) // n_splits`.

      this reference
        MATERIALIZES the sorted encoded label vector, takes the round-robin
        stripes `y_order[i::n_splits]`, counts each class in each stripe, and
        repeats the fold number that many times, which is the definition the
        closed form is a shortcut for. An off-by-one in the modular
        arithmetic, an `offset` that does not accumulate, or a rotation of the
        result all disagree with it.

      The KFold branch is recomputed the same way, as contiguous blocks with
      the first `n % n_splits` folds one row longer.

  (3) AN ORDER ARM, because the lane's own finding is that the fold indices do
      NOT pin the split. It holds the documented holes open on purpose: a
      within-class permutation must leave the stratified indices identical and
      must move the fold CONTENT, and `split_descriptor`'s sha256 must MOVE
      under it. That last one is the check that catches a descriptor that
      silently stopped reading X, which is the whole reason `split_descriptor`
      exists.

WHAT THESE ORACLES CAN CATCH

  * any wrong fold ASSIGNMENT, including the one-row rotation the lane's own
    negative control applies and every invariant survives;
  * an off-by-one in the residue arithmetic of the stratified branch;
  * an `offset` that does not accumulate across classes, which makes every
    class start its allocation at fold 0;
  * a KFold branch that distributes the remainder to the wrong folds;
  * folds that overlap, leave a row out, or are empty;
  * a stratified branch that is not stratified (a class spread unevenly);
  * an assignment that is not reproducible from the same arguments;
  * a `split_descriptor` that does not read X, or does not read y, or whose
    canonical encoding does not depend on the fields it names.

WHAT THEY CANNOT CATCH

  * A ROW ORDER THAT IS WRONG FOR THE CALLER'S PURPOSE. `_default_folds`
    never reads X and cannot; the order is the caller's. The order arm
    RECORDS that hole rather than closing it.
  * THE CLASS ENCODING ORDER, as a correctness question. Ours encodes classes
    FIRST SEEN; scikit-learn encodes them SORTED. The reference here is
    parameterized by that choice and arm ENCODING measures which one we
    implement and whether the two differ on the lane's labels. It reports;
    it does not decide, because which encoding is right is a compatibility
    question and not an arithmetic one.
  * a defect in `flatten_labels`, which both sides call.
  * anything about `cross_val_score`'s scoring, cloning or fitting. The lane
    is the fold partition alone.

SEEN TO FAIL. `--sabotage-expected` requires at least one disagreement. The
evidence includes the differing values printed by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

N_ROWS = 2048                 # the lane's row count, and the lane's fixture
SPLITS = (2, 3, 5, 7)         # the lane hashes 3 and 5; 2 and 7 reach the
                              # remainder cases 2048 % k = 0 and 2048 % k = 4


def _load_identity_break():
    spec = importlib.util.spec_from_file_location(
        "_identity_break_fixture", ROOT / "tools" / "identity_break.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FoldClf:
    """A classifier stand-in, as the lane uses. Nothing is fitted."""
    _estimator_type = "classifier"

    def get_params(self, deep=False):
        return {}


class _FoldReg:
    _estimator_type = "regressor"

    def get_params(self, deep=False):
        return {}


# --------------------------------------------------------------------------
# the reference, written from the definition
# --------------------------------------------------------------------------

def ref_kfold(n, n_splits):
    """Unshuffled KFold: contiguous blocks of POSITIONS, the first
    `n % n_splits` of them one row longer."""
    sizes = [n // n_splits + (1 if f < n % n_splits else 0) for f in range(n_splits)]
    tests, at = [], 0
    for size in sizes:
        tests.append(list(range(at, at + size)))
        at += size
    assert at == n
    return tests


def ref_stratified(labels, n_splits, encoding="first-seen"):
    """Unshuffled StratifiedKFold, by MATERIALIZING the sorted encoded label
    vector rather than by closed-form residue counting.

    The definition: encode the classes, sort the encoded vector, and read the
    round-robin stripes `y_order[i::n_splits]`. The number of class-k entries
    in stripe i is how many rows of class k go to fold i. Those fold numbers,
    in ascending fold order, are handed to that class's rows IN ARRIVAL ORDER.
    """
    order = []
    for label in labels:
        if label not in order:
            order.append(label)
    if encoding == "sorted":
        order = sorted(order)
    code = {label: k for k, label in enumerate(order)}
    encoded = [code[label] for label in labels]
    n_classes = len(order)

    y_order = sorted(encoded)
    allocation = []
    for i in range(n_splits):
        stripe = y_order[i::n_splits]
        allocation.append([stripe.count(k) for k in range(n_classes)])

    rows_of = {k: [] for k in range(n_classes)}
    for index, k in enumerate(encoded):
        rows_of[k].append(index)

    assignment = [-1] * len(labels)
    for k in range(n_classes):
        folds_for_class = []
        for i in range(n_splits):
            folds_for_class += [i] * allocation[i][k]
        assert len(folds_for_class) == len(rows_of[k])
        for row, fold in zip(rows_of[k], folds_for_class):
            assignment[row] = fold
    assert -1 not in assignment
    return [[i for i, f in enumerate(assignment) if f == fold] for fold in range(n_splits)]


def ref_folds(labels, n_splits, classifier, encoding="first-seen"):
    """`(train, test)` per fold, as `_default_folds` yields them."""
    n = len(labels)
    discrete = all(float(v).is_integer() for v in labels) if labels and not isinstance(labels[0], str) else True
    tests = (ref_stratified(labels, n_splits, encoding) if (classifier and discrete)
             else ref_kfold(n, n_splits))
    out = []
    for test in tests:
        heldout = set(test)
        out.append(([i for i in range(n) if i not in heldout], sorted(test)))
    return out


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

class Report:
    def __init__(self, out):
        self.out = out
        self.failures = []
        self.disagreements = []

    def ok(self, arm, message, detail=""):
        print(f"  ok    {arm}: {message}{detail}", file=self.out)

    def fail(self, arm, message, detail="", disagreement=False):
        print(f"  FAIL  {arm}: {message}{detail}", file=self.out)
        self.failures.append(f"{arm}: {message}")
        if disagreement:
            self.disagreements.append(f"{arm}: {message}")

    def same(self, arm, message, ours, theirs, show=None):
        if ours == theirs:
            return self.ok(arm, message)
        shown = show(ours, theirs) if show else f"\n          ours = {ours!r}\n          ref  = {theirs!r}"
        self.fail(arm, message, shown, disagreement=True)

    def differ(self, arm, message, a, b):
        """Two values that MUST NOT be equal (a switch that must flip)."""
        if a != b:
            return self.ok(arm, message)
        self.fail(arm, message, f"\n          both sides are {a!r}", disagreement=True)


def _show_folds(ours, theirs):
    """The first fold and the first row on which two assignments differ."""
    lines = [f"\n          folds ours={len(ours)} ref={len(theirs)}"]
    for f in range(min(len(ours), len(theirs))):
        a, b = ours[f], theirs[f]
        if a == b:
            continue
        lines.append(f"\n          fold {f}: sizes ours={len(a)} ref={len(b)}")
        only_ours = sorted(set(a) - set(b))[:6]
        only_ref = sorted(set(b) - set(a))[:6]
        lines.append(f"\n            rows only in ours: {only_ours}")
        lines.append(f"\n            rows only in ref : {only_ref}")
        for k in range(min(len(a), len(b))):
            if a[k] != b[k]:
                lines.append(f"\n            first position {k}: ours={a[k]} ref={b[k]}")
                break
        break
    else:
        lines.append("\n          every fold matched elementwise")
    return "".join(lines)


def _show_indices(ours, theirs):
    """Two index lists, SUMMARIZED. Printing 2,048 integers twice is not
    printing the values, it is hiding them."""
    missing = sorted(set(theirs) - set(ours))
    extra = sorted(set(ours) - set(theirs))
    duplicated = sorted({i for i in ours if ours.count(i) > 1}) if len(ours) < 5000 else []
    at, a, b = _first_difference(ours, theirs)
    return (f"\n          len ours={len(ours)} ref={len(theirs)}"
            f"\n          first difference at position {at}: ours={a} ref={b}"
            f"\n          rows missing from ours (first 8): {missing[:8]}"
            f"\n          rows extra in ours  (first 8): {extra[:8]}"
            f"\n          rows appearing twice (first 8): {duplicated[:8]}")


def _first_difference(ours, theirs):
    for k in range(min(len(ours), len(theirs))):
        if ours[k] != theirs[k]:
            return k, ours[k], theirs[k]
    return min(len(ours), len(theirs)), None, None


def _sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

def cases(identity_break):
    """The lane's own labels first, then label shapes it never sees."""
    X, yc, yr = identity_break.fixture("base")
    Xn = np.ascontiguousarray(X[:N_ROWS])
    clf_labels = np.ascontiguousarray(yc[:N_ROWS]).tolist()
    reg_labels = np.ascontiguousarray(yr[:N_ROWS]).tolist()
    out = [("lane-strat", clf_labels, True, Xn),
           ("lane-kfold", reg_labels, False, Xn)]
    # A label sequence whose classes are UNBALANCED and whose first-seen order
    # is not the sorted order; `offset` must accumulate and the encoding
    # choice is visible here if it is visible anywhere.
    unbalanced = [2] * 700 + [0] * 61 + [1] * 300 + [2] * 987
    out.append(("unbalanced-3-class", unbalanced, True, Xn))
    # Classes interleaved so no class occupies a contiguous block of rows.
    interleaved = [(i * 7) % 4 for i in range(N_ROWS)]
    out.append(("interleaved-4-class", interleaved, True, Xn))
    # A class whose count is not a multiple of any split, so every stripe
    # count is a different number.
    ragged = [0] * 1021 + [1] * 1027
    out.append(("ragged-2-class", ragged, True, Xn))
    return out


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def arm_partition(rep, ms, work):
    """Invariants that RAISE instead of hashing, plus the two the cell does
    not make at all: fold sizes within one, and class counts within one."""
    for name, labels, classifier, _X in work:
        n = len(labels)
        for k in SPLITS:
            folds = list(ms._default_folds(labels, k, classifier))
            train = [list(a) for a, _ in folds]
            test = [list(b) for _, b in folds]
            rep.same("PARTITION", f"{name}/{k}: exactly {k} folds", len(folds), k)
            flat = sorted(i for b in test for i in b)
            rep.same("PARTITION", f"{name}/{k}: the test folds hold every row exactly once",
                     flat, list(range(n)), _show_indices)
            rep.same("PARTITION", f"{name}/{k}: train is exactly the complement, ascending",
                     train, [[i for i in range(n) if i not in set(b)] for b in test],
                     _show_folds)
            sizes = sorted(len(b) for b in test)
            rep.same("PARTITION", f"{name}/{k}: fold sizes differ by at most one "
                                  f"(min {sizes[0]}, max {sizes[-1]})",
                     sizes[-1] - sizes[0] <= 1, True)
            rep.same("PARTITION", f"{name}/{k}: no fold is empty", sizes[0] > 0, True)
            if classifier:
                worst = 0
                for label in set(labels):
                    per = [sum(1 for i in b if labels[i] == label) for b in test]
                    worst = max(worst, max(per) - min(per))
                rep.same("PARTITION", f"{name}/{k}: every class is spread within one fold "
                                      f"(worst spread {worst})", worst <= 1, True)
            again = [list(b) for _, b in ms._default_folds(labels, k, classifier)]
            rep.same("PARTITION", f"{name}/{k}: the same arguments give the same assignment",
                     test, again, _show_folds)


def arm_reference(rep, ms, work, encoding):
    """The assignment against an independent recomputation."""
    for name, labels, classifier, _X in work:
        for k in SPLITS:
            ours = [list(b) for _, b in ms._default_folds(labels, k, classifier)]
            theirs = [list(b) for _, b in ref_folds(labels, k, classifier, encoding)]
            rep.same("REFERENCE", f"{name}/{k}: the fold assignment agrees with the "
                                  f"recomputation ({encoding} encoding)", ours, theirs, _show_folds)


def arm_switch(rep, ms, work):
    """Switches that MUST flip. A check whose two sides are always equal is
    the same non-check as a grep that always returns zero."""
    labels = work[0][1]
    a = [list(b) for _, b in ms._default_folds(labels, 5, True)]
    rep.differ("SWITCH", "a different n_splits gives a different assignment",
               a, [list(b) for _, b in ms._default_folds(labels, 3, True)])
    rep.differ("SWITCH", "the stratified branch differs from the KFold branch on the same labels",
               a, [list(b) for _, b in ms._default_folds(labels, 5, False)])
    moved = list(labels)
    # Move ONE row across the class boundary, so the label SEQUENCE changes.
    for i, v in enumerate(moved):
        if v != moved[0]:
            moved[i] = moved[0]
            break
    rep.differ("SWITCH", "changing one label moves the stratified assignment",
               a, [list(b) for _, b in ms._default_folds(moved, 5, True)])
    # THE HOLE, HELD OPEN ON PURPOSE. The KFold branch is a function of
    # len(y) alone, so changing a label must leave it ALONE. If this ever
    # starts to move, the branch began reading something it never read and
    # the lane's docstring is stale.
    rep.same("SWITCH", "the KFold branch is unmoved by a changed label (a function of "
                       "len(y) alone, the lane's documented hole)",
             [list(b) for _, b in ms._default_folds(labels, 5, False)],
             [list(b) for _, b in ms._default_folds(moved, 5, False)], _show_folds)


def arm_order(rep, ms, work, rng):
    """What the assignment pins and what only `split_descriptor` pins."""
    name, labels, _classifier, X = work[0]
    n = len(labels)
    # A permutation that keeps the LABEL SEQUENCE: rotate the rows of each
    # class among themselves, so position i keeps its label and gets a
    # different row.
    positions = {}
    for i, label in enumerate(labels):
        positions.setdefault(label, []).append(i)
    perm = list(range(n))
    for _label, rows in positions.items():
        for k, row in enumerate(rows):
            perm[row] = rows[(k + 1) % len(rows)]
    moved_labels = [labels[perm[i]] for i in range(n)]
    rep.same("ORDER", f"{name}: the rotation preserves the label sequence",
             moved_labels, labels)

    before = [list(b) for _, b in ms._default_folds(labels, 5, True)]
    after = [list(b) for _, b in ms._default_folds(moved_labels, 5, True)]
    rep.same("ORDER", f"{name}: the fold INDICES do not move under it "
                      "(the lane's finding, held open on purpose)", before, after, _show_folds)

    Xp = np.ascontiguousarray(X[np.asarray(perm, dtype=np.int64)])
    rep.differ("ORDER", f"{name}: the fold CONTENT does move under it",
               _sha(*[X[np.asarray(b)] for b in before]),
               _sha(*[Xp[np.asarray(b)] for b in after]))

    y = np.ascontiguousarray(np.asarray(labels, dtype=np.int32))
    d0 = ms.split_descriptor(X, y, estimator=_FoldClf(), cv=5)
    d1 = ms.split_descriptor(Xp, y, estimator=_FoldClf(), cv=5)
    rep.same("ORDER", f"{name}: split_descriptor reproduces its own sha256",
             d0["sha256"], ms.split_descriptor(X, y, estimator=_FoldClf(), cv=5)["sha256"])
    rep.differ("ORDER", f"{name}: split_descriptor's sha256 MOVES when the rows move "
                        "(what catches a descriptor that stopped reading X)",
               d0["sha256"], d1["sha256"])
    rep.differ("ORDER", f"{name}: split_descriptor's X_sha256 moves when the rows move",
               d0["X_sha256"], d1["X_sha256"])
    rep.same("ORDER", f"{name}: split_descriptor's fold_assignment_sha256 does NOT move, "
                      "which is why it is a summary and not the claim",
             d0["fold_assignment_sha256"], d1["fold_assignment_sha256"])
    shuffled = np.ascontiguousarray(X[rng.permutation(n)])
    rep.differ("ORDER", f"{name}: an arbitrary permutation moves split_descriptor too",
               d0["sha256"], ms.split_descriptor(shuffled, y, estimator=_FoldClf(), cv=5)["sha256"])


def arm_encoding(rep, ms, work):
    """Which class encoding we implement, measured rather than assumed."""
    disagreeing = []
    for name, labels, classifier, _X in work:
        if not classifier:
            continue
        for k in SPLITS:
            ours = [list(b) for _, b in ms._default_folds(labels, k, True)]
            first_seen = [list(b) for _, b in ref_folds(labels, k, True, "first-seen")]
            in_sorted = [list(b) for _, b in ref_folds(labels, k, True, "sorted")]
            which = []
            if ours == first_seen:
                which.append("first-seen")
            if ours == in_sorted:
                which.append("sorted")
            if first_seen != in_sorted:
                disagreeing.append(f"{name}/{k}")
            rep.same("ENCODING", f"{name}/{k}: ours matches at least one encoding "
                                 f"(matches {which or 'NEITHER'}; the two encodings "
                                 f"{'differ' if first_seen != in_sorted else 'agree'} here)",
                     bool(which), True)
    print(f"  note  ENCODING: the two encodings give different folds on "
          f"{len(disagreeing)} case(s): {disagreeing}", file=rep.out)
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        print("  note  ENCODING: scikit-learn is not importable here, so the "
              "compatibility arm did not run", file=rep.out)
        return
    for name, labels, classifier, _X in work:
        if not classifier:
            continue
        for k in (3, 5):
            ours = [list(b) for _, b in ms._default_folds(labels, k, True)]
            sk = [sorted(int(i) for i in b) for _, b in
                  StratifiedKFold(n_splits=k, shuffle=False).split(np.zeros((len(labels), 1)),
                                                                   np.asarray(labels))]
            if ours == sk:
                rep.ok("ENCODING", f"{name}/{k}: the assignment equals scikit-learn's")
            else:
                print(f"  note  ENCODING: {name}/{k}: the assignment DIFFERS from "
                      f"scikit-learn's{_show_folds(ours, sk)}", file=rep.out)


def main(argv=None, out=sys.stdout):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sabotage-expected", action="store_true",
                        help="require at least one disagreement; exit nonzero on agreement")
    parser.add_argument("--encoding", default="first-seen", choices=("first-seen", "sorted"),
                        help="the class encoding the REFERENCE arm uses")
    parser.add_argument("--only", default="", help="comma separated arm names")
    args = parser.parse_args(argv)

    from mojolearn import model_selection as ms

    identity_break = _load_identity_break()
    work = cases(identity_break)
    rep = Report(out)
    rng = np.random.default_rng(20260916)
    print(f"cross_val_folds_oracle_check: {len(work)} label sequences, splits {SPLITS}, "
          f"reference encoding {args.encoding!r}, "
          f"MOJOLEARN_FOLD_ORDER_SABOTAGE="
          f"{os.environ.get('MOJOLEARN_FOLD_ORDER_SABOTAGE', '')!r} "
          f"MOJOLEARN_HOST_ALLOW_SABOTAGE="
          f"{os.environ.get('MOJOLEARN_HOST_ALLOW_SABOTAGE', '')!r}", file=out)

    arms = [("PARTITION", lambda: arm_partition(rep, ms, work)),
            ("REFERENCE", lambda: arm_reference(rep, ms, work, args.encoding)),
            ("SWITCH", lambda: arm_switch(rep, ms, work)),
            ("ORDER", lambda: arm_order(rep, ms, work, rng)),
            ("ENCODING", lambda: arm_encoding(rep, ms, work))]
    wanted = set(filter(None, args.only.split(",")))
    for name, run in arms:
        if wanted and name not in wanted:
            continue
        print(f"[{name}]", file=out)
        run()

    if args.sabotage_expected:
        if rep.disagreements:
            print(f"\nSABOTAGE CAUGHT: {len(rep.disagreements)} disagreement(s); the first is",
                  file=out)
            print(f"  {rep.disagreements[0]}", file=out)
            return 0
        print("\nSABOTAGE NOT CAUGHT: every arm agreed, so this oracle cannot see the "
              "defect it was run against", file=out)
        return 1
    if rep.failures:
        print(f"\nFAILED: {len(rep.failures)} check(s)", file=out)
        for line in rep.failures:
            print(f"  {line}", file=out)
        return 1
    print("\nOK: the invariants hold and the assignment agrees with the recomputation", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
