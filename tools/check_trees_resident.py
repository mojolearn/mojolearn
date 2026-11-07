#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The tree family's resident fitted state: kept, read-only, replaced, released.

Lane ne1-a-trees (2026-11-06; docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md
workstream A). A fitted forest keeps its prepared snapshot (the device copy,
or the host column's strict trees) in the tree door's native resident table
(bindings/trees_resident_table.mojo), keyed by the model's identity and
generation word and the identities of the five model arrays. This check fits
a forest and a DART model and, for each:

1. predicts twice: the second predict uploads nothing (the table's upload
   count is unchanged) and returns the same bits;
2. writes a fitted array in place through NumPy: NumPy's read-only error;
3. replaces the fitted array (DART: the fitted sequence) with equal copies:
   the next predict uploads again (one more upload), same bits;
4. deletes the model: the number of kept snapshots falls.

    python tools/check_trees_resident.py
    pytest tools/check_trees_resident.py

Prints `TREES_RESIDENT_CHECK PASS` or `... FAIL <what>` and exits 1 on FAIL.
A FAIL is a bug in the resident path, never a result.
"""
import gc
import sys

import numpy as np


def _stats():
    from mojolearn import _native_trees_bridge
    entry, context = _native_trees_bridge.connect()
    kept, held_bytes, uploads = entry(None, "trees_resident_stats", (), context)
    return int(kept), int(held_bytes), int(uploads)


def _data(classifier):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((256, 6)).astype(np.float32)
    if classifier:
        y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int32)
    else:
        y = (x[:, 0] - x[:, 2]).astype(np.float32)
    return x, y


def _bits(a):
    return np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()


def _fields(model):
    return object.__getattribute__(model, "__dict__")


def _array_copy(old):
    from mojolearn._array import Array
    code = "i" if str(old.dtype) == "<i4" else "f"
    return Array.from_buffer(memoryview(bytes(memoryview(np.asarray(old)))).cast(code))


def _forest_write(model):
    np.asarray(_fields(model)["_leaves"])[:1] = 0


def _forest_replace(model):
    _fields(model)["_leaves"] = _array_copy(_fields(model)["_leaves"])


def _dart_write(model):
    np.asarray(_fields(model)["tree_values_"][0])[:1] = 0


def _dart_replace(model):
    # The fitted sequence replaced by assignment (a new list of equal
    # copies): its identity changes, so the kept forest is uploaded again.
    _fields(model)["tree_values_"] = [_array_copy(v) for v in _fields(model)["tree_values_"]]


def _check(make, classifier, predict, write, replace, label):
    x, y = _data(classifier)
    model = make()
    model.fit(x, y)
    kept0, _, up0 = _stats()
    first = _bits(predict(model, x))
    kept1, _, up1 = _stats()
    if up1 != up0 + 1 or kept1 != kept0 + 1:
        return f"{label}: the first predict did not keep one snapshot (uploads {up0}->{up1}, kept {kept0}->{kept1})"
    second = _bits(predict(model, x))
    _, _, up2 = _stats()
    if up2 != up1:
        return f"{label}: the second predict uploaded again ({up1}->{up2})"
    if second != first:
        return f"{label}: the kept snapshot changed the prediction bits"
    try:
        write(model)
    except ValueError as error:
        if "read-only" not in str(error):
            return f"{label}: in-place write raised another error: {error}"
    else:
        return f"{label}: an in-place write into a fitted array was accepted"
    replace(model)
    third = _bits(predict(model, x))
    _, _, up3 = _stats()
    if up3 != up2 + 1:
        return f"{label}: replacing a fitted array did not upload again ({up2}->{up3})"
    if third != first:
        return f"{label}: the re-uploaded snapshot changed the prediction bits"
    kept_before, _, _ = _stats()
    del model
    gc.collect()
    kept_after, _, _ = _stats()
    if kept_after >= kept_before:
        return f"{label}: deleting the model kept its snapshot ({kept_before}->{kept_after})"
    return None


def run():
    import mojolearn as ml
    cases = [
        ("RandomForestClassifier", lambda: ml.RandomForestClassifier(n_estimators=8, max_depth=4, random_state=3),
         True, lambda m, x: m.predict_proba(x), _forest_write, _forest_replace),
        ("ExtraTreesRegressor", lambda: ml.ExtraTreesRegressor(n_estimators=8, max_depth=4, random_state=3),
         False, lambda m, x: m.predict(x), _forest_write, _forest_replace),
    ]
    if hasattr(ml, "DARTRegressor"):
        cases.append(("DARTRegressor", lambda: ml.DARTRegressor(n_estimators=6, max_depth=3, random_state=3),
                      False, lambda m, x: m.predict(x), _dart_write, _dart_replace))
    failures = []
    for label, make, classifier, predict, write, replace in cases:
        failure = _check(make, classifier, predict, write, replace, label)
        if failure:
            failures.append(failure)
    return failures


def test_trees_resident():
    failures = run()
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    found = run()
    if found:
        print("TREES_RESIDENT_CHECK FAIL " + "; ".join(found))
        sys.exit(1)
    print("TREES_RESIDENT_CHECK PASS")
