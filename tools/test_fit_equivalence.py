#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FIT-EQUIVALENCE check and the dataset REFUSAL, against fixtures.

    python3 tools/test_fit_equivalence.py

Main lane alone executes this file. No GPU, no benchmark, no library: every
"model" below is a few lines of Python shaped like the real estimator's
readback surface, so the extraction logic and the verdict logic are exercised
on a laptop in milliseconds. What it CANNOT prove is that CatBoost 1.2.10 or
cuML 26.08 still spell their accessors the way the readers here assume --
only a box with those libraries can say that, and the readers report
UNAVAILABLE BY NAME rather than crashing when they do not.

BOTH SIDES OF EVERY SWITCH (ENGINEERING_RULES.md section 8). The refusal case
and the accept case are separate named checks: a dataset that is missing must
REFUSE, and a generated fixture asked for BY NAME must still load. A check
that only ever ran the refusing side would not notice the day the accepting
side broke, and vice versa.
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import speed_gbdt_arm as spec           # noqa: E402


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def arm(name, library):
    """An `Arm` with inert callables: only `name` and `library` are read by
    the shape dispatch, and giving it real ones would invite this file to
    start fitting something."""
    return spec.Arm(name, lambda: None, lambda m, d: None, lambda m, d: [],
                    library=library)


def captured(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# -------------------------------------------------------------- the fixtures

class FakeOurGbdt(object):
    """`mojolearn.GradientBoosting`: leaf counts through the public accessor,
    and the model text that `_tree_metadata` parses beside it."""

    def __init__(self, depths=(2, 2), declared=None, early=False):
        self._counts = [1 << d for d in depths]
        lines = ["format mojolearn-model 2", "features 4 0",
                 "trees %d" % (len(depths) if declared is None else declared),
                 "losses 0"]
        for t, d in enumerate(depths):
            lines.append("tree %d depth %d dim 1 weights 0" % (t, d))
        self.model_ = "\n".join(lines)
        self.stopped_early_ = early
        self.best_iteration_ = 1 if early else -1

    def get_tree_leaf_counts(self):
        return list(self._counts)


class FakeOurForest(object):
    """`mojolearn.RandomForestClassifier`: the flat cuML-shaped arrays."""

    def __init__(self, offsets, left_child):
        self._offsets = np.asarray(offsets, dtype=np.int32)
        self._left_child = np.asarray(left_child, dtype=np.int32)


class FakeCatBoost(object):
    def __init__(self, trees=2, leaf_values=8):
        self.tree_count_ = trees
        self._leaves = leaf_values

    def get_leaf_values(self):
        return np.zeros(self._leaves, dtype=np.float64)


class FakeXgb(object):
    class _Booster(object):
        def get_dump(self, with_stats=False):
            return ["0:[f0<1] yes=1,no=2\n\t1:leaf=0.1\n\t2:leaf=0.2\n",
                    "0:[f1<2] yes=1,no=2\n\t1:leaf=0.3\n\t2:[f0<3] yes=3,no=4"
                    "\n\t\t3:leaf=0.4\n\t\t4:leaf=0.5\n"]

    def get_booster(self):
        return FakeXgb._Booster()


class _SkTree(object):
    def __init__(self, children_left, max_depth):
        self.children_left = np.asarray(children_left, dtype=np.int64)
        self.node_count = len(children_left)
        self.max_depth = max_depth


class _SkEstimator(object):
    def __init__(self, tree):
        self.tree_ = tree


class FakeSklearn(object):
    def __init__(self, n=2):
        self.estimators_ = [_SkEstimator(_SkTree([1, -1, -1], 1))
                            for _ in range(n)]


class FakeUnreadable(object):
    """A library whose accessor raises. The reader must survive it."""

    @property
    def tree_count_(self):
        raise RuntimeError("this build exposes nothing")

    def get_leaf_values(self):
        raise RuntimeError("nor this")


# ------------------------------------------------------- the shape extractors

def test_shapes():
    s = spec.model_shape(arm("ours", "mojolearn"), FakeOurGbdt((2, 2)))
    check("ours gbdt trees", s["trees"] == 2, s)
    check("ours gbdt leaves", s["leaves"] == 8, s)
    # two depth-2 oblivious trees: 3 internal + 4 leaves each
    check("ours gbdt nodes", s["nodes"] == 14, s)
    check("ours gbdt depth", s["depth_max"] == 2, s)
    check("ours gbdt source named", s["source"] == "get_tree_leaf_counts", s)

    # The text and the accessor disagreeing is itself a defect, and it has to
    # SURFACE rather than be resolved in favour of whichever was read last.
    s = spec.model_shape(arm("ours", "mojolearn"),
                         FakeOurGbdt((2, 2), declared=5))
    check("ours gbdt disagreement surfaces", "disagreement" in s, s)

    s = spec.model_shape(arm("ours", "mojolearn"),
                         FakeOurForest([0, 7], [1, 3, 5, -1, -1, -1, -1]))
    check("ours forest trees", s["trees"] == 1, s)
    check("ours forest nodes", s["nodes"] == 7, s)
    check("ours forest leaves", s["leaves"] == 4, s)
    check("ours forest depth", s["depth_max"] == 2, s)

    # A child index outside its own tree's block makes the DEPTH unknown and
    # must not silently produce a number from a walk that does not match the
    # layout. The counts, which do not depend on the walk, still stand.
    s = spec.model_shape(arm("ours", "mojolearn"),
                         FakeOurForest([0, 3], [9, -1, -1]))
    check("ours forest bad walk -> depth unknown", s["depth_max"] is None, s)
    check("ours forest bad walk keeps counts", s["leaves"] == 2, s)

    s = spec.model_shape(arm("catboost-gpu", "catboost"), FakeCatBoost())
    check("catboost trees", s["trees"] == 2, s)
    check("catboost leaves", s["leaves"] == 8, s)
    check("catboost names its source", s["source"] == "get_leaf_values", s)

    s = spec.model_shape(arm("xgboost-gpu", "xgboost"), FakeXgb())
    check("xgboost trees", s["trees"] == 2, s)
    check("xgboost leaves", s["leaves"] == 5, s)
    # 3 nodes in the first tree (1 internal, 2 leaves) and 5 in the second
    # (2 internal, 3 leaves). Written out because the first draft of this
    # assertion said 7 and the reader was the one that was right.
    check("xgboost nodes", s["nodes"] == 8, s)
    check("xgboost depth", s["depth_max"] == 2, s)

    s = spec.model_shape(arm("sklearn-rf-cpu", "sklearn"), FakeSklearn(2))
    check("sklearn trees", s["trees"] == 2, s)
    check("sklearn nodes", s["nodes"] == 6, s)
    check("sklearn leaves", s["leaves"] == 4, s)
    check("sklearn depth", s["depth_max"] == 1, s)

    s = spec.model_shape(arm("catboost-gpu", "catboost"), FakeUnreadable())
    check("unreadable arm does not raise", s.get("leaves") is None, s)
    check("unreadable arm says why", bool(s.get("note")), s)


# ------------------------------------------------------------- the verdicts

class FakeCuml(object):
    """cuML's only door onto a fitted forest. `payload` is whatever its
    `get_json()` returns, as a JSON string."""

    def __init__(self, payload):
        self._payload = payload

    def get_json(self):
        import json as _json
        return _json.dumps(self._payload)


def test_cuml_schema_guard():
    """The reader's three cuML outcomes, and the one it cannot have.

    ABSENCE IS SAFE BECAUSE IT IS LOUD: no accessor means UNAVAILABLE means
    verdict=UNKNOWN. The hazard is a dump that PARSES into a shape the walk
    misreads and yields a plausible wrong leaf count that flows into the
    verdict dressed as a measurement, so the reader carries
    `nodes >= leaves >= trees >= 1` and degrades to UNAVAILABLE when it fails.
    """
    # A tree the walk understands: root with two leaf children.
    good = [{"children": [{"leaf_value": 1.0}, {"leaf_value": 2.0}]}]
    s = spec.model_shape(arm("cuml-rf-gpu", "cuml"), FakeCuml(good))
    check("cuml good schema trees", s["trees"] == 1, s)
    check("cuml good schema nodes", s["nodes"] == 3, s)
    check("cuml good schema leaves", s["leaves"] == 2, s)

    # A flat per-tree node LIST -- the schema this walk does not understand.
    # Every entry is skipped, the counts come out impossible, and the
    # invariant must refuse them rather than report leaves=0.
    flat = [[{"nodeid": 0, "left": 1}, {"nodeid": 1}, {"nodeid": 2}]]
    s = spec.model_shape(arm("cuml-rf-gpu", "cuml"), FakeCuml(flat))
    check("cuml flat schema -> UNAVAILABLE", s.get("leaves") is None, s)
    check("cuml flat schema says why", "nodes >= leaves" in (s.get("note") or ""), s)

    # An accessor that is absent entirely.
    s = spec.model_shape(arm("cuml-rf-gpu", "cuml"), object())
    check("cuml no accessor -> UNAVAILABLE", s.get("leaves") is None, s)

    # An UNAVAILABLE cuML arm must never let a cell read COMPARABLE.
    arms = [arm("ours", "mojolearn"), arm("cuml-rf-gpu", "cuml")]
    models = {"ours": FakeOurForest([0, 7], [1, 3, 5, -1, -1, -1, -1]),
              "cuml-rf-gpu": FakeCuml(flat)}
    _, out = captured(spec.check_fit_equivalence, "rf", arms, models, None)
    check("cuml misread never reads COMPARABLE",
          "verdict=COMPARABLE" not in out, out)


def test_verdict_comparable():
    arms = [arm("ours", "mojolearn"), arm("catboost-gpu", "catboost")]
    models = {"ours": FakeOurGbdt((2, 2)), "catboost-gpu": FakeCatBoost()}
    _, out = captured(spec.check_fit_equivalence, "gbdt-symmetric", arms,
                      models, dict(n_estimators=2))
    check("comparable verdict", "verdict=COMPARABLE" in out, out)
    check("fit line per arm", out.count("FSPEED-FIT lane=") == 2, out)
    check("no refusal when equal", "FSPEED-REFUSED" not in out, out)


def test_verdict_not_comparable():
    """The case this whole check exists for: same config, different model."""
    arms = [arm("ours", "mojolearn"), arm("catboost-gpu", "catboost")]
    models = {"ours": FakeOurGbdt((2, 2)),                # 8 leaves
              "catboost-gpu": FakeCatBoost(trees=2, leaf_values=4)}
    _, out = captured(spec.check_fit_equivalence, "gbdt-symmetric", arms,
                      models, dict(n_estimators=2))
    check("not-comparable verdict", "verdict=NOT-COMPARABLE" in out, out)
    check("not-comparable is loud", "FSPEED-NOTE" in out and "leaves" in out, out)
    check("both counts are printed", "ours:8" in out and "catboost-gpu:4" in out,
          out)


def test_verdict_unknown_is_not_a_pass():
    arms = [arm("ours", "mojolearn"), arm("catboost-gpu", "catboost"),
            arm("cuml-rf-gpu", "cuml")]
    models = {"ours": FakeOurGbdt((2, 2)), "catboost-gpu": FakeCatBoost(),
              "cuml-rf-gpu": object()}
    _, out = captured(spec.check_fit_equivalence, "rf", arms, models, None)
    check("unread arm forces UNKNOWN", "verdict=UNKNOWN" in out, out)
    check("unread arm is named", "unread=cuml-rf-gpu" in out, out)
    check("UNKNOWN never reads as comparable",
          "verdict=COMPARABLE" not in out, out)


def test_single_arm_is_unknown():
    arms = [arm("ours", "mojolearn")]
    models = {"ours": FakeOurGbdt((2, 2))}
    _, out = captured(spec.check_fit_equivalence, "rf", arms, models, None)
    check("one arm cannot be a comparison", "verdict=UNKNOWN" in out, out)


def test_short_ensemble_refuses():
    """An arm that built fewer trees than the cell asked for is REFUSED: its
    time is the time of a smaller job than the header names."""
    arms = [arm("ours", "mojolearn")]
    models = {"ours": FakeOurGbdt((2, 2))}
    _, out = captured(spec.check_fit_equivalence, "gbdt-symmetric", arms,
                      models, dict(n_estimators=100))
    check("short ensemble refuses", "FSPEED-REFUSED" in out, out)
    check("refusal names both counts",
          "FITTED 2 trees" in out and "asked for 100" in out, out)


def test_declared_early_stopping_is_not_a_refusal():
    arms = [arm("ours", "mojolearn")]
    models = {"ours": FakeOurGbdt((2, 2), early=True)}
    _, out = captured(spec.check_fit_equivalence, "gbdt-symmetric", arms,
                      models, dict(n_estimators=100))
    check("declared early stop is not refused", "FSPEED-REFUSED" not in out, out)
    check("declared early stop is still noted", "FSPEED-NOTE" in out, out)


# ---------------------------------------------- the dataset refusal, both sides

def test_missing_dataset_refuses():
    """A dataset that is not on the box must REFUSE, naming the key and the
    manifest. It must NOT return a synthetic fixture."""
    with tempfile.TemporaryDirectory() as empty:
        old = os.environ.get("GBM_BENCH_DATA")
        os.environ["GBM_BENCH_DATA"] = empty
        try:
            spec.load_with_fallback("taxi", "smoke", 1000)
        except SystemExit as exc:
            text = str(exc)
            check("missing dataset refuses", "REFUSING to run" in text, text)
            check("refusal names the dataset", "'taxi'" in text, text)
            check("refusal names the manifest", "manifest.tsv" in text, text)
            check("refusal names the store key",
                  "gbm-bench/taxi/taxi_speed.npz" in text, text)
            check("refusal names the escape", "--dataset synthclf" in text, text)
        except Exception as exc:                   # noqa: BLE001
            check("missing dataset refuses", False,
                  "raised %r instead of SystemExit" % exc)
        else:
            check("missing dataset refuses", False,
                  "RETURNED DATA for an absent dataset: the fallback is back")
        finally:
            if old is None:
                os.environ.pop("GBM_BENCH_DATA", None)
            else:
                os.environ["GBM_BENCH_DATA"] = old


def test_generated_dataset_still_loads_by_name():
    """The other side of the same switch: generated data asked for BY NAME is
    legitimate and still works, and it is tagged as what it is."""
    data = spec.load_with_fallback("synthclf", "smoke", 2000)
    check("synthclf by name still loads", data.name == "synthclf", data.name)
    check("synthclf tags its own shape", data.tag.startswith("synthclf-"),
          data.tag)


def main():
    test_shapes()
    test_cuml_schema_guard()
    test_verdict_comparable()
    test_verdict_not_comparable()
    test_verdict_unknown_is_not_a_pass()
    test_single_arm_is_unknown()
    test_short_ensemble_refuses()
    test_declared_early_stopping_is_not_a_refusal()
    test_missing_dataset_refuses()
    test_generated_dataset_still_loads_by_name()
    if FAILURES:
        print("\nFAILED %d: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
