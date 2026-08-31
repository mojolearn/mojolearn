#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The HIGGS load ladder's two load-bearing properties, and a sabotage of
each that proves the assertions are live.

`bench/results/fast_speed/2026-08-25-nvidia-trees-PARTIAL.md` compares our
tree learners against CatBoost, XGBoost and cuML at 463,715 and 522,911
rows. Every ratio in it is from that size. The question that table cannot
answer is whether the gap closes with load, and this repository has already
measured the mechanism that predicts it would: per-row work FASTER than
CatBoost CPU, fixed per-tree cost 5.7x theirs.

`load_higgs(size, rows_cap)` is the ladder that tests it. Two things have to
be true of it or a rung comparison is meaningless:

  1. NESTED PREFIXES. Rung 1,000,000 must be the first million rows of rung
     5,000,000. If the rungs were disjoint samples, a difference between
     them would be partly a difference of data.
  2. A FIXED HELD-OUT TAIL. Every rung must score the SAME 500,000 rows. If
     the test block moved with `n_train`, each rung would be scored on a
     different problem and the accuracy column could not be read down.

Neither is visible in a timing. Both are one slice-index typo away, and the
typo in (2) -- `x[n_train:n_train + n_test]` -- is the one that looks right.

    python3 tools/test_higgs_ladder.py

Runs on a synthetic 600,000-row fixture written to a temp directory. It
downloads nothing, touches no GPU, and is safe on a laptop.
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "speed_gbdt_arm.py")


def _fixture(root, n=600000, feats=4):
    """A cache the loader will read instead of downloading 2.6 GB. Values
    are the row index so a mis-slice is visible as a wrong NUMBER and not
    just a wrong shape."""
    os.makedirs(os.path.join(root, "higgs"), exist_ok=True)
    x = np.arange(n * feats, dtype=np.float32).reshape(n, feats)
    y = (np.arange(n, dtype=np.float32) % 2)
    np.savez(os.path.join(root, "higgs", "higgs_speed.npz"), x=x, y=y)
    return x, y


def _load_module(source):
    """Import a STRING as the module, so a sabotage can be applied without
    editing the file on disk. Editing the real file to test it is how a
    sabotage gets left in."""
    mod = types.ModuleType("_higgs_under_test")
    mod.__file__ = SOURCE
    exec(compile(source, SOURCE, "exec"), mod.__dict__)   # noqa: S102
    return mod


def _properties(mod):
    full = mod.load_higgs("shipped")
    r1 = mod.load_higgs("shipped", 10000)
    r2 = mod.load_higgs("shipped", 50000)
    return {
        "nested_prefix": bool(np.array_equal(r1.X_train, r2.X_train[:10000])
                              and np.array_equal(r2.X_train,
                                                 full.X_train[:50000])),
        "fixed_tail": bool(np.array_equal(r1.X_test, full.X_test)
                           and np.array_equal(r2.X_test, full.X_test)
                           and np.array_equal(r1.y_test, full.y_test)),
    }, full, r1, r2


def main():
    root = tempfile.mkdtemp(prefix="higgs-ladder-")
    os.environ["GBM_BENCH_DATA"] = root
    try:
        x, _ = _fixture(root)
        src = open(SOURCE).read()
        mod = _load_module(src)
        props, full, r1, r2 = _properties(mod)

        # --- the shape of the ladder ------------------------------------
        assert full.X_train.shape[0] == 100000, full.X_train.shape
        assert full.X_test.shape[0] == 500000, full.X_test.shape
        assert r1.X_train.shape[0] == 10000
        assert r2.X_train.shape[0] == 50000
        # The tail is the END of the file.
        assert np.array_equal(full.X_test[-1], x[-1])
        # A rung larger than the data cannot invent rows.
        assert mod.load_higgs("shipped", 10 ** 9).X_train.shape[0] == 100000
        # `smoke` can never be mistaken for a shipped-size fit.
        assert mod.load_higgs("smoke").X_train.shape[0] == 50000
        # Every emitted line carries the row count, so two rungs in one log
        # are distinguishable even with the header lost.
        assert r1.tag == "higgs-10000x4", r1.tag
        assert r2.tag == "higgs-50000x4", r2.tag
        assert r1.tag != r2.tag
        # The task is binary; the gbdt lanes' one-dimensional Python surface
        # can run it, which covtype's 7 classes is why they cannot (1838).
        assert full.task == "binary" and full.n_classes == 2

        # --- the two load-bearing properties ----------------------------
        assert props["nested_prefix"], props
        assert props["fixed_tail"], props
        print("as committed          %s" % props)

        # --- and the sabotages that prove those two checks are live -----
        sab_prefix = src.replace(
            "    x_train = np.ascontiguousarray(x[:n_train])\n"
            "    y_train = np.ascontiguousarray(y[:n_train])",
            "    x_train = np.ascontiguousarray(x[-n_test - n_train:-n_test])\n"
            "    y_train = np.ascontiguousarray(y[-n_test - n_train:-n_test])")
        assert sab_prefix != src, "sabotage 1 did not apply; the loader moved"
        p1, _, _, _ = _properties(_load_module(sab_prefix))
        print("SABOTAGE take-from-end %s" % p1)
        assert p1["nested_prefix"] is False, \
            "rungs stopped being nested and the check did not notice"
        assert p1["fixed_tail"] is True, \
            "sabotage 1 moved the OTHER check too; the checks are not separable"

        sab_tail = src.replace(
            "    x_test = np.ascontiguousarray(x[-n_test:])\n"
            "    y_test = np.ascontiguousarray(y[-n_test:])",
            "    x_test = np.ascontiguousarray(x[n_train:n_train + n_test])\n"
            "    y_test = np.ascontiguousarray(y[n_train:n_train + n_test])")
        assert sab_tail != src, "sabotage 2 did not apply; the loader moved"
        p2, _, _, _ = _properties(_load_module(sab_tail))
        print("SABOTAGE moving-tail   %s" % p2)
        assert p2["fixed_tail"] is False, \
            "the held-out rows moved with the rung and the check did not notice"
        assert p2["nested_prefix"] is True, \
            "sabotage 2 moved the OTHER check too; the checks are not separable"

        print("\nhiggs ladder OK: 13 assertions, and BOTH sabotages moved "
              "their own check and only their own check.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
