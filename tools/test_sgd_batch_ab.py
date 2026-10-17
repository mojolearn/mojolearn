"""sgd_batch_ab: the arm parameter builder and the summary line (no mojolearn import)."""
import sys

import pytest

import sgd_batch_ab as S


def test_no_mojolearn_import():
    assert "mojolearn" not in sys.modules


@pytest.mark.parametrize("est,fixed", [("SGDClassifier", 4096), ("SGDRegressor", 4096),
                                       ("Perceptron", 256), ("PassiveAggressiveClassifier", 256)])
def test_arms_differ_only_in_batch_size(est, fixed):
    a = S.arm_params(est, "auto")
    b = S.arm_params(est, "fixed")
    assert a["batch_size"] == "auto"
    assert b["batch_size"] == fixed
    assert {k: v for k, v in a.items() if k != "batch_size"} == {k: v for k, v in b.items() if k != "batch_size"}
    assert set(a) == set(b)
    assert a["random_state"] == 0 and "max_iter" in a and "tol" in a
    # the builder hands out copies
    a["tol"] = 99
    assert S.arm_params(est, "auto")["tol"] != 99


def test_bad_inputs():
    with pytest.raises(ValueError):
        S.arm_params("LinearSVC", "auto")
    with pytest.raises(ValueError):
        S.arm_params("SGDClassifier", "both")


def test_resolved_batch():
    assert S.resolved_batch("SGDClassifier", "fixed", 800) == 4096
    assert S.resolved_batch("SGDClassifier", "auto", 800) == 25
    assert S.resolved_batch("SGDClassifier", "auto", 80_000) == 2500
    assert S.resolved_batch("Perceptron", "auto", 80_000) == 256
    assert S.resolved_batch("Perceptron", "auto", 10) == 1
    assert S.n_train_rows(1_000) == 800 and S.n_train_rows(100_000) == 80_000


def test_summary_line():
    line = S.summary_line("SGDClassifier", 10000, 12.5, 10.0, 0.91, 0.9)
    assert line == ("SGDAB SGDClassifier rows=10000 auto_ms=12.500 fixed_ms=10.000 ratio=1.2500 "
                    "auto_q=0.910000 fixed_q=0.900000")
    assert S.summary_line("Perceptron", 1000, None, 3.0, None, 0.5) == (
        "SGDAB Perceptron rows=1000 auto_ms=nan fixed_ms=3.000 ratio=nan auto_q=nan fixed_q=0.500000")
