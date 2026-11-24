# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SGD family's fit loop record on the host column (lane alm-linear,
ab-loop-mode 2026-11-24): a fit capped at K epochs with stopping off
(tol None, the -3e38 word) reports K iterations, equal to its n_iter word,
and a non-negative loop time, through the shipped host fit
(x_linear/x_linear_host_impl.mojo `x_linear_fit_host`, which the host
binding runs) and the result tail reader (x_linear/sgd_loop_words.mojo).

Four fixtures: the per-sample fit (batch 0) and the minibatch fit (batch
'auto'), each for a regression (squared_error) and a three-class
one-vs-rest classification (hinge). The standalone wrapper is
x_linear/checks/sgd_fit_loop_check.mojo."""

from x_linear.ops import FP
from x_linear.dispatch import ALGO_SGD
from x_linear.x_linear_host_impl import x_linear_fit_host
from x_linear.sgd_loop_words import sgd_result_words, sgd_n_iter_word, sgd_fit_loop

comptime _K = 3
comptime _N = 12
comptime _D = 2
comptime _TOL_NONE = Float32(-3.0e38)


def _fit_loop_case(n_classes: Int, loss: Int, lr: Int, batch: Int, name: String) raises:
    """One capped fit; refuses unless the record holds K iterations, the
    n_iter word, and a non-negative time."""
    var n = _N
    var d = _D
    var problems = n_classes if n_classes > 2 else 1
    var x = List[Float32](length=n * d, fill=Float32(0))
    var y = List[Float32](length=n, fill=Float32(0))
    for i in range(n):
        x[i * d] = Float32(i % 5) * Float32(0.25) - Float32(0.5)
        x[i * d + 1] = Float32((i * 7) % 3) * Float32(0.5) - Float32(0.25)
        if n_classes == 0:
            y[i] = x[i * d] * Float32(2) - x[i * d + 1] + Float32(0.1)
        else:
            y[i] = Float32(i % n_classes)
    var n_out = sgd_result_words(problems, d)
    var res = List[Float32](length=n_out, fill=Float32(0))
    # [n_classes, loss, penalty l2, lr, fit_intercept, max_iter, n_iter_no_change, shuffle, seed_lo, seed_hi,
    #  sample_weight, class_weight, batch, batch_sum]
    var ip: List[Int32] = [Int32(n_classes), Int32(loss), 1, Int32(lr), 1, Int32(_K), 5, 1, 7, 0, 0, 0,
                           Int32(batch), Int32(1 if n_classes != 0 else 0)]
    # [alpha, l1_ratio, eta0, power_t, epsilon, tol None]
    var fp: List[Float32] = [Float32(1.0e-4), Float32(0.15), Float32(0.01), Float32(0.25), Float32(0.1), _TOL_NONE]
    var dims: List[Int] = [n, d, n * d, n, n_out, problems * (n + d + 1), problems * n, len(ip), len(fp)]
    _ = x_linear_fit_host(ALGO_SGD, Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), dims, ip, fp, Int(res.unsafe_ptr()))
    var words = FP(unsafe_from_address=Int(res.unsafe_ptr()))
    var loop = sgd_fit_loop(words, problems, d)
    var n_iter = Int(res[sgd_n_iter_word(problems, d)])
    var status = Int(res[sgd_n_iter_word(problems, d) + 1])
    _ = x^
    _ = y^
    _ = res^
    if status != 0:
        raise Error("sgd fit_loop " + name + ": the fit refused (status " + String(status) + ")")
    if not loop.recorded:
        raise Error("sgd fit_loop " + name + ": no record")
    if loop.iterations != _K or n_iter != _K:
        raise Error("sgd fit_loop " + name + ": iterations " + String(loop.iterations) + ", n_iter "
                    + String(n_iter) + ", expected " + String(_K))
    if loop.loop_ns < 0 or loop.milliseconds() < 0.0:
        raise Error("sgd fit_loop " + name + ": negative loop time " + String(loop.loop_ns) + " ns")


def run_checks() raises:
    """The four capped fits on the host column."""
    # loss codes: squared_error 10, hinge 0; rates: invscaling 2, optimal 1;
    # batch 0 the per-sample fit, -4096 'auto' (the minibatch fit)
    _fit_loop_case(0, 10, 2, 0, "regression per-sample")
    _fit_loop_case(0, 10, 2, -4096, "regression minibatch")
    _fit_loop_case(3, 0, 1, 0, "one-vs-rest per-sample")
    _fit_loop_case(3, 0, 1, -4096, "one-vs-rest minibatch")
    print("sgd fit_loop: 4 host fits report K =", _K, "epochs and a non-negative loop time")
