# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SGD fit's result tail (SGDClassifier, SGDRegressor, Perceptron,
PassiveAggressiveClassifier / Regressor, SGDOneClassSVM and the SGD members
of a meta-estimator): the words after [coef (P*d) | intercept (P)].

lane alm-linear (ab-loop-mode, 2026-11-24): the fit reports its epoch loop
as `fit_loop_` (core/fit_loop_clock.mojo). The binding that runs the loop
(host column x_linear/sgd_host.mojo `sgd_fit`; device x_linear/device.mojo
`_sgd_mb_grid`, `_sgd_mb_ovr_grid`, `_sgd_ps_grid`) reads the clock twice
around the epoch loop and writes the loop's nanoseconds into two result
words, so the record crosses the descriptor call like every other fitted
word. The iterations are the n_iter word (epochs run, == `n_iter_`).

Tail layout, P problems of d weights:
  P*d + P      n_iter (epochs run, the largest over the problems)
  P*d + P + 1  status (0 ok, -1 non-finite)
  P*d + P + 2  loop nanoseconds >> 24
  P*d + P + 3  loop nanoseconds & (2^24 - 1)
Each nanosecond word is an integer below 2^24, exact in float32 (a loop up to
2^48 ns, about three days, is exact; longer clamps). Measurement plumbing
only: no computed word changes."""

from x_linear.ops import FP
from core.fit_loop_clock import FitLoop

#: words after [coef | intercept]: n_iter, status, the two loop-time words
comptime SGD_RESULT_TAIL = 4
comptime _SGD_LOOP_SPLIT = 1 << 24


def sgd_result_words(problems: Int, d: Int) -> Int:
    """The SGD fit's result length: [coef | intercept | the tail]."""
    return problems * d + problems + SGD_RESULT_TAIL


def sgd_n_iter_word(problems: Int, d: Int) -> Int:
    """The index of the n_iter word (the status word follows it)."""
    return problems * d + problems


def sgd_loop_store(res: FP, problems: Int, d: Int, loop: FitLoop):
    """Write the loop's nanoseconds into the tail's two time words. Called
    once per fit after the loop's clock stopped (and, on the device, after
    the result's readback, which would otherwise overwrite them)."""
    var at = sgd_n_iter_word(problems, d) + 2
    var ns = max(loop.loop_ns, 0)
    var hi = min(ns >> 24, _SGD_LOOP_SPLIT - 1)
    res.unsafe_store(at, Float32(hi))
    res.unsafe_store(at + 1, Float32(ns & (_SGD_LOOP_SPLIT - 1)))


def sgd_fit_loop(res: FP, problems: Int, d: Int) -> FitLoop:
    """The fit's loop record from its result words: the epochs run (the
    n_iter word) and the loop's nanoseconds."""
    var it = sgd_n_iter_word(problems, d)
    var ns = Int(res.unsafe_load(it + 2)) * _SGD_LOOP_SPLIT + Int(res.unsafe_load(it + 3))
    return FitLoop(Int(res.unsafe_load(it)), ns, True)
