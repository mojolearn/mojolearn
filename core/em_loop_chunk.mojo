# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""How many EM iterations the mixtures' device loops enqueue between two
reads of their stop word (lane n1-gmm-bgmm-loops, 2026-11-09).

GaussianMixture (`mixture/estimator.mojo`) and BayesianGaussianMixture /
the x_cluster GaussianMixture (`x_cluster/bgmm.mojo`) run EM as a bounded
device loop: the criterion is a device value, the parameters of the
stopping iteration are committed by a gated copy, and the host reads one
word every `check_every` iterations. The E-step and M-step kernels are the
families' own and are not gated (the identical GEMM takes no gate), so an
iteration enqueued past the stop runs in full and its result is discarded.
The chunk therefore trades one host read (a drain and the launch bubble
after it, tens of microseconds) per iteration against up to
`check_every - 1` wasted iterations once per fit:

- an iteration whose work is small (`work = n * k * d * d`, the
  multiply-adds of the Mahalanobis products and the covariance Grams) is
  bounded by its launches, so the read is a large share of it: read every
  EM_LOOP_CHUNK iterations;
- above EM_LOOP_LAUNCH_BOUND_WORK an iteration costs well over a read, so a
  wasted iteration would cost more than the reads it saves: read every
  iteration (the loop is then the per-iteration loop it replaced).

The rule is a cost bound over the work of an iteration, the same on every
vendor; it keys on no dimension. `-D MOJOLEARN_EM_LOOP_CHECK_EVERY=<k>`
(k >= 1) pins the interval for the A/B that prices the bound. The interval
moves no bit: the committed parameters are the stopping iteration's for any
interval.
"""
from std.sys.compile import get_defined_int

#: `-D MOJOLEARN_EM_LOOP_CHECK_EVERY=<k>`: the interval, pinned (0: the rule).
comptime EM_LOOP_CHECK_EVERY_PIN = get_defined_int["MOJOLEARN_EM_LOOP_CHECK_EVERY", 0]()

#: Iterations per read while an iteration is launch-bound.
comptime EM_LOOP_CHUNK = 8

#: The work (multiply-adds) below which an EM iteration is launch-bound:
#: 2**24 is a few microseconds of arithmetic on the boards' GPUs against
#: some twenty launches an iteration.
comptime EM_LOOP_LAUNCH_BOUND_WORK = 1 << 24


def em_loop_check_every(n: Int, k: Int, d: Int) -> Int:
    """The read interval of an EM device loop over `n` rows, `k` components
    and `d` features (see the module note)."""
    comptime if EM_LOOP_CHECK_EVERY_PIN > 0:
        return EM_LOOP_CHECK_EVERY_PIN
    var work = n * k * d * d
    if work <= EM_LOOP_LAUNCH_BOUND_WORK:
        return EM_LOOP_CHUNK
    return 1
