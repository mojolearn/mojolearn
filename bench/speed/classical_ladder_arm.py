#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The scikit-learn arm of the classical SIZE LADDER, and the conductor that
interleaves it with ours.

    nice -n 19 python3 bench/speed/classical_ladder_arm.py \\
        --lane gmm --binary build/classical_ladder

ONE LANE PER INVOCATION. There is no all-lanes mode. See
`bench/speed/classical_ladder_main.mojo`'s module docstring for the whole
design; this file repeats only what it must.

=============================================================== WHAT THIS IS
Two things in one file, on purpose:

  THE OPPONENT.  scikit-learn on the CPU, on the same synthesized data at the
  same rungs, with every shared parameter set EXPLICITLY rather than left to
  two libraries' defaults.

  THE CONDUCTOR.  The thing that alternates the two arms inside a rung so a
  thermal drift hits both equally. It cannot be done from either arm alone:
  our arm is a compiled Mojo binary in its own process and the opponent is
  in this one, so somebody has to hold both. This file holds both, invoking
  the Mojo binary once per BLOCK with `MOJOLEARN_LADDER_CONDUCTED=1`.

  Order inside a rung, always:  ours, theirs, ours, theirs.
  Block granularity and not rep granularity, because a subprocess per rep
  would charge every one of our reps a fresh warm-up. A block is bounded by
  the per-cell budget, so the drift inside one is bounded too.

============================================ WHAT THIS FILE MAY NOT BE ASKED
There is NO `--n`, NO `--only`, NO `--skip`, NO `--from` and NO `--max-size`,
and adding one would break the argument that makes this harness legitimate.
The ladder is DECLARED in the Mojo source before any run; every declared rung
is run from the bottom up; and every rung that ran is PRINTED, wins and
losses in the same line format with the same fields in the same order. The
ladder may stop EARLY, and only because the machine ran out of time or
memory, never because a rung went badly. Every stop names the rung reached,
the rung skipped, the rule that fired and its arithmetic.

`--theirs-only` is not an exception to that; it runs the same full ladder
with the opponent alone, for when no Mojo binary has been built yet.

=========================================================== THE LADDER ITSELF
`LADDER` below is this side's copy. It is NOT the source of truth; the Mojo
driver's `_ladder()` is. On every race this file PROBES the binary
(`MOJOLEARN_LADDER_PROBE=1`), reads back the ladder the binary declared, and
REFUSES to run if the two disagree. Two spellings of a ladder that are never
compared are two ladders. The copy exists only so `--theirs-only` works with
no binary present, and it is asserted against the original whenever there is
one, which is the same discipline `bench/bench_sklearn.py::u01_at` is held to.

============================================================ THE SAME DATA
By REGENERATION plus a CHECKSUM, not by a dump.

`u01` is IMPORTED from `bench/bench_sklearn.py`, the file that already owns
the numpy twin of `bench/bench_main.mojo`'s splitmix64. `u01_at` is a
row-indexed variant of the same recurrence and is ASSERTED against the
imported `u01` on every run, so a re-spelling cannot drift.

Both arms print `datahash=` per array per rung and the conductor refuses the
rung if they differ.

WHY THE CHECKSUM AND NOT A DUMP. `bench/speed/classical_speed_main.mojo`
writes its fixtures out as hex words, one per line, and that is right for a
48-row correctness fixture. Here the top gmm rung alone is 8,000,000 floats,
about 72 MB of hex text, and a ladder of four rungs across six lanes would
write and re-parse close to a gigabyte on a laptop before a millisecond was
measured. The generator is a closed form in (row, column, salt), so both
sides can just compute it; there is no file to go stale between the two
commands and no parser to be slow.

The checksum is `datahash=<W>.<X>` where X is the XOR of every float32's bit
pattern and W is `sum_i bits[i] * (i + 1)` mod 2^64. Both cover the whole
array and both vectorize here. It is a CHECKSUM and not a digest, and its job is to
catch two implementations of one generator drifting apart, which the position
weight makes it do. A plain XOR would pass a permutation, which is exactly
the failure mode `uniform test data hides permutation` warns about.

=================================================================== THREADS
scikit-learn RUNS AS A USER WOULD RUN IT. This file does not set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or
`n_jobs`, and it never will, because capping the opponent's threads to make a GPU
look better is not a measurement. It PRINTS how many threads the pools report
and it prints whether those variables were already set in the environment it
inherited, so the comparison is legible either way.

`nice -n 19` is a scheduling priority, not a thread cap, and it is inherited
by BOTH arms from the invoking command, so it does not bias the comparison.
It is in the documented command because the standing rule for this laptop is
`no heavy local compute` and one heavy thing at a time.

============================================================ THE OPPONENTS
RAPIDS ships no GPU implementation of any of these six, so scikit-learn on
the CPU is the honest opponent for all of them and every arm is labeled
`sklearn-cpu`. Three of the six carry a difference that cannot be removed
from their side and that is printed rather than buried:

  gp          scikit-learn works in FLOAT64 and we work in FLOAT32. That is
              a real difference in the amount of arithmetic and
              `sklearn.gaussian_process` has no float32 path.
  nystroem    the basis sample differs (a pinned Philox stream against
  rbfsampler  numpy's `RandomState`), so the two fit different rows and draw
              different weights. The WORK is the same shape, which is what is
              timed; the outputs are not comparable and no hash is claimed.
  kde         scikit-learn's `KernelDensity` has no brute-force option; it
              builds a KD-tree or ball tree. With `rtol = atol = 0` that tree
              cannot prune, so it computes the same sum by a different data
              structure. Their fit and score times are printed separately so
              a reader can see how much of it was the tree build.

=============================================================== OUTPUT LINES
The opponent's lines mirror the Mojo driver's with `arm=sklearn-cpu`. The
conductor adds four of its own:

    FLADDER-DATACHECK lane=<l> n=<n> array=<a> ours=<h> theirs=<h>
        verdict=<match|MISMATCH>
    FLADDER-VERDICT lane=<l> n=<n> ratio=<f>x result=<WIN|LOSS|
        TIE-BANDS-OVERLAP> ours_median_ms=<f> theirs_median_ms=<f>
        ours_band_ms=<f>-<f> theirs_band_ms=<f>-<f> bands_overlap=<yes|no>
    FLADDER-STOP lane=<l> reached=<n> skipped=<n> rule=<t> detail=<text>
    FLADDER-SUMMARY lane=<l> rungs_declared=<k> rungs_run=<k>
        contiguous=<yes|no> wins=<k> losses=<k> ties=<k> first_win=<n|->
        stopped=<rule|->

`ratio` is THEIRS divided by OURS, so a ratio above 1 means we are faster.
A WIN LINE AND A LOSS LINE ARE THE SAME LINE with different values; there is
no field that appears only when the news is good and no shorter line for a
bad rung.

A RUNG WHOSE TWO MIN-MAX BANDS OVERLAP IS NOT A FINDING. It is printed as
`result=TIE-BANDS-OVERLAP` and that is the whole of what may be said about
it. Do not quote its ratio.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARM = "sklearn-cpu"
MASK64 = (1 << 64) - 1

OUTHASH_MAX = 1 << 20
"""Output values folded before `_outhash` stops and says so in `@<count>`.
The same cap as the Mojo driver's `HASH_MAX`, for the same reason. The
nystroem embedding at n = 10^6 and q = 64 is 64,000,000 floats and hashing
it six times a rung would cost more than the rung. THE DATA CHECKSUM IS NOT
CAPPED and never will be; that one has to cover every byte or it is not a
check that the two arms saw the same dataset."""


# ---------------------------------------------------------------------------
# THE LADDER, this side's copy. The Mojo source is the original; see THE
# LADDER ITSELF above. Every race asserts these two agree.
# ---------------------------------------------------------------------------

LADDER = {
    "gmm": [1000, 10000, 100000, 1000000],
    "kde": [1000, 10000, 100000, 1000000],
    "nystroem": [1000, 10000, 100000, 1000000],
    "rbfsampler": [1000, 10000, 100000, 1000000],
    # CAPPED at 16,000: the affinity is an exact brute-force self-kNN, so the
    # pair count is n^2, which is 2.56e8 here and 4.1e9 at the 64,000 that is NOT
    # in this list. See the Mojo source for the full arithmetic.
    "spectral": [1000, 4000, 16000],
    # CAPPED at 8,000: their float64 Gram is 8n^2 = 512 MB here and the
    # factorization is n^3/3 = 1.7e11 flops. 16,000 needs 2.048 GB for that
    # Gram alone, which is already over the 2 GB cap.
    "gp": [500, 1000, 2000, 4000, 8000],
}

EXPONENT = {
    "gmm": 1, "kde": 1, "nystroem": 1, "rbfsampler": 1,
    "spectral": 2, "gp": 3,
}

LANES = list(LADDER.keys())


# ---------------------------------------------------------------------------
# THE PINNED PARAMETERS. The same values the Mojo driver pins, spelled here so
# a reader can diff the two lists without opening a second file. Nothing here
# is a library default.
# ---------------------------------------------------------------------------

GMM_D, GMM_K, GMM_MAX_ITER, GMM_TOL, GMM_REG_COVAR, GMM_SEED = 8, 8, 20, 1e-3, 1e-6, 0
KDE_D, KDE_N_QUERY, KDE_BANDWIDTH = 8, 1000, 1.0
KM_D, KM_Q, KM_GAMMA, KM_SEED = 8, 64, 0.125, 20260825
SPECTRAL_D, SPECTRAL_CLUSTERS, SPECTRAL_COMPONENTS = 4, 8, 8
SPECTRAL_N_INIT, SPECTRAL_NEIGHBORS, SPECTRAL_TOL, SPECTRAL_SEED = 10, 10, 1e-5, 7
GP_D, GP_N_STAR, GP_LENGTH_SCALE = 8, 1000, 1.0

GP_ALPHA = 2.0 ** -20
"""`gaussian_process/estimator.mojo::gp_profile_alpha()` IS 2^-20
(`chol_jitter_pinned`, DEVIATION 1751). Spelled as a POWER OF TWO and not as
a decimal because a power of two is the same real number in float32 and in
float64; `1e-3` on the two sides would be two different ridges. The Mojo
header prints the bits and this file checks them."""


# ---------------------------------------------------------------------------
# THE GENERATOR, `splitmix64-blob-v1`.
# ---------------------------------------------------------------------------

N_BLOBS = 8
BLOB_SEP = 8.0        # a power of two: exact in f64 and f32, so FMA-safe
CENTER_ROW_BASE = 1000003
SALT_CENTER, SALT_JITTER, SALT_QUERY, SALT_Y = 41, 11, 17, 29

M1 = 0x9E3779B97F4A7C15
M2 = 0xBF58476D1CE4E5B9
M3 = 0x94D049BB133111EB


def _load_bench_sklearn():
    """`bench/bench_sklearn.py` as a module, by path.

    By path rather than by `import bench.bench_sklearn` so this works from any
    working directory and so there is no doubt about WHICH file was loaded.
    Its `main()` is behind an `if __name__ == "__main__"` guard, so importing
    it runs nothing.
    """
    path = os.path.join(REPO, "bench", "bench_sklearn.py")
    spec = importlib.util.spec_from_file_location("_mojolearn_bench_sklearn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BS = _load_bench_sklearn()
u01 = _BS.u01


def u01_at(rows, cols, salt):
    """`u01` for an EXPLICIT vector of row indices instead of `arange(n)`.

    The blob centers live at rows 1,000,003 upward and materializing
    `u01(1000011, 8, 41)` to take eight rows of it would allocate 64 MB to
    read 512 bytes. Same recurrence; `_check_u01_variant` asserts on every run
    that it agrees with the imported `u01` on the rows they share, so a
    re-spelling that is checked against its original cannot drift.
    """
    r = np.asarray(rows, dtype=np.uint64)[:, None]
    k = np.arange(cols, dtype=np.uint64)[None, :]
    z = (r * np.uint64(M1) + (k + np.uint64(1)) * np.uint64(M2)
         + np.uint64(salt + 1) * np.uint64(M3))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M3)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _check_u01_variant():
    a = u01(37, 11, 3)
    b = u01_at(np.arange(37), 11, 3)
    if not np.array_equal(a, b):
        raise SystemExit(
            "u01_at disagrees with bench/bench_sklearn.py::u01. The opponent "
            "would be racing on a different dataset. Fix u01_at before "
            "quoting any number from this run."
        )


def gen_matrix(n, d, jitter_salt):
    """`n x d` float32, generator `splitmix64-blob-v1`.

    The whole combination happens in FLOAT64 and rounds to float32 EXACTLY
    ONCE, which is what `Float32(_u01(...) + cen[...])` does on the Mojo side.
    Two roundings in different places would be two datasets.

    `center * 8.0` is exact because 8.0 is a power of two, so this expression
    gives the same bits whether or not either compiler contracts the multiply
    and the add into an FMA. `FMA contraction is PER SEAM` has already cost
    this tree one comparison.
    """
    cen = u01_at(np.arange(CENTER_ROW_BASE, CENTER_ROW_BASE + N_BLOBS),
                 d, SALT_CENTER) * BLOB_SEP
    jit = u01(n, d, jitter_salt)
    return (jit + cen[np.arange(n) % N_BLOBS]).astype(np.float32)


def gen_targets(n):
    """One target per row, column 0 of the recurrence at `SALT_Y`.

    The gp lane's timed work does not depend on these values; the Gram, its
    ridge and its factorization never see `y`.
    """
    return u01(n, 1, SALT_Y)[:, 0].astype(np.float32)


def datahash(a):
    """`<W 16 hex>.<X 8 hex>`, the same checksum the Mojo side computes.

    Chunked at a million elements so the temporary index array stays under
    8 MB. The top rung is 8,000,000 floats and this runs on a laptop that has
    other work to do.
    """
    u = np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)
    x = np.uint32(0)
    w = 0
    step = 1 << 20
    for lo in range(0, u.size, step):
        blk = u[lo:lo + step]
        x = np.bitwise_xor(x, np.bitwise_xor.reduce(blk))
        idx = np.arange(lo + 1, lo + 1 + blk.size, dtype=np.uint64)
        w = (w + int(np.sum(blk.astype(np.uint64) * idx, dtype=np.uint64))) & MASK64
    return "%016x.%08x" % (w, int(x))


# ---------------------------------------------------------------------------
# THE MEMORY CEILING, their side, in float64.
#
# Estimates of the DOMINANT allocations, stated as such: a guard, not an
# accounting, and deliberately generous rather than tight. The formula is
# printed beside the number so a refusal can be read and argued with.
# ---------------------------------------------------------------------------

def footprint(lane, n):
    if lane == "gmm":
        # X, the responsibilities, the log-prob block and the k-means init
        return 16 * n * GMM_D + 16 * n * GMM_K, \
            "16*n*d+16*n*K=16*%d*%d+16*%d*%d" % (n, GMM_D, n, GMM_K)
    if lane == "kde":
        # X plus the tree's own copy of it and its node bookkeeping
        return 24 * n * KDE_D + 8 * KDE_N_QUERY * KDE_D, \
            "24*n*d+8*q*d=24*%d*%d+8*%d*%d" % (n, KDE_D, KDE_N_QUERY, KDE_D)
    if lane in ("nystroem", "rbfsampler"):
        # X, the n x q cross kernel (or projection) and the n x q result
        return 8 * n * KM_D + 16 * n * KM_Q, \
            "8*n*d+16*n*Q=8*%d*%d+16*%d*%d" % (n, KM_D, n, KM_Q)
    if lane == "spectral":
        # X, the sparse kNN graph (n*k of data+indices+indptr) and the
        # n x n_components embedding
        return 8 * n * SPECTRAL_D + 16 * n * SPECTRAL_NEIGHBORS \
            + 8 * n * SPECTRAL_COMPONENTS, \
            "8*n*d+16*n*k+8*n*c=8*%d*%d+16*%d*%d+8*%d*%d" % (
                n, SPECTRAL_D, n, SPECTRAL_NEIGHBORS, n, SPECTRAL_COMPONENTS)
    if lane == "gp":
        # K and its Cholesky factor, both n x n float64, plus K_trans and V
        return 16 * n * n + 16 * n * GP_N_STAR, \
            "16*n*n+16*n*nstar=16*%d*%d+16*%d*%d" % (n, n, n, GP_N_STAR)
    raise SystemExit("classical_ladder_arm: unknown lane %r" % lane)


# ---------------------------------------------------------------------------
# THE OPPONENT. One block = build the data once, one untimed warm-up call,
# then `reps` timed calls, exactly the shape the Mojo side runs.
#
# Every shared parameter is set EXPLICITLY. Where a scikit-learn default
# already equals the pinned value it is STILL written out, because a default
# that silently moves in a future release is a comparison that silently
# changes.
# ---------------------------------------------------------------------------

def _outhash(a):
    """`<W>.<X>@<count>`, the same cheap checksum `datahash` uses, over the
    arm's OUTPUT.

    Present so a reader can see that the opponent is stable across reps. IT
    IS NEVER COMPARED WITH OUR HASH and it is not the same function ours
    uses, because scikit-learn is a different implementation of the same algorithm
    and is not supposed to produce our bits, so a cross-arm hash comparison
    would be meaningless whichever function either side ran. A vectorized
    checksum rather than a byte-at-a-time FNV because a pure-Python fold over
    four megabytes costs more than several of the cells it is attached to.
    """
    v = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)[:OUTHASH_MAX]
    return "%s@%d" % (datahash(v), v.size)


def _data_line(lane, n, name, arr):
    rows = arr.shape[0]
    cols = arr.shape[1] if arr.ndim > 1 else 1
    h = datahash(arr)
    print("FLADDER-DATA lane=%s n=%d array=%s rows=%d cols=%d datahash=%s"
          % (lane, n, name, rows, cols, h))
    return h


def block_gmm(lane, n, block, reps, samples):
    from sklearn.mixture import GaussianMixture
    x = gen_matrix(n, GMM_D, SALT_JITTER)
    hashes = {"x": _data_line(lane, n, "x", x)}
    for r in range(reps + 1):
        gm = GaussianMixture(
            n_components=GMM_K, covariance_type="full", tol=GMM_TOL,
            reg_covar=GMM_REG_COVAR, max_iter=GMM_MAX_ITER, n_init=1,
            init_params="kmeans", random_state=GMM_SEED, warm_start=False,
        )
        t0 = time.perf_counter()
        gm.fit(x)
        ms = (time.perf_counter() - t0) * 1000.0
        h = _outhash(gm.means_)
        # `iters` rides on every line because `tol` may stop the two arms at
        # DIFFERENT iterations on the same data. If the counts differ, the two
        # cells did different amounts of work; the conductor says so rather
        # than dividing them.
        _line(lane, n, block, r, ms, h, "iters%d" % gm.n_iter_, samples)
    return hashes


def block_kde(lane, n, block, reps, samples):
    from sklearn.neighbors import KernelDensity
    train = gen_matrix(n, KDE_D, SALT_JITTER)
    query = gen_matrix(KDE_N_QUERY, KDE_D, SALT_QUERY)
    hashes = {"train": _data_line(lane, n, "train", train),
              "query": _data_line(lane, n, "query", query)}
    for r in range(reps + 1):
        t0 = time.perf_counter()
        kd = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel="gaussian",
                           metric="euclidean", algorithm="auto",
                           rtol=0.0, atol=0.0).fit(train)
        t_fit = time.perf_counter()
        out = kd.score_samples(query)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        if r > 0:
            # The tree build is inside the clock because our host entry does
            # the whole thing from host arrays too. It is split out here so a
            # reader can see how much of their number it was.
            print("FLADDER-NOTE lane=%s arm=%s n=%d rep=%d fit_ms=%.6f "
                  "score_ms=%.6f" % (lane, ARM, n, r, (t_fit - t0) * 1000.0,
                                     (t1 - t_fit) * 1000.0))
        _line(lane, n, block, r, ms, _outhash(out), "-", samples)
    return hashes


def block_nystroem(lane, n, block, reps, samples):
    from sklearn.kernel_approximation import Nystroem
    x = gen_matrix(n, KM_D, SALT_JITTER)
    hashes = {"x": _data_line(lane, n, "x", x)}
    for r in range(reps + 1):
        ny = Nystroem(kernel="rbf", gamma=KM_GAMMA, n_components=KM_Q,
                      random_state=KM_SEED)
        t0 = time.perf_counter()
        # `fit` then `transform` rather than `fit_transform`, because that is
        # the pair our host entries are: `nystroem_fit_host` then
        # `nystroem_transform_host`.
        ny.fit(x)
        z = ny.transform(x)
        ms = (time.perf_counter() - t0) * 1000.0
        _line(lane, n, block, r, ms, _outhash(z), "-", samples)
    return hashes


def block_rbfsampler(lane, n, block, reps, samples):
    from sklearn.kernel_approximation import RBFSampler
    x = gen_matrix(n, KM_D, SALT_JITTER)
    hashes = {"x": _data_line(lane, n, "x", x)}
    for r in range(reps + 1):
        rs = RBFSampler(gamma=KM_GAMMA, n_components=KM_Q,
                        random_state=KM_SEED)
        t0 = time.perf_counter()
        rs.fit(x)
        z = rs.transform(x)
        ms = (time.perf_counter() - t0) * 1000.0
        _line(lane, n, block, r, ms, _outhash(z), "-", samples)
    return hashes


def block_spectral(lane, n, block, reps, samples):
    from sklearn.cluster import SpectralClustering
    x = gen_matrix(n, SPECTRAL_D, SALT_JITTER)
    hashes = {"x": _data_line(lane, n, "x", x)}
    for r in range(reps + 1):
        sc = SpectralClustering(
            n_clusters=SPECTRAL_CLUSTERS, n_components=SPECTRAL_COMPONENTS,
            n_init=SPECTRAL_N_INIT, n_neighbors=SPECTRAL_NEIGHBORS,
            affinity="nearest_neighbors", eigen_solver="arpack",
            eigen_tol=SPECTRAL_TOL, assign_labels="kmeans",
            random_state=SPECTRAL_SEED,
        )
        t0 = time.perf_counter()
        labels = sc.fit_predict(x)
        ms = (time.perf_counter() - t0) * 1000.0
        # Label NUMBERING is arbitrary in both implementations, so this hash
        # is a within-arm stability probe and nothing else.
        _line(lane, n, block, r, ms,
              _outhash(labels.astype(np.float32)), "-", samples)
    return hashes


def block_gp(lane, n, block, reps, samples):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    x = gen_matrix(n, GP_D, SALT_JITTER)
    y = gen_targets(n)
    xs = gen_matrix(GP_N_STAR, GP_D, SALT_QUERY)
    hashes = {"x": _data_line(lane, n, "x", x),
              "y": _data_line(lane, n, "y", y),
              "x_star": _data_line(lane, n, "x_star", xs)}
    kern = RBF(length_scale=np.full(GP_D, GP_LENGTH_SCALE, dtype=np.float64))
    for r in range(reps + 1):
        # `optimizer=None` because `gpr_fit_host` implements no
        # hyperparameter optimizer (DEVIATION 1761). `normalize_y=False`
        # because ours does not normalize.
        gpr = GaussianProcessRegressor(kernel=kern, alpha=GP_ALPHA,
                                       optimizer=None, normalize_y=False)
        t0 = time.perf_counter()
        gpr.fit(x, y)
        mean, std = gpr.predict(xs, return_std=True)
        ms = (time.perf_counter() - t0) * 1000.0
        _line(lane, n, block, r, ms, _outhash(mean), "float64", samples)
    return hashes


BLOCKS = {
    "gmm": block_gmm, "kde": block_kde, "nystroem": block_nystroem,
    "rbfsampler": block_rbfsampler, "spectral": block_spectral,
    "gp": block_gp,
}


def _line(lane, n, block, rep, ms, h, note, samples):
    """Rep 0 is the untimed warm-up and is PRINTED, never hidden and never in
    the table. Every other rep is a sample."""
    if rep == 0:
        print("FLADDER-WARMUP lane=%s arm=%s n=%d block=%d ms=%.6f"
              % (lane, ARM, n, block, ms))
        return
    print("FLADDER lane=%s arm=%s n=%d block=%d rep=%d ms=%.6f hash=%s note=%s"
          % (lane, ARM, n, block, rep, ms, h, note))
    samples.append(ms)


# ---------------------------------------------------------------------------
# Environment reporting. Threads are REPORTED, never SET.
# ---------------------------------------------------------------------------

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def thread_report():
    threads, pools = "unknown", "unknown"
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
        if info:
            threads = str(max(int(p.get("num_threads", 0)) for p in info))
            pools = ";".join(
                "%s/%s=%s" % (p.get("user_api"), p.get("internal_api"),
                              p.get("num_threads"))
                for p in info
            ) or "none"
        else:
            threads, pools = "0", "none"
    except Exception as exc:                       # noqa: BLE001
        pools = "threadpoolctl-unavailable(%s)" % type(exc).__name__
    preset = ";".join("%s=%s" % (v, os.environ[v])
                      for v in THREAD_VARS if v in os.environ) or "none"
    return threads, pools, preset


def header_theirs(lane, rungs, args):
    import sklearn
    threads, pools, preset = thread_report()
    print("FLADDER-HEADER family=classical-ladder lane=%s arm=%s "
          "device=cpu generator=splitmix64-blob-v1 ladder=%s exponent=%d "
          "per_cell_s=%d total_s=%d mem_cap_bytes=%d reps=%d threads=%s "
          "cpu_count=%s pools=%s thread_env=%s numpy=%s sklearn=%s python=%s"
          % (lane, ARM, ",".join(str(r) for r in rungs), EXPONENT[lane],
             args.cell_seconds, args.total_seconds, args.mem_mb << 20,
             args.reps, threads, os.cpu_count(), pools, preset,
             np.__version__, sklearn.__version__,
             ".".join(str(v) for v in sys.version_info[:3])))


# ---------------------------------------------------------------------------
# Statistics. A win and a loss are the same line.
# ---------------------------------------------------------------------------

def median(xs):
    s = sorted(xs)
    m = len(s)
    if m == 0:
        return float("nan")
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def rung_line(lane, arm, n, samples):
    print("FLADDER-RUNG lane=%s arm=%s n=%d ms_median=%.6f ms_min=%.6f "
          "ms_max=%.6f samples=%d"
          % (lane, arm, n, median(samples), min(samples), max(samples),
             len(samples)))


def verdict_line(lane, n, ours, theirs):
    om, tm = median(ours), median(theirs)
    olo, ohi, tlo, thi = min(ours), max(ours), min(theirs), max(theirs)
    overlap = not (ohi < tlo or thi < olo)
    ratio = tm / om if om > 0 else float("inf")
    if overlap:
        result = "TIE-BANDS-OVERLAP"
    elif ratio > 1.0:
        result = "WIN"
    else:
        result = "LOSS"
    print("FLADDER-VERDICT lane=%s n=%d ratio=%.4fx result=%s "
          "ours_median_ms=%.6f theirs_median_ms=%.6f ours_band_ms=%.6f-%.6f "
          "theirs_band_ms=%.6f-%.6f bands_overlap=%s"
          % (lane, n, ratio, result, om, tm, olo, ohi, tlo, thi,
             "yes" if overlap else "no"))
    return result, om, tm


# ---------------------------------------------------------------------------
# Our arm, one block at a time, in its own process.
# ---------------------------------------------------------------------------

def run_ours_block(binary, lane, rung_index, block, reps, args):
    """Invoke the prebuilt Mojo binary for ONE rung, ONE block.

    A PREBUILT BINARY and not `pixi run mojo run`, because this is called twice a
    rung, and `mojo run` would recompile the whole import graph every time,
    which would both dwarf the measurement and heat the laptop for nothing.

    The subprocess start cost is NOT in any reported number. The binary times
    its own calls with `perf_counter_ns` inside the process and prints the
    milliseconds; this side reads those, never the wall time of the
    subprocess. `nice` is inherited from whatever invoked this file, so both
    arms carry the same scheduling priority.
    """
    env = dict(os.environ)
    env["MOJOLEARN_LADDER_LANE"] = lane
    env["MOJOLEARN_LADDER_CONDUCTED"] = "1"
    env["MOJOLEARN_LADDER_RUNG_INDEX"] = str(rung_index)
    env["MOJOLEARN_LADDER_BLOCK"] = str(block)
    env["MOJOLEARN_LADDER_REPS"] = str(reps)
    env["MOJOLEARN_LADDER_MEM_MB"] = str(args.mem_mb)
    proc = subprocess.run([binary], env=env, capture_output=True, text=True)
    samples, hashes, refused = [], {}, None
    for raw in proc.stdout.splitlines():
        print(raw)
        f = raw.split()
        if not f:
            continue
        if f[0] == "FLADDER":
            samples.append(float(_field(f, "ms")))
        elif f[0] == "FLADDER-DATA":
            hashes[_field(f, "array")] = _field(f, "datahash")
        elif f[0] == "FLADDER-REFUSED":
            refused = _field(f, "rule")
    if proc.stderr.strip():
        for raw in proc.stderr.splitlines():
            print("FLADDER-NOTE lane=%s arm=ours stderr=%s" % (lane, raw))
    if proc.returncode != 0 and refused is None:
        refused = "exit%d" % proc.returncode
    return samples, hashes, refused


def _field(fields, name):
    pre = name + "="
    for f in fields:
        if f.startswith(pre):
            return f[len(pre):]
    return ""


def probe(binary, lane):
    """Ask the binary for the ladder it DECLARED, and refuse if this file's
    copy disagrees. Two spellings of a ladder that are never compared are two
    ladders."""
    env = dict(os.environ)
    env["MOJOLEARN_LADDER_LANE"] = lane
    env["MOJOLEARN_LADDER_PROBE"] = "1"
    proc = subprocess.run([binary], env=env, capture_output=True, text=True)
    declared, mode, device, alpha = None, "?", "?", "?"
    for raw in proc.stdout.splitlines():
        print(raw)
        f = raw.split()
        if not f:
            continue
        if f[0] == "FLADDER-PROBE":
            declared = [int(v) for v in _field(f, "ladder").split(",") if v]
        elif f[0] == "FLADDER-HEADER":
            mode = _field(f, "mode")
            device = _field(f, "device")
            alpha = _field(f, "alpha_bits")
    if declared is None:
        raise SystemExit(
            "classical_ladder_arm: %s printed no FLADDER-PROBE line. Build it "
            "with `pixi run mojo build -I . -o <path> "
            "bench/speed/classical_ladder_main.mojo` first.\nstderr:\n%s"
            % (binary, proc.stderr))
    if declared != LADDER[lane]:
        raise SystemExit(
            "classical_ladder_arm: the Mojo driver declares ladder %s for "
            "lane %s and this file's copy says %s. THE TWO ARMS WOULD BE "
            "MEASURING DIFFERENT SIZES. Fix the copy in classical_ladder_arm."
            "py to match classical_ladder_main.mojo, which is the original."
            % (declared, lane, LADDER[lane]))
    if mode != "FAST":
        print("FLADDER-NOTE lane=%s arm=ours mode=%s is not FAST; this is a "
              "speed question and only the FAST arm answers it" % (lane, mode))
    if lane == "gp" and alpha not in ("0x35800000", "?"):
        print("FLADDER-NOTE lane=gp arm=ours alpha_bits=%s does not match "
              "2^-20 (0x35800000); the two arms are ridging differently"
              % alpha)
    return device


# ---------------------------------------------------------------------------
# The conductor.
# ---------------------------------------------------------------------------

def race(lane, args):
    device = probe(args.binary, lane)
    rungs = LADDER[lane]
    header_theirs(lane, rungs, args)
    print("FLADDER-NOTE lane=%s conductor=1 order=ours,theirs,ours,theirs "
          "blocks_per_arm=2 device_ours=%s" % (lane, device))

    cap = args.mem_mb << 20
    t_start = time.perf_counter()
    ran, wins, losses, ties, first_win, stopped = [], 0, 0, 0, None, None
    prev_n, prev_worst = 0, 0.0

    for i, n in enumerate(rungs):
        nxt = rungs[i + 1] if i + 1 < len(rungs) else -1

        # (1) the memory ceiling, THEIR side. Ours is checked inside the
        # binary, before it allocates anything.
        tbytes, tformula = footprint(lane, n)
        print("FLADDER-MEM lane=%s arm=%s n=%d bytes=%d cap=%d formula=%s "
              "verdict=%s" % (lane, ARM, n, tbytes, cap, tformula,
                              "ok" if tbytes <= cap else "over"))
        if tbytes > cap:
            print("FLADDER-REFUSED lane=%s arm=%s n=%d rule=memory detail=%s "
                  "bytes exceeds cap %d bytes;formula=%s"
                  % (lane, ARM, n, tbytes, cap, tformula))
            stopped = "memory"
            _stop(lane, prev_n, n, "memory",
                  "every rung above %d is larger still" % n)
            break

        # (2) the PREDICTIVE rule, on the DECLARED exponent and never a
        # fitted one, run BEFORE the rung so an unaffordable rung is never
        # started rather than started and abandoned.
        if prev_n:
            predicted = prev_worst * (n / prev_n) ** EXPONENT[lane]
            if predicted > args.cell_seconds * 1000.0:
                stopped = "predicted-cell"
                _stop(lane, prev_n, n, "predicted-cell",
                      "predicted %.1f ms = %.1f ms * (%d/%d)^%d exceeds "
                      "per_cell_s=%d" % (predicted, prev_worst, n, prev_n,
                                         EXPONENT[lane], args.cell_seconds))
                break

        # (3) the total budget.
        spent = time.perf_counter() - t_start
        if spent > args.total_seconds:
            stopped = "total-budget"
            _stop(lane, prev_n, n, "total-budget",
                  "%.1f s elapsed exceeds total_s=%d"
                  % (spent, args.total_seconds))
            break

        # (4) THE ALTERNATION: ours, theirs, ours, theirs, at this size,
        # before the next size.
        ours, theirs = [], []
        ours_hashes, theirs_hashes, refused = {}, {}, None
        for block in (1, 2):
            s, h, ref = run_ours_block(args.binary, lane, i, block,
                                       args.reps, args)
            ours += s
            ours_hashes.update(h)
            refused = refused or ref
            theirs_hashes = BLOCKS[lane](lane, n, block, args.reps, theirs)

        if refused or not ours or not theirs:
            stopped = refused or "no-samples"
            _stop(lane, prev_n, n, stopped,
                  "the rung produced no usable pair of cells")
            break

        # (5) both arms on the same bytes, or the rung does not count.
        bad = False
        for name in sorted(set(ours_hashes) | set(theirs_hashes)):
            o, t = ours_hashes.get(name, "-"), theirs_hashes.get(name, "-")
            ok = (o == t and o != "-")
            bad = bad or not ok
            print("FLADDER-DATACHECK lane=%s n=%d array=%s ours=%s theirs=%s "
                  "verdict=%s" % (lane, n, name, o, t,
                                  "match" if ok else "MISMATCH"))
        if bad:
            stopped = "datacheck"
            _stop(lane, prev_n, n, "datacheck",
                  "the two arms did not generate the same bytes;no ratio "
                  "from this rung means anything")
            break

        # (6) the rung, printed whatever it says.
        rung_line(lane, "ours", n, ours)
        rung_line(lane, ARM, n, theirs)
        result, om, tm = verdict_line(lane, n, ours, theirs)
        ran.append(n)
        if result == "WIN":
            wins += 1
            first_win = first_win if first_win is not None else n
        elif result == "LOSS":
            losses += 1
        else:
            ties += 1

        # (7) the MEASURED rule.
        prev_n, prev_worst = n, max(om, tm)
        if prev_worst > args.cell_seconds * 1000.0 and nxt > 0:
            stopped = "measured-cell"
            _stop(lane, n, nxt, "measured-cell",
                  "the slower arm's median %.1f ms exceeds per_cell_s=%d"
                  % (prev_worst, args.cell_seconds))
            break

    contiguous = ran == rungs[:len(ran)]
    print("FLADDER-SUMMARY lane=%s rungs_declared=%d rungs_run=%d "
          "contiguous=%s wins=%d losses=%d ties=%d first_win=%s stopped=%s"
          % (lane, len(rungs), len(ran), "yes" if contiguous else "no",
             wins, losses, ties,
             first_win if first_win is not None else "-",
             stopped if stopped else "-"))
    if not contiguous:
        print("FLADDER-NOTE lane=%s the rungs that ran are not a contiguous "
              "prefix of the declared ladder; this run is NOT a ladder and "
              "no crossover may be read off it" % lane)


def _stop(lane, reached, skipped, rule, detail):
    print("FLADDER-STOP lane=%s reached=%s skipped=%d rule=%s detail=%s"
          % (lane, reached if reached else "-", skipped, rule, detail))


def theirs_only(lane, args):
    """The opponent alone, over the full declared ladder. For when no binary
    has been built yet. It obeys the same four stop rules; it just has one arm
    to apply them to."""
    rungs = LADDER[lane]
    header_theirs(lane, rungs, args)
    cap = args.mem_mb << 20
    t_start = time.perf_counter()
    prev_n, prev_med = 0, 0.0
    for i, n in enumerate(rungs):
        nxt = rungs[i + 1] if i + 1 < len(rungs) else -1
        tbytes, tformula = footprint(lane, n)
        print("FLADDER-MEM lane=%s arm=%s n=%d bytes=%d cap=%d formula=%s "
              "verdict=%s" % (lane, ARM, n, tbytes, cap, tformula,
                              "ok" if tbytes <= cap else "over"))
        if tbytes > cap:
            _stop(lane, prev_n, n, "memory",
                  "footprint %d bytes exceeds cap %d bytes;formula=%s"
                  % (tbytes, cap, tformula))
            return
        if prev_n:
            predicted = prev_med * (n / prev_n) ** EXPONENT[lane]
            if predicted > args.cell_seconds * 1000.0:
                _stop(lane, prev_n, n, "predicted-cell",
                      "predicted %.1f ms = %.1f ms * (%d/%d)^%d exceeds "
                      "per_cell_s=%d" % (predicted, prev_med, n, prev_n,
                                         EXPONENT[lane], args.cell_seconds))
                return
        if time.perf_counter() - t_start > args.total_seconds:
            _stop(lane, prev_n, n, "total-budget",
                  "elapsed exceeds total_s=%d" % args.total_seconds)
            return
        theirs = []
        for block in (1, 2):
            BLOCKS[lane](lane, n, block, args.reps, theirs)
        rung_line(lane, ARM, n, theirs)
        prev_n, prev_med = n, median(theirs)
        if prev_med > args.cell_seconds * 1000.0 and nxt > 0:
            _stop(lane, n, nxt, "measured-cell",
                  "median %.1f ms exceeds per_cell_s=%d"
                  % (prev_med, args.cell_seconds))
            return


def main():
    ap = argparse.ArgumentParser(
        description="the classical size ladder: scikit-learn arm + conductor")
    ap.add_argument("--lane", required=True, choices=LANES,
                    help="ONE lane per invocation; there is no all-lanes mode")
    ap.add_argument("--binary", default="build/classical_ladder",
                    help="the PREBUILT Mojo driver (mojo build, not mojo run)")
    ap.add_argument("--theirs-only", action="store_true",
                    help="run the opponent alone over the full declared "
                         "ladder, for when no binary has been built")
    ap.add_argument("--reps", type=int, default=3,
                    help="timed reps per block; two blocks per arm per rung")
    ap.add_argument("--cell-seconds", type=int, default=60,
                    help="per-cell wall budget")
    ap.add_argument("--total-seconds", type=int, default=600,
                    help="total wall budget")
    ap.add_argument("--mem-mb", type=int, default=2048,
                    help="footprint cap in MiB, per arm")
    # There is deliberately no --n, --only, --skip, --from or --max-size. See
    # WHAT THIS FILE MAY NOT BE ASKED at the top of this file.
    args = ap.parse_args()

    _check_u01_variant()
    if args.theirs_only:
        theirs_only(args.lane, args)
    else:
        race(args.lane, args)


if __name__ == "__main__":
    main()
