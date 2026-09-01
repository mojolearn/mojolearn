# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SIZE LADDER for the classical lanes, asking whether any of them wins
at a size a user would actually run.

    MOJOLEARN_LADDER_LANE=gmm \\
        pixi run mojo build -I . -o build/classical_ladder \\
        bench/speed/classical_ladder_main.mojo

ONE LANE PER INVOCATION. There is no all-lanes mode and there will not be
one. The machine this runs on is the project owner's laptop, the standing
rule is one heavy thing at a time, and a driver that walks six lanes in one
process is a driver that cannot be stopped between them.

================================================================ THE QUESTION
`bench/speed/classical_speed_main.mojo` measures every classical lane at the
fixture its own lane ships. For six of them that fixture is a CORRECTNESS
fixture of a few dozen rows (gp is 12 x 3, gmm is 24 x 2, kernel_methods is
16 x 5, spectral is 48 x 4), and at those sizes the number is per-call FIXED
COST, not throughput. We lose to scikit-learn on the CPU by 13x to 35x there.
Those numbers are honest and they answer a different question from the one a
user asks, which is "at MY size, which one should I run?". Nothing in this
tree has ever measured these lanes at a size where a GPU could plausibly win,
so nobody knows whether there is a story. This file asks.

======================================== WHY A LADDER IS NOT A DATASET CHOICE
The house rule (`never build to datasets`) forbids picking, dropping,
deferring or tuning a benchmark dataset by whether it flatters us.
`bench/speed/CLASSICAL_SPEED.md` adds that the lanes' fixture builders take a
fixture id and not an `n`, so inventing a bigger one would be inventing a
dataset.

A LADDER IS NOT THAT, AND THE DESIGN HERE IS BUILT SO THAT IT STRUCTURALLY
CANNOT BECOME THAT:

  1. THE LADDER IS DECLARED UP FRONT, IN THIS SOURCE, BEFORE ANY RUN.
     `_ladder()` below is the whole of it. It is a constant of the file, not
     a function of a measurement, not a function of the environment, and not
     a function of an argument.
  2. EVERY DECLARED RUNG IS RUN, from the bottom up, in order.
  3. EVERY RUNG IS PRINTED, INCLUDING EVERY RUNG WE LOSE, and a loss is
     printed in EXACTLY the same line format as a win, with the same fields in
     the same order. Only the `ratio` and `result` values differ. There is no
     shorter line for a bad result and no field that appears only when the
     news is good.
  4. THERE IS NO `--only-this-size` ARGUMENT AND NO WAY TO DROP A RUNG.
     A caller cannot name an `n`. The one selector that exists,
     `MOJOLEARN_LADDER_RUNG_INDEX`, names an INDEX INTO THE DECLARED LADDER
     and exists only so the conductor (`bench/speed/classical_ladder_arm.py`)
     can interleave the two arms; it is refused unless
     `MOJOLEARN_LADDER_CONDUCTED=1` is also set, every conducted invocation
     PRINTS THE WHOLE DECLARED LADDER on its `FLADDER-CONDUCTED` line, and
     the conductor refuses to emit a summary unless it walked indices
     0..k contiguously. A hand-picked rung is therefore visible in the log
     rather than invisible in the shell history.

  The ladder MAY stop early, and only for three reasons, all of which are
  about the machine and none of which is about the answer. They are a
  measured cell over the per-cell wall budget, a PREDICTED cell over it (on the
  DECLARED scaling exponent, never a fitted one), and a memory footprint
  over the cap. Every stop prints the rung reached, the rung skipped, the
  rule that fired and its arithmetic. A ladder that stops because the work
  got big is a result. A laptop that swaps for twenty minutes is not.

===================================================== THE LADDERS AND THE CAPS
Worked out here rather than guessed at run time. `n` is the row count.

  gmm         1,000 / 10,000 / 100,000 / 1,000,000        exponent 1
      Per EM iteration the E-step is O(n K d^2) through the precision
      Cholesky and the M-step is O(n K d) plus O(K d^2) of per-component
      work; with K and d fixed both are LINEAR in n. Memory is X plus the
      responsibilities, 4n(2d + 2K) bytes here, which is 128 MB at n = 10^6
      with d = 8 and K = 8. It climbs.

  kde         1,000 / 10,000 / 100,000 / 1,000,000        exponent 1
      `score_samples` is a brute-force pass over every (query, train) pair,
      O(n_query * n_train * d). THE QUERY COUNT IS HELD FIXED at 1,000 and
      the TRAIN count climbs, so the work is linear in the rung. Memory is
      the train matrix twice, 8nd bytes, 64 MB at n = 10^6. It climbs.

  nystroem    1,000 / 10,000 / 100,000 / 1,000,000        exponent 1
      `fit` is O(n) basis selection plus O(q^2 d) for the basis kernel and
      O(q^3) for its eigendecomposition, and q is FIXED at 64, so all of
      that is constant in n except the selection. `transform` over the same
      n rows is O(n q d) for the cross kernel plus O(n q^2) for the matmul.
      LINEAR in n. Memory is dominated by the n x q embedding and the n x q
      cross kernel, 12nq bytes plus 8nd, which is 800 MB at n = 10^6 with
      q = 64. It climbs, and it is the rung most likely to meet the cap.

  rbfsampler  1,000 / 10,000 / 100,000 / 1,000,000        exponent 1
      `fit` does not look at X at all (it draws a d x D matrix and a D
      vector from n_features, n_components and the seed; neither does
      scikit-learn's). `transform` is one n x d by d x D matmul plus a
      cosine, O(n d D). LINEAR in n. Same memory shape as nystroem.

  spectral    1,000 / 4,000 / 16,000                      exponent 2
      CAPPED, AND HERE IS THE ARITHMETIC. The affinity is an EXACT
      brute-force self-kNN: `create_connectivity_graph` calls
      `neighbors.estimator::knn_search` with index = queries = the dataset
      (`spectral_embedding.mojo:169`), so the pair count is n^2 --
      10^6 at 1,000, 1.6 x 10^7 at 4,000, 2.56 x 10^8 at 16,000. THE NEXT
      RUNG, 64,000, IS 4.1 x 10^9 PAIRS, sixteen times the top rung, and it
      is refused BY DECLARATION rather than by the budget. 100,000 would be
      10^10 and 1,000,000 would be 10^12; a driver that tried 1,000,000
      here would take the laptop down and that is why the number is not in
      the list.
      NOTE THE CORRECTION TO A COMMON BELIEF ABOUT THIS LANE. It does NOT
      materialize a dense n x n affinity. `knn_search` tiles the queries
      256 at a time, so the distance buffer is 256 * n * 4 bytes (25.6 MB
      at n = 16,000) and the graph is a COO of n * n_neighbors entries.
      MEMORY IS NOT THE BINDING CONSTRAINT ON THIS LANE. TIME IS, through the
      n^2 pair count, plus a HOST loop that builds n * n_neighbors COO triples
      one element at a time and a HOST merge sort over 2n * n_neighbors of
      them (`coo_ops.mojo`, DEVIATION 775). The cap is set by the pair
      count and the host plumbing, not by an allocation.

  gp          500 / 1,000 / 2,000 / 4,000 / 8,000         exponent 3
      CAPPED, AND HERE IS THE ARITHMETIC. `gpr_fit_host` forms the n x n
      Gram, ridges it and factorizes it: O(n^2 d) to build and O(n^3 / 3)
      to factor. `gpr_predict_host` forms an n x n_star cross-covariance
      and triangular-solves it, O(n^2 n_star). So TIME IS CUBIC and MEMORY
      IS QUADRATIC.
      Ours in float32, the Gram is 4n^2 bytes, 256 MB at n = 8,000, plus
      the factor and the n x n_star cross-covariance.
      scikit-learn in float64: K is 8n^2 = 512 MB at n = 8,000 and
      `cholesky` returns a SECOND n x n array, so their side is about
      1.1 GB at the top rung.
      THE NEXT RUNG, 16,000, IS REFUSED BY DECLARATION: 8 * 16,000^2 =
      2.048 GB for their K ALONE, which is already over the 2 GB cap, and
      the factorization is 8x the flops of the top rung
      (16,000^3 / 3 = 1.4 x 10^12).

  There is no rung below 500 anywhere here. The sub-100-row regime already
  has its measurement in `bench/speed/CLASSICAL_SPEED.md` and re-taking it
  would just be the same fixed-cost number under a new name.

============================================================ THE DATA, AND WHY
The lanes' own fixture builders take a fixture id and not an `n`, so this
file cannot use them. It uses ONE generator for all six lanes, declared here,
seeded, and reproducible on any machine and in numpy:

    GENERATOR `splitmix64-blob-v1`

    centers[b][j] = _u01(CENTER_ROW_BASE + b, j, SALT_CENTER) * 8.0   (f64)
    x[i][j]       = Float32( _u01(i, j, jitter_salt) + centers[i % N_BLOBS][j] )

`_u01` is the splitmix64 mixer TRANSCRIBED from `bench/bench_main.mojo`;
`bench/bench_sklearn.py::u01` is its vectorized numpy twin and has been the
proven twin of that recurrence since the k-means lane. The opponent
regenerates from the same recurrence rather than reading a dump; see THE TWO
ARMS AGREE ON DATA below.

WHY THE SEPARATION IS 8.0 AND NOT 10.0. It is a power of two, so
`center * 8.0` is EXACT in both float64 and float32, with no rounding, and
therefore `a + b * 8.0` gives the same bits whether the compiler contracts it
into an FMA or not. `FMA contraction is PER SEAM` has already cost this tree
a comparison; a generator whose bits depend on whether one of two compilers
fused a multiply would hand the two arms different datasets while looking
identical on the page.

WHY THE DATA IS CLUSTERED AT ALL. Eight blobs, because a Gaussian mixture on
uniform noise collapses components and `gaussian_mixture_fit` RAISES on a
collapsed component (DEVIATION 1723), and because k-means on a spectral
embedding of uniform noise is not a defined problem. Clustering is what makes
the algorithms RUN; it is not a size and it is not chosen rung by rung. THE
SAME GENERATOR, THE SAME SALTS AND THE SAME EIGHT BLOBS SERVE EVERY LANE AND
EVERY RUNG. Nothing here is tuned per rung, and there is no second generator
anyone could quietly switch to.

The per-lane PARAMETERS (d, K, q, the bandwidth, the length scales, gamma,
n_neighbors) are pinned as comptime constants below and set EXPLICITLY ON
BOTH SIDES, never left to two libraries' defaults. That is the fairness rule
`bench/speed/CLASSICAL_SPEED.md` already obeys.

===================================================== THE TWO ARMS AGREE ON DATA
By REGENERATION plus a CHECKSUM, not by a dump. Both sides run the same
recurrence and both print `datahash=`; the conductor refuses the rung if they
disagree. The dump route that `classical_speed_main.mojo` uses, one hex word
per line read back by a pure-Python parser, is right for a 48-row fixture
and wrong here. The top gmm rung alone is 8,000,000 floats, about 72 MB of hex
text per rung, and a ladder of four rungs across six lanes would write and
re-parse the better part of a gigabyte on a laptop before a single millisecond
was measured. Regeneration is a closed form in (row, column, salt), costs no
disk, and cannot go stale between the two commands.

The checksum is `datahash=<W>.<X>` where

    X = the XOR of every float32's bit pattern              (32 bits)
    W = sum over i of (bits[i] as u64) * (i + 1), mod 2^64  (64 bits)

Both are O(n) with cheap operations and both vectorize in numpy, so the full
array is covered on both sides rather than a sampled prefix. THIS IS A
CHECKSUM AND NOT A DIGEST. Its job is to catch two implementations of one
generator drifting apart, which the position weight makes it do (a plain XOR
would pass a permutation). It is not adversarial and is not claimed to be.

======================================================== SAFETY ON THIS LAPTOP
The orchestrator runs this on the project owner's M4, which is also the
machine he works on, and that box's GPU governor drifts up to 1.7x within one
session under heat (`the M4 drifts 1.7x in 20 minutes`). Four things follow,
and they are requirements rather than preferences:

  ONE LANE PER INVOCATION.  Stated above. No all-lanes mode.

  A TIME BUDGET.  `MOJOLEARN_LADDER_CELL_S` (default 60) is the per-cell
  wall budget and `MOJOLEARN_LADDER_TOTAL_S` (default 600) the total. After
  each rung, if that rung's median cell exceeded the per-cell budget, the
  climb STOPS and says so, naming the rung reached and the rung skipped.
  Before each rung, the NEXT rung's cost is PREDICTED as
  `last_median * (n_next / n_last) ^ exponent` using the DECLARED exponent
  from the table above, and a prediction over budget also stops the climb.
  The predictor only ever stops a climb; it can never drop a lower rung,
  reorder the ladder, or change a number.

  A MEMORY CEILING.  `MOJOLEARN_LADDER_MEM_MB` (default 2048). The rung's
  footprint is computed BEFORE anything is allocated and a rung over the cap
  is REFUSED, with the arithmetic printed. `_footprint_bytes` is an estimate
  of the DOMINANT allocations, stated as such; it is a guard, not an
  accounting, and it is deliberately generous rather than tight.

  ALTERNATION.  A rung's two arms are interleaved so a thermal drift hits
  both equally, in the order ours, theirs, ours, theirs, at that size, before the next
  size. The alternation is driven by the conductor
  (`bench/speed/classical_ladder_arm.py`), which invokes this binary once per
  BLOCK. Block granularity and not rep granularity because our arm is a
  separate process and a subprocess per rep would charge every one of our
  reps a fresh warm-up; the block is bounded by the per-cell budget, so the
  drift inside one is bounded too. Each arm's samples are reported as a
  MEDIAN and a MIN-MAX BAND, and A RUNG WHOSE TWO BANDS OVERLAP IS NOT A
  FINDING; the conductor prints `result=TIE-BANDS-OVERLAP` for it and that
  is the whole of what may be said about that rung.

  `nice -n 19` is applied by the invoking command and is INHERITED by this
  process and by the scikit-learn arm alike, so it does not bias the
  comparison. It is a scheduling priority; it is not a thread cap and
  nothing here caps anyone's threads.

============================================================== OUTPUT CONTRACT
Whitespace delimited, one record per line, and nothing else on those lines.

    FLADDER-HEADER family=classical-ladder lane=<l> arm=ours mode=<FAST|...>
        device=<name> generator=splitmix64-blob-v1 ladder=<n,n,n>
        exponent=<e> per_cell_s=<s> total_s=<s> mem_cap_bytes=<b> reps=<r>
    FLADDER-CONDUCTED lane=<l> rung_index=<i> block=<b> n=<n> ladder=<n,n,n>
    FLADDER-DATA lane=<l> n=<n> array=<name> rows=<r> cols=<c> datahash=<W>.<X>
    FLADDER-MEM lane=<l> arm=ours n=<n> bytes=<b> cap=<b> formula=<text>
        verdict=<ok|over>
    FLADDER-WARMUP lane=<l> arm=ours n=<n> block=<b> ms=<f>
    FLADDER lane=<l> arm=ours n=<n> block=<b> rep=<i> ms=<f> hash=<16hex>@<c>
        note=<token>
    FLADDER-RUNG lane=<l> arm=ours n=<n> ms_median=<f> ms_min=<f> ms_max=<f>
        samples=<k>
    FLADDER-STOP lane=<l> reached=<n> skipped=<n> rule=<token> detail=<text>
    FLADDER-REFUSED lane=<l> arm=ours n=<n> rule=<token> detail=<text>

`mode=` is read from the COMPTIME constant `GLOBAL_NUMERIC_MODE` through
`numeric_mode_name()`, never from the environment and never from the flag
that was passed. Three mislabeled measurements were caught by that witness on
2026-08-23. A row whose mode is not `FAST` is not an answer to this question.

`hash=` is FNV-1a64 over the output's float bits, `@<c>` saying how many
values were folded; above `HASH_MAX` only the first `HASH_MAX` are folded and
`@<c>` says so. It is a WITHIN-ARM determinism probe. UNDER FAST IT IS
ALLOWED TO MOVE between reps and a move is a report, not a failure
(`FAST is NOT supposed to be identical`). It is NEVER a cross-arm equality
check, because scikit-learn is a different implementation and is not supposed to
produce our bits.

========================================================= WHAT IS IN THE CLOCK
One timed call is one full host-entry call plus a `ctx.synchronize()`, on a
fixture built ONCE before the block's reps. FIVE OF THE SIX LANES GO THROUGH
HOST ENTRIES THAT CONSTRUCT THEIR OWN `DeviceContext` INSIDE THE CALL
(`gaussian_mixture_fit`, `kde_score_samples_host`, `nystroem_*_host`,
`rbf_sampler_*_host`, `gpr_fit_host` / `gpr_predict_host`), so every timed
call pays a context construction and an upload. That is exactly what a caller
pays today and it is the point of the ladder, because it is a FIXED cost and
the thing that should wash out as n climbs. If a lane's ratio improves up the
ladder, that fixed cost is what it was losing to. `spectral` is the exception
(`fit_predict_dataset` takes this driver's `ctx`) and it is called out
here so nobody reads its shape as the same shape.

Environment:
    MOJOLEARN_LADDER_LANE        required; gmm|kde|nystroem|rbfsampler|
                                 spectral|gp
    MOJOLEARN_LADDER_REPS        timed reps per block, default 3
    MOJOLEARN_LADDER_BLOCKS      blocks per rung in STANDALONE mode, default 2
    MOJOLEARN_LADDER_CELL_S      per-cell wall budget in seconds, default 60
    MOJOLEARN_LADDER_TOTAL_S     total wall budget in seconds, default 600
    MOJOLEARN_LADDER_MEM_MB      footprint cap in MiB, default 2048
    MOJOLEARN_LADDER_PROBE       print the declared ladder and exit
    MOJOLEARN_LADDER_CONDUCTED   set to 1 by the conductor only
    MOJOLEARN_LADDER_RUNG_INDEX  index into the declared ladder; conducted only
    MOJOLEARN_LADDER_BLOCK       block number for the log line; conducted only
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from core.identity_trace import FNV_OFFSET, FNV_PRIME, IdentityTrace, _hex16
from checks.numerics import numeric_mode_name

# ---- gmm -------------------------------------------------------------------
from mixture.estimator import (
    COV_FULL,
    INIT_KMEANS,
    GmmParams,
    gaussian_mixture_fit,
)

# ---- kde -------------------------------------------------------------------
from kde.estimator import kde_score_samples_host

# ---- nystroem / rbfsampler -------------------------------------------------
from kernel_methods.estimator import (
    nystroem_fit_host,
    nystroem_transform_host,
    rbf_sampler_fit_host,
    rbf_sampler_transform_host,
)
from kernel_methods.checks.kernel_matrix import KM_KERNEL_RBF
from svm.impl.svm.svm_parameter import KernelParams

# ---- spectral --------------------------------------------------------------
from spectral.impl.cuvs.cluster.detail.spectral import (
    SpectralClusteringParams,
    fit_predict_dataset,
)

# ---- gp --------------------------------------------------------------------
from gaussian_process.estimator import (
    gp_profile_alpha,
    gpr_fit_host,
    gpr_predict_host,
)
from gaussian_process.checks.kernels import gp_kernel_rbf


# ============================================================================
# THE GENERATOR. One recurrence, one set of salts, every lane, every rung.
# ============================================================================

comptime N_BLOBS = 8
"""Eight blob centers. See THE DATA, AND WHY in the module docstring; this
is what makes a mixture and a spectral clustering DEFINED problems, and it
is fixed across every lane and every rung."""

comptime BLOB_SEP = 8.0
"""A POWER OF TWO on purpose. `center * 8.0` is exact in float64 and in
float32, so `jitter + center * 8.0` gives the same bits whether or not a
compiler contracts it into an FMA. `FMA contraction is PER SEAM`."""

comptime CENTER_ROW_BASE = 1000003
comptime SALT_CENTER = 41
comptime SALT_JITTER = 11
comptime SALT_QUERY = 17
comptime SALT_Y = 29


def _u01(row: Int, k: Int, salt: Int) -> Float64:
    """TRANSCRIBED from `bench/bench_main.mojo::_u01`, the splitmix64 mixer
    whose vectorized numpy twin is `bench/bench_sklearn.py::u01`. The two
    arms get the same dataset because they run the same recurrence, not
    because two people wrote down the same intention."""
    var z = (
        UInt64(row) * 0x9E3779B97F4A7C15
        + UInt64(k + 1) * 0xBF58476D1CE4E5B9
        + UInt64(salt + 1) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


def _centers(d: Int) -> List[Float64]:
    """`N_BLOBS x d`, row-major, already multiplied by `BLOB_SEP`."""
    var c = List[Float64](capacity=N_BLOBS * d)
    for b in range(N_BLOBS):
        for j in range(d):
            c.append(_u01(CENTER_ROW_BASE + b, j, SALT_CENTER) * BLOB_SEP)
    return c^


def _matrix(n: Int, d: Int, jitter_salt: Int) -> List[Float32]:
    """`n x d` row-major, generator `splitmix64-blob-v1`.

    The whole combination happens in FLOAT64 and rounds to float32 EXACTLY
    ONCE, which is what numpy's `(jit + cen).astype(np.float32)` does. Two
    roundings in different places would be two datasets."""
    var cen = _centers(d)
    var out = List[Float32](capacity=n * d)
    for i in range(n):
        var b = (i % N_BLOBS) * d
        for j in range(d):
            out.append(Float32(_u01(i, j, jitter_salt) + cen[b + j]))
    return out^


def _targets(n: Int) -> List[Float32]:
    """One target per row, column index 0 of the recurrence at `SALT_Y`.

    The gp lane's timed work does NOT depend on these values, because the Gram, its
    ridge and its factorization never see `y`, which enters only the
    triangular solve. They are here because `gpr_fit_host` needs a `y`, and
    they are generated rather than zeroed so nobody reads a degenerate
    right-hand side as a shortcut."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(Float32(_u01(i, 0, SALT_Y)))
    return out^


# ============================================================================
# THE DATA CHECKSUM. See THE TWO ARMS AGREE ON DATA in the module docstring.
# ============================================================================


comptime HEX_DIGITS = "0123456789abcdef"


def _hex8(u: UInt32) -> String:
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(HEX_DIGITS[byte=nib])
    return out


def _datahash(xs: List[Float32]) -> String:
    """`<W 16 hex>.<X 8 hex>`: a position-weighted wrapping sum and an XOR
    over the float32 bit patterns. Covers every element on both sides and
    vectorizes in numpy. A CHECKSUM, not a digest."""
    var w = UInt64(0)
    var x = UInt32(0)
    for i in range(len(xs)):
        var bits = bitcast[DType.uint32](xs[i])
        x = x ^ bits
        w = w + UInt64(bits) * UInt64(i + 1)
    return _hex16(w) + "." + _hex8(x)


# ============================================================================
# THE OUTPUT HASH. FNV-1a64 over float bits, TRANSCRIBED from
# `bench/speed/classical_speed_main.mojo`, which transcribed it from
# `core/identity_trace.mojo::fnv1a64_bytes`. Spelled out rather than imported
# because that file carries a `main`.
# ============================================================================

comptime HASH_MAX = 1048576
"""Values folded before the fold gives up and says so in `@<count>`. The
nystroem embedding at n = 10^6 and q = 64 is 64,000,000 floats and folding it
six times a rung would cost more than the benchmark it is attached to."""


def _fold_f32_list(h: UInt64, xs: List[Float32]) -> UInt64:
    var out = h
    var m = len(xs)
    if m > HASH_MAX:
        m = HASH_MAX
    for i in range(m):
        var v = UInt64(bitcast[DType.uint32](xs[i]))
        for b in range(4):
            out = (out ^ ((v >> UInt64(8 * b)) & UInt64(0xFF))) * FNV_PRIME
    return out


def _fold_i32_list(h: UInt64, xs: List[Int32]) -> UInt64:
    var out = h
    var m = len(xs)
    if m > HASH_MAX:
        m = HASH_MAX
    for i in range(m):
        # `[[mojo-int-widening-sign-extends]]`: mask AFTER the widen or every
        # negative label folds eight bytes of sign instead of four of value.
        var v = UInt64(Int(xs[i])) & UInt64(0xFFFFFFFF)
        for b in range(4):
            out = (out ^ ((v >> UInt64(8 * b)) & UInt64(0xFF))) * FNV_PRIME
    return out


def _hashed(h: UInt64, count: Int) -> String:
    var c = count
    if c > HASH_MAX:
        c = HASH_MAX
    return _hex16(h) + "@" + String(c)


# ============================================================================
# THE LADDER. DECLARED HERE, BEFORE ANY RUN. Nothing below reads a
# measurement, an argument or an environment variable to decide these.
# ============================================================================


def _lanes() -> List[String]:
    var l: List[String] = [
        String("gmm"),
        String("kde"),
        String("nystroem"),
        String("rbfsampler"),
        String("spectral"),
        String("gp"),
    ]
    return l^


def _ladder(lane: String) raises -> List[Int]:
    """THE LADDER, and the only place it is written down on this side.

    The scaling argument for every one of these lists, and the arithmetic
    behind the two caps, is in THE LADDERS AND THE CAPS in the module
    docstring. Do not add a rung here without adding its arithmetic there,
    and do not remove one because it went badly."""
    if lane == "gmm" or lane == "kde" or lane == "nystroem" or lane == "rbfsampler":
        var a: List[Int] = [1000, 10000, 100000, 1000000]
        return a^
    if lane == "spectral":
        # CAPPED at 16,000: 2.56e8 self-kNN pairs. 64,000 would be 4.1e9.
        var b: List[Int] = [1000, 4000, 16000]
        return b^
    if lane == "gp":
        # CAPPED at 8,000: 512 MB for their float64 Gram and 1.7e11 flops.
        # 16,000 needs 2.048 GB for that Gram alone, over the 2 GB cap.
        var c: List[Int] = [500, 1000, 2000, 4000, 8000]
        return c^
    raise Error(
        "classical_ladder: unknown lane '" + lane + "'. The six lanes are"
        " gmm, kde, nystroem, rbfsampler, spectral, gp."
    )


def _exponent(lane: String) raises -> Int:
    """The DECLARED scaling exponent used by the predictive stop rule. It is
    a property of the algorithm, argued in the module docstring, and it is
    never fitted to a measurement, because a fitted exponent would let a fast rung
    talk the harness into a rung it cannot afford."""
    if lane == "gmm" or lane == "kde" or lane == "nystroem" or lane == "rbfsampler":
        return 1
    if lane == "spectral":
        return 2
    if lane == "gp":
        return 3
    raise Error("classical_ladder: unknown lane '" + lane + "'")


def _ladder_text(rungs: List[Int]) -> String:
    var s = String("")
    for i in range(len(rungs)):
        if i > 0:
            s += ","
        s += String(rungs[i])
    return s


# ============================================================================
# THE PINNED PARAMETERS. Set EXPLICITLY here and EXPLICITLY on the other side;
# never left to two libraries' defaults. `bench/speed/classical_ladder_arm.py`
# carries the same values and prints them.
# ============================================================================

comptime GMM_D = 8
comptime GMM_K = 8
comptime GMM_MAX_ITER = 20
comptime GMM_TOL = Float32(1.0e-3)
comptime GMM_REG_COVAR = Float32(1.0e-6)
comptime GMM_SEED: UInt64 = 0

comptime KDE_D = 8
comptime KDE_N_QUERY = 1000
comptime KDE_BANDWIDTH = Float32(1.0)

comptime KM_D = 8
comptime KM_Q = 64
comptime KM_GAMMA = 0.125
comptime KM_SEED: UInt64 = 20260825

comptime SPECTRAL_D = 4
comptime SPECTRAL_CLUSTERS = 8
comptime SPECTRAL_COMPONENTS = 8
comptime SPECTRAL_N_INIT = 10
comptime SPECTRAL_NEIGHBORS = 10
comptime SPECTRAL_TOL = Float32(1.0e-5)
comptime SPECTRAL_SEED: UInt64 = 7

comptime GP_D = 8
comptime GP_N_STAR = 1000
comptime GP_LENGTH_SCALE = Float32(1.0)
"""`gp_profile_alpha()` is the ridge and it IS 2^-20 (`chol_jitter_pinned`,
DEVIATION 1751). The Python arm spells it `2.0 ** -20` rather than a decimal
because a POWER OF TWO is the same real number in float32 and in float64;
`1e-3` would be two different ridges on the two arms. The header prints its
bits so a divergence is visible rather than assumed away."""


# ============================================================================
# The environment, the budgets and the small numeric helpers.
# ============================================================================


def _env(name: String) -> String:
    return String(getenv(name))


def _env_int(name: String, default: Int) raises -> Int:
    var s = _env(name)
    if s == "":
        return default
    return Int(atol(s))


def _no_spaces(s: String) -> String:
    """`device=` is one whitespace-delimited field and `ctx.name()` returns
    things like `Apple M4`."""
    var out = String("")
    for i in range(s.byte_length()):
        var c = String(s[byte=i])
        out += String("_") if c == String(" ") else c
    return out


def _sorted(xs: List[Float64]) -> List[Float64]:
    var s = xs.copy()
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    return s^


def _median(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = _sorted(xs)
    var m = len(s)
    if m % 2 == 1:
        return s[m // 2]
    return (s[m // 2 - 1] + s[m // 2]) * 0.5


def _min_of(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var v = xs[0]
    for i in range(1, len(xs)):
        if xs[i] < v:
            v = xs[i]
    return v


def _max_of(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var v = xs[0]
    for i in range(1, len(xs)):
        if xs[i] > v:
            v = xs[i]
    return v


def _powi(base: Float64, e: Int) -> Float64:
    var out = 1.0
    for _e in range(e):
        out = out * base
    return out


# ============================================================================
# THE MEMORY CEILING. Computed BEFORE anything is allocated.
#
# These are ESTIMATES OF THE DOMINANT ALLOCATIONS and are stated as such, a
# guard rather than an accounting. They are deliberately generous rather than
# tight, because the failure this exists to prevent, a laptop swapping for
# twenty minutes, is much worse than refusing a rung that would have just fit.
# The formula string is printed beside the number so a refusal can be read
# and argued with rather than merely obeyed.
# ============================================================================


def _footprint_bytes(lane: String, n: Int) raises -> Int:
    if lane == "gmm":
        # host X + device X + device resp + device log-prob
        return 8 * n * (GMM_D + GMM_K)
    if lane == "kde":
        # host train + device train (the 1,000 queries are constant)
        return 8 * n * KDE_D + 8 * KDE_N_QUERY * KDE_D
    if lane == "nystroem" or lane == "rbfsampler":
        # host X + device X, then the n x q cross kernel and the n x q
        # embedding on the device and once more on the host
        return 8 * n * KM_D + 12 * n * KM_Q
    if lane == "spectral":
        # host X + device X, the 256-query distance tile, and the COO
        # (n * k triples, doubled by symmetrize, three arrays each)
        return 8 * n * SPECTRAL_D + 1024 * n + 36 * n * SPECTRAL_NEIGHBORS
    if lane == "gp":
        # the n x n Gram and its factor in float32, plus the n x n_star
        # cross-covariance, plus X twice
        return 8 * n * n + 4 * n * GP_N_STAR + 8 * n * GP_D
    raise Error("classical_ladder: unknown lane '" + lane + "'")


def _footprint_formula(lane: String, n: Int) raises -> String:
    if lane == "gmm":
        return (
            "8*n*(d+K)=8*" + String(n) + "*(" + String(GMM_D) + "+"
            + String(GMM_K) + ")"
        )
    if lane == "kde":
        return (
            "8*n*d+8*q*d=8*" + String(n) + "*" + String(KDE_D) + "+8*"
            + String(KDE_N_QUERY) + "*" + String(KDE_D)
        )
    if lane == "nystroem" or lane == "rbfsampler":
        return (
            "8*n*d+12*n*Q=8*" + String(n) + "*" + String(KM_D) + "+12*"
            + String(n) + "*" + String(KM_Q)
        )
    if lane == "spectral":
        return (
            "8*n*d+1024*n+36*n*k=8*" + String(n) + "*" + String(SPECTRAL_D)
            + "+1024*" + String(n) + "+36*" + String(n) + "*"
            + String(SPECTRAL_NEIGHBORS)
        )
    if lane == "gp":
        return (
            "8*n*n+4*n*nstar+8*n*d=8*" + String(n) + "*" + String(n) + "+4*"
            + String(n) + "*" + String(GP_N_STAR) + "+8*" + String(n) + "*"
            + String(GP_D)
        )
    raise Error("classical_ladder: unknown lane '" + lane + "'")


# ============================================================================
# The line emitters. A WIN AND A LOSS ARE THE SAME LINE.
# ============================================================================


def _emit(
    lane: String, n: Int, block: Int, rep: Int, ns: Int, tag: String,
    note: String,
):
    print(
        "FLADDER lane=" + lane + " arm=ours n=" + String(n) + " block="
        + String(block) + " rep=" + String(rep) + " ms="
        + String(Float64(ns) / 1.0e6) + " hash=" + tag + " note=" + note
    )


def _warmup(lane: String, n: Int, block: Int, ns: Int):
    print(
        "FLADDER-WARMUP lane=" + lane + " arm=ours n=" + String(n) + " block="
        + String(block) + " ms=" + String(Float64(ns) / 1.0e6)
    )


def _data_line(lane: String, n: Int, name: String, rows: Int, cols: Int, xs: List[Float32]):
    print(
        "FLADDER-DATA lane=" + lane + " n=" + String(n) + " array=" + name
        + " rows=" + String(rows) + " cols=" + String(cols) + " datahash="
        + _datahash(xs)
    )


def _refused(lane: String, n: Int, rule: String, detail: String):
    print(
        "FLADDER-REFUSED lane=" + lane + " arm=ours n=" + String(n) + " rule="
        + rule + " detail=" + detail
    )


def _stop(lane: String, reached: Int, skipped: Int, rule: String, detail: String):
    print(
        "FLADDER-STOP lane=" + lane + " reached=" + String(reached)
        + " skipped=" + String(skipped) + " rule=" + rule + " detail="
        + detail
    )


def _rung_line(lane: String, n: Int, samples: List[Float64]):
    """The rung's summary for OUR arm. It is printed for every rung that
    ran, whatever the number says; there is no branch here on the value."""
    print(
        "FLADDER-RUNG lane=" + lane + " arm=ours n=" + String(n)
        + " ms_median=" + String(_median(samples))
        + " ms_min=" + String(_min_of(samples))
        + " ms_max=" + String(_max_of(samples))
        + " samples=" + String(len(samples))
    )


# ============================================================================
# THE LANES. One block = build the fixture once, one untimed warm-up call,
# then `reps` timed calls. Every one of these calls a HOST entry, because the
# lanes' host entries are their public surface and because the fixture is
# synthesized here rather than taken from a lane's fixture builder.
# ============================================================================


def _block_gmm(
    lane: String, n: Int, block: Int, reps: Int, mut samples: List[Float64]
) raises:
    """`gaussian_mixture_fit`: K = 8 full-covariance components on n x 8,
    `init_params = kmeans`, `max_iter = 20`, scikit-learn's `tol = 1e-3` and
    `reg_covar = 1e-6`, `n_init = 1`, `random_state = 0`.

    `max_iter` is pinned at 20 rather than scikit-learn's 100 so that one
    cell is a bounded amount of work on a laptop. IT IS PINNED TO THE SAME
    VALUE ON BOTH ARMS and `tol` is scikit-learn's default on both, so the
    two may still stop at DIFFERENT iterations on the same data. That is why
    `note=iters<N>` rides on every line, so that if the two arms' iteration counts
    differ, the two cells did different amounts of work and the conductor
    says so rather than dividing them.

    `gaussian_mixture_fit` constructs its OWN `DeviceContext` per call
    (`mixture/estimator.mojo:704`). At 24 rows that context was most of the
    measurement; that is the fixed cost this ladder exists to climb out of."""
    var d = GMM_D
    var x = _matrix(n, d, SALT_JITTER)
    _data_line(lane, n, "x", n, d, x)
    var p = GmmParams.default()
    p.n_components = GMM_K
    p.covariance_type = COV_FULL
    p.tol = GMM_TOL
    p.reg_covar = GMM_REG_COVAR
    p.max_iter = GMM_MAX_ITER
    p.init_params = INIT_KMEANS
    p.random_state = GMM_SEED
    for r in range(reps + 1):
        var t0 = perf_counter_ns()
        var model = gaussian_mixture_fit(x, n, d, p)
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, model.weights)
        h = _fold_f32_list(h, model.means)
        h = _fold_f32_list(h, model.covariances)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(
                lane, n, block, r, t1 - t0,
                _hashed(h, len(model.means)),
                "iters" + String(model.n_iter),
            )
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = model^
    _ = x^


def _block_kde(
    lane: String, n: Int, block: Int, reps: Int, mut samples: List[Float64]
) raises:
    """`kde_score_samples_host`: gaussian kernel, euclidean metric, bandwidth
    1.0, no sample weights, 1,000 query rows against n train rows in 8
    dimensions.

    THE QUERY COUNT IS HELD FIXED and the train count climbs, so the rung is
    linear in n. Both arms time the WHOLE call, which on their side is
    `KernelDensity(...).fit(X).score_samples(Q)` including the tree build --
    scikit-learn has no brute-force option for this estimator and a tree with
    `rtol = atol = 0` cannot prune, so it computes the same sum by a
    different data structure. The arm reports its fit and score separately so
    a reader can see the split."""
    var d = KDE_D
    var train = _matrix(n, d, SALT_JITTER)
    var query = _matrix(KDE_N_QUERY, d, SALT_QUERY)
    _data_line(lane, n, "train", n, d, train)
    _data_line(lane, n, "query", KDE_N_QUERY, d, query)
    var weights = List[Float32]()
    for r in range(reps + 1):
        var t0 = perf_counter_ns()
        var out = kde_score_samples_host(
            train, n, query, KDE_N_QUERY, d, KDE_BANDWIDTH,
            String("gaussian"), String("euclidean"), weights, False,
        )
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, out)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(lane, n, block, r, t1 - t0, _hashed(h, len(out)), "-")
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = out^
    _ = train^
    _ = query^
    _ = weights^


def _block_nystroem(
    lane: String, n: Int, block: Int, reps: Int, mut samples: List[Float64]
) raises:
    """`nystroem_fit_host` then `nystroem_transform_host` on the SAME n rows:
    RBF, gamma 0.125, 64 components, seed 20260825.

    Fitting and transforming the same matrix is what makes this rung linear
    in n, because the fit's q x q kernel and q x q eigendecomposition are constant,
    and the transform's n x q cross kernel and n x q by q x q matmul are the
    work that climbs.

    THE BASIS SAMPLE WILL NOT MATCH THE OPPONENT'S. Ours permutes with a
    pinned Philox stream and scikit-learn's with numpy's `RandomState`, so
    the two fit different basis rows. The WORK is the same shape, which is
    what is timed; the outputs are not comparable and neither arm claims they
    are."""
    var d = KM_D
    var x = _matrix(n, d, SALT_JITTER)
    _data_line(lane, n, "x", n, d, x)
    var kp = KernelParams(KM_KERNEL_RBF, 3, KM_GAMMA, 1.0)
    for r in range(reps + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var model = nystroem_fit_host(x, n, d, kp, KM_Q, KM_SEED, trace)
        var emb = nystroem_transform_host(model, x, n, trace)
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, emb)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(lane, n, block, r, t1 - t0, _hashed(h, len(emb)), "-")
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = emb^
        _ = model^
    _ = x^


def _block_rbfsampler(
    lane: String, n: Int, block: Int, reps: Int, mut samples: List[Float64]
) raises:
    """`rbf_sampler_fit_host` then `rbf_sampler_transform_host` on n rows:
    random Fourier features, 64 components, gamma 0.125, seed 20260825.

    `fit` DOES NOT LOOK AT X and neither does scikit-learn's; it draws from
    `n_features`, `n_components` and the seed alone. All of the work that
    climbs is in the transform, which is one n x 8 by 8 x 64 matmul, an offset, a
    cosine and a scale. Same caveat as nystroem about the draws differing
    between the two arms."""
    var d = KM_D
    var x = _matrix(n, d, SALT_JITTER)
    _data_line(lane, n, "x", n, d, x)
    for r in range(reps + 1):
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        var model = rbf_sampler_fit_host(
            d, KM_Q, Float32(KM_GAMMA), KM_SEED, trace
        )
        var z = rbf_sampler_transform_host(model, x, n, trace)
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, z)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(lane, n, block, r, t1 - t0, _hashed(h, len(z)), "-")
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = z^
        _ = model^
    _ = x^


def _block_spectral(
    ctx: DeviceContext, lane: String, n: Int, block: Int, reps: Int,
    mut samples: List[Float64],
) raises:
    """`fit_predict_dataset`: a 10-NN connectivity graph, a normalized
    Laplacian, a restarted Lanczos embedding at 8 components and k-means at
    8 clusters with `n_init = 10`, tolerance 1e-5, seed 7, on n x 4.

    THE ONLY LANE HERE THAT TAKES THIS DRIVER'S CONTEXT rather than building
    its own inside the call, so its timed region does not carry a context
    construction and its shape is not the other five lanes' shape.

    THE COST THAT CAPS THIS LADDER IS THE SELF-kNN. Index and queries are
    both the dataset, so the exact search is n^2 pairs, and the COO plumbing
    behind it (`coo_ops.mojo`, DEVIATION 775) runs on the HOST over
    n * n_neighbors triples with a host merge sort over twice that."""
    var d = SPECTRAL_D
    var x = _matrix(n, d, SALT_JITTER)
    _data_line(lane, n, "x", n, d, x)
    var cfg = SpectralClusteringParams(
        n_clusters=SPECTRAL_CLUSTERS,
        n_components=SPECTRAL_COMPONENTS,
        n_init=SPECTRAL_N_INIT,
        n_neighbors=SPECTRAL_NEIGHBORS,
        tolerance=SPECTRAL_TOL,
        seed=SPECTRAL_SEED,
    )
    for r in range(reps + 1):
        var labels = List[Int32]()
        var emb = List[Float32]()
        var trace = IdentityTrace.disabled()
        var t0 = perf_counter_ns()
        fit_predict_dataset(ctx, cfg, x, n, d, labels, emb, trace)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var h = _fold_i32_list(FNV_OFFSET, labels)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(lane, n, block, r, t1 - t0, _hashed(h, len(labels)), "-")
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = labels^
        _ = emb^
    _ = x^


def _block_gp(
    lane: String, n: Int, block: Int, reps: Int, mut samples: List[Float64]
) raises:
    """`gpr_fit_host` then `gpr_predict_host(return_std = True)`: an ARD RBF
    kernel whose eight length scales are all 1.0, the profile's ridge
    (`gp_profile_alpha()`, which IS 2^-20), no hyperparameter optimizer
    (there is none to run; DEVIATION 1761), n training rows x 8 features and
    1,000 test rows.

    THE OPPONENT WORKS IN FLOAT64 AND WE WORK IN FLOAT32. That is a real
    difference in the amount of arithmetic and it cannot be turned off on
    their side; `sklearn.gaussian_process` has no float32 path. It is stated
    on every gp row rather than absorbed into the ratio.

    A rung whose Gram does not factor is REFUSED by name (`model.info`) and
    contributes no timing, because a failed factorization is not a fast
    one."""
    var d = GP_D
    var x = _matrix(n, d, SALT_JITTER)
    var y = _targets(n)
    var xs = _matrix(GP_N_STAR, d, SALT_QUERY)
    _data_line(lane, n, "x", n, d, x)
    _data_line(lane, n, "y", n, 1, y)
    _data_line(lane, n, "x_star", GP_N_STAR, d, xs)
    var ls = List[Float32](capacity=d)
    for _j in range(d):
        ls.append(GP_LENGTH_SCALE)
    var spec = gp_kernel_rbf(ls)
    var alpha = gp_profile_alpha()
    for r in range(reps + 1):
        var t0 = perf_counter_ns()
        var model = gpr_fit_host(x, n, d, y, spec, alpha)
        if model.info != 0:
            _refused(
                lane, n, "factorization",
                "the ridged Gram did not factor: info=" + String(model.info),
            )
            _ = model^
            return
        var pred = gpr_predict_host(model, xs, GP_N_STAR, True)
        var t1 = perf_counter_ns()
        var h = _fold_f32_list(FNV_OFFSET, pred.mean)
        h = _fold_f32_list(h, pred.std)
        if r == 0:
            _warmup(lane, n, block, t1 - t0)
        else:
            _emit(
                lane, n, block, r, t1 - t0, _hashed(h, len(pred.mean)),
                "clamped" + String(pred.n_clamped),
            )
            samples.append(Float64(t1 - t0) / 1.0e6)
        _ = pred^
        _ = model^
    _ = x^
    _ = y^
    _ = xs^


def _block(
    ctx: DeviceContext, lane: String, n: Int, block: Int, reps: Int,
    mut samples: List[Float64],
) raises:
    if lane == "gmm":
        _block_gmm(lane, n, block, reps, samples)
    elif lane == "kde":
        _block_kde(lane, n, block, reps, samples)
    elif lane == "nystroem":
        _block_nystroem(lane, n, block, reps, samples)
    elif lane == "rbfsampler":
        _block_rbfsampler(lane, n, block, reps, samples)
    elif lane == "spectral":
        _block_spectral(ctx, lane, n, block, reps, samples)
    elif lane == "gp":
        _block_gp(lane, n, block, reps, samples)
    else:
        raise Error("classical_ladder: unknown lane '" + lane + "'")


# ============================================================================
# main
# ============================================================================


def main() raises:
    var lane = _env("MOJOLEARN_LADDER_LANE")
    if lane == "":
        var names = _lanes()
        var joined = String("")
        for i in range(len(names)):
            if i > 0:
                joined += ","
            joined += names[i]
        print(
            "FLADDER-REFUSED lane=- arm=ours n=0 rule=no-lane detail="
            "MOJOLEARN_LADDER_LANE is required;one_of=" + joined
        )
        return

    var rungs = _ladder(lane)
    var exponent = _exponent(lane)
    var reps = _env_int("MOJOLEARN_LADDER_REPS", 3)
    var blocks = _env_int("MOJOLEARN_LADDER_BLOCKS", 2)
    var cell_s = _env_int("MOJOLEARN_LADDER_CELL_S", 60)
    var total_s = _env_int("MOJOLEARN_LADDER_TOTAL_S", 600)
    var mem_cap = _env_int("MOJOLEARN_LADDER_MEM_MB", 2048) * 1024 * 1024
    var conducted = _env("MOJOLEARN_LADDER_CONDUCTED") == "1"

    var ctx = DeviceContext()
    print(
        "FLADDER-HEADER family=classical-ladder lane=" + lane + " arm=ours"
        + " mode=" + numeric_mode_name()
        + " device=" + _no_spaces(ctx.name())
        + " generator=splitmix64-blob-v1"
        + " ladder=" + _ladder_text(rungs)
        + " exponent=" + String(exponent)
        + " per_cell_s=" + String(cell_s)
        + " total_s=" + String(total_s)
        + " mem_cap_bytes=" + String(mem_cap)
        + " reps=" + String(reps)
        + " alpha_bits=0x" + _hex8(bitcast[DType.uint32](gp_profile_alpha()))
    )

    if _env("MOJOLEARN_LADDER_PROBE") == "1":
        # THE PROBE. The conductor asks this binary for the declared ladder
        # rather than trusting its own copy of it, then asserts the two
        # agree. Two spellings of a ladder that are never compared are two
        # ladders. Nothing is measured and nothing is allocated here.
        print(
            "FLADDER-PROBE lane=" + lane + " ladder=" + _ladder_text(rungs)
            + " exponent=" + String(exponent) + " generator="
            "splitmix64-blob-v1"
        )
        return

    if conducted:
        # CONDUCTED MODE. One rung, one block, no aggregation and no verdict:
        # the conductor owns both. The whole declared ladder is printed on
        # this line every time, so a hand-picked rung index is a visible fact
        # in the log rather than an invisible one in the shell history.
        var idx = _env_int("MOJOLEARN_LADDER_RUNG_INDEX", -1)
        var block = _env_int("MOJOLEARN_LADDER_BLOCK", 1)
        if idx < 0 or idx >= len(rungs):
            _refused(
                lane, 0, "bad-rung-index",
                "MOJOLEARN_LADDER_RUNG_INDEX=" + String(idx)
                + " is not an index into ladder=" + _ladder_text(rungs),
            )
            return
        var n = rungs[idx]
        print(
            "FLADDER-CONDUCTED lane=" + lane + " rung_index=" + String(idx)
            + " block=" + String(block) + " n=" + String(n) + " ladder="
            + _ladder_text(rungs)
        )
        var bytes = _footprint_bytes(lane, n)
        print(
            "FLADDER-MEM lane=" + lane + " arm=ours n=" + String(n)
            + " bytes=" + String(bytes) + " cap=" + String(mem_cap)
            + " formula=" + _footprint_formula(lane, n) + " verdict="
            + (String("ok") if bytes <= mem_cap else String("over"))
        )
        if bytes > mem_cap:
            _refused(
                lane, n, "memory",
                "footprint " + String(bytes) + " bytes exceeds cap "
                + String(mem_cap) + " bytes;formula="
                + _footprint_formula(lane, n),
            )
            return
        var samples = List[Float64]()
        _block(ctx, lane, n, block, reps, samples)
        return

    # STANDALONE MODE. The whole declared ladder, our arm only, bottom up,
    # with all four stop rules enforced here. Useful on its own and it is
    # what proves the ladder runs before the conductor pairs it with an
    # opponent; the RACE goes through the conductor, because only the
    # conductor can interleave the two arms.
    var elapsed_ns = 0
    var t_start = perf_counter_ns()
    var last_median = 0.0
    var last_n = 0
    for i in range(len(rungs)):
        var n = rungs[i]
        var nxt = -1
        if i + 1 < len(rungs):
            nxt = rungs[i + 1]

        var bytes = _footprint_bytes(lane, n)
        print(
            "FLADDER-MEM lane=" + lane + " arm=ours n=" + String(n)
            + " bytes=" + String(bytes) + " cap=" + String(mem_cap)
            + " formula=" + _footprint_formula(lane, n) + " verdict="
            + (String("ok") if bytes <= mem_cap else String("over"))
        )
        if bytes > mem_cap:
            _refused(
                lane, n, "memory",
                "footprint " + String(bytes) + " bytes exceeds cap "
                + String(mem_cap) + " bytes;formula="
                + _footprint_formula(lane, n),
            )
            _stop(
                lane, last_n, n, "memory",
                "every rung above " + String(n) + " is larger still",
            )
            return

        # The PREDICTIVE rule, using the DECLARED exponent and never a
        # fitted one. It runs BEFORE the rung, so a rung the machine cannot
        # afford is never started rather than started and abandoned.
        if last_n > 0:
            var ratio = Float64(n) / Float64(last_n)
            var predicted_ms = last_median * _powi(ratio, exponent)
            if predicted_ms > Float64(cell_s) * 1000.0:
                _stop(
                    lane, last_n, n, "predicted-cell",
                    "predicted " + String(predicted_ms) + " ms = "
                    + String(last_median) + " ms * (" + String(n) + "/"
                    + String(last_n) + ")^" + String(exponent)
                    + " exceeds per_cell_s=" + String(cell_s),
                )
                return

        elapsed_ns = perf_counter_ns() - t_start
        if Float64(elapsed_ns) / 1.0e9 > Float64(total_s):
            _stop(
                lane, last_n, n, "total-budget",
                String(Float64(elapsed_ns) / 1.0e9) + " s elapsed exceeds"
                " total_s=" + String(total_s),
            )
            return

        var samples = List[Float64]()
        for b in range(1, blocks + 1):
            _block(ctx, lane, n, b, reps, samples)
        if len(samples) == 0:
            _stop(lane, last_n, n, "no-samples", "the rung produced no timing")
            return
        _rung_line(lane, n, samples)

        last_median = _median(samples)
        last_n = n

        # The MEASURED rule.
        if last_median > Float64(cell_s) * 1000.0 and nxt > 0:
            _stop(
                lane, n, nxt, "measured-cell",
                "median " + String(last_median) + " ms exceeds per_cell_s="
                + String(cell_s),
            )
            return
