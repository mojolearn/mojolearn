# The classical SIZE LADDER, or do these lanes win at a size anyone runs?

**Nothing in this file is a measurement.** It describes a harness that has
been written and has never been built or run. The first run belongs to the
orchestrator, on the M4 laptop, one lane at a time.

## The question, and why the existing numbers do not answer it

`bench/speed/CLASSICAL_SPEED.md` measures every classical lane at the fixture
its own lane ships. Six of those fixtures are **correctness** fixtures of a
few dozen rows. gp is 12 x 3, gmm is 24 x 2, kernel_methods is 16 x 5 and
spectral is 48 x 4. At those sizes the number is per-call **fixed cost**, not
throughput, and we lose to scikit-learn on the CPU by 13x to 35x. Those
numbers are honest and they are answering a different question from the one a
user asks, which is "at my size, which one should I run?".

Nothing in this tree has measured these lanes at a size where a GPU could
plausibly win, so nobody knows whether there is a story. This harness asks.

## Why a ladder is not a violation of `never build to datasets`

The rule forbids picking, dropping, deferring or tuning a benchmark dataset
by whether it flatters us. It does not forbid measuring how a lane **scales**.
The design is built so that the difference is structural and not a promise:

* the ladder is **declared up front** as a constant in
  `bench/speed/classical_ladder_main.mojo::_ladder()`, before any run;
* **every declared rung is run**, from the bottom up, in order;
* **every rung is printed**, including every rung we lose, and a loss prints
  in exactly the same line format as a win, same fields in the same order;
* there is **no `--only-this-size` argument** and no way for a caller to drop
  a rung. `classical_ladder_arm.py` has no `--n`, `--only`, `--skip`,
  `--from` or `--max-size`. The one selector that exists,
  `MOJOLEARN_LADDER_RUNG_INDEX`, names an index into the declared ladder,
  is refused unless the conductor set `MOJOLEARN_LADDER_CONDUCTED=1`, prints
  the whole declared ladder on every invocation, and is only ever driven from
  index 0 upward. The conductor's `FLADDER-SUMMARY` carries
  `contiguous=yes|no` and a `no` says in plain words that the run is not a
  ladder and no crossover may be read off it.

The ladder **may** stop early, and only for reasons about the machine, which
are a measured cell over the per-cell wall budget, a predicted cell over it,
and a memory footprint over the cap. Every stop names the rung reached, the rung
skipped, the rule that fired and its arithmetic.

## The two files

| file | what it is |
| --- | --- |
| `bench/speed/classical_ladder_main.mojo` | our arm, and the **original** declaration of the ladder. One lane per process. |
| `bench/speed/classical_ladder_arm.py` | the scikit-learn arm **and** the conductor that interleaves the two. |

Neither file touches any lane's fixtures, gates or `impl/` code, and neither
touches `bench/speed/classical_speed_main.mojo`. `FLADDER` lines are a
separate namespace from that file's `FSPEED` lines on purpose; nothing parses
both.

## The ladders, the scaling argument and the caps

| lane | ladder | exponent | why |
| --- | --- | --- | --- |
| `gmm` | 1,000 / 10,000 / 100,000 / 1,000,000 | 1 | E-step O(n K d^2), M-step O(n K d); linear in n at fixed K = 8, d = 8. Memory 4n(2d + 2K) = 128 MB at the top rung. |
| `kde` | 1,000 / 10,000 / 100,000 / 1,000,000 | 1 | brute-force over (query, train) pairs; **the query count is fixed at 1,000** and the train count climbs, so the rung is linear. Memory 8nd = 64 MB at the top rung. |
| `nystroem` | 1,000 / 10,000 / 100,000 / 1,000,000 | 1 | fit is O(q^2 d) + O(q^3) with q fixed at 64; transform over the same n rows is O(n q d) + O(n q^2). Linear. Memory 12nq + 8nd, about 800 MB at the top rung: this is the rung most likely to meet the cap. |
| `rbfsampler` | 1,000 / 10,000 / 100,000 / 1,000,000 | 1 | fit does not look at X at all (nor does scikit-learn's); transform is one n x 8 by 8 x 64 matmul plus a cosine. Linear. |
| `spectral` | 1,000 / 4,000 / 16,000 | 2 | **capped.** See below. |
| `gp` | 500 / 1,000 / 2,000 / 4,000 / 8,000 | 3 | **capped.** See below. |

The exponent is the **declared** scaling exponent and is used only by the
predictive stop rule. It is never fitted to a measurement; a fitted exponent
would let a fast rung talk the harness into a rung the laptop cannot afford.

### The spectral cap, and a correction

The affinity is an **exact brute-force self-kNN**, because
`create_connectivity_graph` calls `neighbors.estimator::knn_search` with index
= queries = the dataset (`spectral_embedding.mojo:169`). So the pair count is
n^2: 10^6 at 1,000, 1.6 x 10^7 at 4,000, **2.56 x 10^8 at 16,000**. The next
rung, 64,000, would be 4.1 x 10^9 pairs, sixteen times the top rung, and it is
refused **by declaration** rather than by a budget. 100,000 would be 10^10 and
1,000,000 would be 10^12; a driver that tried 1,000,000 here would take the
laptop down, which is why the number is not in the list.

**Correction to a common belief about this lane. It does not build a dense
n x n affinity.** `knn_search` tiles the queries 256 at a time, so the
distance buffer is 256 * n * 4 bytes (25.6 MB at n = 16,000) and the graph is
a COO of n * n_neighbors entries. **Memory is not the binding constraint on
this lane. Time is**, through the n^2 pair count, plus a host loop that builds
n * n_neighbors COO triples one element at a time and a host merge sort over
twice that (`spectral/impl/sparse/op/coo_ops.mojo`, DEVIATION 775). The cap
is set by the pair count and the host plumbing, not by an allocation.

### The gp cap

`gpr_fit_host` forms the n x n Gram, ridges it and factorizes it: O(n^2 d) to
build, O(n^3 / 3) to factor. `gpr_predict_host` forms an n x n_star
cross-covariance and triangular-solves it. **Time is cubic, memory is
quadratic.**

* ours, float32: 4n^2 = 256 MB for the Gram at n = 8,000, plus the factor and
  the n x 1,000 cross-covariance;
* scikit-learn, float64: K is 8n^2 = 512 MB at n = 8,000 and `cholesky`
  returns a **second** n x n array, so their side is about 1.1 GB at the top
  rung;
* the next rung, 16,000, is refused **by declaration**: 8 * 16,000^2 =
  **2.048 GB for their K alone**, already over the 2 GB cap, and the
  factorization is 8x the flops of the top rung (1.4 x 10^12).

There is no rung below 500 anywhere here. The sub-100-row regime already has
its measurement in `CLASSICAL_SPEED.md` and re-taking it would be the same
fixed-cost number under a new name.

## The safety limits and their defaults

The orchestrator runs this on the project owner's M4, which is also the
machine he works on, and that box's GPU governor drifts up to 1.7x within a
session under heat.

| limit | default | flag / variable | behavior |
| --- | --- | --- | --- |
| one lane per invocation | always | `--lane` / `MOJOLEARN_LADDER_LANE` | there is no all-lanes mode |
| per-cell wall budget | 60 s | `--cell-seconds` / `MOJOLEARN_LADDER_CELL_S` | a rung whose slower arm's median exceeds it **stops the climb** |
| predicted-cell rule | same 60 s | (same) | before each rung, `last_worst * (n_next / n_last) ^ exponent`; over budget **stops the climb** before the rung starts |
| total wall budget | 600 s | `--total-seconds` / `MOJOLEARN_LADDER_TOTAL_S` | checked before each rung |
| memory ceiling | 2048 MiB | `--mem-mb` / `MOJOLEARN_LADDER_MEM_MB` | the footprint is computed **before** anything is allocated; a rung over the cap is refused with the arithmetic printed |
| timed reps | 3 per block | `--reps` / `MOJOLEARN_LADDER_REPS` | two blocks per arm per rung, so six samples per arm |

Each arm also gets one **untimed warm-up call per block**, printed as
`FLADDER-WARMUP` and never in the table.

The footprint formulas are **estimates of the dominant allocations**, stated
as such in both files, a guard rather than an accounting, and deliberately generous
rather than tight. The formula string is printed beside the number so a
refusal can be read and argued with rather than merely obeyed.

`nice -n 19` is in the documented commands because the standing rule for this
laptop is no heavy local compute and one heavy thing at a time. It is a
scheduling priority, not a thread cap, and **both arms inherit it** from the
invoking command, so it does not bias the comparison.

### Alternation, and what a band means

Inside a rung the arms alternate **ours, theirs, ours, theirs**, at that size,
before the next size, so a thermal drift hits both equally. The conductor
drives it, invoking the Mojo binary once per **block**; block granularity and
not rep granularity because our arm is a separate process and a subprocess per
rep would charge every one of our reps a fresh warm-up. A block is bounded by
the per-cell budget, so the drift inside one is bounded too. The subprocess
start cost is in nobody's number, because the binary times its own calls internally
and prints the milliseconds, and the conductor reads those rather than the
wall time of the subprocess.

Each arm reports a **median** and a **min-max band**.
**A rung whose two bands overlap is not a finding.** It prints as
`result=TIE-BANDS-OVERLAP` and that is the whole of what may be said about it.
Do not quote its ratio.

## How the two arms agree on data

By **regeneration plus a checksum**, not by a dump.

Generator `splitmix64-blob-v1`, declared in both files:

```
centers[b][j] = _u01(1000003 + b, j, 41) * 8.0                       (float64)
x[i][j]       = float32( _u01(i, j, jitter_salt) + centers[i % 8][j] )
```

`_u01` is the splitmix64 mixer transcribed from `bench/bench_main.mojo`;
`bench/bench_sklearn.py::u01` is its proven vectorized numpy twin and the
Python arm **imports it by path** rather than re-spelling it. The row-indexed
variant `u01_at` is asserted against that import on every run.

* The separation is **8.0, a power of two**, so `center * 8.0` is exact in
  float64 and float32 and the expression gives the same bits whether or not
  either compiler contracts the multiply and the add into an FMA. FMA
  contraction is per seam and has already cost this tree one comparison.
* The whole combination happens in float64 and rounds to float32 **exactly
  once** on both sides. Two roundings in different places would be two
  datasets.
* The data is **clustered into eight blobs** because a Gaussian mixture on
  uniform noise collapses components and `gaussian_mixture_fit` raises on a
  collapsed component (DEVIATION 1723), and because k-means on a spectral
  embedding of uniform noise is not a defined problem. The same generator,
  the same salts and the same eight blobs serve every lane and every rung.
  Nothing is tuned per rung and there is no second generator to switch to.

**The checksum, and why it is preferred to a dump.** Both sides print
`datahash=<W>.<X>` per array per rung, where X is the XOR of every float32's
bit pattern and W is `sum_i bits[i] * (i + 1)` mod 2^64. The conductor prints
a `FLADDER-DATACHECK` line per array and **refuses the rung** on a mismatch.

The dump route that `classical_speed_main.mojo` uses, one hex word per line
read back by a pure-Python parser, is right for a 48-row correctness fixture
and wrong here. The top gmm rung alone is 8,000,000 floats, about 72 MB of hex
text, and a ladder of four rungs across six lanes would write and re-parse
close to a gigabyte on a laptop before a millisecond was measured. The
generator is a closed form in (row, column, salt), so both sides can just
compute it; there is no file to go stale between the two commands and no
parser to be slow. Both checksums cover the **whole** array and both
vectorize on the Python side.

It is a checksum, not a digest. Its job is to catch two implementations of one
generator drifting apart, which the position weight makes it do; a plain XOR
would pass a permutation.

The ladder itself has the same discipline. The Mojo source is the original,
the Python file keeps a copy so `--theirs-only` works with no binary, and
every race **probes the binary** for the ladder it declared and refuses to run
if the two disagree.

## Threads

scikit-learn runs **as a user would run it**. Neither file sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or any `n_jobs`,
and neither ever will, because capping the opponent's threads to make a GPU
look better is not a measurement. The arm's header prints `threads=`, `cpu_count=`,
the per-pool breakdown, and `thread_env=` naming any of those variables that
were already set in the inherited environment, so the comparison is legible
either way.

## The opponents

RAPIDS ships no GPU implementation of any of these six, so scikit-learn on the
CPU is the honest opponent for all of them and every arm is labeled
`sklearn-cpu`. Every shared parameter is set explicitly on both sides. Three
of the six carry a difference that cannot be removed from their side, and it
is printed rather than buried:

* **gp** works in float64 on their side and float32 on ours;
  `sklearn.gaussian_process` has no float32 path. The ridge is pinned to
  `2^-20` on both, spelled as a power of two because that is the same real
  number in both widths; the Mojo header prints its bits and the conductor
  checks them.
* **nystroem** and **rbfsampler** draw differently (a pinned Philox stream
  against numpy's `RandomState`), so the two fit different basis rows and
  different weights. The work is the same shape, which is what is timed; the
  outputs are not comparable and no cross-arm hash is claimed.
* **kde**, where scikit-learn's `KernelDensity` has no brute-force option and builds
  a tree. With `rtol = atol = 0` that tree cannot prune, so it computes the
  same sum by a different data structure. Their fit and score times are
  printed separately on an `FLADDER-NOTE` line so a reader can see how much of
  their number was the tree build.
* **gmm**: `max_iter` is pinned at 20 on both arms rather than
  scikit-learn's 100, so one cell is a bounded amount of work; `tol` is
  scikit-learn's default on both. The two may still stop at different
  iterations on the same data, so `note=itersN` rides on every line. If the
  counts differ, the two cells did different amounts of work.

Five of our six lanes go through host entries that **construct their own
`DeviceContext` inside the call**, so every timed call pays a context
construction and an upload. That is what a caller pays today and it is the
point of the ladder, because it is a fixed cost and should wash out as n climbs. If
a lane's ratio improves up the ladder, that fixed cost is what it was losing
to. `spectral` is the exception; it takes the driver's context.

## What the output looks like

`ratio` is **theirs divided by ours**, so above 1 means we are faster. A win
line and a loss line are the same line with different values.

A win:

```
FLADDER-RUNG lane=nystroem arm=ours n=100000 ms_median=41.2 ms_min=39.8 ms_max=44.1 samples=6
FLADDER-RUNG lane=nystroem arm=sklearn-cpu n=100000 ms_median=203.7 ms_min=198.2 ms_max=219.4 samples=6
FLADDER-VERDICT lane=nystroem n=100000 ratio=4.9442x result=WIN ours_median_ms=41.2 theirs_median_ms=203.7 ours_band_ms=39.8-44.1 theirs_band_ms=198.2-219.4 bands_overlap=no
```

A loss, same fields in the same order:

```
FLADDER-RUNG lane=nystroem arm=ours n=1000 ms_median=8.9 ms_min=8.4 ms_max=9.7 samples=6
FLADDER-RUNG lane=nystroem arm=sklearn-cpu n=1000 ms_median=2.1 ms_min=2.0 ms_max=2.4 samples=6
FLADDER-VERDICT lane=nystroem n=1000 ratio=0.2360x result=LOSS ours_median_ms=8.9 theirs_median_ms=2.1 ours_band_ms=8.4-9.7 theirs_band_ms=2.0-2.4 bands_overlap=no
```

A rung that says nothing:

```
FLADDER-VERDICT lane=gmm n=1000 ratio=1.0400x result=TIE-BANDS-OVERLAP ours_median_ms=5.0 theirs_median_ms=5.2 ours_band_ms=4.6-5.9 theirs_band_ms=4.9-5.5 bands_overlap=yes
```

A stop, which is a result:

```
FLADDER-STOP lane=gp reached=4000 skipped=8000 rule=predicted-cell detail=predicted 96000.0 ms = 12000.0 ms * (8000/4000)^3 exceeds per_cell_s=60
FLADDER-SUMMARY lane=gp rungs_declared=5 rungs_run=4 contiguous=yes wins=0 losses=4 ties=0 first_win=- stopped=predicted-cell
```

A refusal, with its arithmetic:

```
FLADDER-MEM lane=gp arm=sklearn-cpu n=8000 bytes=1152000000 cap=2147483648 formula=16*n*n+16*n*nstar=16*8000*8000+16*8000*1000 verdict=ok
```

**The numbers in these blocks are illustrations of the format. They are not
measurements and must never be quoted.**

## Exactly what the orchestrator runs, one lane at a time, cheapest first

Build once, then race one lane per command. Wait for each to finish; never run
two at once.

```sh
mkdir -p build run

# 0. build the driver ONCE. A prebuilt binary and not `mojo run`, because
#    the conductor invokes it twice a rung and `mojo run` would recompile
#    the whole import graph every time.
nice -n 19 tools/with_build_lock.sh pixi run mojo build -I . \
    -o build/classical_ladder bench/speed/classical_ladder_main.mojo

# 1. rbfsampler, the cheapest. fit does not look at X; the whole rung is
#    one n x 8 by 8 x 64 matmul and a cosine.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane rbfsampler --binary build/classical_ladder \
    | tee run/ladder.rbfsampler.log

# 2. nystroem, same shape, plus a 64 x 64 eigendecomposition per fit.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane nystroem --binary build/classical_ladder \
    | tee run/ladder.nystroem.log

# 3. kde, linear, with 1,000 fixed queries against n train rows.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane kde --binary build/classical_ladder \
    | tee run/ladder.kde.log

# 4. gmm, linear but 20 EM iterations deep, and their k-means init is real
#    work on both sides.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane gmm --binary build/classical_ladder \
    | tee run/ladder.gmm.log

# 5. spectral, quadratic. Expect the climb to stop; that is the result.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane spectral --binary build/classical_ladder \
    | tee run/ladder.spectral.log

# 6. gp, cubic in time and quadratic in memory. The most likely to stop early
#    and the most likely to make the laptop breathe. Run it last and run it
#    when nothing else is running.
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane gp --binary build/classical_ladder \
    | tee run/ladder.gp.log
```

If the build has not happened yet, the opponent alone still runs the full
declared ladder and needs no binary:

```sh
nice -n 19 python3 bench/speed/classical_ladder_arm.py \
    --lane rbfsampler --theirs-only
```

Our arm alone, the full ladder in one process, is useful for checking that
the driver runs before pairing it with an opponent. It is **not** the race;
only the conductor can interleave the arms.

```sh
MOJOLEARN_LADDER_LANE=rbfsampler nice -n 19 ./build/classical_ladder
```

## Reading the result

The thing to look for is a **crossover**, the smallest n at which
`result=WIN`, reported as `first_win=` on the `FLADDER-SUMMARY` line. A lane
whose `first_win` is `-` did not win anywhere on its declared ladder, and that
is a finding about the lane and not a reason to extend the ladder. A lane
whose ladder stopped before the top has a `stopped=` rule saying which limit
of this laptop it hit; that is a statement about the machine, and the rung it
skipped is named so a later run on a rented box can start there.

A summary line with `contiguous=no` is not a ladder. Do not read a crossover
off it.
