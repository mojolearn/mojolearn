# resample: a bit-identical GPU bootstrap, permutation test and Monte Carlo integrator

**A p-value or a confidence interval that changes when the researcher changes
machines is a defect in published work, and nobody today ships a reproducible
GPU bootstrap.** That is the design goal of this lane and it is the whole
reason it exists. Every other property below is a consequence of one decision,
stated once here and once in `checks/index_map.mojo`.

**The resample index for replicate `r` at position `i` is a pure function of
`(seed, kind, r, i)` through the counter-based Philox generator, not a draw
from a sequential stream.**

Four things follow, and each has a gate written for it.

| property | what it means | gate |
|---|---|---|
| embarrassingly parallel | no cross-thread RNG state, no atomic, no shuffle, nothing to order | the construction |
| **batch invariance** | replicate 7 computed alone is bit-identical to replicate 7 inside a batch of a million | `check_batch_invariance` |
| **prefix stability** | extending a run from 10,000 to 100,000 resamples leaves the first 10,000 replicates unchanged | `check_prefix_stability` |
| cross-vendor identity | the map is integer arithmetic; the folds are one pinned tree; there are no atomics anywhere | `check_bootstrap_vs_oracle`, `check_launch_invariance`, the card |

Stated as a design goal, not as a measurement. See Status.

## Status, honestly

**BUILT AND GATED ON ONE APPLE M4, BOTH MODES, 2026-08-25. NO SECOND VENDOR
HAS RUN THIS.**

`pixi run check-resample` is green in FAST and green under
`tools/with_identical_mode.sh`, fifteen checks in each mode. The headline
results of the IDENTICAL run, which are the ones this lane exists for:

    check_batch_invariance OK: replicate 7's 32 drawn indices and its mean
      are bit-identical in a window of 2 at offset 7, in a batch of 1000 and
      in a batch of 100000, on a full-chunk fixture (n=200) and a
      partly-filled one (n=8)
    check_prefix_stability OK: 10000 of 10000 replicates bit-identical
      between an n_resamples = 10000 run and an n_resamples = 100000 run
      from one seed
    check_bootstrap_vs_oracle OK: 6144 replicates bit-equal to the serial
      host replay across 4 fixtures x 3 statistics, 0 differ
    check_launch_invariance OK: 1024 replicates identical across fold tpb
      256/128/64, map tpb 256/128/64/32, 37 padded rows and 37 poisoned rows

Under FAST `check_bootstrap_vs_oracle` REPORTS 834 of 6144 replicates
differing, because the FAST fold goes through `block.sum` whose cross-lane
stage is the hardware's warp width. That contrast is what makes the
IDENTICAL result evidence rather than a tautology.

- **There is no performance number for this lane and none is claimed.**
  Nothing here has been timed against anything.
- **There is no cross-vendor claim.** One Apple M4, one process.
- **BCa is refused by name in BOTH modes** on the missing `ndtri`
  (DEVIATION 1699). The bias percentile, the jackknife and the acceleration
  are all built and gated; only the interval is refused.

### Three arms that could not fire, and what each one taught

**`RSAB_SKIP_REJECTION` cannot be reached by sampling at any sample size.**
Lemire rejects when the low 32 bits of the product fall below `2^32 mod n`,
so the rejection rate is `(2^32 mod n) / 2^32`. At n = 200 that is
`96 / 2^32 = 2.2e-08`, and the driver takes 51,200 draws, for 0.0011
expected rejections. One expected rejection needs 44.7 million draws. Since
`2^32 mod n < n` for every n, the rate is bounded by `n / 2^32` always, so
no fixture fixes this. The lane had anticipated only the power-of-two case
(n = 8, rate exactly zero). The arm is now RECORDED with its arithmetic and
a direct probe of the rejection zone is owed.

**`RSAB_REVERSED_POSITIONS` is inert under IDENTICAL because the pin makes
it inert.** Reversing which slot holds which position leaves the drawn
multiset untouched and only reverses the summand order. A balanced halving
fold pairs slot `i` with slot `i + half`, and under reversal that pair is
the SAME unordered pair the unreversed fold formed at `j = n-1-i-half`.
Float addition is commutative to the bit even though it is not associative,
so every pair sum is unchanged and the argument repeats up the tree. **The
pinned fold is reversal-invariant by construction.** The arm is reached, and
the proof is the other mode: under FAST the same call moves 179 of 256,
because `block.sum`'s cross-lane stage is not reversal-symmetric.

**`RSAB_FOLD_DESCENDING` is inert at n = 200 and moves at n = 600**, because
200 is one chunk and 600 is three. This one the lane predicted correctly.

### One defect the gate found in itself

`check_launch_invariance` used to vary `tpb` and `map_tpb` at the same time
as the data length, and then raise IN BOTH MODES with a message blaming a
read past `n`. Two variables moved and one was accused. The padding and
poison arms now hold both widths fixed, so that arm is a statement about the
data length and nothing else, and a separate map-width arm asserts in both
modes because the index map is a per-position pure function with no fold in
it.

## There is no upstream for this, and it matters

**cuML does not ship a bootstrap, a permutation test or a Monte Carlo
integrator.** Neither does cuVS, and neither does RAFT. So
`PORTING_RULES.md`'s charter, **COPY, DO NOT IMPROVE**, does not apply to this
lane, and a reader must not assume this directory is a transliteration of
someone else's file. It is not. `DERIVATION_MAP.tsv` says "no upstream" on almost
every row and says it per row rather than once at the top.

What governs instead is narrower and is worth being exact about.

- The algorithms are published and standard. `scipy.stats.bootstrap`,
  `scipy.stats.permutation_test` and `sklearn.utils.resample` define the
  SEMANTICS, and they are read as the ORACLE. Their designs are serial and
  CPU shaped, and they are never the design source.
- **Every parameter this lane accepts means exactly what SciPy's parameter of
  that name means, or is named differently.** The mapping table is below and
  it is part of the contract rather than documentation of it.
- The one thing that IS a port is the generator underneath the map.
  `core/philox.mojo` holds RAFT's `PhiloxGenerator` (cuRAND's
  `curandStatePhilox4_32_10_t`) and RAFT's Lemire range reduction, both
  transcribed line by line and held to an oracle built by compiling their own
  code. This lane spends that generator at positions of its own choosing and
  reimplements none of it.

### SciPy semantics mapping

| ours | SciPy | meaning, and any departure |
|---|---|---|
| `bootstrap_host(x, n, n_features, statistic, n_resamples, seed, method, confidence_level, alternative, q_or_prop, r_first, tpb, with_bca_diagnostics, map_tpb)` | `scipy.stats.bootstrap(data, statistic, *, n_resamples, rng, method, confidence_level, alternative, paired)` | same |
| `x` with `n_features == 2` | `paired=True` over a two-sample `data` | the resample draws a ROW, so the pairing is preserved. This is what "1-D or 2-D sample" means here |
| `n_resamples` | `n_resamples` | same. SciPy's default is 9999; this surface has no default and the caller states it |
| `seed` (UInt64) | `rng` | NOT the same object. SciPy seeds a sequential `Generator`; this seeds a counter-based key (DEVIATION 1690, 1691). The bit patterns do not correspond and no run of this lane reproduces a SciPy run |
| `method` in `percentile`, `basic`, `BCa` | `method` | same three names, same spellings. `BCa` RAISES here (DEVIATION 1699) |
| `confidence_level` | `confidence_level` | same, default 0.95 |
| `alternative` in `two-sided`, `less`, `greater` | `alternative` | same three spellings, hyphen and all. `two_sided` and `twosided` are refused, because they are not SciPy's spellings either |
| `r_first` | (none; SciPy's nearest is passing `bootstrap_result` back in) | the batch handle. SciPy's extension CONTINUES a stream; `r_first` names a POSITION, which is why prefix stability is a property here and a coincidence there |
| `batch` | `batch` | **NOT PORTED and not needed.** SciPy's `batch` is a memory knob whose value changes the answer, because it changes how many stream draws precede a replicate. Here the answer is independent of it by construction, so there is nothing to expose |
| `vectorized` | `vectorized` | **NOT PORTED.** Statistics are comptime arms, not callables, so there is nothing to vectorize |
| `standard_error` (returned) | `standard_error` | `np.std(theta_hat_b, ddof=1)`, same convention (DEVIATION 1697) |
| `permutation_test_host(x, y, statistic, n_resamples, seed, alternative, r_first, tpb)` | `scipy.stats.permutation_test((x, y), statistic, permutation_type='independent', n_resamples, rng, alternative)` | same, for the two-sample independent case only |
| `permutation_type` | `permutation_type` | fixed at `'independent'`. `'samples'` and `'pairings'` are NOT PORTED and are refused by name |
| the p-value | `pvalue` | SciPy's expression exactly, `gamma` tolerance and `+1` adjustment included (DEVIATION 1702) |
| `statistic='mean'`, `'std'`, `'quantile'`, `'pearson'`, `'diff_means'`, `'trimmed_mean'` | `np.mean`, `np.std(ddof=1)`, `np.quantile(method='linear')`, `scipy.stats.pearsonr(...).statistic`, a difference of means, `scipy.stats.trim_mean` | same semantics; `median` is accepted as a name for `quantile` at `q = 0.5` |
| `q_or_prop` | `q` for `np.quantile`, `proportiontocut` for `trim_mean` | ONE parameter serving two arms, and therefore NAMED DIFFERENTLY rather than called `q`. It means whichever the selected statistic means |
| `monte_carlo_integrate_host[f_id](lower, upper, n_samples, seed, ...)` | **none** | `scipy.stats.monte_carlo_test` is a HYPOTHESIS TEST, not an integrator, and `scipy.integrate` has no Monte Carlo rule. The name says what it does and does not borrow theirs |

`sklearn.utils.resample`'s `replace=True` with `n_samples=None` is the single
step this lane's bootstrap repeats. Its `stratify` and `sample_weight`
arguments are NOT PORTED (`NOT_IMPLEMENTED.tsv`).

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Everything in this section was grepped for before a line of this lane was
written. **If a construction below ever gets a second copy inside
`resample/`, that copy is the bug.**

| reused | from | what it does here |
|---|---|---|
| Philox-4x32-10, `PhiloxState`, `skipahead_sequence`, `skipahead`, `next_u32`, `philox_next_u64` | `core/philox.mojo` | the generator, whole. This lane calls `PhiloxState.init(key, subsequence, offset)` and nothing else |
| `custom_next_uniform_int_u32` (RAFT's Lemire bounded draw with its rejection loop) | `core/philox.mojo` | the range reduction for every drawn observation index. Not `x % n`, which is a different and wrong distribution |
| `virtual_block_sum`, `host_tree_sum`, `host_fold_partials`, `chunk_count`, `linear_block_id`, `physical_block_count`, `canonicalize_nan`, `PINNED_SUM_W` | `metrics/checks/pinned_sum.mojo` | **every fold in this lane, without exception.** That file already solves the problem `check_launch_invariance` poses, by folding a fixed 256-wide tree that a 64-thread block and a 256-thread block fill identically |
| `segmented_sort_keys_f32`, `float_to_sortable` (CUB's `TwiddleIn`), `SORT_BLOCK` | `core/segmented_sort.mojo` | the per-replicate sort for the order statistics and the sort of the bootstrap distribution for the interval. Stable LSD radix, so ties keep replicate order, and `-0.0` sorts below `+0.0` |
| `ftz`, `identical_mul_add`, `identical_mul`, `identical_div`, `identical_sqrt`, `GLOBAL_NUMERIC_MODE` | `checks/numerics.mojo` | every float seam. No `std.math` transcendental appears on any path reaching an output |
| `IdentityTrace`, `first_divergence`, `FNV_OFFSET`, `FNV_PRIME` | `core/identity_trace.mojo` | the card, and the FNV-1a64 constants the derived key is hashed with (rather than a second hash function, for that file's own stated reason) |
| `mix64` / `bits_value` (splitmix64 into a float32 by bitcast) | `kde/checks/kde_fixture.mojo` | the hashed fixtures, so no host floating-point operation builds an input |
| `log_launch` | `core/launch_log.mojo` | reached through `segmented_sort_keys_f32`; this lane adds no launch-log site of its own yet (see WHAT IS OWED) |

Deliberately NOT reused, with the reason stated so it is a decision and not an
oversight.

- **`core/pinned_reduce.mojo::pinned_block_sum`.** It folds `block_size`
  values and therefore makes the fold a function of the launch, which is
  exactly what `check_launch_invariance` forbids here.
  `metrics/checks/pinned_sum.mojo` exists because of that difference and
  its header says so; this lane takes the one that fits.
- **`core/block_reduce.mojo` and `core/block_scan.mojo`.** Nothing in this
  lane reduces or scans outside the pinned tree and the segmented sort, both
  of which already own their primitives.
- **`checks/fixed_point.mojo`.** Fixed-point accumulation is the REPLACE
  move for a float atomic (IDENTITY_PATHS rows 1 and 2). **This lane has no
  atomic to replace.** Every fold is a block-local tree writing one scalar
  per replicate, so there is no contended accumulator anywhere and a
  fixed-point one would buy nothing and cost range.
- **`core/gemm.mojo`, `core/expand_distances.mojo`, `core/row_norms.mojo`.**
  No matrix product occurs in this lane.

### The word "bootstrap" already means something else in this repository

`grep` finds "bootstrap" 133 times in this tree and **every one of them is
CatBoost's row WEIGHTING** inside `gbdt/`. `gbdt/gpu_util/kernel/bootstrap.mojo`
is the port of `BayesianBootstrapImpl`, `UniformBootstrapImpl` and
`PoissonBootstrapImpl` (`catboost/cuda/cuda_util/kernel/bootstrap.cu`), driven
by `gbdt/gpu_data/bootstrap.h`'s dispatch, and it was read before this lane
was opened.

**The difference, in one sentence.** CatBoost's bootstrap multiplies each
training row's gradient and weight by a random draw so that every tree sees a
differently weighted copy of the SAME rows, and its output is a weight vector
consumed inside one model fit; this lane's bootstrap draws WHICH ROWS a
replicate contains, with replacement, and its output is a distribution of a
statistic used to put an interval around an estimate.

They share a name, a generator family and nothing else. In particular
CatBoost's arm is REGULARIZATION and this lane's is INFERENCE, and neither one
can stand in for the other.

## The identity table

In `IDENTITY_PATHS.md`'s shape. **Every status cell says DESIGNED, because
nothing has run.**

| n | pathway | order-dependent in the obvious spelling? | what this lane does | status |
|---|---|---|---|---|
| R1 | **the resample index map** | SciPy's is a sequential `Generator` stream, so replicate `r` depends on how many draws preceded it, which depends on `n_resamples`, on `batch`, and on whether the run was extended | **REPLACE** with a counter-based position map, DEVIATION 1690. `subsequence = (r << 32) \| i`, injective, into the HIGH half of the Philox counter; the Lemire rejection loop walks the LOW half of that position's own stream | DESIGNED. Gates written: `check_index_map_is_positional`, `check_batch_invariance`, `check_prefix_stability`. Sabotage `RSAB_BATCH_DEPENDENT` targets it |
| R2 | **the derived key** | one seed feeding three different uses would correlate them | **PIN** to FNV-1a64 over `(seed, kind)`, DEVIATION 1691, with `core/identity_trace.mojo`'s constants. Host only, integer only | DESIGNED. Recorded as `resample.key`, which is pure integer arithmetic and cannot differ between two runs given the same seed |
| R3 | **the per-replicate statistic fold** | any block reduction whose cross-lane stage runs at the hardware warp width (32 on Apple and NVIDIA, 64 on AMD CDNA) | **PIN** to `metrics/checks/pinned_sum.mojo::virtual_block_sum`, a 256-wide halving tree filled identically by any admitted block width. `PINNED_SUM_W` is NUMERIC; threads per block and grid shape are SCHEDULING | DESIGNED. `check_launch_invariance` (fold tpb 256/128/64), `check_bootstrap_vs_oracle` |
| R4 | **the chunk chain** | folding `ceil(n/256)` chunk totals in an arrival order | **PIN** to ascending, through `ftz`, executed by thread 0 in the kernel and by `host_fold_partials` on the host | DESIGNED. Sabotage `RSAB_FOLD_DESCENDING` |
| R5 | **float atomics** | the usual REPLACE row | **NONE EXIST.** No `Atomic.fetch_add`, no `Atomic.max`, no `Atomic.min` on any float anywhere in `resample/`. Each replicate writes one scalar from thread 0 | DESIGNED, and it is a grep rather than an argument |
| R6 | **warp and lane primitives** | `warp.sum`, `warp.shuffle_*`, `lane_id()` on a numeric path | **NONE EXIST** in `resample/`. The block broadcast goes through one threadgroup float, not `warp.broadcast` | DESIGNED, also a grep |
| R7 | **transcendentals** | `std.math.sqrt` is APPROXIMATE on NVIDIA (176,577 of 2^20 normals off by one ulp, IDENTITY_PATHS row 10) | **PIN** to `identical_sqrt` and `identical_div` at every seam; no `exp`, `log`, `pow` or trig occurs in this lane at all | DESIGNED. Sabotage `RSAB_STD_SQRT`, expected inert on Apple and expected to bite on NVIDIA |
| R8 | **the contraction seam** | `u * span + lo` in the affine uniform, and `frac * (a_hi - a_lo) + a_lo` in the interpolation, are one rounding or two at the codegen's whim | **PIN** to `identical_mul_add`, IDENTITY_PATHS row 9 | DESIGNED |
| R9 | **denormals** | Metal flushes, CUDA honors | **PIN** with `ftz` at every stored seam, IDENTITY_PATHS row 10's checklist | DESIGNED. Expected bit-inert on Apple, as everywhere else |
| R10 | **the sort's tie handling and its zeros** | an unstable sort leaves equal keys in an implementation-chosen order, and `-0.0`/`+0.0` compare equal as floats | **PIN** to the total order `(float_to_sortable(theta_r), r)`. CUB's `TwiddleIn` makes `-0.0` (key `0x7FFFFFFF`) sort strictly below `+0.0` (key `0x80000000`), and the LSD radix is stable, so bitwise-equal values keep ascending replicate order | DESIGNED, and **HONESTLY BOUNDED**: the replicate-index half is NOT OBSERVABLE in this lane's output, because the interval reads VALUES at ranks and two bitwise-equal values are the same bits either way. `check_percentile_interval` ASSERTS the zero ordering on a planted distribution and REPORTS the stability |
| R11 | **the order-statistic interpolation** | inheriting a library's default `method`, which is free to change between versions | **PIN** to Hyndman-Fan type 7 spelled out, DEVIATION 1698, with the `frac = +0.0` knife edge gated on both sides | DESIGNED. `check_percentile_interval` |
| R12 | **the permutation** | `numpy.random.Generator.permutation` is Fisher-Yates, so position `i` depends on every swap before it and the number of stream words consumed depends on `n_obs` | **REPLACE** with a rank over the total order `(philox_u64(key, r, j), j)`. 64-bit keys so ties are ~3e-14, index tie-break so the order is total and the rank is a bijection. Integer compares only | DESIGNED. `check_permutation_separable` asserts the bijection; sabotage `RSAB_PERM_KEY_ONLY` narrows the key to 4 bits and drops the tie-break |
| R13 | **BCa's interval endpoints** | needs `ndtri`, the inverse normal CDF, which has no portable construction in `checks/numerics.mojo` | **REFUSE**, by name, in BOTH modes, DEVIATION 1699. Everything BCa needs besides `ndtri` is built, identical and recorded (`resample.bca.z0p`, `resample.jackknife`, `resample.bca.ahat`) | REFUSED. `check_resample_refusals` asserts the raise; `check_jackknife_and_bca` gates the half that works |
| R14 | **NaN payloads** | a computed NaN carries the vendor's payload (Apple `0x7fc00000`, NVIDIA `0x7fffffff`, AMD `0xffc00000`), IDENTITY_PATHS row 39 FACT 2 | **PIN** to `canonicalize_nan` at every stored statistic; non-finite INPUT is refused by name before any launch | DESIGNED. Sabotage `RSAB_NO_CANON_NAN` shows the payload reaching `resample.theta` |
| R15 | **signed zero in the folds** | `x + (+0.0) == x` for everything except `x = -0.0` | **STATED AND PROVEN, not pinned**, because there is nothing to pin. Every fold is `+0.0` seeded and `+0.0` padded on every vendor, `x - x` is `+0.0`, and `fma(frac, +0.0, -0.0)` is `+0.0`, so no `-0.0` can reach a recorded stage through a fold statistic. The property that `-0.0` really is ORDERED below `+0.0` is checked by PLANTING a distribution | DESIGNED. `check_percentile_interval`'s signed-zero half |
| R16 | **float64 on the device** | none | **NONE.** Float64 appears only in `resample_oracle.mojo::reference_*_f64`, which runs on the host and is a tolerance reference | DESIGNED, a grep |

## The deviations

Range 1690 to 1719 is this lane's. Thirteen are spent.

| # | what | where |
|---|---|---|
| **1690** | **THE RESAMPLE INDEX IS A POSITION, NOT A STREAM DRAW.** SciPy walks a sequential `Generator`; this lane gives position `(r, i)` its own Philox subsequence. Required rather than preferred, because there is no partition of a sequential stream that is simultaneously parallel, machine-independent and independent of the batch size. Price, paid on every drawn index: two Philox block evaluations per draw where a stream amortises one over four, about 8x the RNG arithmetic | `checks/index_map.mojo` |
| 1691 | The derived key is FNV-1a64 over `(seed, kind)`, not the seed, so the bootstrap draw, the permutation key and the Monte Carlo coordinate cannot share counter positions. cuML derives its per-tree RF seed the same way and for the same reason | `checks/index_map.mojo` |
| 1692 | RAFT's launch geometry is NOT used. `launch_uniform_int`'s `RNG_STRIDE = 110592` mapping is a stream partition and is the thing 1690 gives up; `RNG_STRIDE` appears nowhere in `resample/`. Consequence, stated so nobody looks for it: a `resample/` bootstrap sample does not equal an `ensemble/` Random Forest bootstrap sample from the same seed | `checks/index_map.mojo` |
| 1693 | The uniform float is RAFT's `next_float` (`val = next_u32() >> 8; val / 2^24`), and its division is EXACT on every vendor, so it needs neither `identical_div` nor `ftz`. The affine map onto the box is a different matter and IS pinned | `checks/index_map.mojo` |
| 1694 | The drawn values are RECOMPUTED per pass, not materialised. Storing them costs 40 GB at a million resamples of a thousand observations; the map is a pure function, so the second pass provably sees the same values. The two order statistics are the exception and they materialise, under `RESAMPLE_MAX_SORT_CELLS` | `checks/statistics.mojo` |
| 1695 | `pearson`'s denominator is `sqrt(sxx * syy)`, not `sqrt(sxx) * sqrt(syy)`. One rounding fewer, and stated because a reader diffing against `np.corrcoef` will see the last bit move | `checks/statistics.mojo` |
| 1696 | A degenerate `pearson` returns the CANONICAL NaN, not the vendor's. Reachable with real probability at small `n` | `checks/statistics.mojo` |
| 1697 | `std` and the returned `standard_error` are ddof = 1, matching `scipy.stats.bootstrap`'s `correction=1`. `ddof = 0` is not silently available | `checks/statistics.mojo` |
| 1698 | The interpolation between order statistics is Hyndman-Fan type 7 SPELLED OUT, not inherited. Three reasons, the first of which is that the last line is a multiply-add and therefore a contraction seam | `checks/statistics.mojo` |
| **1699** | **BCa IS REFUSED BY NAME, IN BOTH MODES.** The blocker is `ndtri`, the inverse normal CDF, which `checks/numerics.mojo` does not have. The bias percentile, the jackknife and the acceleration ARE identical and are computed, recorded and gated. Closure condition named in the error text | `checks/intervals.mojo` |
| 1700 | The jackknife's fold keeps the full `n`-slot layout with `+0.0` in the left-out slot rather than compacting to `n - 1`, so `theta_hat_i`'s tree is the same tree as `theta_hat`'s for every `i` | `checks/intervals.mojo` |
| 1701 | `alternative` narrows the interval exactly as SciPy does, infinities and order included | `checks/intervals.mojo` |
| 1702 | The permutation p-value carries SciPy's `gamma` tolerance and its `+1` adjustment, computed from OUR dtype. Consequence, and it is the easy mistake here: the two-sided floor is `2/(n_resamples+1)`, NOT `1/(n_resamples+1)` | `checks/intervals.mojo` |

1703 to 1719 are unspent and are this lane's to use.

## WHAT THE ORCHESTRATOR MUST WIRE

Nothing outside `resample/` was touched. These are the exact lines.

### 1. `pixi.toml`, in the `[tasks]` block beside `check-kde` and `check-linkage`

```toml
check-resample = "mojo run -I . resample/checks/resample_check.mojo"
resample-card = "mojo run -I . resample/resample_main.mojo"
```

FAST is `pixi run check-resample`; IDENTICAL is
`tools/with_identical_mode.sh pixi run check-resample`, exactly as for every
other gate in this tree. No `*-identity` task is wanted.

Both need the GPU, so use the lock.

```
tools/with_build_lock.sh     pixi run mojo run -I . resample/checks/resample_check.mojo
tools/with_identical_mode.sh pixi run mojo run -I . resample/checks/resample_check.mojo

MOJOLEARN_IDENTITY_TRACE=/tmp/mac.resample.identical.card \
    tools/with_identical_mode.sh pixi run mojo run -I . resample/resample_main.mojo
```

### 2. The check writes two card files

`check_card_is_emitted` writes `resample_card_a.trace` and
`resample_card_b.trace` into `$MOJOLEARN_RESAMPLE_CARD_DIR`, defaulting to
`/tmp`. Set it if `/tmp` is not wanted.

### 3. `IDENTITY_PATHS.md` needs a row

The identity table above is written in that file's shape and rows R1 to R16
collapse into one ledger row. Suggested text, for the orchestrator to place
and number.

> **`resample/` -- bootstrap, permutation test and Monte Carlo integration end
> to end.** No cuML, cuVS or RAFT upstream exists; SciPy defines the semantics
> and is the oracle. Their spelling is a sequential `numpy.random.Generator`
> whose replicate `r` depends on how many draws preceded it, a Fisher-Yates
> shuffle that cannot be evaluated at a position, and `numpy` folds free to
> choose any summation order. Ours is DEVIATION 1690's counter-based position
> map (`(r << 32) | i` into the Philox counter's high half, RAFT's Lemire
> reduction unchanged), every fold `metrics/checks/pinned_sum.mojo`'s
> 256-wide tree, every sort `core/segmented_sort.mojo`'s stable radix over
> CUB's `TwiddleIn` key, no float atomic and no warp primitive anywhere, and
> BCa REFUSED on `ndtri` (DEVIATION 1699). **CONSTRUCTION ONLY, 2026-08-25 --
> nothing has been built or run on any vendor.**

### 4. The Python surface, when `bindings/` is wired

Not this lane's directory, so it is named rather than done.

- `mojolearn.stats.bootstrap(x, statistic='mean', n_resamples=9999, seed=0,
  method='percentile', confidence_level=0.95, alternative='two-sided',
  q=0.5)` calling `resample/estimator.mojo::bootstrap_host`, returning a
  `BootstrapResult` with `confidence_interval`, `bootstrap_distribution` and
  `standard_error` so the attribute names match SciPy's.
- `mojolearn.stats.permutation_test(x, y, statistic='diff_means',
  n_resamples=9999, seed=0, alternative='two-sided')` calling
  `permutation_test_host`.
- `mojolearn.stats.monte_carlo_integrate(f, lower, upper, n_samples, seed)`
  where `f` is one of the string names `mc_integrand_from_name` accepts, NOT a
  Python callable. Refuse a callable by name and say why.
- Refuse in Python, by name: `method='BCa'`, `permutation_type` other than
  `'independent'`, `batch`, `vectorized`, `stratify`, `sample_weight`, and
  float64 inputs.

### 5. `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv`

Both exist in this directory and are ready to be concatenated into the
repository-level files if that is how the orchestrator wants them.

## SABOTAGES TO PERFORM

All nine arms are written, comptime-selectable, and driven by
`check_resample_sabotages`. **The result column is empty because nothing has
been run.** The pattern is `hierarchy/checks/sabotage_tile.mojo`'s and the
driver is `hierarchy/checks/linkage_check.mojo`'s.

| arm | what it breaks | fixture | expected | result |
|---|---|---|---|---|
| `RSAB_BATCH_DEPENDENT` | **the headline.** The index map's subsequence is xored with `n_replicates`, so the map becomes a function of the batch size | `FIX_HASHED`, n = 200 | `check_batch_invariance` and `check_prefix_stability` MUST FAIL; every other gate stays green, because any single run is still self-consistent | (not run) |
| `RSAB_MAP_IGNORES_POSITION` | the position is dropped from the subsequence, so every position of a replicate draws the same row | `FIX_HASHED` | `check_index_map_is_positional`'s separation clause MUST FAIL | (not run) |
| `RSAB_LOW_HALF_SUBSEQUENCE` | the subsequence goes into the LOW counter half, the trap `core/philox.mojo::_incr_hi` names by hand | `FIX_HASHED` | `check_bootstrap_vs_oracle` MUST FAIL | (not run) |
| `RSAB_SKIP_REJECTION` | Lemire's rejection loop dropped | `FIX_HASHED` (n = 200) **and** `FIX_ANALYTIC` (n = 8) | MUST FAIL at n = 200; **EXPECTED INERT at n = 8**, because `2^32 mod 8 == 0` and the loop is unreachable there. The per-branch demonstration | (not run) |
| `RSAB_REVERSED_POSITIONS` | slot `s` of the fold holds position `n - 1 - s`. Same multiset, different tree pairing | `FIX_HASHED` | `check_bootstrap_vs_oracle` MUST FAIL | (not run) |
| `RSAB_FOLD_DESCENDING` | the chunk chain folded the other way | a 600-row hashed sample (three chunks) **and** `FIX_HASHED` (n = 200, one chunk) | MUST FAIL at n = 600; **EXPECTED INERT at n = 200**, because reversing a one-element chain is the identity. The second per-branch demonstration | (not run) |
| `RSAB_STD_SQRT` | `std.math.sqrt` instead of `identical_sqrt` at the `std` and `pearson` seams | `FIX_HASHED` | **REPORT.** Apple's `sqrt` is correctly rounded, so 0 cells are expected to move there; NVIDIA's lowers to an approximate PTX sqrt, 176,577 of 2^20 normals off by one ulp (IDENTITY_PATHS row 10), so it is expected to bite on the second vendor | (not run) |
| `RSAB_NO_CANON_NAN` | `canonicalize_nan` dropped on the degenerate `pearson` path | `FIX_ANALYTIC` | **RECORDED.** Prints the vendor's NaN payload reaching `resample.theta` | (not run) |
| `RSAB_PERM_KEY_ONLY` | the permutation key narrowed to 4 bits AND the index tie-break dropped | `FIX_SEPARABLE` | `check_permutation_separable`'s bijection assertion MUST FAIL. Two changes in one arm on purpose, because at 64 bits the tie-break is unreachable and an unreachable branch cannot be sabotaged | (not run) |

Two sabotages considered and NOT written, with the reason, so nobody adds them
thinking they were forgotten.

- **A rotation of the slab fill inside a chunk.** Measured INERT by the metrics
  lane (`metrics/checks/pinned_sum.mojo::sabotage_shifted_host_tree_sum`)
  for every rotation amount, because the halving tree's subtrees at level `k`
  are the residue classes mod `2^k` and a rotation maps residue classes onto
  residue classes. `RSAB_REVERSED_POSITIONS` is the sabotage of that tree that
  actually bites.
- **Dropping `ftz`.** Bitwise inert on Apple by construction (Metal flushes in
  hardware) and this lane has no denormal-reachable seam identified yet. It
  belongs on the NVIDIA leg, where it is not inert, and it is listed under
  WHAT IS OWED rather than written as an arm that is guaranteed to print zero.

## The fixtures, and the job each one alone does

| fixture | what it is | why it exists |
|---|---|---|
| `FIX_ANALYTIC` | column 0 is `1..8`, column 1 is `9 - column 0` | **three hand-derived facts about EVERY replicate.** Each mean is an exact multiple of 0.125 in `[1, 8]`; each `pearson` is exactly `-1.0` or the canonical NaN and no third value is legal; each `diff_means` is an exact multiple of 0.25. The exactness is argued in the fixture's docstring, bound by bound |
| `FIX_SEPARABLE` | 16 values at `+100 + i`, 16 at `-100 - i` | the permutation p-value must sit on its floor. `C(32,16) = 601,080,390` against 999 draws, so the accidental collision has probability 1.7e-6 and the gate cannot flap |
| `FIX_SAME` | 32 values from one hashed population, split 16/16 | the null is TRUE by construction, so the p-value should be roughly uniform. Reported over 16 seeds, never asserted |
| `FIX_DUPES` | twelve rows over three distinct values | exact duplicates, so the with-replacement TIE path is reached: runs of equal keys in the sort, and interpolation between two equal order statistics |
| `FIX_SIGNED_ZERO` | `-0.0`, `+0.0`, `+-1`, `+-2`, written by bits | a sample carrying signed zeros, run end to end. Plus `planted_signed_zero_distribution`, which puts both zeros into a DISTRIBUTION, because the folds provably cannot |
| `FIX_HASHED` | 200 rows, two columns, splitmix64 into float32 by bitcast | a general sample nothing about which is special |
| the Monte Carlo boxes | the unit square and `[-2,2] x [0.5,2.5]` | `const` integrates to the volume EXACTLY (bit for bit, for any draws), `sum` to `V*(mid0+mid1)`, `product` to `V*mid0*mid1`. All three closed forms are derived by hand, not by quadrature |

## The card

Eleven stages when the BCa diagnostics ride along, eight otherwise.

```
resample.key         u32 x2   the derived key
resample.index_map   i32      4 replicates x 32 positions of the map
resample.theta       f32      the bootstrap distribution, replicate order
resample.sorted      f32      the same values, sorted
resample.point       f32 x1   theta_hat
resample.order_pos   i32 x2   the two order-statistic positions
resample.se          f32 x1   the standard error
resample.interval    f32 x2   the two endpoints
resample.bca.z0p     f32 x1   BCa's bias percentile
resample.jackknife   f32      the n leave-one-out statistics
resample.bca.ahat    f32 x1   BCa's acceleration
```

The permutation entry writes `resample.key`, `resample.null`,
`resample.observed` and `resample.pvalue`; the Monte Carlo entry writes
`resample.key`, `resample.mc.points`, `resample.mc.partials`,
`resample.mc.mean` and `resample.mc.integral`.

A cross-vendor diff has an ADDRESS and each address has a cause.
`resample.key` and `resample.index_map` are pure integer arithmetic and CANNOT
differ; if they do, either the two runs were not given the same seed or the
Philox port is broken, and `ensemble/bench/philox_oracle.txt` is where that
goes. `resample.theta` is the first float stage. `resample.sorted` differing
while `resample.theta` matches is a sort defect and nothing else.

## WHAT IS OWED

1. **A FIRST BUILD.** Nothing here has been compiled. The most likely places
   for a compile error are the comptime kernel bindings in
   `estimator.mojo::_launch_*_at` (the `comptime kern = f[a, b]` form), the
   `comptime for` lane loops writing SIMD lanes inside `_chunked_sum`, and the
   `stack_allocation` of `Scalar[DType.uint64]` in threadgroup memory in
   `perm_stat_kernel`.
2. **A FIRST GATE RUN, in both modes,** and the sabotage table's result column
   filled in from it. Until then every claim in this file is a design claim.
3. **THE SECOND VENDOR (NVIDIA H100).** Two things are expected to move there
   and are worth watching for by name. `RSAB_STD_SQRT` should stop being inert
   (row 10's approximate PTX sqrt). `ftz` should stop being bitwise inert
   (CUDA honors denormals), which is when a dropped-`ftz` sabotage becomes
   worth writing. The IDENTICAL card must be bit-identical to Apple's, stage
   for stage.
4. **THE THIRD VENDOR (AMD MI325X).** The wavefront is 64 wide, which is
   exactly what `virtual_block_sum` exists to be independent of, so this leg is
   the one that tests the reuse decision rather than the arithmetic. Also
   check that no kernel here trips the block-primitive width constraint that
   broke PCA's build on that box (`core/pinned_reduce.mojo`'s block comment) --
   this lane calls no `max.gpu.primitives.block` primitive except through
   `virtual_block_sum`'s FAST arm and `core/segmented_sort.mojo`'s
   `prefix_sum`, and the latter runs at `SORT_BLOCK = 512`, above every warp
   width.
5. **DEVIATION 1699's closure.** `portable_ndtri` / `identical_ndtri` in
   `checks/numerics.mojo`, gated the way `check-division` gates
   `portable_divf`, then BCa ships.
6. **A launch-log site per kernel.** `core/launch_log.mojo::log_launch` is
   reached only through `segmented_sort_keys_f32` today, so a Metal trace
   cannot name this lane's own kernels. One line per `enqueue_function`.
7. **Batching the sort path.** `RESAMPLE_MAX_SORT_CELLS` refuses a large
   `quantile` or `trimmed_mean` bootstrap rather than tiling it. The refusal
   names `r_first` as the workaround and tiling as the closure.
8. **`permutation_type='samples'` and `'pairings'`,** and `stratify` /
   `sample_weight` on the resample. All four are in `NOT_IMPLEMENTED.tsv` with their
   reasons.
9. **A SciPy cross-check as a REPORT.** Running `scipy.stats.bootstrap` on the
   same fixture cannot reproduce our numbers (different generator, DEVIATION
   1690) but SHOULD land within Monte Carlo error of them, and a tool that
   says so would be the only comparison in this lane whose other side is not
   our own code. `tools/mamba_corpus_check.py` is the template.
