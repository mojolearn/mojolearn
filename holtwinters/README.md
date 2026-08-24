# holtwinters: cuML's Holt-Winters exponential smoothing, ported for bitwise identity

DEVIATIONS 660-665 and 697-698. COPY, DO NOT IMPROVE.

## Status: Apple M4 only, both modes green, and PART OF IT IS UNVERIFIED

Read this before trusting anything below.

* The gate `mojo_only/hw_check.mojo` was built and run on one Apple M4 on
  2026-08-23 and printed `ALL OK` under IDENTICAL and under FAST.
* **No second vendor has run this.** Every cross-vendor claim here is a
  claim about what the spelling is designed to guarantee, not a
  measurement. NVIDIA and AMD re-prints are OWED.
* **Every arm and both modes have now been built and run against the
  current tree** (2026-08-23, serially, one compile at a time): the clean
  IDENTICAL gate, the clean FAST gate, and all nine sabotage arms. The
  earlier "uncompiled since the green run" caveat is discharged. Seven
  arms fail as designed; two are Apple-null and are counted as REACH
  FAILURES, not passes.

## UPSTREAM IS DEPRECATING THIS ALGORITHM IN 26.12

`holtwinters.pyx` in the pinned tree carries, and the 25.08 tree does not:

    .. deprecated:: 26.08
        ``cuml.tsa.ExponentialSmoothing`` and ``cuml.ExponentialSmoothing``
        are deprecated and will be removed in the cuML 26.12 release.

and `__init__` calls `warn_deprecated_tsa_api("cuml.tsa.ExponentialSmoothing")`.
So the thing this lane mirrors is on its way out of cuML. That does not
change the port -- v26.08.00 is the pinned target and this is a faithful
mirror of it -- but it does mean there will be no upstream to re-sync
against after 26.12, and any future "check us against a real cuML run"
has a deadline. Flagged for the orchestrator, not decided here.

## Where the upstream is, and the trap

    /Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00   <- THIS ONE, 265b9da6, v26.08.00
    /Users/andrewhendel/CascadeProjects/upstream/cuml             <- NOT THIS ONE, 00094f7e, branch-25.08

Both contain a complete `cpp/src/holtwinters`. The wrong one is a whole
RELEASE behind (VERSION 25.08.00), not a near-copy. They differ: 25.08 has
the 13-line Apache header where the pinned tree has a 2-line SPDX one,
which shifts every line number by about eleven; it lacks the
`checked_arithmetic.hpp` guards and the `RAFT_FAIL` in
`stl_decomposition_gpu`; its `holtwinters.pyx` has no `check_array` call
at all (so DEVIATION 664's premise does not even exist there); and it has
a `holtwinters_api.cpp` that v26.08.00 does not. An audit of this lane was
run against 25.08 once and produced two findings that were false (it
flagged our `RAFT_FAIL` port as invented code, and missed that our
`trend_len` parenthesization mirrors the pinned `checked_sub`).

**Provenance of every finding in this file.** After catching the mistake,
all six `.cuh`/`runner.cuh` files, the params header, `holtwinters.cu` and
`holtwinters.pyx` were diffed BETWEEN the two trees, so the exact regions
where 25.08 could have misled a reader are known rather than assumed:

| file | 25.08 vs pinned | consequence for the audit |
|---|---|---|
| `hw_utils.cuh` | license header ONLY | first-pass read valid as-is |
| `hw_eval.cuh` | license header ONLY | first-pass read valid as-is |
| `hw_forecast.cuh` | license header ONLY | first-pass read valid as-is |
| `holtwinters.cu` | license header ONLY | first-pass read valid as-is |
| `hw_optim.cuh` | header + a `checked_arithmetic` include + one `checked_mul` on the `pseason` allocation | the gradient, the BFGS body, the line search and the Hessian update are byte-identical between trees; DEVIATION 697 re-verified directly in the pinned tree at `:573-578` |
| `hw_decompose.cuh` | header + the `RAFT_FAIL` guard, `checked_sub` `trend_len`, `batch_trend_n` | only `stl_decomposition_gpu` was affected, and it was re-read in the pinned tree; the kernels were not |
| `runner.cuh` | header + `hw_narrow_size` and the int64 `HoltWintersBufferSize` rewrite | only `HoltWintersBufferSize` was affected, and it was re-read; the other seven entry points were not |
| `holtwinters_params.h` | header + `namespace CUML_EXPORT ML` and an `export.hpp` include | a visibility attribute; no semantic change to the enums or `OptimParams` |
| `holtwinters.pyx` | substantial: the deprecation above, `handle` removed, `CumlArrayDescriptor` -> `ReflectedAttr`, `input_to_cupy_array` -> `check_array(..., ensure_all_finite=False)` | the VALIDATION raises this lane mirrors are unchanged between trees, and the two cited lines (`:164` eps default, `:197` seasonal refusal) plus DEVIATION 664's `check_array` at `:237-241` were each verified in the pinned tree directly |

Every line span in `PORTED_MAP.tsv` and in the `.mojo` headers is against
the PINNED tree and was recomputed by brace-matching on 2026-08-23.

## What is here

    cuml/cpp/include/cuml/tsa/holtwinters_params.h -> ported/tsa/holtwinters_params.mojo
    cuml/cpp/src/holtwinters/internal/hw_utils.cuh -> ported/holtwinters/internal/hw_utils.mojo
    cuml/cpp/src/holtwinters/internal/hw_eval.cuh  -> ported/holtwinters/internal/hw_eval.mojo
    cuml/cpp/src/holtwinters/internal/hw_decompose.cuh -> .../hw_decompose.mojo
    cuml/cpp/src/holtwinters/internal/hw_forecast.cuh  -> .../hw_forecast.mojo
    cuml/cpp/src/holtwinters/internal/hw_optim.cuh     -> .../hw_optim.mojo
    cuml/cpp/src/holtwinters/runner.cuh            -> ported/holtwinters/runner.mojo
    cuml/cpp/src/holtwinters/holtwinters.cu        -> ported/holtwinters/holtwinters.mojo

`PORTED_MAP.tsv` says per file what is transliterated and what is partial.
`UNPORTED.tsv` has nineteen rows and is meant to be complete, not short.
Float32 only: cuML instantiates `float` and `double`, Metal has no
Float64.

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . holtwinters/mojo_only/hw_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . holtwinters/mojo_only/hw_check.mojo

A sabotage arm is a build define and edits nothing:

    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_HW_SABOTAGE_SWAP_FMA=1 -I . holtwinters/mojo_only/hw_check.mojo

No pixi task is registered (`pixi.toml` is not this lane's). The task line
to land is under HAND-OFF.

## Why this algorithm is its own identity problem

Holt-Winters is a strictly SEQUENTIAL recurrence over level, trend and
season, one thread per series, nothing crossing a thread. There is no
reduction fold order to freeze, which is where the gemm lane's identity
risk lived. The risk moves to three other places:

1. **Contraction.** Every update is `a * x + b * y`. C++ does not say
   which of the two products nvcc fuses into the add, so THEIR source does
   not determine an answer and three compilers will pick freely. A pin was
   required. DEVIATION 698 is that pin and the seam table below is it,
   written out.
2. **Denormals.** Any term here can go subnormal (`pts - xhat_` on a
   nearly recovered series, `1 - alpha_` at the clamp, `clevel - plevel`
   at convergence). Apple flushes, NVIDIA and AMD keep. Every stored
   intermediate goes through `ftz` under IDENTICAL.
3. **Division and the clamp.** The multiplicative arm divides by `stmp_eps`
   and by `clevel`; the optimizer divides by `2 eps` and by `rho_`. The
   clamp `bound_device` sees `-0.0` and NaN. Division is correctly rounded
   on every column measured (row 10); the clamp is DEVIATION 663's compare
   chain because a hardware `max` is not.

## The seam table: fused or unfused, per seam, with the reason

This is DEVIATION 698 in full. `fma(a, b, c)` means `identical_mul_add`
(a real fused multiply-add under IDENTICAL, the naive chain under FAST);
every result named below is stored through `ftz`.

| where | theirs | ours | fused? | why |
|---|---|---|---|---|
| eval: `leveltrend` | `plevel + ptrend` | same | n/a | no multiply to fuse |
| eval: `xhat_` additive | `plevel + ptrend` then `+= stmp` | `ftz(leveltrend + stmp)` | n/a | reuses `leveltrend`, same operands and op, so same bits |
| eval: `xhat_` multiplicative | `(plevel + ptrend) * stmp` | `ftz(leveltrend * stmp)` | n/a | one multiply, nothing to fuse into |
| eval: SSE | `error_ += diff * diff` | `fma(diff, diff, error_)` | FUSED | one multiply and one add in one expression is the canonical contraction site; nvcc contracts it by default, so fusing matches their most likely codegen AND is the cheaper pin |
| eval: level, trend, season (`_mix`) | `a * x + (1 - a) * y` | `fma(a, x, ftz((1-a) * y))` | FIRST product FUSED, second STORED | two products, one add: exactly one can fuse and C++ does not say which. Chose the FIRST (the parameter times the observation term) so the rule reads the same left to right everywhere. `SAB_SWAP_FMA` fuses the other one and FAILS the gate, which is the evidence the choice is load-bearing rather than cosmetic |
| decompose: `conv1d` | `out += filter[i] * input[...]` | `fma(filter[i], input[...], out)` | FUSED | same shape as the SSE seam |
| decompose: `batched_ls_solver` | `level_ += rq[2i] * b` | `fma(rq[2i], b, level_)` | FUSED | same |
| decompose: `season_mean` sums | `period_mean += season[...]` | `ftz(period_mean + ...)` | n/a | no multiply |
| decompose: `season_mean` divisions | `/= count`, `/= frequency`, `/= mean` | same, through `ftz` | n/a | division, correctly rounded |
| forecast | `level + trend * (i + 1) [+ or * season]` | `fma(trend, i+1, level)` then `+ season` or `* season` | FUSED (the level/trend part) | theirs parenthesizes `(level + trend*(i+1))` in the multiplicative arm, so the fusable pair is unambiguous; the additive arm's `+ season` is a separate add in both |
| optim: every 3-term dot (`_dot3`) | `a1*b1 + a2*b2 + a3*b3` | `ftz(a1*b1)`, then `fma(a2,b2,.)`, then `fma(a3,b3,.)` | first STORED, next two FUSED | ascending, so the order is a property of the source and not of a compiler. The first term has no accumulator to fuse into yet |
| optim: `nx = x + step * p` | `*x1 + step_size * p1` | `fma(step_size, p1, x1)` | FUSED | canonical site |
| optim: line-search target | `loss_ref + step_size * cauchy` | `fma(step_size, cauchy, loss_ref)` | FUSED | canonical site |
| optim: `k` | `rho * rho * (dot + rho_)` | every product stored | UNFUSED | theirs parenthesizes left to right and there is no add for the products to fuse INTO |
| optim: Hessian `k*s*s` and `2*rho*s*Hy` chains | `k * s1 * s1`, `2. * rho * s1 * Hy1` | every product stored, `two_rho` hoisted | UNFUSED | same reason: a chain of multiplies with the add outside it. Hoisting `2*rho` moves no bit because it is their first operation |
| optim: Hessian off-diagonals | `rho * (s2 * Hy1 + s1 * Hy2)` | `rho * ftz(fma(s2, Hy1, ftz(s1 * Hy2)))` | FIRST product FUSED | this IS an `a*x + b*y` seam, so it takes the same rule as `_mix`. One rule for the lane, not two |

An earlier revision of `hw_optim.mojo`'s header claimed every product in
the Hessian update was stored. It never was; the sentence was corrected
rather than the code, because the fused spelling is the rule above.

## The deviations

**DEVIATION 660** (`hw_decompose.mojo`): `batched_ls` without cuSOLVER.
`R^-1 Q^T` is `pinv([1, t])`, a function of `trend_len` alone and of no
series, reached in theirs through two closed libraries. Ours writes it on
the host in float64 from the closed form and casts. Their data-touching
solver kernel is ported unchanged. Priced: our `R1Qt` is not their bits in
the last places.

**DEVIATION 661** (`runner.mojo`): the card records one NaN payload
(`0x7FC00000`) through a copy; the live buffers are untouched. A computed
NaN's payload is the vendor's (Apple `0x7fc00000`, NVIDIA `0x7fffffff`,
AMD `0xffc00000`), so a raw hash of a stage holding one is a fingerprint,
not an identity.

**DEVIATION 662** (`hw_optim.mojo`): a zero search direction returns
`OPTIM_MIN_GRAD_NORM` instead of stepping by `0.866 / sqrt(0)`. Their
unguarded step makes `nx = x + inf * 0 = NaN`, and the fit then REPORTS
`(0, 0, 0)` for a series that was already at a stationary point by their
own criterion. This is cuml#888 in its simplest form.

**DEVIATION 663** (`hw_utils.mojo`): `bound_device` is a compare chain,
not `fminf(fmaxf(val, 0), 1)`. `-0.0 -> +0.0` and `NaN -> +0.0` on every
vendor, which is NVIDIA's answer on both hazards of IDENTITY_PATHS row 39.

**DEVIATION 664** (`runner.mojo`): non-finite input, and non-positive
input under MULTIPLICATIVE, refused by name with the series, the position
and the value. Theirs passes `ensure_all_finite=False`. SSE overflow is
deliberately not guarded (`+inf` is the same bits everywhere).

**DEVIATION 665** (`hw_optim.mojo`): the optimizer writes its
per-iteration parameters, its iteration count and its criterion, so a
cross-vendor difference in a fitted parameter has an ITERATION as its
address instead of only a verdict. No arithmetic added or moved.

**DEVIATION 697** (`hw_optim.mojo`): the Hessian diagonal in float32.
`hw_optim.cuh:573-578` reads `2.` in the H11 and H33 updates and `2` in
H22. On the float instantiation `2.` is a DOUBLE literal, so H11's and
H33's whole subtrahend is evaluated in float64 and the `+=` rounds back,
while H22's identical expression is float32 throughout. Three diagonal
entries of one symmetric matrix, two precisions, decided by a typed
period. Ours does all three in H22's float32 spelling, because (a) Metal
has no float64 so their arm cannot be mirrored on one of the three
vendors at all, and (b) it is a typo, and the standing rule is to fix
their bugs numbered rather than port them. Priced: H11 and H33 lose
precision that was never load-bearing, and this is the second of the two
places the lane is not cuML's numbers.

**DEVIATION 698** (`hw_eval.mojo`): the flush-and-fuse rule, above.

## Sabotage table

Every arm is a `-D MOJOLEARN_HW_SABOTAGE_<NAME>=1` build under IDENTICAL;
nothing is edited and nothing is reverted. All nine were built and run
serially on one M4 on 2026-08-23 against the current tree. **Stages and
cells are counted, not just first divergence**, because the mamba lane
measured an arm that moves 13 of 16 stages and still leaves the final
output bit-identical. Cell counts are device vs oracle on the `additive`
fixture (2800 cells) unless the row says otherwise.

| arm | what it breaks | stages | cells | verdict |
|---|---|---|---|---|
| `NO_FTZ` | `ftz` dropped at every stored intermediate | STRUCTURAL 42 vs 37 | 1398/2800 `[sse=7 alpha=7 beta=1 gamma=7 iter=317 level=418 trend=60 season=420 fcast=161]` | FAILS, hardest arm in the lane |
| `SWAP_FMA` | the OTHER product fused in the eval's level update | STRUCTURAL 39 vs 37 | 1471/2800 `[sse=7 alpha=7 beta=1 gamma=7 iter=387 level=416 trend=60 season=420 fcast=166]` | FAILS. DEVIATION 698's fuse choice is load-bearing, not cosmetic |
| `ROTATE_CONV` | `conv1d`'s filter sum starts at `block_idx.x % filter_size` | 34/41 on `mixed`, **0/37 on the other five** | 506/2800 on `mixed` `[sse=2 alpha=2 beta=1 gamma=2 iter=132 level=90 trend=120 season=115 fcast=42]`, 0 elsewhere | FAILS, but REACHED BY ONE FIXTURE ONLY. See below |
| `LS_TIE` | line-search acceptance loosened `>` to `>=` | 9/37 (first `hw.opt.iter008.params`) | 64/2800 `[beta=1 iter=7 trend=56]` **FINAL FORECAST UNCHANGED** | FAILS. See below |
| `CRIT_ORDER` | the two stop criteria tested in the other order | 1/37 (`hw.opt.criterion`) | **0/2800** | FAILS. See below |
| `NO_ZERO_DIR_GUARD` | DEVIATION 662 off | fails earlier, at `check_hw_zero_series_keeps_start` | n/a | FAILS: `params 0.0/0.0/0.0 criterion OPTIM_MIN_ERROR_DIFF niter 1 iter000 0x7fc00000 x3` |
| `CLAMP_GE` | DEVIATION 663's lower test loosened `>` to `>=` | fails earlier, at `check_hw_signed_zero_clamp` | n/a | FAILS: `bound_device(-0.0) = 0x80000000, expected 0x00000000` |
| `STD_SQRT` | `std.math.sqrt` for the BFGS step size | **0/37, 0/36, 0/15, 0/14, 0/29, 0/41** | **0 on all six fixtures** | **APPLE-NULL: a REACH FAILURE, not a pass** |
| `HW_MAX_CLAMP` | `bound_device` as `min(max(0.0, v), 1.0)` | **0/37, 0/36, 0/15, 0/14, 0/29, 0/41** | **0 on all six fixtures** | **APPLE-NULL: a REACH FAILURE, not a pass** |

### Three arms that an output-only comparison would have called inert

This lane reproduces the mamba lane's result, twice, and it is the single
most important thing on this page.

* **`CRIT_ORDER` moves ZERO cells.** Not one of 2800. It changes exactly
  one stage, `hw.opt.criterion`, and nothing else: the arithmetic is
  untouched and only the REPORTED criterion changes. A gate that compared
  fitted parameters and components would have passed it for ever. It is
  caught only because the criterion is a recorded stage (DEVIATION 665),
  which is the entire argument for recording it. It also settles a
  question the arm was written to ask: **the tie IS reached** -- both stop
  criteria fire on the same iteration on the additive fixture, so their
  test ORDER is a real decision and not a theoretical one.
* **`LS_TIE` leaves the final forecast bit-identical.** It moves 9 stages
  and 64 cells, including 56 trend cells, and the forecast -- the thing a
  caller actually reads -- does not move at all. An output-only comparison
  sees nothing. This is the mamba lane's `S1_FOLD_DESCENDING` finding
  reproduced in a completely different algorithm.
* **`ROTATE_CONV` is reached by one fixture in six.** Five fixtures show
  0/37 stages and 0 cells, and that is not a weak arm: with `batch_size`
  7 at `tpb` 32 there is exactly ONE block, so `block_idx.x %
  filter_size` is 0 and the rotation is the identity. Only `mixed`, which
  the gate deliberately runs at `tpb` 4 to get two blocks, reaches it --
  and there it moves 34 of 41 stages. REACH IS PER-BRANCH: had `mixed`
  not existed, this arm would have looked inert while the defect it
  models was fully live on any batch large enough to need two blocks.

### The two Apple-null arms are reach failures, and are counted as such

Neither is a pass. `STD_SQRT` is null because on Metal `std.math.sqrt`
and `identical_sqrt` are the same correctly-rounded hardware instruction;
it is the NVIDIA arm of that seam (row 10: approximate there) and cannot
be exercised here. `HW_MAX_CLAMP` is null because the clamp's zero is a
compile-time constant, so `llvm.maxnum(0.0, v)` folds to a compare-select
answering `+0.0`; LLVM may do that because maxnum's zero-tie answer is
unspecified. Both now carry counts across all six fixtures rather than a
single "no difference", so a future vendor run has a baseline to move.
`CLAMP_GE` is the arm that bites where `HW_MAX_CLAMP` does not.

### Negative controls, so the passes are not vacuous

`check_hw_launch_invariance` is a chain of `_first_diff(...) == ""`. If
`_all_bits` returned an empty list, or `_device_fit` ignored its tpb, pad
and poison arguments, every comparison would compare a thing to itself
and the gate would pass for ever on every vendor. Two controls now run
before any of it is believed, and both PASS: a fit of a DIFFERENT series
must produce different bits over a non-empty cell list, and rows 1-7 of
the batch of 512 must not all carry row 0's SSE. They raise the word
VACUOUS rather than FAILED, because a vacuous gate and a broken kernel
are different diagnoses.

## What the gates check

    check_hw_refusals                13 refusals by name (their pyx rules + DEVIATION 664)
    check_hw_decompose_vs_reference  R1Qt == the float64 pinv at 172 weights with the four
                                     pinv identities; planted slope/intercept/season recovered
    check_hw_optimizer_reduces_sse   14 series, fitted SSE <= start SSE everywhere, parameters
                                     in [0,1], float32 vs float64 SSE reported
    check_hw_forecast_continues_pattern  planted noiseless series continued within 0.5% over
                                     two seasons
    check_hw_signed_zero_clamp       DEVIATION 663 on the host helper, as the alpha/beta/gamma
                                     of a direct eval launch, and at the RECORDED clamp
    check_hw_zero_series_keeps_start DEVIATION 662: the all-zero additive series keeps
                                     (0.4, 0.3, 0.3), MIN_GRAD_NORM, niter 0, SSE 0
    check_hw_device_equals_oracle    6 fixtures, every stage and cell, device == oracle
    check_hw_launch_invariance       THE HEADLINE: every output byte identical across
                                     tpb_decomp 32/256, tpb_optim 128/64, pad 0/37, two
                                     poisons, run twice, and batch 1 == 7 == 512
    check_hw_card_is_emitted         the stage list and the run-to-run control

Measured 2026-08-23 on one M4, IDENTICAL: all nine OK, 6 of 6 fixtures
bit-identical at every stage and cell, 37 stages recorded, both negative
controls pass.

FAST, rebuilt and run the same day, RECORDED not asserted (exit 0): all
nine checks run and `check_hw_device_equals_oracle` REPORTS rather than
asserts, 1 of 6 fixtures identical (`zero`). The divergence is now
QUANTIFIED per fixture instead of being summarised as "the counts
differ":

| fixture | stages | cells |
|---|---|---|
| additive | STRUCTURAL 41 vs 35 | 1471/2800 |
| multiplicative | STRUCTURAL 45 vs 43 | 1124/2800 |
| constant | 9/15 (first `hw.opt.iter000.params`) | 215/2800 |
| zero | 0 | 0 (identical) |
| short | STRUCTURAL 27 vs 29 | 565/1792 |
| mixed | 31/37 (first `hw.opt.iter000.params`) | 905/2800 |

The divergence starts at `hw.opt.iter000.params` wherever it is not
structural, i.e. in the optimizer's FIRST iteration, and then compounds:
the two runs stop after different iteration counts, which is why four of
the six are structural rather than per-cell. That is the expected FAST
split (no `ftz`, `identical_mul_add` as the naive chain), and it is
recorded rather than asserted because FAST is the vendor-default arm by
construction. `check_hw_launch_invariance` is OK under FAST too, controls
included: launch invariance does not depend on the numeric mode.

## Two places this port is NOT cuML's bits, stated plainly

1. DEVIATION 660's `R1Qt` (host float64 closed form, not cuSOLVER float32).
2. DEVIATION 697's H11 and H33 (float32, not their accidental float64).

Everything else is designed to be their arithmetic in a pinned spelling.
Neither has been checked against a real cuML run, because cuML does not run
on this machine; both are expected to differ in the last places and neither
is expected to change a criterion or a decision.

## HAND-OFF

**pixi task lines** (I do not own `pixi.toml`; land these next to
`check-metrics-*`):

    check-holtwinters = "mojo run -I . holtwinters/mojo_only/hw_check.mojo"

**IDENTITY_PATHS row text** (I do not own `IDENTITY_PATHS.md`; the row
number is the orchestrator's to assign):

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| NN | holtwinters: Holt-Winters exponential smoothing (`runner.cuh`, `internal/hw_{eval,optim,decompose,forecast}.cuh`) | a strictly sequential recurrence with no fold to pin, so the exposure is elsewhere: every `a*x + b*y` update leaves the contraction to the compiler (C++ does not say which product fuses); every stored intermediate inherits the vendor's denormal mode; `bound_device` is `fminf(fmaxf(v,0),1)` and meets both `-0.0` and a computed NaN; `batched_ls` reaches `pinv([1,t])` through cuSOLVER geqrf/orgqr + cuBLAS gemm; the Hessian diagonal is computed in float64 in H11/H33 and float32 in H22 through a `2.` vs `2` literal; the optimizer can compute NaN on a legal all-zero series and report (0,0,0) | DEVIATION 698 (one flush-and-fuse rule, first product fused and second stored at every seam, ftz at every stored intermediate; seam table in the README); DEVIATION 663 (the clamp as a compare chain); DEVIATION 660 (R1Qt as a host float64 closed form); DEVIATION 697 (the Hessian diagonal in float32 on all three entries -- Metal has no float64, so their arm is unportable); DEVIATION 662 (a zero direction returns their own MIN_GRAD_NORM criterion); DEVIATION 661 (one NaN payload in the card); DEVIATION 664 (non-finite and non-positive-under-multiplicative refused by name); DEVIATION 665 (per-iteration parameters, niter and criterion recorded) | device == oracle bitwise under IDENTICAL on one Apple M4, 6 fixtures x every stage and cell; launch-invariant across two block widths, two paddings, two poisons, and batch 1 == 7 == 512; 7 sabotage arms fail, 2 recorded Apple-null; NO SECOND VENDOR; part of the tree uncompiled since the green run (README OWED list) |

**The Python surface** is not this lane's directory. `estimator.mojo` is
the entry `bindings/` should reach.

## OWED, and it is not a short list

Items 1, 2 and 3 of the previous list are DONE and are the tables above.
What remains:

1. **NVIDIA and AMD re-prints** of the whole gate. `STD_SQRT` and
   `HW_MAX_CLAMP` are reach failures on Apple and can only be exercised
   there; every "identical across vendors" claim in this file is a claim
   about the spelling, not a measurement.
2. **A fixture that reaches `bfgs_iter_limit`** so the cuml#888
   line-search behaviour recorded in `UNPORTED.tsv` is exercised rather
   than argued about. No current fixture hits the 1000-iteration limit
   (the gate prints `0 hit the 1000-iteration limit`).
3. **A fixture that reaches the second NaN route** (`rho_ = 0` from two
   bitwise-equal consecutive gradients).
4. **`get_num_blocks`'s 65535 cap** is ported but not on the launch path
   and no fixture approaches it. Ungated.
5. **A `ROTATE_CONV`-style multi-block fixture for the OTHER kernels.**
   `mixed` gives two blocks for the decomposition path; the optimizer and
   forecast kernels are only ever run at one block by five of the six
   fixtures. Their launch-geometry independence rides on the batch-512
   arm of `check_hw_launch_invariance` rather than on a sabotage.
