# holtwinters: cuML's Holt-Winters exponential smoothing, ported for bitwise identity

DEVIATIONS 660-665 and 697-698. COPY, DO NOT IMPROVE.

## Status: Apple M4 only, both modes green, and PART OF IT IS UNVERIFIED

Read this before trusting anything below.

* The gate `mojo_only/hw_check.mojo` was built and run on one Apple M4 on
  2026-08-23 and printed `ALL OK` under IDENTICAL and under FAST.
* **No second vendor has run this.** Every cross-vendor claim here is a
  claim about what the spelling is designed to guarantee, not a
  measurement. NVIDIA and AMD re-prints are OWED.
* **Some of this tree has not been compiled since that green run.** The
  lane was made read-only mid-round (the machine was crashed by seven
  parallel compiles), so the commits after the green one carry source
  edits that are unbuilt. The OWED list at the bottom names every one of
  them. Do not read a green line in this file as covering them.

## Where the upstream is, and the trap

    /Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00   <- THIS ONE, 265b9da6
    /Users/andrewhendel/CascadeProjects/upstream/cuml             <- NOT THIS ONE, 00094f7e

Both contain a complete `cpp/src/holtwinters`. They differ: the newer
checkout still has the 13-line Apache header where the pinned tree has a
2-line SPDX one, which shifts every line number by about eleven; it lacks
the `checked_arithmetic.hpp` guards; and it has a `holtwinters_api.cpp`
that v26.08.00 does not. An audit of this lane was already run against the
wrong tree once and produced two findings that were false (it flagged our
`RAFT_FAIL` port as invented code, and missed that our `trend_len`
parenthesization mirrors the pinned `checked_sub`). Every line span in
`PORTED_MAP.tsv` and in the `.mojo` headers is against the PINNED tree and
was recomputed by brace-matching on 2026-08-23.

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
nothing is edited and nothing is reverted.

| arm | what it breaks | result |
|---|---|---|
| `ROTATE_CONV` | `conv1d`'s filter sum starts at `block_idx.x % filter_size` and wraps: the same terms in a launch-dependent order | FAILS. `mixed card: hw.decomp.trend f32 84 c85b27cf43373bd2 VS b08af783e8b81cc1` |
| `NO_FTZ` | `ftz` dropped at every stored intermediate | FAILS. `additive card: record counts differ: 42 vs 37` -- a STRUCTURAL divergence, because the flush changes the optimizer's path and the two runs stop after different numbers of iterations |
| `SWAP_FMA` | the OTHER product fused in the eval's level update | FAILS. `additive card: record counts differ: 39 vs 37` |
| `NO_ZERO_DIR_GUARD` | DEVIATION 662 off: their unguarded `0.866 / sqrt(0)` | FAILS. `series 0 params 0.0/0.0/0.0 criterion OPTIM_MIN_ERROR_DIFF niter 1 sse 0x00000000 iter000 0x7fc00000/0x7fc00000/0x7fc00000` -- the NaN, canonicalized by DEVIATION 661, and the (0,0,0) the deviation exists to prevent |
| `CLAMP_GE` | DEVIATION 663's lower test loosened from `val > lo` to `val >= lo`, so a `-0.0` compares equal to `+0.0` and survives the clamp | FAILS. `bound_device(-0.0) = 0x80000000, expected 0x00000000` |
| `LS_TIE` | the line-search acceptance test loosened from `>` to `>=`: on an exact tie the step is rejected and halved again | FAILS. `additive card: hw.opt.iter008.params f32 21 4bcc383a5858277b VS 6da3d5f0c043ed14` |
| `CRIT_ORDER` | the two stop criteria tested in the other order, so an iteration where both fire reports the other criterion | **UNVERIFIED.** The build was cut off by the compile freeze. OWED |
| `STD_SQRT` | `std.math.sqrt` for the BFGS step size instead of `identical_sqrt` | **NULL ON APPLE**, recorded. The whole gate is byte-identical to the clean build, because on Metal both spellings are the same correctly-rounded hardware sqrt. It is the NVIDIA arm of that seam (row 10: approximate there) and stays unrun until a second vendor prints |
| `HW_MAX_CLAMP` | `bound_device` as `min(max(0.0, v), 1.0)`, the hardware max with the zero FIRST | **NULL ON APPLE**, recorded, and the reason is the interesting part. Row 39 measured `max(+0.0, -0.0)` on two RUNTIME operands. Here one operand is the compile-time constant `+0.0`, and `llvm.maxnum(0.0, v)` is folded to a compare-select whose tie answer is `+0.0`. LLVM may do that, because maxnum's zero-tie answer is unspecified. That null is itself the argument for the compare chain: a `max` here answers correctly only if a constant happened to get folded, which is not a property to rest a recorded stage on. `CLAMP_GE` is the arm that bites |

Two arms being Apple-null is a finding, not a gap: both are other
vendors' arms and both are recorded as owed rather than counted as passes.

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
bit-identical at every stage and cell, 37 stages recorded. FAST: all nine
run, and `check_hw_device_equals_oracle` REPORTS rather than asserts (1 of
6 identical) -- the expected FAST split, and the divergence is inside the
optimizer's iterations, so the two runs take different numbers of stages
and the card difference is structural rather than per-cell.

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

Everything here needs a compile slot. None of it has been run.

1. **`CRIT_ORDER` sabotage** -- written, build cut off by the freeze.
   Must FAIL `hw.opt.criterion`. If it turns out Apple-null (both criteria
   never fire on the same iteration on these fixtures), that is a REACH
   failure, not a pass: it means the tie the arm targets is unreached and
   a fixture must be built that reaches it.
2. **A FAST-mode rebuild** of everything after the green run.
3. **The six pre-existing sabotage arms re-run** against the current tree.
   They passed against the pre-edit tree; the edits are believed
   bit-neutral (the clean IDENTICAL gate was rebuilt once after them and
   printed identical hashes) but the arms themselves were not re-run.
4. **NVIDIA and AMD re-prints** of the whole gate. `STD_SQRT` and
   `HW_MAX_CLAMP` are specifically waiting on these.
5. **A fixture that reaches the `bfgs_iter_limit`** so the cuml#888
   line-search behaviour recorded in `UNPORTED.tsv` is exercised rather
   than argued about.
6. **A fixture that reaches the second NaN route** (`rho_ = 0` from two
   bitwise-equal consecutive gradients) to confirm it terminates the way
   `UNPORTED.tsv` says it does.
7. **`get_num_blocks`'s 65535 cap** is ported but not on the launch path
   and no fixture approaches it. Ungated.
