# arima: cuML's batched ARIMA Kalman filter

Rung 2 of the time series ladder. Rung 1 (the KPSS stationarity test and
auto_arima's `d` loop, DEVIATIONS 671-672) landed in `tsa/`.

**This README covers `arima/` AND carries the lane's shared material:
DEVIATION 670, the hand-off list, and the row text for both directories.
`tsa/README.md` points here for all three.** COPY, DO NOT IMPROVE.

---

## Status: BUILDS, GATED GREEN, AND BIT-IDENTICAL APPLE <-> AMD. NVIDIA owed.

    the whole tree builds                      YES
    check-arima, IDENTICAL                     PASSES
    check-arima, FAST                          PASSES (23 RECORDED lines)
    arima-card                                 229 stage records
    sabotages run                              11 of 11 (8 bite, 3 null)
    inherited MEASURED claims judged           4 of 4 (2 earned, 2 struck)
    a second vendor                            YES, AMD MI325X, 2026-08-28
                                               137 card records, 0 differing
    a third vendor                             NVIDIA owed: the 2026-08-28
                                               legs ran 7 and 9 lanes and
                                               arima was in neither list

    ALL OF THE 2026-08-24 WORK NOW BUILDS AND PASSES, both modes: DEVIATION
    677, the six new card stages, orders `arma44` and `ar2_tie`, the two new
    gates, the widened fold gate, and sabotage arms (h) and (i). The card
    carries 229 stage records, up from the 138 recorded on 2026-08-23
    (`arima/PORTED_MAP.tsv` lines 3 and 13). 11 of 11 sabotage arms have run.

    A PREDICTION OF MINE THAT WAS WRONG, recorded because it was written
    down first: I expected signature errors around `guards`, `Fs` and the
    `P` writeback, "they thread new arguments through two kernels". All
    three binaries compiled clean on the first attempt.

Shapes are the smallest that still reach every branch: `n_obs = 24`,
`batch = 6`, `fc_steps = 3`, eight orders. No timing was taken, at any
point, for any reason.

**The headline.** The Kalman filter is bit-identical device-versus-oracle on
all eight orders across all eleven stages -- `Z R T RQ RQR P0 alpha0 pred vs
loglike fc` -- with ZERO cells differing anywhere, including at `rd = 8`,
the largest state this lane accepts. So is `predict`, and so is the
finite-difference gradient.

**What that does and does not mean.** It means the device and a serial host
replay through the same numeric helpers agree bit for bit on one machine. It
does NOT mean bitwise identity across vendors, which is the actual goal and
which nothing here has tested. Row 58 stays SOURCE-ONLY on the vendor
question until an NVIDIA or AMD card is diffed against this one.

## What the gates found (2026-08-23, Apple M4, both modes)

    check_jones_device_equals_oracle    16 stage comparisons, p = 1..4, AR
                                        and MA, forward and inverse: 0 cells
                                        differ. Round trip inverse(forward)
                                        worst relative 2.73e-6 (~23 Float32
                                        ulp), REPORTED not asserted
    check_jones_refuses_by_name         batchSize < 1, parameter < 1,
                                        parameter > 8, all raised by name
    check_jones_contraction_is_visible  of 160 accumulations the two
                                        contraction spellings differ on 27
    check_kalman_device_equals_oracle   8 orders x 11 stages: 0 cells differ
    check_lyapunov_solves_the_equation  worst relative residual 3.5e-6
                                        (ar2_unit); 7.3e-8 to 4.3e-7 on the
                                        other seven
    check_predict_sentinel_is_reached   4 of 8 orders reach it, res_offset
                                        1, 1, 3, 4
    check_predict_device_equals_oracle  8 orders x 162 cells: 0 differ;
                                        0/0/0/6/6/0/18/24 sentinel cells
                                        checked, all 0x7fc00000
    check_grad_device_equals_oracle     24 gradient cells, 0 differ
    check_grad_reset_preserves_neg_zero d_x untouched; x_pert back to d_x
                                        bitwise
    check_kalman_matches_float64        n_diff = 0: 4.7e-8 to 1.5e-7.
                                        n_diff > 0: up to 1.8e-3
    check_arima_refuses_by_name         11 refusals, each by name
    check_kalman_launch_invariant       block 32/64/128, 41 poisoned floats,
                                        batch of 6 vs batch of 3, run twice:
                                        0 cells differ
    check_unit_root_guard_is_reached    fires on 6 of 6 series; pre-guard
                                        phi_2 is the clamp 0xbf7ff972
    check_lu_pivot_tie_is_reached       an EXACT tie (0.45327485 twice),
                                        maximal by 0.20, on 6 of 6 series
    check_guard_decisions_are_recorded  three arms: bit 0 set, bit 1 set,
                                        neither set; 0 of 6 series disagree
                                        on each
    check_fold_order_is_visible         55 of 1110 cells over 10 orders
                                        (FAST 63), 8 orders moving,
                                        strongest 25

**Three findings worth carrying forward.**

`check_fold_order_is_visible` was thin: 2 of 24 cells on one order. WIDENED
2026-08-24 to all ten orders, and the result overturned the model I had
written down in advance. Measured (IDENTICAL / FAST), 55 / 63 cells of 1110,
8 of 10 orders moving, strongest single order 25 / 24:

    order         rd  n_phi   cells   moved   moved%
    ar1            1      1       6     0/0     0.0
    ma2            3      0      54     0/0     0.0
    sarima_rd8     8      1     384     1/2     0.3
    arima111       3      1      54     2/1     3.7
    ar2_unit       2      2      24     1/3     4.2
    sarima_full    7      3     294    13/18    4.4
    arima212       4      2      96     5/7     5.2
    arma11_k       2      1      24     2/1     8.3
    arma44         5      4     150    25/24   16.7
    ar2_tie        2      2      24     6/7    25.0

**CHAIN LENGTH IS NOT THE DRIVER, and my recorded prediction was wrong.** It
said `ar1` moves zero (right), `rd = 2` orders move very few (wrong:
`ar2_tie` at `rd = 2` moves the MOST of anything, 25%), and `rd >= 4` moves
a substantial fraction (wrong: `sarima_rd8` has the LONGEST chain at `rd = 8`
and moves 0.3%, the least of anything that moved). One clause of three held.

THE DRIVER IS `n_phi`, the count of NON-TRIVIAL entries in `T`. A differenced
or seasonal `T` is mostly structural: exact `0.0` and exact `1.0` from the
differencing rows, the seasonal shift and the companion superdiagonal. A fold
over exact values is order-independent, so those cells cannot move however
long the chain is. Only the `n_phi` hashed AR coefficients carry rounding.
`ma2` has `n_phi = 0` and moves NOTHING despite `rd = 3`.

The lesson generalizes past this gate, which is why the wrong prediction is
left in the docstring rather than quietly replaced: widening a fixture means
widening the part that CARRIES ARITHMETIC, not the part that carries
structure. `rd` was the wrong knob; `n_phi` is the right one.

Floors raised from "at least 2 orders move something" to the observed numbers
with headroom: at least 6 orders, at least 35 cells total, and at least one
order moving 15.

`check_kalman_matches_float64` splits by `n_diff`, not by anything else, and
`arima212` is the exception that shows why: `n_diff = 1` yet it agrees to 7
digits like the undifferenced orders. The diffuse `kappa = 1e6` block costs
accuracy in proportion to how much weight the filter puts on those states,
not merely by existing.

The loosest bounds in the gate are DERIVED and sit three orders of magnitude
above what was measured. They are not yet bounds anyone has earned. The
table below says what each should be replaced with and, more importantly,
what evidence makes the replacement legitimate; a compile slot can work
straight down it.

| bound | where | asserted | measured worst | replace with | the evidence that earns it |
|---|---|---|---|---|---|
| Lyapunov residual | `check_lyapunov_solves_the_equation` | `5e-3` | `3.5e-6` (`ar2_unit`); `7.3e-8` to `4.3e-7` elsewhere | `2e-5` | Must ALSO hold at `arma44`, whose `r = 5` makes the LU 25x25 and is the worst conditioning the lane accepts; that order did not exist when `3.5e-6` was measured. Then sabotage `lu_inverse`'s trailing update and confirm the residual blows past the new bound, so the bound is known to be a bound and not a ceiling nothing touches. |
| Float64 gap, `n_diff = 0` | `check_kalman_matches_float64` | `5e-3` | `1.5e-7` | `1e-6` | Straight tightening; the four undifferenced orders sit at `4.7e-8` to `1.5e-7`, so `1e-6` is roughly 7x headroom. |
| Float64 gap, `n_diff > 0` | `check_kalman_matches_float64` | `2e-2` | `1.8e-3` | `5e-3` | Keep the headroom wide here and say why in the message: the diffuse `kappa = 1e6` cost is a MECHANISM, not noise, and `arima212` shows it does not scale with `n_diff` (`n_diff = 1` yet `7.2e-8`). A tight bound would be fitted to the fixture rather than to the mechanism. |
| Jones round trip | `check_jones_device_equals_oracle` | `1e-2` | `2.73e-6` | `1e-5` | This one is REPORTED, not a correctness claim, so the bound's only job is to catch a gross regression. `1e-5` is about 4x the measured value and still far below anything that would indicate a real break. |

None of these should be changed by a lane that cannot run the gate: a bound
edited without a run is a bound that turns the next build red for a reason
nobody can see. They are written here and left alone deliberately.

## Metal's 31-argument kernel cap, and this lane's headroom

The holtwinters lane hit it on 2026-08-24: one added `Int32` output took a
kernel from 31 arguments to 32 and the build died with `Metal Compiler
failed to compile metallib`, naming no argument, no parameter, and pointing
at line 1 of the CHECK file. It is a backend limit, so CUDA and HIP would
not have shown it, and the message says nothing about the real cause.

This lane adds six card stages and is an obvious candidate, so the arguments
were COUNTED rather than assumed:

    kalman_init_state_kernel                21 arguments, 10 to spare
    batched_kalman_loop_kernel              19 arguments, 12 to spare
    init_batched_kalman_matrices_kernel     16 arguments, 15 to spare
    in_sample_prediction_kernel             14 arguments, 17 to spare
    grad / perturb / reset / copy_forecast   5 to  7 arguments

That headroom is why the six new stages built first time. `kalman_init_
state_kernel` is the one to watch: SIX of its twenty-one arguments are
Lyapunov scratch buffers (`ImAA`, `ImAA_inv`, `piv`, `vecq`, `ImT`,
`ImT_inv`), and the day a seventh output is needed they should collapse into
ONE buffer with named slices split out in the launcher, which is the fix
that worked in holtwinters. A future output REUSES A SLICE; it does not add
an argument.

If a build here ever fails with a metallib error pointing somewhere useless,
count the arguments before hunting anything else.

## More hashes: the decisions this pipeline makes

Andrew asked whether there are more hashes to take. There were. Every stage
this lane recorded was a float BUFFER, and a buffer records a VALUE; the
things below are CHOICES, and a choice can differ between vendors while
every value stays bit-identical. That is not hypothetical: the holtwinters
lane's `CRIT_ORDER` sabotage moves ZERO of 2800 float cells and is caught
solely because the criterion is a recorded stage.

**THE RULE (CARD_GAPS.md): a decision worth hashing is one the ALGORITHM
makes, not one the SCHEDULER makes.** Launch geometry must never enter an
identity card. The card is ASSERTED launch-invariant, so recording block
width would make two settings differ BY CONSTRUCTION and destroy the exact
property `check_kalman_launch_invariant` exists to prove. Anything whose
value depends on the machine, the occupancy or the dispatch is out; anything
the algorithm itself decides from computed values is in. Every entry in the
"added" list below is a comparison the arithmetic performed. Every entry in
the "listed, not added" tail is either derivable from the order or, in the
last case, scheduler state that is barred outright.

**Added to the card (2026-08-24).** Four, chosen because they are
load-bearing AND cheap:

    piv          the LU permutation, per column, for BOTH solves. This IS
                 DEVIATION 674's tie rule, which this lane CHOSE because
                 cuBLAS's is unreadable. NOT derivable from any float stage:
                 `P0` is the product of the solve, not of the permutation
                 that produced it. Sabotage (h) can leave `P0` bit-identical
                 and still be wrong; `piv` is then the only witness.
    guards       one byte per series. Bit 0: the `rd == 2 && p == 2`
                 unit-root guard rewrote `T[1]`. Bit 1: the `r == 1`
                 intercept guard nudged `I - T*`. Bit 0 was only inferable
                 from a suspicious `-0.99` in `T` if you already knew to
                 look; bit 1 was NOT INFERABLE AT ALL, because `ImT` is
                 overwritten in place by its own LU before any stage is read.
    info_init    the singular-solve column, per series.
    info_loop    the refused step, per series. Both are all-zero on a
                 healthy run, and a buffer of zeros is worth recording
                 because it proves THE REFUSAL PATH WAS NOT TAKEN.

Three more, added 2026-08-24 after a repo-wide audit ranked them:

    Fs           the innovation variance `F = Z P Z'` at EVERY timestep. It
                 is the quantity the DEVIATION 673 / 677 guard READS, so a
                 refusal that fires on one vendor and not another shows up
                 here one stage before it shows up as a raise. And
                 `identical_log` compresses the whole series into a single
                 `sum_logFs` fold, so a per-step difference that cancels
                 inside that fold otherwise leaves no trace at all.
    P_final      the covariance AFTER the filter. Only `P0`, the covariance
                 BEFORE it, was recorded, so everything the filter did to the
                 covariance and carried forward was invisible.
    guards bit 2 WHICH SIGN the `r == 1` intercept nudge chose.
                 `raft::signPrim`'s double specialization is `signbit`, so
                 `-0.0` and `+0.0` send the intercept to OPPOSITE SIGNS while
                 looking identical in every float stage. A row-10 signed-zero
                 hazard sitting directly on a model parameter.

**A limit worth naming, because it is an argument for one more stage.**
`_numerical_stability` runs at the end of every iteration: `A = 0.5(A + A')`
AVERAGES AWAY any antisymmetric difference between two vendors, and
`A_ii = |A_ii|` ABSOLUTE-VALUES AWAY any diagonal sign difference, `-0.0`
against `+0.0` included. The covariance is therefore a SINK for exactly the
class of divergence this repo hunts. `P_final` catches what survives that
sink; it cannot catch what the sink erased. The strictly better instrument
is `P` BEFORE the stabilization, and it is not yet taken -- listed below.

**Listed, not added.** Each with why it was left:

    P before             the covariance BEFORE `_numerical_stability`, per
    stabilization        iteration or at least at the last one. Strictly
                         better than `P_final` for the reason above: the
                         stabilization is a sink that erases antisymmetric
                         and diagonal-sign divergence, so a vendor difference
                         can exist, be erased, and never appear in any stage.
                         Left for now because it needs either a per-iteration
                         buffer (`n_obs * rd2 * batch`, no longer free) or a
                         deliberate choice of which single iteration to keep.
                         The cheap version is a per-series MAX ASYMMETRY
                         scalar, `max |P_ij - P_ji|` before stabilization,
                         which is one float per series and would show that a
                         difference existed even after it was averaged away.
    jones clamp mask     which parameters the Jones clamp bound, per series.
                         A clamped parameter is a DIFFERENT MODEL, so this is
                         load-bearing. Left out because it is DERIVABLE from
                         `t_params`, which is already a recorded stage
                         (compare against +-0.9999), and because capturing it
                         at the source would change `jones_transform`'s
                         signature, which is shared by four call sites.
    n_obs_ll             how many observations entered the likelihood.
                         Derivable from `n_obs` and `n_diff`; recorded
                         nowhere, but it is a function of the order, not of
                         the data, so it cannot differ between vendors.
    predict's shape      `dD`, `res_offset`, `p_start`, `p_end`, `period1`,
                         `period2`. All host-side integers derived from the
                         order; a vendor cannot disagree about them.
    prepare_data branch  `d + D` in {0, 1, 2}. Same: a function of the order.
    finalize_forecast    single vs double undifferencing. Same.
    trans vs copy        whether the Jones transform ran or `_copy_params`
                         did. A caller's argument, not a device decision.
    the loop kernel's    `n_diff == 0` vs `n_diff > 0` arms in `pred` and
    n_diff arms          `F`. Determined by the order; derivable.
    kalman_tpb, grid     BARRED, not merely omitted. These are SCHEDULER
                         state, and CARD_GAPS.md's rule puts them out of an
                         identity card categorically: the card is asserted
                         launch-invariant, so recording block width would
                         make two settings differ BY CONSTRUCTION and break
                         the very property `check_kalman_launch_invariant`
                         proves. `kalman_tpb` is a parameter of
                         `batched_kalman_filter` precisely so that gate can
                         vary it and show nothing moves.

The pattern in the "listed" column is worth stating once: almost everything
left out is a function of the ORDER rather than of the arithmetic, and two
vendors running the same order cannot disagree about it. What is worth
hashing is a decision made from COMPUTED VALUES -- a pivot comparison, a
guard threshold, a refusal test -- because those are where two vendors can
legitimately part company.

## The inherited MEASURED claims, judged

Every "MEASURED" sentence in this lane was written by an agent that died
before compiling anything. All four have now been checked against real runs.

    STRUCK  jones_transform.mojo  "round trip within 4 ulp" -- it is 2.73e-6,
                                  about 23 ulp, nearly six times the claim.
                                  Replaced with the number and its cause.
    STRUCK  arima_common.mojo     "agrees to the sixth digit" and the
                                  attribution to simple_differencing. It is
                                  the seventh digit undifferenced and the
                                  third differenced, and the split is n_diff,
                                  not a switch this lane implements.
    EARNED  matrix.mojo           residual <= 1e-5 holds (worst 3.5e-6);
                                  per-order table substituted for the bound.
    EARNED  batched_kalman.mojo   the refusal names series and step, quoted
                                  verbatim now. One correction (it named a
                                  gate that does not exist) and one addition
                                  (it fired for real, not just when planted).

## Which upstream tree this was audited against

**`/Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00`**, VERSION
`26.08.00`, commit `265b9da6a0e75dbef071a3168398b993a5ff6f0e`.

There is a trap here that cost another lane a round. The sibling checkout
`/Users/andrewhendel/CascadeProjects/upstream/cuml` is **branch-25.08**
(VERSION `25.08.00`, commit `00094f7e`), and all six ARIMA files differ
between the two trees. Every symbol this lane cites sits 10 to 11 lines
LATER in 25.08 than in 26.08, so an audit read against the wrong tree
produces citations that look plausible, resolve to a neighboring
statement, and are wrong everywhere.

This lane's audit was read against v26.08.00 throughout, and that was
re-verified after the trap was reported by resolving all 27 header and
PORTED_MAP citations plus all 29 `SEAMS.tsv` rows in BOTH trees and
comparing. Every recorded number matched v26.08.00 exactly and none matched
25.08. No finding was an artifact of the wrong tree. The two rows the whole
audit hangs on both resolve to the exact statement:

    jones_transform.cuh:47  tmp[k] += sign * (a * myNewParams[j - k - 1]);
    jones_transform.cuh:79  tmp[k] = (myNewParams[k] + sign * (a * myNewParams[j - k - 1])) / (1 - (a * a));

Eleven `SEAMS.tsv` rows were nonetheless re-pinned in that pass, because
they had been pointing at a closing brace or a comment a few lines off the
statement they described. None changed a finding; a citation that lands on
`}` is still a wrong citation.

## What is here

    cuml/cpp/include/cuml/tsa/arima_common.h      -> arima/ported/tsa/arima_common.mojo
    cuml/cpp/src_prims/timeSeries/jones_transform.cuh
                                                  -> arima/ported/timeSeries/jones_transform.mojo
    cuml/cpp/src_prims/timeSeries/arima_helpers.cuh
                                                  -> arima/ported/timeSeries/arima_helpers.mojo
    cuml/cpp/src_prims/linalg/batched/matrix.cuh  -> arima/ported/linalg/batched/matrix.mojo  (the r <= 5 Lyapunov path only)
    cuml/cpp/src/arima/batched_kalman.cu          -> arima/ported/arima/batched_kalman.mojo
    cuml/cpp/src/arima/batched_arima.cu           -> arima/ported/arima/batched_arima.mojo

`arima/PORTED_MAP.tsv` pins the commit (cuML 265b9da6, v26.08.00) and says
per file what is transliterated and what is partial. `arima/UNPORTED.tsv`
lists what is not carried and why, one line each; it is long on purpose.
`arima/SEAMS.tsv` records fused versus unfused PER SEAM with the reason,
read off the C++ expression tree. `arima/SABOTAGES.md` is the sabotage
list, every row of it still unrun.

## Commands

    tools/with_build_lock.sh     pixi run mojo run -I . arima/mojo_only/arima_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/mojo_only/arima_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/arima_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/arima.card tools/with_identical_mode.sh \
        pixi run mojo run -I . arima/arima_main.mojo
    python3 tools/identity_trace_diff.py /tmp/arima.mac.card /tmp/arima.other.card

No pixi task is registered; `pixi.toml` is not this lane's file. The task
line to land is under HAND-OFF.

---

## DEVIATION 670: their `double` is our Float32 on the device

Stated once in `arima/ported/tsa/arima_common.mojo` and carried by every
file in `arima/` and `tsa/`.

THEIRS. Every ARIMA kernel is instantiated on `double` only, and
`arima.pyx:326` checks the input to `float64`. `b_lyapunov`'s own comment
(`matrix.cuh:1892-1893`) says the single-precision direct solver "is not
good, use double".

OURS. Metal exposes no Float64 on the device, so the device arithmetic is
Float32 through the IDENTICAL helpers and every host oracle that must match
it bitwise is Float32 too. A Float64 HOST reference sits beside it
(`kalman_oracle.mojo::kalman_host_f64`) so the gap between the two is
MEASURED rather than assumed. The gate that measures it,
`check_kalman_matches_float64`, is written and has not run: its bounds
(5e-3 relative on the undifferenced orders, 2e-2 where `n_diff > 0` because
the diffuse `kappa = 1e6` costs Float32 about `ulp(1e6) = 0.0625` of
absolute error in the stationary block after the first step) are DERIVED,
not observed, and the compile slot must replace them with what it sees. A
Float64 device arm is not offered; the refusal is by dtype.

## DEVIATION 673: a non-positive innovation variance is refused, not filtered

THEIRS. `F = Z P Z'` goes straight into `log(_Fs)` and `vs*vs / _Fs`
(`batched_kalman.cu:209-210`); a non-positive `F` makes the log-likelihood
NaN or -inf, and that payload is the vendor's, in a stage the card records.

OURS (ADDENDUM 11: no computed NaN in a hashed stage). A non-positive `F`
at a SUMMED step (`it >= n_diff`) sets `info[bid] = it + 1` and the host
raises by name after the launch; the loop still runs to the end so every
series is checked. Note what this does NOT cover: for `it < n_diff` theirs
does not check either, and neither do we, so those steps can still produce a
non-finite `_1_Fs`. That is faithful and it is also a hole, and it is
recorded here rather than papered over.

## DEVIATION 677: the same refusal at the diffuse steps too

DEVIATION 673's check was guarded by `it >= n_diff`, so the first `n_diff`
steps were not checked. That looked harmless, because those steps contribute
no term to the log-likelihood. It was not harmless: `_1_Fs = 1.0 / _Fs` is
computed UNCONDITIONALLY at step 3, so a non-positive `F` at a diffuse step
makes the Kalman gain infinite, `alpha` NaN at the next update and `P` NaN
for the rest of the series. Every one of those is a recorded card stage, so
the hole was a live route to a vendor-dependent NaN payload inside a hash.

Theirs checks at no step at all, so this is not a fresh disagreement with
cuML. Having already decided in 673 that a non-positive innovation variance
is refused by name rather than filtered, checking only some steps was an
inconsistency in OUR code. `info` is now signed: `it + 1` for a summed step,
`-(it + 1)` for a diffuse one, which raises a distinct message. The
arithmetic is untouched, so no oracle and no recorded stage changes.

UNREACHED BY EVERY CURRENT FIXTURE, and NOT PROVEN UNREACHABLE. Those are
two different states and the distinction is the point. The diffuse diagonal
is initialized to `kappa = 1e6`, so nothing drives `F` non-positive there
today; but `F = Z P Z'` mixes the diffuse and stationary blocks and `P`
evolves, so no argument on hand says it CANNOT happen. The branch is
therefore defensive and unchecked, which is honest, and it is listed as
OWED with the two ways to settle it: construct a fixture that reaches it, as
the pivot tie was constructed, or prove it unreachable and close it.

"Ungated" would be the wrong word here and is deliberately not used. A
branch nobody can reach does not want gating, it wants a proof and a
closure; saying "ungated" invites someone to spend a round writing a gate
that can never fire.

## DEVIATION 674: cuBLAS getrf/getri/gemm are closed; the solve is spelled here

THEIRS. `Matrix::inv` is `cublasgetrfBatched` then `cublasgetriBatched`
(`matrix.cuh:383-389`). `b_gemm` is `cublasgemmStridedBatched`. Both are
closed: their association order and their pivot tie rule are not readable.
The `info` array is computed and NEVER CHECKED by the caller
(`batched_kalman.cu:1088-1089`, `matrix.cuh:1876`), so a singular
`I - T (x) T` produces an unusable `P0` silently.

OURS (PORTING_RULES 0b-i). `lu_inverse` per series, column-major:
pivot = the FIRST index of the strictly largest `|a|` in the column (a
strict `>` scan, so no signed zero can displace the other), row swap,
`L` column `= a / pivot`, trailing update `fma(-l, u, a)`; then the inverse
column by column by permuted forward (unit `L`) and backward (`U`)
substitution, serial ascending, `fma`. A ZERO PIVOT sets `info[bid] =
column + 1` and the host RAISES BY NAME rather than filtering with a
non-finite `P0`.

**The tie rule and the association are OURS, not theirs.** That is why
`SABOTAGES.md` row (e) perturbs the pivot comparison, and why
`check_lyapunov_solves_the_equation` checks `P0` against the EQUATION
`Ts P Ts' - P + RQRs = 0` in Float64 rather than only against an oracle that
shares its spelling. Neither has run.

## DEVIATION 675: tanh and atanh through identical_exp / identical_log

THEIRS. `raft::tanh(x * 0.5)` and `2 * raft::atanh(v)`: the vendor's
`tanhf` / `atanhf`, whose last bit is a vendor choice (IDENTITY_PATHS row
12's class; `numerics.mojo` carries exp/log/sqrt/cos/pow but no tanh).

OURS, under IDENTICAL:

    tanh(x/2)  = (e^x - 1) / (e^x + 1)   with e^x = identical_exp(x),
                 +-1 returned outright for |x| > 80
    2 atanh(v) = log((1 + v) / (1 - v))  through identical_log

one local per op through `ftz`. Under FAST they are `std.math.tanh(x*0.5)`
and `2 * std.math.atanh(v)`. Neither identity is the libm algorithm, so
IDENTICAL bits differ from FAST bits by design, as every row-12 seam does;
what is purchased is one arithmetic on every backend.

### THE DECISION (2026-08-24): ACCEPT FOR NOW, AND TAKE THE ONE MEASUREMENT THAT WOULD OVERTURN IT

Asked for a decision rather than another measurement. Here it is, with the
reasoning, because the reasoning contains a finding.

**What is measured.** `inverse(forward(x))` returns to x with worst relative
error 2.73e-6, about 23 Float32 ulp, uniform over p = 1..4 and both AR and
MA. The cause is cancellation: `identical_exp(x)` is accurate to about a ulp
of `e^x`, but for small `|x|` the subtraction `e^x - 1` is about `x`, so the
numerator's relative error is roughly `ulp(1)/|x|`. At `|x| ~ 0.03` that is
2e-6, which is what the gate reports.

**FINDING: only HALF of that spelling is on the ported path, and it is the
half that cannot be fixed here.** `batched_jones_transform` is called from
exactly one place, `batched_arima.mojo:111`, with `is_inv = false`. Nothing
in this lane ever calls the inverse: `two_atanh` is reached only by
`check_jones_device_equals_oracle`. The inverse transform belongs to the
optimizer, which maps fitted parameters back to unconstrained coordinates,
and the optimizer is NOT PORTED. So:

    tanh half  (identical_exp)   ON the ported path, runs every fit
    atanh half (identical_log)   OFF it, gate-only, optimizer's rung

**The options, and what each costs.**

  A. ACCEPT and document. Costs nothing. Keeps every recorded card valid.
  B. Fix the atanh half with `identical_log1p`, which ALREADY EXISTS
     (`mojo_only/numerics.mojo`, row 51's seam): `2*atanh(v) =
     log1p(2v/(1-v))` removes that cancellation entirely. Available today,
     no new primitive.
  C. Fix the tanh half with `expm1`: `tanh(x/2) = expm1(x)/(expm1(x)+2)`.
     `identical_expm1` DOES NOT EXIST and `mojo_only/numerics.mojo` is not
     this lane's file, so this is a HAND-OFF to the numerics lane, not
     something that can be done here.
  D. Refuse small `|x|`. Not sensible: small coefficients are the normal
     case, not an edge.

**RECOMMENDATION: A now, C later if and only if one measurement says so, and
NOT B.**

Not B, even though B is free and available. B improves code that the ported
pipeline never executes, and it would move bits: every `arima.jones.inv.*`
stage on every card changes, invalidating the recorded baseline, in exchange
for accuracy on a path only the gate walks. Spending a deviation number and
a card revision on that is the wrong trade. It becomes right the day the
optimizer is ported, and it is listed as a hand-off so that day is not a
rediscovery.

A now, because the cost of the tanh half is genuinely unknown and the only
honest thing to do with an unknown cost is not to pay for it blind.

**THE MEASUREMENT THAT WOULD OVERTURN THIS, and why it has not been taken.**
`kalman_host_f64` is fed the SAME Float32 transformed parameters the device
uses (`kalman_oracle.mojo` says so in its docstring: it "isolates the
filter's precision"). That is deliberate and it is also the reason the
existing numbers CANNOT settle this question: the Float64 reference starts
DOWNSTREAM of the transform, so DEVIATION 675's contribution is not in the
1.5e-7 / 1.8e-3 figures at all. Nobody has ever measured what 23 ulp in a
model parameter does to a log-likelihood.

The fix is cheap and moves no device bit: give `kalman_host_f64` its own
Float64 Jones transform, applied to the UNTRANSFORMED parameters with the
stdlib `tanh`, so the reference spans the whole pipeline. Then the gate
reports the end-to-end cost of DEVIATION 675 directly.

    DECISION RULE, fixed in advance so the answer is not chosen after
    seeing it: if the end-to-end contribution is below the filter's own
    Float32 error (1.5e-7 undifferenced, 1.8e-3 differenced), ACCEPT
    permanently and delete this section's OWED entry. If it is comparable or
    larger, raise the `identical_expm1` hand-off with the numerics lane and
    spend DEVIATION 678 on option C.

## DEVIATION 676: the undefined predictions are the canonical NaN, by constant

THEIRS. `d_y_p[..] = nan("")` (`batched_arima.cu:209`), the vendor's quiet
NaN (Apple `0x7fc00000`, NVIDIA `0x7fffffff`), in a buffer the card records.

OURS. The bit pattern `0x7FC00000` written as a constant, never computed, so
the recorded bytes are the same on every vendor and the caller sees a NaN
exactly where theirs does.

---

## THE RUNG 0 AUDIT: are we mirroring or reinventing?

Mirroring. Every kernel in `arima/ported/` follows cuML branch for branch
and loop for loop, and the divergences are the numbered ones above plus the
spelling collapses recorded below. What the symbol-by-symbol read found was
not reinvention; it was four defects and about thirty wrong citations.

### 1. The Jones contraction was associated wrong (bit-moving)

`jones_transform.cuh:47`, `tmp[k] += sign * (a * myNewParams[j-k-1])`. The
multiply that FEEDS the add is `sign * (...)`, so `a * x` feeds a MULTIPLY,
does not fuse, and rounds. The contraction is
`fma(sign, round(a*x), tmp)`: TWO roundings.

The pile spelled it `fma(sign*a, x, tmp)`: ONE rounding. Worse, its own
header asserted that reading as fact, so the wrong spelling was documented
as the right one. It was wrong in four places at once, the kernel and the
host replay, forward and inverse. All four corrected; the header now
explains the tree. `SABOTAGES.md` row (d) restores the old spelling and must
fail, and `check_jones_contraction_is_visible` asserts on the host that the
two spellings differ on this fixture BEFORE that sabotage is applied, so a
passing (d) cannot be mistaken for an unobservable one.

### 2. The gradient reset was not a copy (bit-moving)

`batched_arima.cu:587` resets with `d_x_pert[N*bid+i] = d_x[N*bid+i]`.
The pile reused `perturb_kernel` with `h = 0`. `-0.0 + 0.0` is `+0.0`, so a
negative-zero parameter came back positive zero after its own iteration and
every LATER parameter's log-likelihood was evaluated on a vector one bit
away from `d_x`. `reset_param_kernel` is the assignment.
`check_grad_reset_preserves_negative_zero` plants `-0.0` in parameter 0 of
every series and checks the gradient of the LAST parameter, which is the one
that has seen every reset.

### 3. An upstream bug was documented as benign and is not

`batched_arima.cu:207`, `d_y_p[0] = 0.0`, is the first statement of the
per-series lambda and writes element 0 of the WHOLE output, from every one
of `batch_size` threads. The pile called it "always overwritten by the loop
below it, a benign race with no effect". Wrong. Thread 0 writes the real
`d_y_p[0]` inside its own loop, and threads `bid > 0` are unordered against
it, so any of them can land its `0.0` afterwards. `d_y_p[0]` can come back
`0.0` instead of the prediction or the sentinel, for any batch of more than
one series. Not ported; recorded as their defect in `UNPORTED.tsv`.

### 4. `predict` had no oracle at all

`in_sample_prediction_kernel` and `copy_forecast_kernel` are device-only
code with no host replay anywhere, so `predict` was the one entry point no
gate could ever have seen, including the DEVIATION 676 sentinel that the
lane's whole vendor-identity claim for that buffer rests on.
`in_sample_prediction_host` and `copy_forecast_host` are now in
`kalman_oracle.mojo` and `check_predict_device_equals_oracle` uses them.

### 5. It did not compile

`LoglikeResult` and `PredictResult` are constructed field-wise and had no
`@fieldwise_init`. Two errors, both fixed. After the fix
`batched_arima.mojo` reported only "module does not contain a `main`
function", which is the whole ported tree parsing.

### 6. About thirty wrong upstream line citations

Every one re-derived against the real file. A sample:
`_batched_kalman_filter` is `:889` not `:891`;
`init_batched_kalman_matrices` `:1141` not `:1142`;
`batched_kalman_filter` `:1248` not `:1245`; `kappa` `:1003` not `:992`;
the `isnan` arms `:191, 193, 219, 236, 246` not `:185, 199, 253, 263`; the
unit-root guard `:1243-1244` not `:1241`; `Mv_l` `:34` and `:45`, `MM_l`
`:57`, `numerical_stability` `:76`. The brief said to verify the header
claims rather than trust them, and this is why.

### 7. One variable of theirs split into two of ours, found and kept

Their `l_tmp[rd2_max]` is BOTH step 4's `T*alpha` vector and step 5's `L`
matrix (`batched_kalman.cu:234, 241`). Ours splits it into `l_v` + `l_tmp`.
The split is SAFE: step 5's first statement rewrites all `rd2` cells of
`l_tmp` from `l_T` before any read, so no cell of theirs is ever read
carrying step 4's value and no bit depends on the reuse. Kept, and spelled
out at the declaration, because a silent split is exactly the class of
defect the audit hunts and the next reader deserves to see it named.

Likewise their two `Mv_l` overloads (`:34-43` unscaled, `:45-56` scaled) are
one function here called with `alpha = 1`; `1.0 * x` is exact for every
IEEE-754 value including subnormals, signed zeros and infinities, so no bit
moves. And their two `P0` branches (`n_diff > 0` on the `r x r` sub-block,
`n_diff == 0` on the whole matrix) are one spelling here, correct because
`rd == r` exactly when `n_diff == 0`.

### 8. Verified correct against the instinct to "fix" it

The pile spells `raft::signPrim` as `signbit(x) ? -1 : +1`. The GENERIC
template (`raft/util/cuda_utils.cuh:559-562`) is `x < 0 ? -1 : +1`, which
returns `+1` for `-0.0` and would have been a real divergence. But cuML
instantiates on `double`, and the `double` SPECIALIZATION at `:569` is
signbit. The port is right, and reading only the generic template is how
that gets "corrected" into a bug. Written into `UNPORTED.tsv` as a trap.

### 9. What was NOT found

No invented algorithm. No missing kernel inside the arms this lane claims.
`init_batched_kalman_matrices`, the `Z` / `R` / `T` construction, the
diffuse `kappa` diagonal, the Kronecker Lyapunov solve, the `r == 1`
intercept guard, the `rd == 2 && p == 2` unit-root guard, the whole
observation loop and the forecast loop all follow theirs statement for
statement. The pile was a good port with four defects in it, not a
reinvention.

---

## The gates, written and unrun

`arima/mojo_only/arima_check.mojo`, fourteen of them. Each is listed in the
file's header with what it covers. The ones that carry the most weight:

    check_jones_device_equals_oracle    p = 1..4, AR and MA, forward and
                                        inverse, device vs host replay
    check_kalman_device_equals_oracle   eleven stages per cell -- Z R T RQ
                                        RQR P0 alpha0 pred vs loglike fc --
                                        across all seven orders
    check_lyapunov_solves_the_equation  DEVIATION 674 against the EQUATION,
                                        not only against its own oracle
    check_predict_device_equals_oracle  the entry point that had no oracle
    check_grad_reset_preserves_negative_zero   audit defect 2
    check_jones_contraction_is_visible         audit defect 1, made observable
    check_unit_root_guard_is_reached    the rare branch, entered deliberately
    check_fold_order_is_visible         the bitwise gates have teeth
    check_kalman_launch_invariant       block width, padding poison, batch
                                        composition, run twice

Two of these exist because `reached but inert` is a real failure mode here.
`check_unit_root_guard_is_reached` asserts `T[1]` IS `-0.99` on the
`ar2_unit` order AND that the transformed `phi_2` was not already `-0.99`,
so the guard and not the fixture is what wrote it.
`check_predict_sentinel_is_reached` asserts the order table contains at
least two orders with `res_offset > 0`, because on most orders the sentinel
loop runs zero times and a gate that never enters it proves nothing.

---

## HAND-OFF

### pixi task line (I do not own `pixi.toml`)

    check-arima = "mojo run -I . arima/mojo_only/arima_check.mojo"
    arima-card  = "mojo run -I . arima/arima_main.mojo"

### IDENTITY_PATHS row 58 (I do not own `IDENTITY_PATHS.md`)

Landed by the orchestrator as row 58 (55 is mamba's). The status cell needs
replacing now that the lane has run. Corrected text below.

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| 58 | arima: the batched Kalman filter log-likelihood, prediction and finite-difference gradient (`batched_kalman.cu`, `batched_arima.cu`, `jones_transform.cuh`) | every ARIMA kernel is `double` only and Metal has no Float64; `P0` is a cuBLAS batched `getrf`/`getri` whose association and pivot tie rule are closed and whose `info` the caller never reads; `RQR` and the Lyapunov solve are `cublasgemmStridedBatched`; `raft::tanh` / `raft::atanh` are the vendor's transcendentals (row 12); the undefined in-sample predictions are `nan("")`, whose payload differs per vendor in a recorded buffer; `d_y_p[0] = 0.0` is a cross-thread race in their lambda | DEVIATION 670 (Float32 device, Float64 host reference beside it); DEVIATION 673 (`F <= 0` refused by name); DEVIATION 677 (the same refusal at the diffuse steps, where an unchecked `1/F` reached the gain); DEVIATION 674 (the LU, both substitutions and both gemm shapes written out serial ascending through `identical_mul_add`, `info` raised by name); DEVIATION 675 (`tanh`/`atanh` through `identical_exp`/`identical_log`, ACCEPTED at ~23 ulp with the decision rule recorded); DEVIATION 676 (the sentinel is the constant `0x7fc00000`); their race not ported. `rd > 8`, `r > 5`, exog, confidence intervals, CSS and missing observations all refused by name | **ONE APPLE DEVICE, both modes, 2026-08-24.** Device == host oracle BITWISE on 10 orders x 11 stages, 0 cells differing, at `rd` up to 8 and `r = 5` (the 25x25 Lyapunov LU); `predict` and the gradient likewise; launch-invariant over block width, poisoned padding, batch composition and a repeat run. Card carries 229 stage records including four DECISION stages (`piv`, `guards`, `info_init`, `info_loop`) plus `Fs` and `P_final`. 11 of 11 sabotage arms run: 8 bite, 3 null. **Arm (i) is decision-only: 1 of 229 card tags, ZERO float stages** -- the model rewrite still happens and only the hashed decision notices. Arm (h) re-armed DEVIATION 674's pivot tie on a CONSTRUCTED exact tie and bites (6 of 229 tags), but `piv` and `P0` BOTH move, so `piv` is corroborating there and not decisive. Two arms remain null for recorded reasons: (c) is structural, (f) needs a second vendor. **NO SECOND VENDOR: the cross-vendor claim this row exists to make is UNTESTED.** |

### A hand-off to whoever owns the deviation ledger

**DEVIATIONS 680-686 ARE ALREADY TAKEN.** My brief allocated me 673-689,
but `isolation_forest/` (landed at `e973623`) uses 680, 681, 682, 683, 684,
685 and 686. My usable range was therefore 673-679 plus 687-689, and I
stayed inside it: this lane uses 673, 674, 675 and 676 only, with 670
inherited from the tsa half. 677, 678, 679, 687, 688 and 689 are free.

The grep that misses collisions is the singular one: this repo writes ranges
in the plural (`DEVIATIONS 680-686`), so `grep "DEVIATION 681"` finds
nothing while the number is very much in use. Check both spellings.

### A hand-off about this checkout

CORRECTED 2026-08-23, twice, because the first two accounts written here
were both wrong within the hour. What actually happened:

The 17 source files of this lane were staged for a commit of their own and
were swept into another lane's commit by a wildcard `git add`. A later
`git reset --soft HEAD~1` in the shared checkout then DROPPED that commit,
so the whole lane survived only as staged index entries while `01df837`
carried nothing but the five doc files. The orchestrator recovered the
source at `162ad1e`. Nothing was lost; nothing was safe either.

Two standing rules come out of it, and the second one is new:

  * `never git add -A in the shared checkout`. Stage explicit paths, and
    commit with `git commit -o <paths>` so the commit is limited to them no
    matter what a peer staged in between.
  * **NEVER REWRITE HISTORY HERE.** No `git reset` of any kind, no
    `--amend`, no `rebase`, no force push. Four resets appear in one
    night's reflog; they destroyed two isolation_forest commits and
    orphaned this entire lane. A reset drops whatever a PEER committed in
    the window. A bad commit is fixed FORWARD with a new commit. An ugly
    honest history beats a clean one that ate someone's work.

The earlier revisions of this paragraph are left described rather than
deleted because the failure mode is the point: in a shared checkout, what
you believe about your own commit can be false by the time you write it
down, and the only cure is to re-read `git log -- <your paths>` rather than
trust your memory of what you staged.

---

## OWED

Reordered 2026-08-24. Items marked WRITTEN exist in source and need only a
build; the rest still need someone to write them.

**Needs a compile slot, nothing else:**

1. **BUILD AND RUN EVERYTHING WRITTEN ON 2026-08-24.** Two new orders, two
   new plants, two new gates, a widened gate, DEVIATION 677, and four new
   card stages. None has been compiled. Expect signature errors around the
   `guards` buffer in particular: it threads a new argument through two
   kernel launches.
2. **Run sabotage (h)**, the pivot tie rule re-armed on `ar2_tie`, and
   record `piv` and `P0` SEPARATELY. If `piv` moves and `P0` does not, that
   is the result worth having and not a weak one: it is this lane's
   holtwinters `CRIT_ORDER`.
3. **Run sabotage (i)**, the guards decision bit, and confirm NO float stage
   moves. If one does, the edit was not confined to the decision.
4. **Re-run the widened `check_fold_order_is_visible`** and raise its floor
   from "at least two orders move something" to what is observed, per order.
   The docstring carries a falsifiable prediction: `ar1` must move zero,
   `rd = 2` few, `rd >= 4` a substantial fraction.
5. **Take the DEVIATION 675 end-to-end measurement** described in that
   section: give `kalman_host_f64` its own Float64 Jones transform so the
   reference spans the transform instead of starting downstream of it. The
   decision rule is fixed in advance there. UNWRITTEN, but small, and it
   moves no device bit.
6. **Replace the four derived bounds** using the table under "What the gates
   found". Do not do this without running.

**Needs a second vendor:**

7. ~~**A SECOND VENDOR.**~~ **DONE 2026-08-28, and this item said "entirely
   untested" for three days after it was.** `bench/results/e1/2026-08-28_203552-mojolearn-e2-amd/lanes/arima.identical.card` against `bench/results/e1/2026-08-28_162228-MacBook-Air-1-terrabyte/lanes/arima.identical.card`: 137 records each, 0 differing. `arima-card`
   emits the four decision stages too, so the diff separates "same decisions,
   different arithmetic" from "different decisions"; it needed neither,
   because nothing differed. **A THIRD VENDOR IS STILL OWED**: the NVIDIA
   legs of 2026-08-28 ran seven and nine lanes and arima was in neither
   `MOJOLEARN_E1_LANES`, so the gap is lane selection, not a failure.
8. **Sabotage (f) cannot be closed on Apple**, where `0.0/0.0` is already
   `0x7fc00000`. Only NVIDIA or AMD can test DEVIATION 676's claim.

**Still unwritten:**

9. **DEVIATION 677's branch is unreached.** It needs a fixture whose `P0`
   has a non-positive diffuse diagonal entry, and `kappa = 1e6` makes that
   hard by construction. Either construct one, as the pivot tie was
   constructed, or record permanently that the branch is defensive and
   unreachable from the ported surface.
10. **The Jones clamp mask** is derivable from `t_params` but is not
    recorded as a decision. If a cross-vendor diff ever shows `t_params`
    moving, the first question will be "did the clamp bind differently", and
    answering that from raw floats is slower than reading a bit.
11. **`numerical_stability`'s symmetrizing half is inert on two orders**
    (`arma11_k`, `arima111`): sabotage (b) moved nothing there. `arma44` may
    now cover it; check when (b) is re-run.
12. **Hand-off: `identical_expm1`** to whoever owns
    `mojo_only/numerics.mojo`, needed only if item 5's measurement says
    DEVIATION 675's tanh half matters. And `identical_log1p` for the atanh
    half the day the optimizer is ported, not before: that half is off the
    ported path today.
