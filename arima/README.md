# arima: cuML's batched ARIMA Kalman filter

Rung 2 of the time series ladder. Rung 1 (the KPSS stationarity test and
auto_arima's `d` loop, DEVIATIONS 671-672) landed in `tsa/`.

**This README covers `arima/` AND carries the lane's shared material:
DEVIATION 670, the hand-off list, and the row text for both directories.
`tsa/README.md` points here for all three.** COPY, DO NOT IMPROVE.

---

## Status: SOURCE ONLY. Nothing here has been gated. Read this first.

    the ported tree parses                     YES (verified 2026-08-23)
    arima/mojo_only/{fixtures,arima_check}     NOT VERIFIED TO PARSE
    any gate has run                           NO. NOT ONCE.
    any sabotage has run                       NO
    a second vendor                            NO
    an identity card produced                  NO

This lane became a no-compile lane part-way through its round (seven agents
were invoking the Mojo compiler at once and the machine went down). What
landed is source and an audit, and both are worth having on their own; what
did NOT land is every number. Treat every claim below as derived from
READING cuML's source against ours, which is exactly what it is, and treat
the OWED list at the foot as the honest remainder.

Any sentence in a docstring under `arima/` that says "MEASURED" and was
written before 2026-08-23 is INHERITED FROM THE AGENT THAT DIED AND IS
UNSUBSTANTIATED. No run produced it. They are left in place rather than
deleted so the compile slot can check each one against a real run and
either earn it or strike it.

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
    MOJOLEARN_IDENTITY_TRACE=/tmp/arima.card tools/with_identical_mode.sh \
        pixi run mojo run -I . arima/mojo_only/arima_check.mojo
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

**A cost this deviation carries and nobody has priced.** `(e^x - 1)` cancels
for small `|x|`: at `x = 1e-4` the absolute error in `e^x` is about
`eps = 1.2e-7` while `e^x - 1` is about `1e-4`, so roughly ten bits are
lost, and `tanh(x/2)` is exactly where the optimizer spends its time near
convergence. `check_jones_device_equals_oracle` reports the round-trip
`inverse(forward(x))` relative error for this reason. It has not run, so
the number in that report does not exist yet. If it turns out large, the
fix is a numbered replacement identity (`expm1(x)/(expm1(x)+2)`) and NOT a
quiet edit.

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

`batched_arima.cu:583-586` resets with `d_x_pert[N*bid+i] = d_x[N*bid+i]`.
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

### IDENTITY_PATHS row (I do not own `IDENTITY_PATHS.md`)

Rung 1's row is 49 (`tsa`). This is the next free number at the time of
writing; renumber as needed.

| n | path | what is vendor-dependent in their spelling | what we did | status |
|---|---|---|---|---|
| 55 | arima: the batched Kalman filter log-likelihood, prediction and finite-difference gradient (`batched_kalman.cu`, `batched_arima.cu`, `jones_transform.cuh`) | every ARIMA kernel is `double` only and Metal has no Float64; `P0` is a cuBLAS batched `getrf`/`getri` whose association and pivot tie rule are closed and whose `info` the caller never reads; `RQR` and the Lyapunov solve are `cublasgemmStridedBatched`; `raft::tanh` / `raft::atanh` are the vendor's transcendentals (row 12); the undefined in-sample predictions are `nan("")`, whose payload differs per vendor in a recorded buffer; `d_y_p[0] = 0.0` is a cross-thread race in their lambda | DEVIATION 670 (Float32 device, Float64 host reference beside it); DEVIATION 673 (`F <= 0` refused by name, not carried into `log`); DEVIATION 674 (the LU, both substitutions and both gemm shapes written out serial ascending through `identical_mul_add`, `info` raised by name); DEVIATION 675 (`tanh`/`atanh` through `identical_exp`/`identical_log`); DEVIATION 676 (the sentinel is the constant `0x7fc00000`, never computed); their race not ported. `rd > 8`, `r > 5`, exog, confidence intervals, CSS and missing observations all refused by name | **SOURCE ONLY. No gate has run, on any vendor. Do not enter this row as a result.** |

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

The 17 source files of this lane were staged for their own commit and were
swept into `561006a` ("spectral: the dead agent's untracked pile LANDS") by
another lane running a wildcard `git add`. The content landed intact and is
byte-identical to what was staged, but the arima source now sits in history
under a commit message about spectral clustering. The status block that
should have accompanied it is this README and the commit that carries it.
`never git add -A in the shared checkout` is the standing rule and it was
not mine that broke it, but the cost is real and is recorded here so the
history is readable later.

---

## OWED. Everything below needs a compile slot.

1. **Build `arima/mojo_only/fixtures.mojo` and `arima_check.mojo`.** Three
   parse errors were found and fixed (a by-value `IdentityTrace` where
   `record_device` needs `mut`, `case` used as a loop variable when it is
   reserved, `arma11_series` called with five arguments where it takes six).
   The rebuild after the third fix never completed. Assume more.
2. **Run the check in both modes** and paste the output into this README
   under a "What the gates found" heading, in tsa's shape.
3. **Run every row of `SABOTAGES.md`** and fill in its OBSERVED column,
   including the two expected nulls, (c) and (f). A row still empty is a
   claim nobody has earned.
4. **Replace every DERIVED bound with an observed one**: the Float64
   tolerances in `check_kalman_matches_float64`, the `5e-3` Lyapunov
   residual bound, and the `1e-2` Jones round-trip bound. All three were
   chosen by reasoning, and a chosen bound must be sabotaged.
5. **Price DEVIATION 675's cancellation.** The round-trip number does not
   exist yet. If it is large, replace the identity with a numbered one.
6. **Write `arima/arima_main.mojo`**, the card driver, in `tsa_main.mojo`'s
   shape, and produce a card. The check records stages into an
   `IdentityTrace` already, but a separate small driver is what the
   cross-vendor diff wants.
7. **A second vendor.** Nothing here has run anywhere but nowhere.
8. **Check whether the `it < n_diff` steps can produce a non-finite
   `_1_Fs`** on any fixture, per DEVIATION 673's recorded hole.
9. **Decide whether the LU tie branch is reachable at all** (sabotage (e)).
   If not, plant a tie in the fixture before claiming DEVIATION 674's tie
   rule is gated.
