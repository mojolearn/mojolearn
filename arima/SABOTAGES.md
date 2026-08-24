# arima sabotages

A green check proves nothing until the path is sabotaged and the gate is
watched to fail. Sabotage is REQUIRED here on every count: the expected
values are OUR OWN tally (the host oracle), the paths are all new, two of
the branches are rare, and DEVIATION 674 chose a bound and a tie rule that
theirs does not expose.

**ALL NINE OF THE ORIGINAL ARMS HAVE BEEN RUN**, 2026-08-23, Apple M4, IDENTICAL, at the
smallest shapes that reach the branches (n_obs 24, batch 6, fc_steps 3,
eight orders). Scoreboard:

    (a)  Kalman fold order, descending      BITES   31 stage comparisons
    (a') Kalman fold order, rotated         BITES   24
    (b)  symmetrization deleted             BITES   23, on 5 of 8 orders
    (c)  Jones inner k loop, descending     NULL    predicted, structural
    (c') Jones outer j loop, descending     BITES    8, jones gate only
    (d)  the one-rounding contraction       BITES   19, INTO the filter
    (e)  LU pivot tie rule > becomes >=     NULL    REACH FAILURE, not a pass
    (f)  computed NaN sentinel              NULL    expected, Apple cannot test
    (g)  gradient reset x + 0.0             BITES    7 of 24 cells -- but ONLY
                                                    after the gate was fixed;
                                                    it was INERT first

ADDED 2026-08-24, WRITTEN BUT NOT RUN (no-compile lane):

    (h)  the pivot tie rule, RE-ARMED       ?       (e) on a fixture that
                                                    actually has a tie
    (i)  the guards decision bit            ?       proves the decision stage
                                                    is load-bearing

Runs use `MOJOLEARN_ARIMA_SURVEY=1`, which makes the gate print
`SABOTAGE-MOVED` and CONTINUE instead of raising on the first differing
stage, so one run enumerates every stage an arm reached. The control is that
the CLEAN tree reports zero such lines. Three of the nine arms taught us
something we did not know, and two of those were about the gates rather than
the port.

Each is applied by hand to the named file, run under
`tools/with_identical_mode.sh`, and then REVERTED. Verify the revert with
`git diff -- arima/` printing nothing before moving on.

## The rule these enforce

`verify reach, not output` and `reached but inert`. A sabotage that moves no
bit is not automatically a failure of the sabotage: it can be a PROPERTY of
the fold, and when it is, the property gets written down here so nobody
reads that null as evidence again. What is never acceptable is a sabotage
whose result was not looked at.

---

## (a) the Kalman fold order

**Required by the brief.** `arima/ported/arima/batched_kalman.mojo`, `_mm`,
the inner `k` accumulation. This is the `MM_l` contraction that computes
both `TP = T*P` and `P = TP*L'`, twice per observation for every series, so
it feeds every stage downstream of step 3.

Replace

```mojo
            var acc = Float32(0.0)
            for k in range(n):
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc = ftz(identical_mul_add(a[i + k * n], bkj, acc))
            out_v[i + j * n] = acc
```

with

```mojo
            var acc = Float32(0.0)
            var k = n - 1
            while k >= 0:
                var bkj = b[j + k * n] if bT else b[k + j * n]
                acc = ftz(identical_mul_add(a[i + k * n], bkj, acc))
                k -= 1
            out_v[i + j * n] = acc
```

Only the DEVICE `_mm` is touched. The oracle's `_mm_host` in
`arima/mojo_only/kalman_oracle.mojo` is a separate spelling on purpose, so
it does not move with it and the two must disagree.

- MUST FAIL: `check_kalman_device_equals_oracle`, first at a `P0`-fed stage
  and then at every later stage of the same series.
- Expected first differing stage: `pred` or `vs` on the first order in the
  table (`arma11_k`), because `n_diff = 0` there makes `_Fs = l_P[0]` and
  the fold reaches the state in one step.
- OBSERVED 2026-08-23, Apple M4, IDENTICAL, survey mode: **BITES, 31 stage
  comparisons moved** across 7 of the 8 orders, plus 7 of 8 `predict`
  blocks. First differing cells, per order:

      arma11_k.pred     144 cells,  2 differ (cell 73: 0xbe265b04 vs ...06)
      arma11_k.vs       144 cells,  1 differ
      ma2.pred          144 cells,  8 differ    ma2.vs      4    ma2.fc     1
      arima111.pred     144 cells,  5 differ    .vs  3    .loglike  1
      arima212.pred     144 cells, 40 differ    .vs 32    .loglike  1  .fc 6
      ar2_unit.pred     144 cells, 43 differ    .vs 32    .loglike  1  .fc 4
      sarima_full.pred  144 cells, 79 differ    .vs 73    .loglike  3  .fc 6
      sarima_rd8        pred/vs/loglike/fc all moved

  TWO NULLS INSIDE THE BITE, and both are the point of having a per-stage
  card. `ar1` moved NOTHING: `rd = 1`, so the `k` loop has exactly one
  iteration and descending is ascending. That is structural immunity, not a
  gap. And `arma11_k.loglike` did NOT move although its `pred` and `vs` did:
  2 differing cells out of 144 were absorbed by the sum. An output-only gate
  would have called `arma11_k` inert here and been wrong about it.

## (a') the Kalman fold order, rotated rather than reversed

Same function, but start the `k` loop at `(i + j) % n` and wrap. A rotation
is the sabotage that the tsa lane found to be INERT on a halving tree, and
`_mm` is a serial chain rather than a tree, so here it should move. Running
it is how that difference gets recorded rather than assumed.

- MUST FAIL: `check_kalman_device_equals_oracle`.
- OBSERVED 2026-08-23: **BITES, 24 stage comparisons moved.** Per order:
  arma11_k 0, ar1 0, ma2 4, arima111 3, arima212 5, ar2_unit 4,
  sarima_full 5, sarima_rd8 3.

  So a rotation DOES move a serial chain, where the tsa lane found it inert
  on a halving tree. The difference is recorded rather than assumed: a tree
  pairs `{j, j+step}` at every level and a uniform rotation maps pairs to
  pairs; a serial `acc = fma(a, b, acc)` chain has no such symmetry. It bites
  on 6 of 8 orders rather than (a)'s 7, and `arma11_k` is the order it loses:
  at `rd = 2` a rotation by `(i+j) % 2` is the identity for half the cells.

## (b) the symmetrization

`arima/ported/arima/batched_kalman.mojo`, `_numerical_stability`. Drop the
`A = 0.5 (A + A')` pass and keep only `A_ii = |A_ii|`:

```mojo
    for i in range(n - 1):
        for j in range(i + 1, n):
            var s = ftz(a[j * n + i] + a[i * n + j])
            var new_val = ftz(Float32(0.5) * s)
            a[j * n + i] = new_val
            a[i * n + j] = new_val
```

becomes nothing (delete the block).

This is a REACH test as much as a fold test: `P` is very nearly symmetric
already by construction, so if this moves no bit the stabilization pass is
inert on the fixture and the fixture is too easy. Either result is
information and both go in the column.

- MUST FAIL: `check_kalman_device_equals_oracle` at `pred` / `vs` /
  `loglike`, on at least one order with `n_diff > 0` (where the diffuse
  `kappa = 1e6` makes `P` least symmetric).
- OBSERVED 2026-08-23: **BITES, 23 stage comparisons moved** -- but on only
  5 of the 8 orders. Per order: arma11_k 0, ar1 0, ma2 4, arima111 0,
  arima212 3, ar2_unit 5, sarima_full 5, sarima_rd8 4.

  `ar1` is structurally immune (`rd = 1` has no off-diagonal, so the pass is
  a no-op by construction). `arma11_k` and `arima111` are NOT immune and
  moved nothing anyway: at `rd = 2` and `rd = 3` this fixture keeps `P`
  symmetric enough that the stabilization has nothing to do. That is a REACH
  GAP, recorded as one: on those two orders `numerical_stability`'s
  symmetrizing half is on the path but inert, and nothing tonight gates it
  there.

## (c) the Jones transform

**Required by the brief.**
`arima/ported/timeSeries/jones_transform.mojo`,
`jones_transform_kernel`, the forward `transform` arm. Run the inner `k`
loop descending:

```mojo
        for j in range(1, parameter):
            var a = mine[j]
            for k in range(j):
                var prod = ftz(a * mine[j - k - 1])
                tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
```

becomes

```mojo
        for j in range(1, parameter):
            var a = mine[j]
            var k = j - 1
            while k >= 0:
                var prod = ftz(a * mine[j - k - 1])
                tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
                k -= 1
```

Note what this does and does not perturb. The `k` loop writes `tmp[k]` at
distinct `k`, so reversing it alone changes no VALUE; it is the `j` loop
that carries the dependence. **That makes this sabotage a deliberate
expected-null**, and it belongs here for exactly that reason: it documents
that the `k` loop is order-free, so a later reader does not mistake a
passing gate under a reversed `k` for a gate with no teeth. The sabotage
that must actually FAIL is (c') below.

- EXPECTED: no bit moves. If a bit DOES move, the audit missed an aliasing
  path between `tmp` and `mine` and that is a finding.
- OBSERVED 2026-08-23: **NULL, 0 stage comparisons moved. Exactly as
  predicted above, and predicted BEFORE the run for the stated reason** --
  the `k` loop writes `tmp[k]` at distinct `k`, so its order carries no
  dependence; the `j` loop does. Recorded as a property of the loop, not as
  evidence about the transform. (c') is the arm with teeth.

## (c') the Jones transform, the loop that carries the dependence

Same file and function, the OUTER `j` loop of `transform`, run descending
(`j` from `parameter - 1` down to `1`). Each `j` reads `mine[j]` and then
overwrites `mine[0..j)` from `tmp`, so the order is load-bearing.

- MUST FAIL: `check_jones_device_equals_oracle` at
  `arima.jones.fwd.p3.ar` and every `p >= 2` case; `p = 1` has no `j` loop
  and must NOT move, which is itself the per-branch reach evidence.
- MUST ALSO FAIL: `check_kalman_device_equals_oracle` at `R` and `T`, since
  the transformed parameters feed `reduced_polynomial`.
- OBSERVED 2026-08-23: **BITES, 8 stage comparisons moved**, all inside the
  standalone Jones gate:

      arima.jones.fwd.p3.ma   18 cells, 11 differ (cell 0: 0x3dc8be31 vs 0x3da45f03)
      arima.jones.fwd.p3.ar   18 cells, 11 differ
      arima.jones.fwd.p4.ma   24 cells, 18 differ
      arima.jones.fwd.p4.ar   24 cells, 18 differ

  `p = 1` and `p = 2` moved NOTHING and cannot: at `p = 1` there is no `j`
  loop, and at `p = 2` the `j` loop has a single iteration, so reversing it
  is the identity. Per-branch reach, confirmed rather than assumed.

  A REACH GAP THIS ARM EXPOSED, and it is the most useful thing (c') did.
  NO filter stage moved -- no `R`, no `T`, no `P0`. The reason is that every
  order in the table has `p <= 2` and `q <= 2`, so the Jones `j` recursion
  NEVER RUNS MORE THAN ONE ITERATION anywhere on the filter path. The
  multi-lag Jones recursion is gated only by the standalone gate, and the
  end-to-end path does not exercise it at all. OWED: an order with `p >= 3`
  or `q >= 3` (both are legal, `p, q <= 8` and `rd <= 8` permitting).

## (d) the contraction the audit corrected

Same file, both arms and both the kernel and the host replay. Put back the
ONE-ROUNDING spelling that the rung-0 audit removed:

```mojo
                var prod = ftz(a * mine[j - k - 1])
                tmp[k] = ftz(identical_mul_add(sign, prod, tmp[k]))
```

back to

```mojo
                tmp[k] = ftz(identical_mul_add(sign * a, mine[j - k - 1], tmp[k]))
```

This is the most important row in the file. If it does not fail, the
correction was cosmetic and `SEAMS.tsv`'s Jones rows are wrong.

`check_jones_contraction_is_visible` in the check is the host-side
companion: it computes both spellings on the same values and asserts they
differ on at least one cell, so the sabotage is known to be observable
BEFORE it is applied.

- MUST FAIL: `check_jones_device_equals_oracle` for `p >= 2` (at `p = 1`
  there is no accumulation and both spellings are trivially equal).
- OBSERVED 2026-08-23: **BITES, 19 stage comparisons moved, AND IT
  PROPAGATES INTO THE FILTER.** This is the row that matters.

      arima.jones.fwd.p2.ma   12 cells,  1 differ (cell 10: 0xbe8c0cf5 vs ...f6)
      arima.jones.fwd.p2.ar   12 cells,  1 differ
      arima.jones.fwd.p3.ma   18 cells,  3 differ
      arima.jones.fwd.p3.ar   18 cells,  3 differ
      arima.jones.fwd.p4.ma   24 cells,  6 differ
      arima.jones.fwd.p4.ar   24 cells,  5 differ
      ma2.R                   18 cells,  1 differ (cell 10: 0x3cddf8bd vs ...be)
      ma2.P0                  54 cells,  3 differ
      ma2.pred               144 cells,  9 differ
      arima212.R              24 cells,  1 differ
      arima212.T              96 cells,  1 differ
      arima212.P0             96 cells,  2 differ
      arima212.pred          144 cells,  5 differ

  So the audit correction was real arithmetic, not a cosmetic re-spelling: a
  single rounding difference in one Jones accumulation reaches `R`, then `T`,
  then `P0`, then every prediction. `p = 1` is null (no accumulation exists),
  which is correct.

  `check_jones_contraction_is_visible` is the companion that makes this
  falsifiable in advance, and it reports "of 160 accumulations, the two
  spellings differ on 27" on the clean tree. The sabotage was known to be
  observable before it was applied.

  ONE PER-BRANCH ODDITY WORTH RECORDING: `ar2_unit` has `p = 2` and so does
  have an accumulation, yet moved nothing. Its `phi_2` sits at the Jones
  clamp and, on these six hashed `ar[0]` values, both spellings round to the
  same bits. A 1-ulp seam is VALUE-dependent, and an order having the right
  shape does not guarantee it exercises the seam.

## (e) the LU pivot tie rule

`arima/ported/linalg/batched/matrix.mojo`, `lu_inverse`, the pivot search.
Change the strict comparison to non-strict:

```mojo
            if m > best_mag:
```

becomes

```mojo
            if m >= best_mag:
```

DEVIATION 674 CHOSE this rule, because cuBLAS's is not readable from
source. A chosen bound must be sabotaged (`sabotage-when-required`).

- EXPECTED: this only moves a bit when a column has a magnitude TIE, which
  a hashed fixture makes unlikely. Two outcomes, both recorded: if nothing
  moves, the tie branch is UNREACHED and the fixture needs a planted tie
  (an `ar2_unit` series with two equal transformed coefficients) before
  DEVIATION 674's tie rule can be claimed to be gated at all; if something
  moves, the column is named here.
- OBSERVED 2026-08-23: **NULL, 0 stage comparisons moved. This is a REACH
  FAILURE and is recorded as one, not as a pass.**

  The prediction written above was that a hashed fixture makes a magnitude
  TIE unlikely and that a null would mean the branch is unreached. That is
  what happened. **DEVIATION 674's pivot tie rule is therefore NOT GATED by
  anything**, on any order, and the `>` in `lu_inverse` could be `>=`
  tonight without a single gate noticing.

  This matters more than an ordinary null because the tie rule is OURS. cuBLAS
  `getrfBatched`'s rule is not readable from source, so this lane CHOSE strict
  `>` and wrote the choice down; a chosen bound that no gate can see is a
  chosen bound nobody is holding. OWED: plant an explicit magnitude tie in a
  column of `I - T (x) T` (two transformed coefficients of equal magnitude
  and opposite sign will do it) and re-run this arm.

## (f) the canonical NaN sentinel

`arima/ported/arima/batched_arima.mojo`, `in_sample_prediction_kernel`.
Replace the constant sentinel with a computed one:

```mojo
        d_y_p.unsafe_store(bid * ld + i, bitcast[DType.float32](CANONICAL_NAN_BITS))
```

becomes

```mojo
        d_y_p.unsafe_store(bid * ld + i, Float32(0.0) / Float32(0.0))
```

DEVIATION 676's whole claim is that the recorded bytes are the same on every
vendor. On Apple, `0.0/0.0` happens to be `0x7fc00000` too, so this is
expected to be INERT on this box and to fail only on NVIDIA
(`0x7fffffff`). That makes it a sabotage that CANNOT be closed on one
vendor, and it is written here so the second-vendor run knows to try it.

- EXPECTED on Apple: no bit moves, and `check_predict_device_equals_oracle`
  still passes. That is the null, not a pass.
- EXPECTED on NVIDIA: `check_predict_device_equals_oracle` FAILS at the
  sentinel assertion with the observed bits printed.
- OBSERVED 2026-08-23, and this row changed the gate. **First run: NULL, 0
  stage comparisons moved. The gate written to catch this defect did not
  catch it.**

  Why it was inert: the gate planted `-0.0` in parameter 0 and looked for the
  damage in the GRADIENT of the last parameter. Parameter 0 of that order is
  `mu`, and a `-0.0` in `mu` never reaches the log-likelihood -- `alpha0[i] =
  ImT_inv[i] * mu` carries the sign of zero, but the filter's first act is
  `pred = 0.0 + alpha[0]`, and `0.0 + (-0.0)` is `+0.0`. The corruption was
  real and the observable was gone one operation later. `reached but inert`.

  THE GATE WAS CHANGED, not the verdict softened. `batched_loglike_grad` now
  takes `d_x_pert` as the CALLER'S scratch, which is also what theirs is
  (`arima_mem.x_pert`, `batched_arima.cu:534`), so a gate can read the buffer
  back and assert directly that it returned to `d_x` bitwise. Re-run against
  that assertion:

      x_pert after the last reset: 7 of 24 cells differ from d_x (must be 0)
        series 0 parameter 0: x_pert 0x00000000 vs d_x 0x80000000
        ... all six series ...

  **BITES.** Seven cells, not six: the six planted `mu` values plus series 0's
  `ma[0]`, which `arima_params_fixture` plants as `-0.0` on its own. The
  fixture's negative zero was doing work nobody had checked. The old indirect
  assertion is kept beside the new one, labelled inert, so the lesson stays
  in the file.

## (g) the gradient reset

`arima/ported/arima/batched_arima.mojo`. Put back the defect the audit
found: call `perturb_kernel` with `h = 0` in place of `reset_param_kernel`.

- MUST FAIL: `check_grad_reset_preserves_negative_zero`, at the gradient of
  the LAST parameter, on every series (all six have `-0.0` planted in
  parameter 0 by that gate).
- OBSERVED 2026-08-23, Apple M4: **NULL, 0 stage comparisons moved -- the
  expected null, and it closes nothing.** On Apple `0.0/0.0` produces
  `0x7fc00000`, the same bits the constant writes, so the sabotage is
  invisible here by construction. DEVIATION 676's entire claim is about a
  payload that differs BETWEEN vendors, and one vendor cannot test it.
  STILL OWED, and only NVIDIA or AMD can discharge it.

## (h) the LU pivot tie rule, ON A FIXTURE THAT HAS A TIE

**This is (e) re-armed.** Same edit, `arima/ported/linalg/batched/matrix.mojo`,
`lu_inverse`, `if m > best_mag:` becomes `if m >= best_mag:`. What changed is
not the sabotage but the fixture: the `ar2_tie` order now plants an EXACT
magnitude tie in column 0 of `I - T (x) T`, maximal by a 0.20 margin, so the
pivot loop actually meets the comparison this arm perturbs.
`check_lu_pivot_tie_is_reached` asserts the tie is present and maximal in the
matrix the device built, so the arm is known to be armed before it is fired.

- MUST FAIL: at minimum `ar2_tie.piv`, the LU permutation decision stage.
- WATCH `P0` SEPARATELY. Swapping which of two EQUAL-magnitude rows becomes
  the pivot changes the permutation for certain, but it does not have to
  change `P0`: the two eliminations can produce the same inverse. If `piv`
  moves and `P0` does not, that is not a weak result, it is the whole reason
  `piv` was added as a stage. It is this lane's holtwinters `CRIT_ORDER`: a
  decision-only sabotage that no float comparison anywhere can see.
- OBSERVED: _(not run: written 2026-08-24 in a no-compile lane)_

## (i) the guards decision stage

`arima/ported/arima/batched_kalman.mojo`. Delete the `guard_bits |= UInt8(1)`
line in `init_batched_kalman_matrices_kernel` (leaving the `T[1] = -0.99`
rewrite itself in place), so the guard still fires but stops SAYING it fired.

The point is to prove the stage is load-bearing rather than decorative. The
rewrite still happens, so every float stage is bit-identical and no
value-comparing gate anywhere can notice.

- MUST FAIL: `check_guard_decisions_are_recorded`, `ar2_unit` arm, all six
  series, and `ar2_unit.guards` on the card.
- MUST NOT FAIL: any float stage. If a float stage moves, the edit was not
  confined to the decision bit.
- OBSERVED: _(not run: written 2026-08-24 in a no-compile lane)_

---

## Revert check

After every row:

```
git diff -- arima/
```

must print nothing. `the shared checkout's mode flip` is the standing
reason: a sabotage left behind gives correctly-labelled measurements of the
wrong arm.
