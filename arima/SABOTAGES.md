# arima sabotages

A green check proves nothing until the path is sabotaged and the gate is
watched to fail. Sabotage is REQUIRED here on every count: the expected
values are OUR OWN tally (the host oracle), the paths are all new, two of
the branches are rare, and DEVIATION 674 chose a bound and a tie rule that
theirs does not expose.

**NONE OF THESE HAS BEEN RUN.** This session was made a no-compile lane
after the machine load spike, so every row below is a written sabotage with
an EMPTY observation column. A compile slot runs them and fills the column
in; a row whose observation is still empty is a claim nobody has earned.

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
- OBSERVED: _(not run)_

## (a') the Kalman fold order, rotated rather than reversed

Same function, but start the `k` loop at `(i + j) % n` and wrap. A rotation
is the sabotage that the tsa lane found to be INERT on a halving tree, and
`_mm` is a serial chain rather than a tree, so here it should move. Running
it is how that difference gets recorded rather than assumed.

- MUST FAIL: `check_kalman_device_equals_oracle`.
- OBSERVED: _(not run)_

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
- OBSERVED: _(not run)_

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
- OBSERVED: _(not run)_

## (c') the Jones transform, the loop that carries the dependence

Same file and function, the OUTER `j` loop of `transform`, run descending
(`j` from `parameter - 1` down to `1`). Each `j` reads `mine[j]` and then
overwrites `mine[0..j)` from `tmp`, so the order is load-bearing.

- MUST FAIL: `check_jones_device_equals_oracle` at
  `arima.jones.fwd.p3.ar` and every `p >= 2` case; `p = 1` has no `j` loop
  and must NOT move, which is itself the per-branch reach evidence.
- MUST ALSO FAIL: `check_kalman_device_equals_oracle` at `R` and `T`, since
  the transformed parameters feed `reduced_polynomial`.
- OBSERVED: _(not run)_

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
- OBSERVED: _(not run)_

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
- OBSERVED: _(not run)_

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
- OBSERVED: _(not run, and cannot be closed on Apple alone)_

## (g) the gradient reset

`arima/ported/arima/batched_arima.mojo`. Put back the defect the audit
found: call `perturb_kernel` with `h = 0` in place of `reset_param_kernel`.

- MUST FAIL: `check_grad_reset_preserves_negative_zero`, at the gradient of
  the LAST parameter, on every series (all six have `-0.0` planted in
  parameter 0 by that gate).
- OBSERVED: _(not run)_

---

## Revert check

After every row:

```
git diff -- arima/
```

must print nothing. `the shared checkout's mode flip` is the standing
reason: a sabotage left behind gives correctly-labelled measurements of the
wrong arm.
