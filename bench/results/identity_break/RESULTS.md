# identity_break: trying to break cross-vendor bit-identity on every estimator

`tools/identity_break.py` fits all 28 public lanes (every estimator,
`linalg.matmul`, the metrics) under the `identical` tier on nine fixtures,
twice per cell, and writes one fingerprint per (lane, fixture, output part).
Three vendors' columns are then diffed cell by cell with `--diff`. The
fixtures are hostile on purpose: integer ties, hash-derived values with no
structure, column magnitudes spanning 1e-4 to 1e4, float32 subnormals, the
same subnormals flushed to signed zero, duplicated rows beside a constant and
an all-zero column, an n = 12345 by d = 17 shape nothing divides evenly, and
all-negative data, plus the stability harness's own fixture as the control.

Every column below was handed hash-identical fixture bytes; `--diff` checks
that and refuses to blame the library for a fixture the vendors did not share.

## Verdict, 2026-08-29

| columns | commit | cells | bit-identical | divergent | moved | refused |
|---|---|---|---|---|---|---|
| Apple M4 (Metal) vs AMD MI325X (HIP) | `e616906e` | 252 | **252** | 0 | 0 | 0 |
| Apple M4 vs NVIDIA | H100 `e16b948b` / 4090 `07ca87ab` | see below (OWED, partial) | | | | |

The AMD column is `bench/results/e1/2026-08-29_150049-mojolearn-e2-amd`; the
Apple column is `apple-m4.identical.json` here; the diff is
`diff.apple-amd.txt`. The same AMD leg's 16-lane stability arm under
`identical` was STABLE 16 of 16.

## What it took to get there, in order

**1. The first diff read 66 DIVERGENT cells, and 64 of them were the
harness's fault.** Every regression lane diverged on every fixture. The
regression target was `X @ w` in float32, which goes through the HOST BLAS,
and Accelerate on the Mac and OpenBLAS on the droplet accumulate in
different orders: 13,876 of 20,000 targets differed on ONE Mac between
`X @ w` and a fixed-order sum. The AMD box had fitted a different target;
the library had never been asked the same question twice. The target is now
a fixed-order elementwise sum, and the fixture hashes travel in the JSON.
A cross-vendor probe must hand every vendor the same bytes, and must prove
it did.

**2. Two cells survived that fix and they were real: `rf-clf/denormal` and
`iforest/denormal`, joined by `rf-reg/denormal` once the target was fixed.**
All three in the cuML-port forest family, all on subnormal input, and all
three agreed on the `denormal_ftz` twin. Part-level hashes localized it: on
iforest, Apple's subnormal answer equaled the flushed answer and AMD's did
not, so AMD's scoring path read the subnormals and Apple's flushed them; on
the random forest both vendors read the subnormals and read them
differently. ExtraTrees, gbdt, k-means, k-NN, PCA, DBSCAN, the linear
surfaces and the SVM all agreed on the same subnormal bytes, so the `ftz`
pin (IDENTITY_PATHS row 10) existed and worked, and these two lanes did not
carry it on the paths that mattered.

**DEVIATION 1942 closed it** (commit `e616906e`): the random forest flushes
its device copy of X once before the quantile pass (every stored threshold
is a gathered X value, so that flushes the thresholds too) and its HOST
predict walk compares the flushed feature (the host honors subnormals on
every vendor, which is why Apple's own `denormal` cell differed from its
`denormal_ftz` cell); the isolation forest flushes in `_upload_f32`, the one
seam every feature value crosses onto the device. FAST is untouched at all
three sites. Reach was shown by sabotage on Apple for both random forest
seams (making either flush inert moves both rf cells away from the
`denormal_ftz` answer), and for the iforest seam, which is bit-inert on
Apple's flush-to-zero hardware exactly as the `ftz` docstring predicts, with
a four-pattern upload and readback probe under the identical build
(0x00000001 to 0, 0x80000001 to negative zero, 0x007fffff to 0, 1.0
unchanged; the FAST build passes all four through). The `base` hashes of all
four forest lanes are byte-identical before and after.

**3. The probe's labels were part of the problem too.** `y_clf` came from
columns 0 and 1, which the denormal fixture perturbs, so the `denormal` and
`denormal_ftz` twins carried 2,493 different labels and the classifier twins
could never be compared. Labels now come from columns 3 and 4, which no
fixture touches. After that, `svc/denormal_ftz` stopped refusing (its first
2,000 rows had been a single class).

## The NVIDIA column, and the RTX 4090 hangs

Two RunPod RTX 4090 legs on 2026-08-29 (drivers 570.195.03 and 550.127.05,
running through the `MODULAR_NVPTX_COMPILER_PATH` escape) hung in every
tier at the first isolation forest fit, while ExtraTrees and RandomForest
fitted in about a second on the same box. A third 4090 leg on a driver 580
host with no escape hung at the same place, so the escape was NOT the
cause. The cause was a DeviceContext lifetime defect in the isolation
forest binding: the estimator opened a second `DeviceContext` beside the
caller's, and on sm_89 the two contexts deadlocked at teardown.
**DEVIATION 1944** (commit `07ca87ab`) makes `IsolationForestModel` use the
caller's context; the fixed binding fits clean on the 4090 in one process.

What the 4090 legs then found is a second defect, and leg 2
(`bench/results/e1/2026-08-29_165644-runpod-nvidia/diag/`) localized it to
the PYTHON BINDING LAYER: in pure Mojo, two isolation forest fits on two
contexts, on one context, and sequentially all PASS (`d3_if_T1..T3`), two
RandomForest fits on two contexts PASS (`d3_rf_two_ctx`), and only the
pre-1944 shape (a model built on a second context) hangs (`d3_if_T4`).
Through the Python bindings on the same box, `IsolationForest.fit` returns
in 1 s and the next call, `score_samples`, never returns; `RandomForest.fit`
hangs inside the binding on its first default-size fit; ExtraTrees twice in
one process passes. It does not reproduce on the H100 or on Apple or AMD.

**DEVIATION 1946 names that defect, and it is NOT the `GILReleased` block
this file blamed on 2026-08-29.** It is the DeviceContext lifetime again,
one call later. Mojo destroys a value at its LAST USE. Both hanging paths
create their context INSIDE the call and then release device-backed values
AFTER the context's last use, so the context is destroyed first and its own
buffers are freed against a context that has already gone:

* `bindings/_mojolearn_rf.mojo` -- `ctx`'s last use was `ctx.synchronize()`,
  and `_ = dx^ _ = dy^ _ = dsw^ _ = hx^ _ = hy^` then freed five buffers,
  two of them PINNED HOST allocations, behind it.
* `isolation_forest/estimator.mojo::iforest_run_host` -- `_ = est^` freed
  the model's EIGHT `DeviceBuffer`s one line after `ctx`'s last use.

The passing bindings do not have the shape: in `_mojolearn_trees.mojo` and
`_mojolearn_gbdt.mojo` the context's last use IS the host call, and every
device value is created and destroyed inside it. That also explains the one
result that looked like an alibi for the bindings: `rf_ctx_probe.mojo`
passed two fits in one process because its `one_fit` takes `ctx` as a
BORROWED argument, so `main` owns the context and every buffer is freed
while it is still alive. The probe accidentally fixed the ordering it was
written to reproduce. It is the same class as DEVIATION 1944 -- a buffer
freed against a context that is not the live one -- which is why one fix
moved the first fit and not the second call.

The fix is that the context dies LAST: an explicit `_ = ctx^` after every
release, at both sites above and at the 33 other library entry points that
had the same ordering (`metrics`, `kde`, `tsa`, `svm`, `mixture`,
`resample`, `cholesky`, `gaussian_process`, `kernel_methods`). It cannot
move a number: nothing is computed after the last `synchronize()`.
`ensemble/mojo_only/rf_ctx_order_probe.mojo` is the ordering alone, both
arms, in 60 lines with no forest and no Python -- that is the on-box
discriminator the next leg should run FIRST.

VERIFIED so far, all on Apple M4: every touched file compiles, the five
rebuilt bindings (`rf`, `svm`, `metrics`, `tsa`, `estimators`) pass their
build smoke gates, whose RandomForest gate alone does eight fits in one
process, two sequential `iforest_run_host` calls return equal values, and
both probe arms print DONE. NOT VERIFIED: that this fixes the 4090. Apple,
AMD and the H100 never showed the defect, so only a 4090 leg can confirm
it, and none has run since the fix. See RUN OWED below.

The probe fits every cell twice, so on a pre-1946 4090 the `rf-clf` cell
cannot complete and the lanes after it are ABSENT, not clean. The probes
are `tools/diag/rtx4090_hang.sh`, `ensemble/mojo_only/rf_ctx_probe.mojo`,
`ensemble/mojo_only/rf_ctx_order_probe.mojo` and
`isolation_forest/mojo_only/if_ctx_probe.mojo`; the verdict file is
`diag/verdicts.txt` in that leg.

The NVIDIA silicon ledger, per column, is therefore: H100 (driver
580.126.09, `E3` round 13, `bench/results/e1/2026-08-28_131651-runpod-nvidia`)
123 stages bit-identical to Apple and AMD, and a 12-lane `identical`
stability arm STABLE on the 4090 at `07ca87ab`. The full 28-lane by 9-fixture
NVIDIA column of this probe is OWED: DEVIATION 1946 is a fix nobody has run
on the silicon that showed the defect, so the column waits on one 4090 leg
(or on an H100, where nothing hung in the first place). OWED is not FAILED
([runpod-failure-is-not-invalidation] in the orchestrator's memory): nothing
above this section moved.

The probe now registers iforest last, rewrites its JSON after every lane,
and the leg wraps it in `timeout`, so a hang can never again take a whole
column with it.

## Reading the table

`IDENTICAL xN` means N columns agree on every output part of that cell.
`DIVERGENT` names the part that disagrees and the parts that agree.
`MOVED` means one column disagreed with itself across its two fits, which
under `identical` is a defect on that box before it is a cross-vendor
question. `REFUSED` carries the exception text. A column marked
`complete=false` was killed mid-run and its later lanes are ABSENT, not
clean.

## RUN OWED

Nothing below has run. Each line names the box it needs and the command.

1. **The DEVIATION 1946 discriminator, one RTX 4090, ~5 minutes.** Before
   anything else, and it needs no Python and no built bindings:

       mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_BAD=1 \
           ensemble/mojo_only/rf_ctx_order_probe.mojo -o /tmp/order_bad
       mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D ORDER_GOOD=1 \
           ensemble/mojo_only/rf_ctx_order_probe.mojo -o /tmp/order_good
       timeout 300 /tmp/order_bad ; echo "bad rc=$?"
       timeout 300 /tmp/order_good ; echo "good rc=$?"

   Expected if 1946 is the whole answer: BAD hangs (rc 124) and GOOD prints
   `orderprobe DONE`. If BOTH print DONE, the ordering is not the poison on
   that box, 1946 is a correctness fix and not the cure, and the hunt goes
   back to the `GILReleased` block -- write that in `diag/verdicts.txt`
   rather than assuming this file's thesis.

2. **The binding verification leg, same box.** Rebuild the identical set and
   run `iforest,rf-clf` stability 6/6 plus `identity_break --lanes rf-clf`;
   the hashes must equal `apple-m4.identical.json`
   (`c2867a6751f32d49`, `2999ccbacdecfa28` on base and hashed).
   `P9_BINDINGS` three bindings, `P9_LANES="iforest,rf-clf"`, `P9_DIAG`.

3. **The full NVIDIA column.** All ten bindings, `P9_BREAK=1`; copy
   `stability/identity_break.identical.json|.txt` to
   `bench/results/identity_break/nvidia-rtx4090.identical.*` and run the
   three-way `--diff` into `diff.apple-amd-nvidia.txt`.

4. **Apple, AMD and H100 re-verification is NOT owed by construction** --
   1946 adds no arithmetic and moves no launch -- but the next scheduled leg
   on each should confirm its lane cards unchanged, as a free check.
