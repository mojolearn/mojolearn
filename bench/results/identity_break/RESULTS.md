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

What the 4090 legs then found is a second, separate defect that is still
OPEN: after one `RandomForest.fit` in a process, the NEXT GPU call in that
process hangs on the 4090 (any lane, any tier). It does not reproduce on
the H100 or on Apple or AMD. The probe fits every cell twice, so on a 4090
the `rf-clf` cell cannot complete and the lanes after it are ABSENT, not
clean. The diag probes for it live in `tools/diag/rtx4090_hang.sh`,
`ensemble/mojo_only/rf_ctx_probe.mojo` and
`isolation_forest/mojo_only/if_ctx_probe.mojo`.

The NVIDIA silicon ledger, per column, is therefore: H100 (driver
580.126.09, `E3` round 13, `bench/results/e1/2026-08-28_131651-runpod-nvidia`)
123 stages bit-identical to Apple and AMD, and a 12-lane `identical`
stability arm STABLE on the 4090 at `07ca87ab`. The full 28-lane by 9-fixture
NVIDIA column of this probe is OWED until the RF second-fit hang is fixed
or the leg runs on an H100. OWED is not FAILED
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
