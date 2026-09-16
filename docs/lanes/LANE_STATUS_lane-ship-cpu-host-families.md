# Lane status: lane/ship-cpu-host-families (2026-09-16)

Andrew's decision: **ship the CPU stuff**, so a user can verify nearly
everything we claim rather than a fifth of it.

**Measured: `python -m mojolearn verify --all` on a CPU-only install now
checks 122 lanes, against 39 before.** Both numbers come from a run on this
Mac, not from arithmetic over a list.

## What changed

Every host family ships. The manifest declared thirty-two and shipped sixteen;
the sixteen held back were the CPU TRAINING families, excluded by the
"inference only, CPU training internal" boundary of
docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md. That boundary was about what
a user may TRAIN with. It was also, as a side effect nobody chose, deciding
what a user may CHECK: a lane whose host binding is not in the wheel cannot be
re-run on the machine it was installed on, whatever the reference table says.

**What did not change is the run-time boundary.** An ordinary `fit` on a
CPU-only install still refuses by name and still tells the user to train on a
GPU and load the saved model (`python/mojolearn/_cpu_reference.py`). These
bindings answer the verifier, which fits inside `reference_training()`.
Shipping the binaries widened what can be CHECKED and changed nothing about
what the library will train. `test_cpu_inference_boundary.py` passes unchanged,
which is the check that says so.

## The measured run

`verify --all` selects lanes from `host_surface.public_reference_lanes()` when
the vendor read-back is `cpu`, so widening the public set IS the deliverable
and the run is what earns it. The rule inherited from
lane/expose-inference-surface is kept: a lane is public only because a run on
a CPU-only install was WATCHED to read IDENTICAL for it.

| | before | after |
|---|---|---|
| host families in the wheel | 16 of 32 | **32 of 32** |
| lanes `verify --all` selects | 39 | **122** |
| lanes it actually compares against a reference | 38 | **121** |

The one lane in each column that is selected without being compared is
`tokenizer`, whose parts read OWED because no committed record carries its
hashes yet (the GPT-2 table left the tree and the lane now loads the synthetic
vocabulary the harness trains itself). It was public before this lane widened
the set, and narrowing it would be a regression, so it stays and the note is
in `PUBLIC_HOST_ONLY_LANES`.

**The after arm.** 131 candidate lanes x 9 fixtures, 4,716 cell parts:

```
IDENTICAL 3855    DIVERGENT 0    REFUSED 0    OWED 393    N/A 468
```

One process per chunk of eight lanes, seventeen chunks plus the portable
models, each through the shared Mac slot helper at `nice -n 19` with every
thread and job knob at 1. 1,488 s of wall time. One process per chunk was not
tidiness: this lane's run had already been killed twice mid-flight, once by
the Mac crash and once by a session rate limit, and chunking made a third
interruption cost one chunk rather than the whole run. Every chunk's JSON
lands on disk as it finishes, under
`~/mojolearn-evidence/ship-cpu-host-families/run-2026-09-16/chunks/`.

**The confirmation**, after the nine were held back, is the shipped command
itself in ONE process on the final set, so the headline number is printed by
`verify --all` rather than computed from the manifest:

```
# verify --all: cpu (cpu), 122 lanes x 1 fixtures, harness checkout, ...
RESULT: VERIFIED (verified 435 of 496 cell parts (0 divergent, 13 owed, 0 refused, 48 n/a)). 118.4s. exit 0
```

**The before arm** is the control, on the same machine, the same harness and
the same fixtures, with origin/main's package and a host directory carrying
exactly origin/main's SIXTEEN `wheel_bindings()`. That is what a CPU-only
install gave a user before this lane, so the 39 is measured here rather than
inherited from another lane's write-up. Evidence in
`~/mojolearn-evidence/ship-cpu-host-families/run-2026-09-16/before/`.

**The verdict these numbers are taken with matters.** On `release/0.8.6`,
`_verify_all.verdict()` returned VERIFIED and exit 0 as soon as ONE part read
IDENTICAL, before it looked at REFUSED, which is a verdict that cannot fail.
Main's 87085a5eb fixes it, any refusal now reads INCOMPLETE and exits 4, and
that fix is merged into this branch. It earned itself here: the first before
arm read INCOMPLETE, exit 4, on 36 REFUSED parts, which is how the stale
binding below was caught at all. Under the old verdict that same run would
have read VERIFIED and the stale binding would have gone unnoticed. The count
above is lanes actually CHECKED rather than lanes that merely failed to refuse
loudly.

### A stale binding the run caught, and what it cost

The rescued host bindings were built at 07:55 from the shared checkout at
`5596c3f08`, which did not yet carry `87085a5eb`. Two families were therefore
stale against this branch's HEAD, `tsa` and `forecast`, both of which that
commit changed when `kpss_test_binding` moved into the shared
`bindings/kpss_host_test.mojo`. It showed up as the before arm reading `kpss`
REFUSED with "the host binding `_mojolearn_forecast_host` is built but exports
no kpss_test".

That looked like a real gap in main's public set and it was not. Isolating it
took two runs: main's package with all thirty-two bindings ran `kpss` fine,
and this branch's package with main's sixteen refused it, so the variable was
the host directory and not the package. `strings` on the binary settled it,
`kpss_test` is absent from the rescued `forecast` binding and present in the
rescued `tsa` one. Both were rebuilt from this worktree at HEAD, both now
carry it, and every lane those two bindings can touch was re-run against the
rebuilt set (`arima`, `arima-011`, `arima-seasonal-c`,
`holtwinters-multiplicative`, `kpss`): 153 of 180 parts IDENTICAL, 0
DIVERGENT, 0 REFUSED. The before arm was then re-run with a correct `forecast`
binding, and `kpss` reads IDENTICAL on origin/main's sixteen as it should.

The general form is worth keeping. A binding built from a checkout that has
since moved is not evidence about the current tree, and the failure it
produces reads exactly like a real defect in someone else's lane.

## Lanes promoted, and lanes held

83 lanes promoted, 0 dropped. Grouped by the family whose binding now ships:

| family | n | lanes |
|---|---|---|
| gbdt | 14 | cross-val, gbdt-adapter-clf, gbdt-adapter-reg, gbdt-categorical-ctr, gbdt-depthwise, gbdt-exact-mae, gbdt-feature-freq, gbdt-lossguide, gbdt-multiclass, gbdt-onevsall, gbdt-ordered-rmse, gbdt-pointwise-l2-bayesian-eval, gbdt-rmse, gbdt-symmetric |
| core | 11 | knn-chebyshev, knn-clf, knn-cosine, knn-manhattan, knn-minkowski-p3, knn-rbc, knn-reg, knn-sqeuclidean, radius-chebyshev, radius-manhattan, radius-minkowski-p3 |
| estimators | 9 | kde-cosine-minkowski, kde-epanechnikov-l1, kde-exponential-chebyshev, kde-linear-cosine, kde-tophat-sqeuclidean, logistic, logistic-multiclass, pca-whiten, tsvd |
| rf | 6 | rf-clf, rf-clf-balanced-parallel, rf-clf-entropy-log2-noboot, rf-reg, rf-reg-gamma-ig, rf-reg-poisson |
| preprocessing | 5 | minmax-scaler, minmax-scaler-clip, standard-scaler, standard-scaler-no-mean, standard-scaler-no-std |
| training | 5 | cross-entropy-arms, mlp, optim-adam-clip, optim-sgd, training-primitives |
| gp | 4 | gp, gp-matern12, gp-matern32, gp-matern52-ard |
| solver | 4 | agglomerative, elasticnet, elasticnet-l2end-no-intercept, lasso |
| trees | 4 | et-clf, et-clf-entropy-bestfirst, et-reg, et-reg-bootstrap-parallel |
| arima | 3 | arima, arima-011, arima-seasonal-c |
| byte_lm | 3 | byte-lm-host-infer, byte-lm-host-infer-threaded, byte-lm-host-train |
| kernel_methods | 3 | kernel-ridge, nystroem, rbf-sampler |
| mamba | 3 | mamba1, mamba2, mamba3 |
| metrics | 3 | metrics, metrics-classification, spectral-precomputed |
| mixture | 2 | gmm, gmm-random-init |
| transformer | 2 | transformer, transformer-window |
| linalg | 1 | gemm-transposed |
| tsa | 1 | holtwinters-multiplicative |

**Held although the run read them IDENTICAL: nine lanes**, in
`PUBLIC_REFERENCE_CANDIDATES` beside `svc-poly`. It is not their arithmetic.
Every IDENTICAL cell they carry rests on the APPLE column alone, with no
NVIDIA and no AMD recording behind it, which is a two-column agreement and
exactly the condition lane/expose-inference-surface held `svc-poly` back for.
Promoting them on this lane's own evidence would apply a weaker rule than the
one that lane applied to itself. They join the day a release record carries
the other two columns.

`gbdt-query-rmse`, `gmm-random-init-sample`, `gmm-sample`, `gp-normalize-y`,
`gp-sample-y`, `gp-sample-y-normalize`, `gpc`, `gpc-multiclass`, `ivf-extend`.

**Held by the derivation: the thirteen `par-*` drivers.** A release record
does not run them (`RECORD_EXCLUDED_PREFIXES`), so nothing refreshes their
references, and unlike a shrunk fixture there is no `LANE_REVISIONS` entry to
catch one going stale. They stay covered lanes and the CPU identity gate still
runs all thirteen.

## Lanes pending on a stale reference

Thirteen, held in `PUBLIC_PENDING_LANES` as `stale reference`. Their fixture
moved past the hash the shipped table carries, which
`tools/identity_break.py` records in `LANE_REVISIONS` and the fixture shrink
(docs/lanes/FIXTURE_SHRINK_SCOPE.md, e2bb9e541) caused. `_verify_all` would
drop and name them anyway; keeping them out means the public set is a set that
PASSES rather than one that reports thirteen lanes it cannot compare. They
return at the next release record, which regenerates the table:

`byte-lm`, `byte-lm-resident`, `gbdt-lossguide-newtoncosine`, `gbdt-nan-modes`,
`gbdt-pair-logit`, `gbdt-parametric-losses`, `hdbscan`, `hdbscan-leaf`,
`holtwinters`, `mamba2-dtlimit`, `samba`, `samba-untied-dropout-accum`,
`spectral`.

Two other reasons hold lanes in the same dict, and they are not stale
references:

- **`no reference`, 8 lanes.** No committed record carries a single hash, so
  every part would read OWED, which is not a pass. `arima-exog`,
  `arima-exog-seasonal`, `gbdt-adapter-score-weighted`,
  `gbdt-categorical-ctr-tables`, `gbdt-tensor-ctr-tables`, `gbdt-yeti-rank`,
  `metrics-fowlkes-mallows`, `rf-score-weighted`.
- **`own record`, 5 lanes.** Diffed against `TRAINING_FIX_COLUMNS` rather than
  the release record. They do have hashes, but `record_covered_lanes()` is
  what the public set is held to. `embedding`, `embedding-sort`, `ivf`,
  `ivf-euclidean`, `kmeans-sqrt`.

A pending list is a memo unless something holds it to what made it true, so
`test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
checks each reason against its source: `stale reference` against the harness's
`LANE_REVISIONS`, `no reference` against the shipped table, `own record`
against `TRAINING_FIX_LANES`. The load-bearing assertion is the last one, that
no lane whose fixture moved is still public, because such a lane would ship a
reference describing different bytes and a user would read DIVERGENT for
something that is not their machine. The test was watched failing in both
directions before it was kept.

## The size objection, measured rather than estimated

The brief's 7 MB was an estimate from a partial set. Measured, from the sixteen
bindings built from this tree on the M4, one core:

| | uncompressed | in the wheel (deflate) |
|---|---|---|
| the sixteen added | 7,663,488 | 2,189,221 |
| mean per binding | 478,968 | 136,826 |
| largest, `gbdt` | 1,295,768 | 435,070 |
| second, `mamba` | 1,057,416 | 383,398 |
| smallest, `embedding` | 229,344 | 44,017 |

The compressed column is a measurement, not a model: deflate level 6 with a
raw window reproduces the published 0.8.6 wheel's recorded `compress_size`
**exactly on all 15 of its 15** host bindings, while level 9 matches none of
them, so these are the bytes a wheel would carry. Re-measured against
`~/mojolearn-evidence/release-0.8.6/macos-wheel/mojolearn-0.8.6-py3-none-macosx_11_0_arm64.whl`,
whose size on disk is the 26,368,494 the table below starts from.

| wheel | before | after | delta |
|---|---|---|---|
| macOS arm64 (measured) | 26,368,494 | 28,639,807 | +2,271,313, **+8.61%** |
| Linux x86-64 (projected) | 70,862,796 | 73,444,604 | +2,581,808, **+3.64%** |

The Linux figure is PROJECTED and labeled as such: Linux binaries cannot be
built on this machine, so per-family bytes are scaled by the ratio measured
over the fifteen families present in both published wheels (0.977 uncompressed,
1.137 compressed). The macOS figure is end to end measured.

These figures were first taken against the bindings built at `5596c3f08`,
where the sixteen came to 7,663,312 uncompressed and 2,188,904 compressed. The
`tsa` rebuild at HEAD moves them by +176 and +317 bytes; the numbers above are
the ones for the tree that merges, and neither percentage changes.

Two baselines differ by one and both are stated: the PUBLISHED 0.8.6 wheel
carries fifteen host bindings, while origin/main's manifest already shipped
sixteen (lane/expose-inference-surface added `resample` after that wheel was
built). The wheel arithmetic uses the published artifact; the family count this
lane changed is sixteen.

The wheel is still dominated by GPU bindings, which is why the percentage is
small: 312 MB uncompressed across 91 files on Linux, against 13.7 MB of host
bindings after this change.

## Every family now says why it ships

lane/expose-inference-surface introduced `wheel_note` and a test that no
exclusion is silent. All thirty-two notes were rewritten to begin `"Ships:"`,
each saying what the binding makes checkable that nothing else could, rather
than a template. Two module-docstring passages that still described the old
boundary were corrected, including one that was already stale on origin/main
(it described `resample` and `kpss_test` as OPEN after that lane had settled
both).

## Deliberately left out

None. `ships_in_wheel=False` now means a stated exclusion and nothing carries
one; `test_public_inference_bindings_ship_and_packaging_reads_the_manifest`
asserts the empty list, so holding a family back again has to delete that
assertion and write a reason into its `wheel_note`.

## The derivation this lane lost once, and how it was found again

Worth recording, because the failure is silent and repeatable. The session
that owned this branch ran `verify --all` over 132 lanes at 07:55 and then
merged origin/main. The merge took main's side for `public_reference_lanes()`,
so the hand-written thirty-nine came back over this lane's DERIVED version,
and the crash-recovery commit preserved that state. The branch then flipped
all thirty-two families to `ships_in_wheel=True` and still told a user they
could check thirty-nine lanes, which is a change that does nothing.

Nothing failed. Every test passed, the families all shipped, and the only
symptom was a number in a log that no longer matched the code. It was found by
reading the dead run's header, `132 lanes x 9 fixtures`, against the committed
`public_reference_lanes()`, which returned 39, and asking what could produce
132. The derivation restored from the session's own pre-merge snapshot
reproduces 132 exactly, which is how it was identified rather than guessed.

`PUBLIC_REFERENCE_CANDIDATES` is now subtracted as well, which is why the
final number is 122 and not 131.

## A pre-existing failure this lane did not cause

`test_tokenizer_manifest.py::test_manifest_declares_the_family` fails on
origin/main and still fails here. It asserts `training_lanes == ()` for the
tokenizer family, which has carried `("tokenizer",)` since
lane/cpu-verifier-gaps-7 (2026-09-15).

Verified rather than asserted. The test file is byte-identical to
origin/main's (same sha256), the operative value reads `('tokenizer',)` on
origin/main, on local main and on this branch's HEAD alike, and the test was
RUN on both sides and watched failing with the same assertion and the same
message. Left for whoever owns the tokenizer manifest, rather than fixed
silently here.

## What I did not touch, and why

- **`python/mojolearn/_verify_all.py`** is another agent's lane
  (lane/expose-inference-surface: `--self-test`, `--json-out`, and the
  INCOMPLETE verdict). This lane did not edit it. Two things there are worth
  that agent's attention rather than mine:
  - `select_lanes` and `_identity.py`'s refusal message interpolate the whole
    public lane list into an error string. At nine lanes that read well; at
    122 it prints a paragraph of lane names.
  - `test_verify_all.py::test_shipped_verifier_hashes_like_the_harness` runs
    the harness and the verifier over every public reference lane in
    subprocesses. Widening the public set makes that test proportionally
    slower; it took 197 s for the nine-plus-thirty set and 211 s for 122.

  One test in that file DID have to change, because this lane broke it.
  `test_full_and_cpu_lane_sets` asserted that a CPU-only install refuses
  `--lanes rf-clf` by name, and `rf-clf` is public now that the rf binding
  ships, so the assertion had nothing left to catch and read
  `DID NOT RAISE ValueError`. The example is now `par-forest`, which is
  excluded by RULE (`PUBLIC_EXCLUDED_PREFIXES`) rather than by happening to be
  off a list, so it stays a real test of the refusal however far the public
  set grows. Checked by pointing it back at a public lane and watching it fail
  again.

## How the host set was built

All thirty-two bindings built one at a time, one core, `nice -n 19`, through
the shared Mac slot helper: 0 failures. `tsa` and `forecast` were rebuilt from
this worktree at HEAD after the merge, into a FRESH output directory, because
the builder refuses to overwrite an existing `.so` and a binding rewritten
under a mapped process is SIGKILL 137 with no output.

## Gates

| gate | result |
|---|---|
| `tools/docs_facts.py --check` | OK, 13 facts, 12 marked spans |
| `packaging/wheel_ci.py pins .` | OK, 56 build scripts |
| `packaging/wheel_ci.py inventory python/mojolearn` | OK, 86 modules reachable |
| `packaging/check_ext_lists.py --host` | OK, 32 of 32 ship, read from the manifest everywhere |
| the manifest and boundary test files | 229 passed, 1 pre-existing failure above |
