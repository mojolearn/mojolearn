# Changelog

## Unreleased

### Added

- **`mojolearn.GaussianProcessRegressor`**, with its kernel classes `RBF`,
  `Matern` (nu in {0.5, 1.5, 2.5}), `ConstantKernel` and `WhiteKernel`,
  composable with `+` and `*`. Exact dense GP regression on the GPU --
  ORIGINAL WORK (cuML/cuVS/RAFT carry no GP at the pinned commits;
  scikit-learn `_gpr.py` is the semantics oracle only), backed by
  `gaussian_process/` (DEVIATIONS 1750-1771) over the Cholesky and gemm
  identity profiles.

  **Why it was held, and why that is resolved.** This was the only
  `_NOT_YET` entry ever placed for a reason other than a missing surface:
  its IDENTICAL card was believed to diverge Apple against AMD on 8 of
  3,494 stages. That reading was WITHDRAWN at `9835094e` (2026-09-01) --
  the eight lines are the sabotaged half of one `GP_SAB_STD_EXP`
  clean-then-sabotaged pair, and the shipped path is byte-identical on the
  other 3,486 lines (`bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md`,
  corrected in place) -- so the sole blocker was gone and the withholding
  text contradicted the evidence. Exposure is a CORRECTNESS claim on the
  vendors the cards cover (Apple M4 and AMD MI325X under `identical`; two
  vendors, not three), NOT a speed claim: the gp speed ladder is unrun.

  No optimizer (`optimizer=None` only, DEVIATION 1761), `return_std` but
  not `return_cov` (DEVIATION 1759), single-target `y`, and the default
  ridge is `alpha=2**-20` rather than sklearn's float32-no-op `1e-10`
  (DEVIATION 1772); every unported parameter is refused by name. A failed
  factorization is a RESULT in `info_`, and `predict` /
  `log_marginal_likelihood` on it are refused by name (DEVIATION 1634).

  Files: `python/mojolearn/_gp_impl.py`,
  `python/mojolearn/tests/test_gp_surface.py`, `python/mojolearn/
  __init__.py` (`_NOT_YET` is empty again), `python/mojolearn/_backend.py`
  (`_mojolearn_gp` in `_MODULES` and `_build_script`, DEVIATION 869).
  **The binding is owed**: `bindings/_mojolearn_gp.mojo` and
  `bindings/build_gp.sh` do not exist yet, and the estimator raises by
  name with the build command until they do.

- **`mojolearn.ARIMA`**, batched ARIMA with `fit`, `predict` and
  `forecast`, backed by `arima/` (DEVIATIONS 670-687 and 990-993): the
  ported cuML batched Kalman filter under an own-written `estimate_x0`
  (Householder QR) and an own-written batched L-BFGS, both landed
  2026-09-01 and gated in both numeric tiers by
  `arima/checks/fit_check.mojo`. This closes 0.2.0's longest-standing
  `_NOT_YET` entry ("`ARIMA` in particular has no `fit`").

  **`y` IS 2-D, `(batch_size, n_obs)`, and that is the point of the lane**:
  every series in the batch is fitted at once with its OWN parameters, one
  set of launches, cuML's design and not a loop. A 1-D `y` is one series;
  outputs stay 2-D. Data goes to `fit`, not the constructor, like the
  package's other estimators and unlike both upstreams. `predict`'s `end`
  is EXCLUDED (cuML's convention, NOT statsmodels'). Default order is
  `(1, 0, 0)`, which is neither upstream's, for reasons the class states.

  **Refused by name, end to end**: exogenous regressors (unported --
  `ARIMAParams` has no `beta`; the count crosses the boundary and
  `validate_order` refuses `n_exog != 0`, so the refusal is reachable from
  every caller), `method='css'`/`'css-ml'`, confidence intervals (`level`),
  missing observations, `start_params`, `trend='t'`/`'ct'`, `rd > 8`,
  `r > 5`. `AutoARIMA`'s search is not ported; its differencing half ships
  as `kpss_test` / `select_d`.

  **What the cross-vendor headline does and does not cover**: the lane's
  KALMAN FILTER is byte-identical on three vendors at `221aa141` (139
  stages); the FIT is gated on ONE Apple M4 only and the class says so
  rather than inheriting the card. The Python surface gate is
  `python/mojolearn/tests/test_arima_surface.py` (`pixi run
  check-arima-surface`, one tier per process, DEVIATION 796). It RAN
  2026-09-02 and printed GREEN in both tiers, 88 checks and 0 failed in
  each, **on ONE APPLE M4 and no other vendor**. The `identical` process
  ASSERTED its bitwise arms; the `fast` process reported "the fast arms
  passed, AND NO BIT WAS CHECKED". A second and third vendor through this
  surface is OPEN; the ordered command ledger is in `arima/README.md`,
  "Public estimator surface".

  Files: `arima/estimator.mojo`, `bindings/_mojolearn_arima.mojo` (the
  eleventh extension), `bindings/build_arima.sh`,
  `python/mojolearn/_arima_impl.py`, `python/mojolearn/__init__.py`,
  `python/mojolearn/_backend.py`, both wheel packagers, and the stale
  "there is no `fit`" paragraphs in `python/mojolearn/_tsa_impl.py` and
  `bindings/_mojolearn_tsa.mojo` deleted 2026-09-02.

- **`python -m mojolearn conformance {export,validate,diff}`**, the identity
  claim as a portable artifact (bundle format v1, `docs/CONFORMANCE.md`). A
  bundle freezes the pinned k-means fixture's inputs, expected stage bytes,
  and identity-trace card under a SHA-256 manifest so an external
  implementation can check itself against IDENTICAL-mode results without
  running Mojo or Python, and write back an `implementation-report.json`
  that `validate` grades and `diff` localizes through
  `tools/identity_trace_diff.py`, the one comparator. SHA-256 is the audit
  layer; FNV-1a64 stays the in-kernel localization checksum (DEVIATION
  928). Manifests hash every loaded binding `.so` with
  `artifact_source_commit: "unverified"` stated outright, because nothing
  yet ties a binary to the commit that built it (DEVIATION 929). Built
  write-only 2026-09-01; every run is owed, and the ordered list is in
  `docs/CONFORMANCE.md`.
- **`verify --confirm-reference`** and **`install-reference`** complete the
  reference-card path as three one-command steps, produce / confirm /
  install; the install is refusal-first (FILL-IN token, profile mismatch,
  missing provenance, differing or unknown commits, a divergent pair,
  DEVIATION 927) and prints the filled `docs/VERIFY.md` provenance block
  for a human to paste. Candidate cards now also record per-binding
  artifact hashes. The reference card itself is still not produced;
  `verify` keeps exiting 5.

- **`mojolearn.DBSCAN(sample_weight=)`** and **`metric='manhattan'`**, two
  configurations the surface used to refuse. A third,
  `algorithm='kd_tree'`, is still refused and now says why on the merits.

  **`sample_weight` enters in exactly one place.** A point is core when the
  SUM OF WEIGHTS in its eps-neighborhood reaches `min_samples`, not when the
  COUNT does (cuML `runner.cuh:300-306`, scikit-learn `_dbscan.py:451-455`,
  which agree). The neighborhood, the adjacency, the CSR, the propagation,
  the merge and both relabels are byte-for-byte the unweighted path. Both of
  cuML's producers are ported: `coalescedReduction` over the dense adjacency
  for `algorithm='brute'` and `accumulateWeights` over the CSR for
  `'rbc'`, with `need_ja_compute`'s `sample_weight` disjunct so loop 1 fills
  the columns on every weighted ball-cover batch (DEVIATION 29).

  **THE WEIGHTED DEGREE IS A FLOAT SUM AND ITS FOLD IS PINNED** (DEVIATION
  28, IDENTITY_PATHS row 64). Both upstream reducers close on a CUB stage
  that folds at the hardware warp width, 32 on Apple and NVIDIA and 64 on a
  CDNA wavefront; since the result is then thresholded against
  `min_samples`, a faithful port would put an AMD fit and a CUDA fit on
  opposite sides of that comparison for any point sitting on it, which is a
  different MODEL rather than a last-bit difference in a reported number.
  Both kernels close on `pinned_block_sum` at the kernel matrix's new
  `K_LIB_WEIGHTED_VERTEX_DEG` width, which is listed in
  `lib_block_bounds_a_float_fold`.

  **`metric='manhattan'` (also `'l1'`, `'cityblock'`) IS ORIGINAL WORK, not
  a port.** cuML's DBSCAN offers euclidean, cosine and precomputed and RAFT
  has no L1 eps-neighborhood kernel, so nothing is credited for the arm
  beyond the per-pair op, RAFT's `distance_ops/l1.cuh:49`. The structural
  point, and the reason it is one kernel with a compile-time metric rather
  than a branch: the ported arm compares a SQUARED distance against
  `eps * eps` and never takes a square root, and **an L1 sum has no squared
  form**, so the threshold is not squared on that arm (DEVIATION 27). It is
  served on `algorithm='brute'` and REFUSED BY NAME on `'rbc'`, whose
  landmark radii and pruning bounds are Euclidean; that is a scope boundary
  in `neighbors/`, not a property of the algorithm, since ball-cover pruning
  rests on the triangle inequality and L1 satisfies it.

  **`algorithm='kd_tree'` stays refused, on engineering grounds.** A
  kd-tree eps query is a recursive, data-dependent descent with a per-thread
  stack, so on a GPU the lanes of a warp diverge at every node and the loads
  are pointer chases; its pruning also degenerates into a scan with overhead
  past roughly ten features. The random ball cover already here prunes
  arithmetically over two flat arrays with coalesced reads and measured
  2.7x-27x over brute force at 16k-200k rows. Offering a kd-tree would hand
  a caller a slower answer under a familiar name.

  Files: `dbscan/impl/neighbors/epsilon_neighborhood.mojo` (the kernel is
  now comptime-parameterized on the metric; ONE tile, ONE reduction, one
  varying accumulate op), `dbscan/impl/dbscan/vertexdeg/algo.mojo`,
  `dbscan/impl/dbscan/corepoints/compute.mojo`,
  `dbscan/impl/dbscan/runner.mojo`, `dbscan/impl/dbscan/dbscan.mojo`,
  `dbscan/estimator.mojo`, `bindings/_mojolearn_estimators.mojo` (params
  slot 7 plus a `weight_addr` argument) and `python/mojolearn/density.py`.
  Seven gates in `dbscan/checks/dbscan_check.mojo`, each with a sabotage arm
  that must move. **ALL SEVEN RUN AND PASS as of 2026-09-01**, and the four
  weighted ones had never compiled until that day: building them raised
  `DeadArgumentElimination surveyUse failed`, an LLVM pass assertion, which
  took the whole lane down and left `sample_weight` IMPLEMENTED AND UNGATED.
  The cure is the OPTIMIZATION LEVEL and not the source -- `check-dbscan`
  now builds at `-O1`, and MEASURED on an Apple M4, 2026-09-01: -O3 and -O2 both assert, -O1 and -O0 both build, and `DeadArgumentElimination` is an -O2-and-above pass.
  Four candidate source rewrites were tried first and every one still
  asserted at -O3. The level is not a weakened gate: with the weighted four
  disabled so that -O3 can build at all, the -O1 and -O3 binaries print
  byte-identical output across all 13 remaining gates, including the
  float-heavy ones. Every weighted sabotage moved: the fold's width shifts
  the sum 1.0000151 / 1.000015 / 1.0, the planted fold sabotage separates the
  two summation orders, weight 5 turns all 12 noise points core, and weight
  1.5 falls back to noise. **Apple only; a three-vendor leg is owed.**

- **`mojolearn.SVR`**, epsilon-support vector regression, the scikit-learn
  surface over cuML's `svrFit`. The solver has carried `EPSILON_SVR` since
  `fea6becc` (2026-08-31), when the six rung-2 pieces were gated 44 of 44
  with the device arm included; what landed on 2026-09-01 is only the path
  from Python to it. **No new compiled extension was needed**: `SVR` rides
  `_mojolearn_svm.so`, which already carried `SVC` and the isolation forest,
  so `_backend.py`'s `_MODULES`, `pyproject.toml` and
  `build_release_wheel.sh`'s `EXT_NAMES` are unchanged.

  Files: `svm/impl/svm/svr_impl.mojo` (`svrFitX`, and `svrPredict` as the
  named wrapper over `svcPredict` with the class epilogue off, which is
  upstream's own arrangement), `svr_fit_host` / `svr_predict_host` in
  `svm/estimator.mojo`, `svr_fit` / `svr_predict` in
  `bindings/_mojolearn_svm.mojo`, and the class in
  `python/mojolearn/_svm_impl.py`.

  **The model is `n_support` wide, not `2 * n_support`.** The solver carries
  `alpha+` and `alpha-` over a doubled `n_train`, but `Results` folds the
  two halves before it selects, so the worst-case output buffers a caller
  allocates are exactly the classifier's. That is the one thing about this
  path that is easy to get wrong without an error, and it is written on the
  entry point, on the binding and on the class.

  **`SVR` MAKES NO CROSS-VENDOR CLAIM AND `SVC`'s CARD DOES NOT EXTEND TO
  IT.** The regression half was gated after the 2026-08-28 three-vendor leg
  and has not been in a round. What stands behind it is 44 of 44 on one box,
  of which four are property gates derived from the epsilon-insensitive
  formulation rather than from this solver, with their independence measured
  at 1.5e-07. A three-vendor SVR card is owed.

  Two defaults differ from scikit-learn's and both are named on the class.
  `gamma` is `'auto'` and not `'scale'` (DEVIATION 870, the same refusal
  `SVC` and `KernelDensity` carry: a host reduction's last bits cannot sit
  inside an identity claim), and `cache_size` is 1024.0 MiB and not 200
  because on this surface it is the PREDICT buffer only (DEVIATION 871).
  `shrinking` is not a parameter at all, because cuML has no shrinking
  heuristic for it to be a port of.

  The boundary gate is `python/mojolearn/tests/test_svr_surface.py`, which
  checks a planted linear recovery, the tube ladder, the model widths, batch
  invariance and every refusal on the path, and which asserts its bitwise
  arms only under `identical`.

### Changed

- **A fit on a dead or saturated GPU device now RAISES a named
  "DEVIATION 2002" error instead of returning a silent empty model.**
  Measured 2026-09-01 (PORTING.md DEVIATION 2002): at box saturation
  (Metal context death under 9-process load) every fit returned a
  coherent-shaped empty model — 0 splits, mse -0.0 — with no error.
  Fits now witness their own device delivery: a per-tree canary word in
  the greedy symmetric driver (rides the tree's existing drain, no
  added synchronize), a poisoned final-loss word in the boosting loop
  (covers every grow policy), and an end-of-fit canary in the
  extratrees and randomforest forest fits (one tiny kernel + drain per
  forest). Legitimately degenerate data (constant labels, unsplittable
  folds) cannot trip any of these — they check delivery, never the
  model. Sabotage arm `-D MOJOLEARN_2002_SABOTAGE=1` fakes the dead
  device, REQUIRED-RED; never in a shipped build. UNVERIFIED, ALL RUNS
  OWED — the exact commands are in PORTING.md's DEVIATION 2002 entry.

- **`RandomForestClassifier` / `RandomForestRegressor`: an unspecified
  `max_depth` now means UNLIMITED depth, not 16.** BEHAVIOR CHANGE on the
  Python surface. Through 0.3.2, `max_depth=None` (the default) substituted
  16, cuML's pre-26.08 default; it now maps to `np.iinfo(np.int32).max`,
  exactly as the pinned cuML marshals `None`
  (`randomforest_common.pyx:480-481`). This mirrors cuML's OWN change --
  v26.08.00, the release this port pins, changed its default from 16 to
  `None` (`.. versionchanged:: 26.08` on both estimators) -- and closes
  DEVIATION 409, which had recorded the divergence between the two surfaces
  of one learner (the Mojo entry points already passed INT32_MAX). An
  unspecified depth now grows to purity: deeper, slower, differently fitted
  trees from the same bare call. **Migration: pass `max_depth=16`
  explicitly to keep the old behavior.** No recorded benchmark number
  moves; every bench and probe arm passes `max_depth` explicitly.
  UNVERIFIED, RUN OWED: rebuild `_mojolearn_rf` on all tiers, then
  `python -m pytest python/mojolearn/tests/ -k rf` and
  `packaging/macos/smoke.py` on all tiers; the E2 RF identity cells re-run
  at the next rented leg.

## 0.3.2 (2026-08-31)

A new estimator, and 1,138 lines of never-executed code taken back out of the
wheel. Nothing in the 0.3.1 surface changes its numbers: the GBDT fix in this
release is on the POINTWISE searcher, which the Python surface does not reach
(`GradientBoosting` runs the greedy subsets searcher), so it is a correctness
fix in the tree rather than a behavior change for anyone installing this.

### Added

- **`mojolearn.RadiusNeighbors`**, every neighbor inside a radius, over
  cuVS's random ball cover (`neighbors/impl/neighbors/ball_cover/`). The
  index has answered DBSCAN's eps neighborhood since 0.1.0 and had no
  caller-facing surface; it does now. The results are EXACT, not approximate:
  the ball cover's pruning is a triangle-inequality bound, and
  `neighbors/checks/radius_check.mojo` asserts the set against a host
  brute-force oracle per cell rather than per total.

  scikit-learn's shape, with two differences named rather than papered over.
  `algorithm` accepts only `'auto'`, because the index is neither `'brute'`
  nor either of scikit-learn's two trees and naming one of them would
  describe the wrong algorithm. And a self-query keeps each point's own
  self-edge, which scikit-learn drops, because the CSR every other consumer
  in this library sees contains it.

  **The distances are recomputed from the finished neighbor list, not stored
  by the search.** The search kernel holds every distance in a register at
  the moment it decides membership and throws them away; adding five stores
  to recover them would have written into a body whose banner reads "Partial.
  Do not improve.", and would have cost `nnz * 4` bytes on every DBSCAN fit
  in the library, in every numeric mode, to serve a surface DBSCAN does not
  call. `neighbors/checks/radius_distances.mojo` carries the reasoning and
  the argument for why the recomputed value is the same value; under
  `identical` the check asserts it bit for bit, and it held for all 50,670
  edges of the fixture on Apple in BOTH modes.

  `sort_results=True` sorts on the host with a STABLE sort, which is what
  makes it free: under `identical` the device already returns each row in
  ascending index order (DEVIATION 551), so a stable sort by distance yields
  `(distance, index)` lexicographic order with no second key and no
  dependence on a lane width.

  Two boundary calls per query, not one, because a radius query's output size
  is not a function of its inputs. The cost is the ball-cover index built
  twice per query, which is written down where it is paid
  (`neighbors/estimator.mojo`, the RADIUS NEIGHBOURS banner) and is what a
  fitted device handle would fix. Nothing in this library holds one yet.


### Removed

- **`torchbridge/` and `mojolearn.torch_ops`, the PyTorch custom-op surface,
  are DELETED as of 2026-08-31.** They shipped in the 0.2.x and 0.3.x wheels
  as `mojolearn/torch_ops.py` and they should not have. This entry is the
  tombstone: the surface existed, it is gone, and nobody should re-plan it
  from the absence of a file.

  What was there: `torchbridge/TORCH_BRIDGE_PLAN.md` (812 lines),
  `torchbridge/identical_ops.mojo` (490 lines),
  `torchbridge/__init__.mojo` (17 lines) and
  `python/mojolearn/torch_ops.py` (1,138 lines), written 2026-08-25 under
  DEVIATIONS 1200 through 1213. It proposed two transport lanes to
  `mojolearn.identical.gemm.fp32.v1` from a `torch.Tensor`, a
  `torch.autograd.Function` backward over the same forward op, and nine
  gates T1 through T9.

  **Why it is gone, and not finished.** Its own three file headers said, in
  capitals, that not one line of it had ever been through the Mojo compiler
  or a Python interpreter. That was still true on the day it was deleted:
  none of the nine gates was ever written, `bindings/build_torchbridge.sh`
  never existed, and no `torch_ops` bytecode was ever produced anywhere in
  the tree. So the cost of keeping it was a plan nobody was going to write
  and 2,457 lines advertising a capability the library did not have.

  **And it was the wrong shape.** This library is a competitor to the
  PyTorch stacks, not a plugin into one. A bit-identical GEMM reached
  through `max.experimental.torch` is a MAX custom op wearing our name; the
  claim this repository actually makes is about its own kernels on three
  vendors, and that claim is made and gated in `gemm/`,
  `python/mojolearn/_linalg_impl.py` and the identical wheel tier, none of
  which needs torch to be installed.

  PyTorch stays in this tree in the ONE role it earns: a benchmark opponent
  and a float64 reference oracle (`pixi.toml`'s `pytorch` dependency,
  `tools/speed_torch_seq.py` and the per-stage cross-checks). Those are
  evidence and they are untouched.


## 0.3.1 (2026-08-31)

**0.3.0's Linux wheel crashes on any x86-64 host without AVX-512. Use this
one.** The macOS wheel was never affected.

`pip install mojolearn` on a Linux box whose CPU lacked AVX-512 imported
cleanly, reported its vendor and architecture correctly, and then died with
`SIGILL` and a core dump on the first `fit()`. Measured on an L40 whose host
was an AMD EPYC 7773X:

    vandps 0x6d7ee(%rip){1to4},%xmm2,%xmm2
    in cluster::estimator::kmeans_fit

`{1to4}` is an EVEX embedded broadcast. **The GPU was never involved**; the
selector had correctly chosen `cuda` and then `sm_80` for that `sm_89`
device. Every AMD Zen 1, 2 and 3 host, most Intel consumer parts and every
Xeon before Skylake-SP would have crashed identically, and all thirty
extensions carried it, not just k-means.

**Cause.** `mojo build` defaults `--target-cpu` to the chip that ran the
compiler, and the Linux branch of every build script passed no CPU flag at
all, so the sets were compiled for whatever the build boxes had. All four had
AVX-512. macOS has pinned `--target-cpu apple-m1` and gated the result with
`packaging/isa_baseline.py` since 0.1.0; Linux pinned nothing and gated
nothing.

**Fix.** All ten build scripts pin `--target-cpu x86-64-v3` on Linux x86_64,
which is AVX2, FMA, BMI2 and SSE4.2, so Haswell 2013 and Zen 1 2017 onward,
and excludes AVX-512. `MOJOLEARN_LINUX_CPU` overrides it. aarch64 Linux keeps
its previous behaviour, since an x86 CPU name means nothing there.

**Gate, so it cannot regress.** `packaging/linux/isa_baseline_linux.py`
disassembles every binary in a set and refuses `zmm` registers, `{1toN}` EVEX
broadcasts, `k0`-`k7` opmasks and the named AVX-512 opcodes; `build_sets.sh`
runs it and exits 5. It refuses 30 of the 34 binaries in the 0.3.0 set, which
is how it was verified.

It is not a blanket ban. The MAX runtime libraries carry AVX-512 and guard it
behind runtime `cpuid` dispatch, so they pass; our extensions contained no
`cpuid` at all and are held to the baseline unconditionally.

**0.3.1 carries FIVE architecture sets, not six.** `cuda/sm_120a`, the
Blackwell and RTX 50 set, is not rebuilt yet: RunPod reported "no instances
currently available" across every GPU type while this was being cut, and
shipping the fix for a crash that hits every AMD Zen host was worth more than
waiting for one more architecture. A Blackwell device now gets a clean
refusal naming its architecture instead of a wheel that would have crashed on
its host anyway. `sm_120a` returns in 0.3.2.

    cuda/sm_80     A100, A40, A10, L4, L40S, RTX 30, RTX 40
    cuda/sm_90a    H100, H200
    hip/gfx942     MI300, MI325
    hip/gfx1100    RX 7900, RDNA3
    hip/gfx90a     MI210, MI250

Everything else in 0.3.0 is unchanged, including the cross-vendor bit
identity: 29 of 29 lanes hash identically between an H100 and an MI325X under
`identical`, while 18 of those same lanes differ under `fast` and
`deterministic`.

## 0.3.0 (2026-08-30)

> **WITHDRAWN IN PRACTICE. 0.3.0's Linux wheel crashes with SIGILL on any
> x86-64 host without AVX-512, which includes all AMD Zen 1, 2 and 3 parts.
> Use 0.3.1. The macOS wheel is unaffected.**

**The Linux wheel.** One PyPI name, `pip install mojolearn`, no extras, no
variants. The vendor AND the GPU architecture are detected at import and the
numeric tier stays a runtime parameter, so this adds a platform and changes
nothing about how the library is used.

**Not yet published at the time of writing, and the reason is the whole
story of this release.** The first Linux wheel was built, audited, put on
TestPyPI and installed from there onto an MI325X, where it passed every lane
in every tier. The same wheel on an A40 failed 27 of 29 lanes with
`CUDA_ERROR_NO_BINARY_FOR_GPU`, because `--target-accelerator` was never
passed and MAX had compiled for the build box's own device and nothing else.
`strings` on the shipped binary said `sm_90a`, 55 times, with no other
architecture and no PTX. So a wheel that installs cleanly, imports cleanly
and reports its vendor correctly would then have failed on essentially every
call for anyone not on an H100. It was held back rather than published, and
the architecture axis below is the fix.

### The Linux wheel

- `mojolearn-0.3.0-py3-none-manylinux_2_35_x86_64.whl`, 26.2 MB, carrying
  BOTH a CUDA and a HIP binary set in all three numeric tiers: sixty
  extensions, plus ONE shared `mojolearn/.libs/` because the two vendors'
  MAX runtime closures turned out to be byte-identical. Layout
  `mojolearn/<vendor>/<arch>/<tier>/*.so`, an ARCHITECTURE AXIS under the
  vendor axis, because one `mojo build` emits code for exactly one GPU
  architecture. A comma-separated list is accepted by the flag parser and
  then rejected by the compiler, `constraint failed: GPU architecture
  'sm_80,sm_86,sm_90a' is not supported`, and there is no PTX in any set, so
  there is no JIT fallback either. The set is chosen at import from the
  device's own compute capability, with a within-family fallback so an
  `sm_86` device runs a carried `sm_80` set. An `a`-suffixed build is never
  chosen by the family rule, because that suffix means architecture-specific.
- **The manylinux level is measured, not chosen: `manylinux_2_35_x86_64`.**
  `pack_wheel.py` deliberately writes the `linux_x86_64` tag that PyPI
  refuses, and `auditwheel` supplies the real one. glibc 2.35 means Ubuntu
  22.04 and Debian 12 are in and RHEL 9, on glibc 2.34, is out by one minor
  version.
- `mojolearn.vendor()` and `estimator.vendor_used()` report the accelerator
  API the loaded binary was COMPILED for, read back out of the binary through
  a `<binding>_vendor()` export on every binding (`checks/vendor.mojo`, a
  compile-time constant from `std.sys.info.has_*_gpu_accelerator()`). The
  selector refuses at import when a binary's answer disagrees with the
  directory it was loaded from, the same refusal as the tier read-back.
  `python -m mojolearn verify` and `mojolearn doctor` print it.
- `MOJOLEARN_VENDOR=cuda|hip` picks the directory on Linux; otherwise a box
  probe (device nodes and driver libraries) picks it. **A box with neither
  refuses at import**, naming every path and library it looked for. There is
  no CPU path and none is planned.
- New classifiers `Operating System :: POSIX :: Linux` and
  `Environment :: GPU :: NVIDIA CUDA`. The second was previously withheld
  with the note "no NVIDIA device has ever run this"; an H100 has now run all
  29 smoke lanes in all three tiers.

### What it was actually tested on, stated narrowly

All on 2026-08-30. `bench/results/wheels/LEGS_2026-08-30.md` records it in
full, including the wrong turns.

- **NVIDIA H100 PCIe**, CUDA set, 29/29 smoke lanes in `fast`,
  `deterministic` and `identical` on the box that built it, sabotage PASS.
- **AMD Instinct MI325X**, HIP set, 29/29 lanes in all three tiers, sabotage
  PASS, and again from a clean `pip install` off TestPyPI into a fresh venv,
  which is the whole install path end to end. All 30 binaries read back
  `gfx942`.
- **NVIDIA A40**, the same wheel installed from TestPyPI: installs, imports,
  picks `cuda` correctly, and then fails 27 of 29 lanes. The two that pass
  name the cause exactly. `gemm-vendor`, which reaches cuBLAS through MAX's
  own matmul, RAN and produced a hash, and `gemm-pinned` refused as designed.
  Everything that launches one of OUR kernels failed and the one thing that
  does not, worked.
- The no-GPU refusal is verified against the finished wheel in a container
  with no device passed through.

**Every set now records the architectures actually inside it.**
`build_sets.sh` reads them back out of each `.so` and REFUSES with exit 4 if
any binary names none. That check earned its place on its first run: given an
unusable architecture list, MAX silently fell back to the build box's own GPU
and 27 of 30 binaries came out as an architecture nobody asked for, with one
compile error as the only other hint.

**A forced vendor whose runtime is absent no longer kills the process.**
Setting `MOJOLEARN_VENDOR=hip` on a Linux box with NO ROCm installed used to
abort during import with no traceback, while forcing `cuda` on a box with no
CUDA behaved correctly, importing and then raising a clean Python exception
at the first fit naming the driver library it could not open. The two vendors
do not degrade alike, and the promise that the first fit reports the
runtime's own error was kept on one branch and not the other.

The selector now refuses a forced vendor for which the box shows no evidence
at all, neither a device node nor a loadable driver library, and names what
it looked for. The refusal is in Python, which is the only place a message
survives a trap in the binary beneath it. It does not close the escape hatch,
because the case that hatch exists for is a device that IS present and a
probe that missed it, and such a box has that vendor's driver library
loadable. `MOJOLEARN_VENDOR_FORCE=1` proceeds anyway and says it may abort.

This was never reachable with `MOJOLEARN_VENDOR` unset, which is how the
library is meant to be used. The ISA explanation was tested three ways and
ruled out; both sets carry the same number of trap instructions and no
runtime CPU dispatch.

### Tooling and gates

- `packaging/linux/`: `build_sets.sh`, `stage_libs.py`, `pack_wheel.py`
  (pure Python, on the Mac), `audit.sh`, `smoke.py`, `sabotage.py`,
  `nogpu.py`, `nogpu_local.sh` and `leg.sh` (one command per vendor).
- `packaging/release_workflow_test.sh`: the release workflow's three
  artifact-handling shell blocks, extracted from the YAML and RUN on the Mac
  against fabricated wheels. 15 cases, no network. It found two bugs before
  CI did, one of which would have broken every macOS-only release.
- A release may now carry two wheels. The Linux wheel cannot be built in CI,
  so it is staged on the runner's disk with a digest sidecar and admitted
  only if the sidecar matches, the version agrees, and the tag is a manylinux
  tag; the publish job then checks the artifact against a digest manifest in
  both directions.
- Three gate bugs found by running gates that had only ever been read:
  `sabotage.py` searched a six-line stderr tail for strings that sit on line
  0; `smoke.py` counted a correct refusal as a failure; `audit.sh` ended a
  pipeline with a `grep -c` whose zero-match SUCCESS case exits 1 under
  `pipefail`, killing the script one line before `twine check`.
- `packaging/macos/smoke.py` asserts the vendor read-back (`metal`).

## 0.2.0 (2026-08-30)

Fourteen new public names and two new submodules, over lanes that were
finished and gated at the kernel level before 0.1.0 shipped and were simply
unreachable from Python. **Their cross-vendor standing is NOT uniform and
each class states its own.** Read the class, not this list.

### Estimators

- `SVC` (cuML's `SVC`; binary C-SVC only. There is no `SVR`: `svmType !=
  C_SVC` raises by name in `svm/impl/svm/svm_parameter.mojo`, and
  epsilon-SVR is rung 2 in `svm/NOT_IMPLEMENTED.tsv`)
- `Lasso`, `ElasticNet` (cuML's `solver='cd'` arm, `cd.cuh::cdFit`;
  DEVIATIONS 610-613 and 880)
- `AgglomerativeClustering` (cuML's `hierarchy/linkage.cu` down through
  cuVS's `cluster/detail/` to RAFT's Boruvka MST; single linkage;
  DEVIATIONS 620-624 and 881)
- `Ridge` (cuML `ridgeFit`, the `eig` arm: `svdEig` + `ridgeSolve`;
  DEVIATION 545) and `LogisticRegression` (cuML `qnFit`, the binary L-BFGS
  sigmoid arm with the Armijo line search; DEVIATIONS 546-549. OWL-QN / l1,
  softmax / multiclass, `sample_weight` and `class_weight` are refused by
  name). Under IDENTICAL every reduction in the objective, the gradient and
  the solver is a pinned fold where cuML's are float atomics, so the
  iteration count is part of the certificate (`qn.n_iter`). Gates:
  `glm/checks/ridge_check.mojo`, `glm/checks/logistic_check.mojo`,
  both modes, in `pixi run check-linalg-identity`; E2U cells `ridge_*`,
  `logreg_*`.
- `KNeighborsClassifier`, `KNeighborsRegressor` (the cuVS/RAFT/FAISS k-NN
  path 0.1.0 already shipped as `NearestNeighbors`, with the vote and the
  mean on top)
- `KernelDensity` (cuML's `kde/`; DEVIATIONS 600-604. `bandwidth='scott'`
  and `'silverman'` are refused by name: compute the number and pass it, so
  the number that ran is the number you passed)
- `IsolationForest` (cuML's `IsolationForest`; see Known issues)
- `SpectralClustering` (cuML's `fit_predict` through cuVS's kNN
  connectivity graph, the normalized graph Laplacian and RAFT's
  thick-restart Lanczos, with every closed vendor library on that path
  replaced by a named, numbered stand-in; DEVIATIONS 770-781)
- `ExponentialSmoothing` (`cuml.tsa.ExponentialSmoothing`, backed by
  `holtwinters/`; DEVIATIONS 660-665 and 697-699)

### Functions and submodules

- `mojolearn.metrics`, fourteen scoring functions: `accuracy_score`,
  `r2_score`, `rand_score`, `adjusted_rand_score`, `homogeneity_score`,
  `completeness_score`, `v_measure_score`,
  `homogeneity_completeness_v_measure`, `entropy`, `mutual_info_score`,
  `kl_divergence`, `silhouette_score`, `silhouette_samples`,
  `trustworthiness`. The names and argument names are scikit-learn's,
  because that is what a caller types; **the defaults and the semantics are
  cuML's**, and the three differences that matter are written on the
  function: `entropy` and `mutual_info_score` are in NATS as RAFT computes
  them, `r2_score` bakes in `force_finite=True` and refuses
  `force_finite=False` by name (DEVIATION 657), and `kl_divergence` does
  not normalize `P` and `Q`.
- `mojolearn.linalg`: `matmul`, plus `profile()`, `numeric_mode()` and
  `require_identical()`. A numerical primitive rather than an estimator,
  for callers who need a matrix product that returns the same bits on three
  vendors and will never fit a model here. Profile
  `mojolearn.identical.gemm.fp32.v1`; contract
  `gemm/IDENTICAL_FP32_CONTRACT.md`.
- `kpss_test` and `select_d` (`cuml.tsa.stationarity.kpss_test` and
  auto_arima's "choose the hyper-parameter d" block, both backed by `tsa/`;
  DEVIATIONS 671-672)

`ARIMA`, `SVR` and `RadiusNeighbors` are still absent and are named in
`mojolearn._NOT_YET`: importing one raises with the line where the thing
that exists stops, rather than an `AttributeError`. (All three shipped
after this release -- `RadiusNeighbors`, `SVR`, and `ARIMA` with a fit on
2026-09-01; see the unreleased section at the top.) `ARIMA` in particular
has no `fit` -- `arima/` ports the batched Kalman filter likelihood, its
gradient and predict, but `estimate_x0` and the batched L-BFGS driver are
NOT PORTED, and those are exactly what produces the coefficients every
existing entry point requires as input. (True at this release; both landed
2026-09-01.)

### Cross-vendor standing

Every lane in this release runs the same pinned path under
`MOJOLEARN_NUMERIC_MODE=identical`. What differs between them is not whether
they are pinned but **where the pinned card has been diffed**: bit-identity
is a claim about two machines agreeing, and checking it requires the lane to
have run on a second and third machine.

On 2026-08-28 all three boxes ran at ONE commit, `a0a0eee` (Apple M4 leg
`bench/results/e1/2026-08-28_130918-MacBook-Air-1-terrabyte`, NVIDIA H100
leg `2026-08-28_131651-runpod-nvidia`, AMD MI325X leg
`2026-08-28_173933-mojolearn-e2-amd`). Every IDENTICAL card is byte-identical
Apple-to-NVIDIA and Apple-to-AMD: cd 23 stages, gemm 61, iforest 124, kde 9,
linkage 10, metrics 64, svm 35, and the `ridge_*` and `logreg_*` E2U cells.
So `mojolearn.linalg`, `Lasso`, `ElasticNet`, `KernelDensity`,
`AgglomerativeClustering`, `SVC`, `mojolearn.metrics`, `IsolationForest`,
`Ridge` and `LogisticRegression` all stand on measured three-vendor cards.
This also closes two items that were open at E3 round 11: `metrics`'s grown
card (34 stages then, 64 now) has had its three-vendor leg, and
`isolation_forest` joined phase 8 and ran on both boxes.

`SpectralClustering`, `ExponentialSmoothing`, `kpss_test` and `select_d` do
not stand on a three-vendor card, and the reason changed on 2026-08-28. All
three lanes now emit identity cards and are listed in the round: `spectral`
and `holtwinters` joined phase 8 in `241aed6`, and `tsa` in `f081b7f`, whose
message records that `tsa/tsa_main.mojo` had built a complete eleven-stage
card since it was written and that the comment claiming otherwise was the
only thing keeping it out.

**CORRECTED 2026-09-01, and the correction goes upward. The shared commit
this section said was missing exists.** At `221aa141` all three lanes have
an Apple card and an AMD card taken at that one commit, and each pair is
byte-identical, `spectral` at 171 stages, `holtwinters` at 182 and `tsa` at
13, with 0 records differing. `arima`, which at this release has no Python
estimator and so no row in the README table (the estimator and its row
landed 2026-09-01; see the unreleased section), is byte-identical at the
same commit on THREE vendors,
Apple against AMD against NVIDIA, 139 stages. The NVIDIA column for the
other three did not fail; it never reached them, because the leg hung in
`holtwinters` on the DEVIATION 1946 context-lifetime defect above. The
certificate, lane by lane and vendor by vendor, is
`bench/results/e1/CERT_2026-08-31.md`. Two vendors is not three, so those
three lanes stay open rather than inheriting arima's third column.

**The FAST arm, which is the default, makes no cross-vendor claim at all.**
Unchanged from 0.1.0, and the FAST cards do differ between vendors for
every lane but gemm; that is recorded, not a defect.

### A third numeric tier, `deterministic`, and it ships

`MOJOLEARN_NUMERIC_MODE` became a three-rung ladder on 2026-08-29
(DEVIATIONS 1940 and 1941). `fast` promises nothing but speed, `deterministic`
promises the same bits run to run on one box and says nothing about a second,
and `identical` promises both plus the same bits on Metal, CUDA and HIP.
Determinism is a strict subset of identity, never a sibling, so
`PIN_DETERMINISM` is true under both upper tiers.

The tier is worth having because the reproducibility most callers actually
want was only purchasable by taking `identical` whole, and identity is not
free. Three consecutive FAST runs of one binary on one fixture on one M4
returned three different sorted k-NN index sets, and across every FAST board
taken on 2026-08-28 our own arm's per-round output hash moved between rounds
on 4 of 179 rows.

**0.2.0 carries all three tiers.** This entry said "`fast` and `identical`
only ... once the lane closes", and the lane closed the same day; the old
text is replaced rather than appended to, because a changelog that describes
a wheel it does not describe is worse than one that says nothing.

What closed it is a MEASUREMENT of the promise, not a pin count.
`tools/repeat_run_stability.py` refits one lane repeatedly in a single
process and compares raw output bytes with no tolerance. At one commit on
three rented columns on 2026-08-29:

| column | `fast` | `deterministic` | `identical` |
|---|---|---|---|
| Apple M4, Metal | MOVED in 8 of 10 attempts | STABLE 10/10 | STABLE 10/10 |
| NVIDIA RTX 4090, CUDA | MOVED, 24 calls 24 answers | STABLE | STABLE |
| AMD MI325X, HIP | MOVED, 6 answers in 24 | STABLE | STABLE |

The pin side is 15 files keyed to `PIN_DETERMINISM`. The class is small
because this tree uses no float `atomicAdd` anywhere -- three of the four
classification scopes came back empty, which is the right answer for a tree
built that way rather than a gap. And the tier is not the top one in
disguise: under `deterministic` the output hashes DIFFER between vendors on
10 of 12 comparable lanes, `gemm-vendor` among them, so it keeps MAX
`matmul`, cuBLAS and rocBLAS at full speed. All three of those were measured
run-to-run stable at 256x4096 @ 4096x128, a wide k chosen to provoke a
split-K epilogue, which is what earns them that exemption.

`ce2e843` is still the reason to be careful with this tier: for the length of
one commit it was defined, empty, and would have taken the float atomic flush
while calling itself deterministic, which is the worst failure a tier named
that way can have. That is what `verify_wheel.sh` now guards -- it installs
the finished wheel into a clean venv under every claimed interpreter and fits
every estimator family in EVERY shipped mode, and a tier that did not build
raises from a missing-binary stub BY NAME rather than serving fast arithmetic
under another label. Which tiers a wheel carries is one variable,
`MOJOLEARN_RELEASE_MODES`, read by both scripts.

### The mode is a PARAMETER now, and there is still ONE install

Until `035922c` the numeric mode was reachable only as
`MOJOLEARN_NUMERIC_MODE`, an environment variable read ONCE by
`_backend.select()` before the first estimator was imported. That was always
one install -- the wheel has carried every tier's binaries since 0.1.0 -- but
it was a choice you had to make from the shell, could not change after import,
and could not make differently for two estimators in one script.

    mojolearn.set_numeric_mode("deterministic")            # process default, in code
    rf = mojolearn.RandomForestClassifier(numeric_mode="identical")   # one estimator
    est.numeric_mode = "fast"                              # after construction
    est.numeric_mode_used()                                # what THIS instance will call
    mojolearn.numeric_mode()                               # the process default

`MOJOLEARN_NUMERIC_MODE` still works and still sets the STARTING default, so
nothing written against the old spelling breaks. There is no extra to install,
nothing to rebuild and nothing to reinstall: `pip install mojolearn` is the
whole surface and it always was.

WHY IT IS A BINDING LOOKUP AND NOT A RUNTIME FLAG. **The tier IS a flag, one
flag over one source: `GLOBAL_NUMERIC_MODE` in `checks/numerics.mojo`, from
which `PIN_DETERMINISM` and `PIN_CROSS_VENDOR` derive. Nothing in this tree
forks per tier.** That flag is comptime, which is what lets the fast build
carry none of the pinning code rather than branch past it, so its three
settings are compiled and shipped side by side and the Python parameter picks
among them instead of flipping something inside one of them. That is why every
call site resolves through `self._bind()` at call time instead of binding a
module-level name at import. The keyword is injected by
`__init_subclass__` rather than written into eleven constructor signatures,
because eleven copies is eleven chances for one to drift silently -- an
estimator that ignored the keyword would run the process default while
reporting the tier it was asked for.

THAT THE TIERS COEXIST IS MEASURED, not assumed. Each `.so` carries its own
Mojo runtime and opens its own device context, so "two of them in one process
will fight" was the real risk. All three were loaded together and called
INTERLEAVED -- fast, deterministic, identical, fast -- twice, on one
256x4096 @ 4096x128 product on an M4, and each returned its own arithmetic
every time; a call made after the identical set did not inherit its answer.
17 estimators x 3 tiers pass.

TWO BUGS THIS FOUND, both in the reporting rather than the arithmetic, and
both shipped in 0.1.0. `mojolearn doctor` reported "import probe: failed" on a
healthy install, because it `json.dumps`'d the `numeric_mode` FUNCTION object.
And `numeric_mode()` reported the IMPORT-TIME tier, so after
`set_numeric_mode("deterministic")` it still answered `fast` -- the one
function whose entire job is that a run cannot be mislabeled was mislabeling
it. Both are fixed in `42abc57`. Neither was caught by a green test, and what
caught them was running what the documentation claimed.

### Tooling

- `python -m mojolearn verify` checks this build against a reference card,
  `--json` for machine-readable output. Exit codes, because people put this
  in continuous integration: 0 VERIFIED, 1 MISMATCH (a diverging stage is
  named), 2 USAGE, 3 REFUSED (the process loaded the FAST binaries, which
  make no identity claim, so nothing was judged), 4 CANNOT RUN, 5 NO
  REFERENCE (this entry originally stopped at 4; exit 5 shipped in 0.2.0
  from the start, and the omission was corrected 2026-09-01 on discovery).
- `python -m mojolearn env` reports what the process loaded without
  touching the GPU; `python -m mojolearn check-fixture` rebuilds and hashes
  the pinned fixture without a GPU and without an extension call.
- The `mojolearn` console script continues to point at
  `mojolearn_diagnostics`, which lives outside the package on purpose so it
  still runs when importing the extensions is the thing being diagnosed.

### Performance (FAST arm)

The FAST path changed substantially this cycle; IDENTICAL is untouched by
it and every A/B fit across the round was byte-identical.

Measured on one MacBook Pro M4 on 2026-08-27 at `20729ea`, FAST both
sides, three timed rounds after a warm-up, against the only arms the
opponents ship on Apple silicon
(`bench/results/fast_speed/2026-08-27-APPLE-trees-evening.md`):

| lane | dataset | opponent | verdict |
|---|---|---|---|
| gbdt-symmetric | higgs 1M x 28 | catboost-cpu | 1.31x FASTER (was parity) |
| gbdt-symmetric | year 464k x 90 | catboost-cpu | 1.20x FASTER (was 1.10x) |
| rf | higgs 1M x 28 | sklearn-rf-cpu | 3.07x FASTER |
| et | higgs 1M x 28 | sklearn-et-cpu | 1.25x FASTER, and ahead on accuracy |
| et | covtype 523k x 54 | sklearn-et-cpu | 1.13x SLOWER |
| rf | covtype 523k x 54 | sklearn-rf-cpu | 1.45x SLOWER |
| gbdt-depthwise | year 464k x 90 | catboost-cpu | 1.28x SLOWER |
| gbdt-lossguide | year 464k x 90 | catboost-cpu | 1.73x SLOWER |

covtype's deep-narrow, launch-bound shape remains the CPU arms' best
ground. depthwise and lossguide get their first Mac rows here and both are
behind.

Landed after that board were DEV 1902 (ridx-only splits), 1911-1914 (quantized
shared-memory histograms), 1916-1919 (random forest launch batching) and
1921-1923 (k-NN warp-select and dispatch), merged and gated at `b90ab1c`.
Measured on Apple at `a8d838e` on 2026-08-28
(`bench/results/fast_speed/2026-08-28-APPLE-forest.md`). There, rf on covtype is
16.89x faster than lightgbm-cpu, gbdt-symmetric on year is 1.14x faster than
catboost-cpu and et on covtype is 1.12x faster than sklearn; depthwise is
1.08x, rf 1.23x and lossguide 1.47x behind their CPU arms. Several arms in
that round are routed to the NVIDIA and AMD FAST columns by the kernel
matrix and do not touch this wheel at all.

### Known issues

- **The Python bindings hung on an RTX 4090 (sm_89), twice, for two
  DeviceContext lifetime defects.** DEVIATION 1944: the isolation forest
  estimator built its empty model on a SECOND `DeviceContext` beside the
  caller's, and the two deadlocked at teardown -- the first `fit` never
  returned. DEVIATION 1946, the same class one call later: `_mojolearn_rf`
  and `iforest_run_host` created their context inside the call and released
  device buffers AFTER its last use, so Mojo destroyed the context first and
  the buffers -- pinned host allocations among them -- were freed behind it;
  the first binding call returned and the NEXT GPU call in the process never
  did, GPU idle, every host thread in futex wait. 1944 is confirmed on the
  box. **1946 is UNRUN on a 4090** (the fix cannot be exercised on Apple,
  AMD or the H100, none of which ever showed the defect). Apple is the only
  platform this wheel targets and it was never affected.

  **CORRECTED 2026-09-01: the 1946 sweep described here was NOT complete.**
  It covered the library entry points and skipped every check driver and
  every `main`. `holtwinters` carried `_ = ctx^` at none of its eleven
  `DeviceContext` sites, and that is where the NVIDIA time-series leg hung
  on 2026-08-31. A mechanical audit on 2026-09-01 found **81 further
  functions in 57 files** freeing device buffers after their context's last
  use, none of them previously swept; all 81 now carry `_ = ctx^`. Verified
  on Apple that the sweep moves no bit. `holtwinters` compiles, its 11 gates
  pass, and the output is byte-identical to the pre-fix reference log. NOT
  verified to cure the hang, which only an sm_89 leg can establish.
  `bench/results/e1/CERT_2026-08-31.md`. `bench/results/identity_break/RESULTS.md` carries the
  evidence, the discriminator
  (`ensemble/checks/rf_ctx_order_probe.mojo`) and the RUN OWED list.

- `ftz`, the denormal-policy helper that IDENTITY_PATHS row 10 is built on,
  DID NOTHING ON THE GPU until 2026-08-28 (DEVIATION 1938). Its guard was two
  float comparisons, and a flush-to-zero backend evaluates those with the
  operand already flushed, so `x != 0.0` was false for every subnormal, the
  branch never fired, and the helper returned subnormals untouched. Measured
  on the M4 under IDENTICAL against a host twin, three of four patterns
  disagreed. It hid because everywhere a denormal also passes through
  arithmetic the hardware flushes it anyway; it is visible only where a value
  reaches a seam BY COPY. **This was a cross-vendor hole, not an Apple one**,
  since on a denormal-honoring backend the guard fired and on an FTZ backend
  it did not. The guard now tests the exponent and mantissa fields, which no
  backend's FTZ can defeat. **Every binary published before this release,
  including the 0.1.0 wheel on PyPI, carries the inert helper.**

- `IsolationForest` carries OPEN DEVIATION 750. cuML's `curand_u64` builds
  a 64-bit draw out of two unsequenced `curand()` calls and C++ does not
  say which becomes the high word. Both readings conform and they give
  DIFFERENT forests from the same seed. This port takes the first draw as
  the high word, by name, and that choice has never been checked against a
  cuML binary. Until it is, agreement with cuML there is a belief and not a
  measurement.
- `check_ols_is_launch_invariant` still fails on both the H100 and the
  MI325X in both numeric modes, most recently in the 2026-08-28 legs at
  `a0a0eee`: two identical OLS fits in one process disagree at coefficient 0
  (`0xbbb60202` vs `0xbbb87825` on the H100 under FAST). The gate's own text
  is the right reading of it -- nothing on that path uses a float atomic, so
  this is not an ordering hazard but an uninitialized read or a race, and it
  is a defect in BOTH modes. It has NOT been reproduced on Apple, which is
  the only platform this wheel targets, and the card fixture's two fits agree
  so there is no traced repro yet. Open since E3 leg 10; localization owed.

### Packaging

- The wheel now carries TEN extensions in THREE numeric tiers rather than
  five: the 0.1.0 set plus `_mojolearn_svm.so`, `_mojolearn_solver.so`,
  `_mojolearn_metrics.so`, `_mojolearn_tsa.so` and `_mojolearn_linalg.so`,
  each built fast, deterministic and identical by
  `packaging/macos/build_release_wheel.sh`, which refuses a stale file by
  mtime.
- Requirements are unchanged: Apple silicon from the M1 up, macOS 11 or
  later, Python 3.10 through 3.14, numpy 1.24 or later, and a GPU. There is
  still no CPU path and no Linux wheel.

## 0.1.0 (2026-08-23)

First public release. One wheel, `pip install mojolearn`, macOS arm64,
Python 3.10 through 3.14, GPU only.

### Estimators

- `GradientBoosting` (CatBoost GPU oblivious trees; 12 losses plus
  MultiClass; depthwise and lossguide growth; CTRs; eval sets; overfitting
  detector; save and load)
- `RandomForestClassifier`, `RandomForestRegressor` (cuML design)
- `ExtraTreesClassifier`, `ExtraTreesRegressor` (cuML design)
- `KMeans`, `NearestNeighbors` (cuVS, RAFT, FAISS designs)
- `DBSCAN`, `PCA`, `TruncatedSVD`, `LinearRegression` (cuML and RAFT designs)

### Numeric modes

- `FAST` (default) and `IDENTICAL` ship in the same wheel as two binary sets;
  `MOJOLEARN_NUMERIC_MODE=identical` at import selects the identical set and
  `mojolearn.numeric_mode()` reads the loaded mode back from the binary.
- Cross-vendor certificates at this release's source: E2 sub-feature matrix
  99 cells, 0 divergent, Apple M4 (Metal) against NVIDIA H100 (CUDA) and
  against AMD MI325X (HIP); see `E2_RESULTS.md`, `E1_RESULTS.md` and the
  ledger `IDENTITY_PATHS.md`.

### Packaging

- The release build compiles all five extensions in both numeric modes from
  a clean checkout, stages the MAX runtime dylibs once for all ten, and gates
  the finished wheel by installing it into a clean venv under every claimed
  interpreter and fitting every estimator family in both modes.
- Built extensions are no longer tracked in git.
- One release workflow, `release-provenance.yml`, on an ephemeral self-hosted
  runner started by `tools/release_runner.sh`; the runbook is
  `docs/PYPI_RELEASE.md`.
- `.zenodo.json` and `CITATION.cff` carry the author ORCID; every GitHub
  release is archived on Zenodo with a DOI.

### Known limits

- No CPU path. No Linux or NVIDIA wheel yet (source builds only).
- Performance measured on one M4; see the README benchmark section and its
  stated accuracy gap on higgs GBDT.
- `MultiClassOneVsAll` is not on the Python surface.
