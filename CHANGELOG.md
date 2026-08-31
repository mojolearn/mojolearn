# Changelog

## 0.3.2 (2026-08-31)

A new estimator, and 1,138 lines of never-executed code taken back out of the
wheel. Nothing in the 0.3.1 surface changes its numbers: the GBDT fix in this
release is on the POINTWISE searcher, which the Python surface does not reach
(`GradientBoosting` runs the greedy subsets searcher), so it is a correctness
fix in the tree rather than a behavior change for anyone installing this.

### Added

- **`mojolearn.RadiusNeighbors`**, every neighbor inside a radius, over
  cuVS's random ball cover (`neighbors/derived/neighbors/ball_cover/`). The
  index has answered DBSCAN's eps neighborhood since 0.1.0 and had no
  caller-facing surface; it does now. The results are EXACT, not approximate:
  the ball cover's pruning is a triangle-inequality bound, and
  `neighbors/original/radius_check.mojo` asserts the set against a host
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
  call. `neighbors/original/radius_distances.mojo` carries the reasoning and
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
  a `<binding>_vendor()` export on every binding (`original/vendor.mojo`, a
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
  C_SVC` raises by name in `svm/derived/svm/svm_parameter.mojo`, and
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
  `glm/original/ridge_check.mojo`, `glm/original/logistic_check.mojo`,
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
that exists stops, rather than an `AttributeError`. (`RadiusNeighbors`
shipped after this release; see the unreleased section at the top.) `ARIMA` in particular
has no `fit` -- `arima/` ports the batched Kalman filter likelihood, its
gradient and predict, but `estimate_x0` and the batched L-BFGS driver are
NOT PORTED, and those are exactly what produces the coefficients every
existing entry point requires as input.

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
only thing keeping it out. `spectral` and `holtwinters` have an Apple card
and an AMD card; `tsa` has an Apple card.

What is missing is a SHARED COMMIT. The Apple cards for `spectral` and
`holtwinters` were taken at `5fd95b3` and `241aed6`, the AMD cards at
`26eb8ba`, and `tsa`'s Apple card at `869d416`, so no two of them are
comparable and no NVIDIA leg has run any of the three. Their pins come from
the same source as the lanes above, so they are expected to match, but the
comparison has not been performed and each class says so rather than
inheriting a neighbor's certificate.

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
flag over one source: `GLOBAL_NUMERIC_MODE` in `original/numerics.mojo`, from
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
  make no identity claim, so nothing was judged), 4 CANNOT RUN.
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
  did, GPU idle, every host thread in futex wait. Both are fixed in this
  tree; 1944 is confirmed on the box, **1946 is UNRUN on a 4090** (the fix
  cannot be exercised on Apple, AMD or the H100, none of which ever showed
  the defect). Apple is the only platform this wheel targets and it was
  never affected. `bench/results/identity_break/RESULTS.md` carries the
  evidence, the discriminator
  (`ensemble/original/rf_ctx_order_probe.mojo`) and the RUN OWED list.

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
