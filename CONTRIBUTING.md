# Contributing to mojolearn

Contributions are welcome. Bug reproductions, documentation, tests, hardware
reports, estimator work, performance improvements and numerical audits all
matter.

New here? [docs/START_HERE.md](docs/START_HERE.md) covers the path from a
clone to a merged change.

## What you need

One machine with [pixi](https://pixi.sh). A GPU supported by the Mojo
toolchain (Apple, NVIDIA or AMD) lets you build and check the GPU paths.
Routine checks also run on the CPU, and CPU-only installs train and predict.
You do not need to rent hardware or own a second vendor.

One machine closes everything except a cross-vendor identity claim. Bug
fixes, host oracles, separating fixtures, sabotage arms, documentation,
performance work on your own hardware and new estimators all fit on one box.
Cross-vendor certification is a maintainer job. **Mark the columns you did
not run `cross-vendor-pending`.** That is a complete contribution. Never infer
a column you did not execute.

Build scripts do not target any one machine. `bindings/build_linalg.sh` pins
`--target-cpu apple-m1` and the Linux builds pin `x86-64-v3`, so wheels run on
other people's hardware. A source build with no `MOJOLEARN_GPU_ARCHS` set
targets your GPU.

## Before opening a pull request

1. Open an issue first for changes to a public API, a numerical profile or an
   algorithm boundary.
2. Keep the change focused and leave unrelated work in the tree alone.
3. Add the smallest test that would have failed before the change.
4. Run the relevant checks for the module you changed (see
   [docs/START_HERE.md](docs/START_HERE.md) and
   [docs/TEST_RUNTIME.md](docs/TEST_RUNTIME.md)).
5. State which hardware and numeric modes you actually ran.

The [pull request template](.github/PULL_REQUEST_TEMPLATE.md) asks for the
reference for any published algorithm the change implements, affected public
APIs, effects on the `fast` and `identical` modes, tests and adversarial
fixtures, hardware columns exercised, and performance evidence when a
performance claim changes.

## Engineering rules

These rules apply to every change. The section titles are stable so code and
documents can cite them.

### Numerical identity

Any change capable of moving `identical` bits must name the numerical-profile
clause or [IDENTITY_PATHS.md](IDENTITY_PATHS.md) row it affects and provide
one of the following.

- Proof that the change is bit-inert.
- A separating fixture, with the numerical profile version bumped.
- A named refusal that prevents an unsupported configuration from claiming
  the contract.

Changes to reductions, RNG mapping, arithmetic contraction, denormal policy,
tie-breaking, serialization and dispatch never merge because ordinary
correctness tests pass. They need explicit review of the numerical DAG.

### Evidence must be able to fail

A test counts as evidence only if a sabotage or separating arm has been shown
to make it fail. When a numerical pin is added, its fixture must first
distinguish the pinned and unpinned spellings. A hash that passes under both
has measured nothing. Uniform or random data can hide a permutation, so
fixtures should be able to expose the defect they guard against. A check
must also prove it reached the code it claims to test. A file no caller
reaches is not done.

### Non-default paths

Every switch is exercised on both sides by a named check that sets it
explicitly. A number taken on a non-default path is provisional until a check
has run that path, and a benchmark prints the path it took. A switch measured
better and identical in output becomes the default. Reach is per-branch.

### Numeric modes

`fast` and `deterministic` ship only for the tree learners (`gbdt`, `rf`,
`trees`). Everything else ships `identical` only, and a withdrawn mode is a
named refusal, never a silent absence. Report our `identical` result against
the opponent's fastest configuration. Our `fast` over our `identical` is an
internal cost, never a result.

### Performance claims

Every speed or quality claim for tree and classical estimators runs on the
same two real datasets, which differ in kind.

- **NYC TLC yellow taxi trips** (January and February 2024). A narrow,
  mixed-type table with missing values, a temporal split, and classification
  and regression targets.
- **Istella-S LETOR.** A wide, numeric, imbalanced table with 220 features
  and graded relevance.

Trees run both at 1,000,000 rows or more. Classical estimators use each
dataset at the shape where the kernel, not launch or transfer, dominates.
Held-out quality is recorded beside every timing. A change becomes the
default when the geometric mean of its time ratios over the two datasets is
below 1 and quality is not worse on either dataset beyond noise
(`tools/flip_verdict.py` prints the verdict). A win on one dataset with the
other unmeasured is not a result. HIGGS and synthetic generators do not
count. Neural claims use two different standard corpora, enwik8 and the
GitHub component of the Pile. Name the opponent's threading and BLAS before
quoting a ratio, never mix GPU models in one ratio, and report measured
ratios rather than general speed claims. There is no fixed percentage
threshold for performance work. Review weighs measured benefit, generality
and regressions against complexity. The full policy is the
[performance acceptance policy](docs/PERFORMANCE_ACCEPTANCE.md).

### Comparing against libraries without a GPU path

The opponent is the fastest thing a user can run on the same box. A library
with a GPU path on that vendor (for example PyTorch ROCm or XGBoost ROCm on
AMD) is measured on the GPU. A library with no GPU path for that vendor (for
example cuML or CatBoost GPU on AMD) is measured on the same box's CPU using
all cores, and the row is labeled CPU. Every row and table cell names GPU or
CPU. Opponent rows are measured once per GPU model, driver, opponent version
and dataset, and recorded in
[bench/OPPONENT_REFERENCE.md](bench/OPPONENT_REFERENCE.md).

### Algorithms and references

Every line of Mojo here is written for this repository. Do not paste code from
another project, whatever its license. Where a contribution implements a
published algorithm, name the reference in the file so its behavior can be
checked against it, and check against the path the reference's dispatch
actually takes for the parameters in question. Do not reproduce a reference
library's bugs. Where a lane keeps a `NOT_IMPLEMENTED.tsv`, record what the
contribution leaves out.

## Automatic checks for external pull requests

External pull requests to the default branch receive two reports from
[External contribution checks](.github/workflows/external-performance.yml).
Owners, members and collaborators are exempt.

- The **admission report** reads changed-file metadata under the trusted base
  commit's policy without fetching or executing the pull request. Narrow
  optimizations of existing code receive `GPU_PENDING`. Changes to
  infrastructure, dependencies, tests, contracts, public APIs or other files
  receive `REVIEW_REQUIRED`.
- The **hosted CPU report** runs packaging, version and CPU-baseline checks
  and comparator negative controls on a disposable GitHub-hosted machine with
  no credentials. It does not build Mojo or run GPU code.

A green CPU report does not close `GPU_PENDING` or permit automatic merging.
No GPU fleet or automatic merge is configured. Untrusted pull request code
never runs on the [manual GPU workflow](.github/workflows/gpu-validation.yml)
or the [release runner](tools/release_runner.sh). Maintainers can run the
admission negative controls locally with
`python3 -m unittest discover -s tools -p test_external_contribution_gate.py`.

## Repository size

No committed file may exceed 50 MiB, and wheels, tarballs and fixture dumps
never go under `bench/results/`. Large raw evidence belongs outside the
repository. Commit a summary, the sha256 of the raw archive and where it
lives. Install the hooks once per clone.

    sh tools/hooks/install.sh

`tools/hooks/pre-commit` refuses oversized or forbidden files, and
`tools/hooks/pre-push` refuses a push carrying any blob over 100 MiB.

## Maintainership and license

Contributors who repeatedly show sound review, preserve the numerical and
provenance contracts and help other contributors may be nominated as
reviewers and then maintainers. See [GOVERNANCE.md](GOVERNANCE.md).

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license and that you have the right to submit it.
