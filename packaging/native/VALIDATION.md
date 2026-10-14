# Native package validation, 2026-10-14

Source base: `37586645dcefaeb6ca87a3e1e79e462140036435` (local `main`, including
the commits added after this worktree was created). Toolchain: Mojo 1.0.0
`ed45d567`, MAX 26.5.0. Execution host: Apple M4, macOS 26.5.2.

| Check | Result |
| --- | --- |
| Packaging unit tests: closure, missing imports, Python rejection, relocation, entry points | 6 passed |
| Complete native package precompile | Passed; 33 entry points, 945 source modules |
| Installed package outside checkout: default mode, CPU GEMM, Adam, model configuration | Passed |
| Apple GPU k-means fit and predict, `identical` | Passed |
| Apple GPU k-means fit and predict, `fast` | Passed |
| Both numeric mode flags at once | Rejected as expected |
| NVIDIA `sm_89` and AMD `gfx942` GPU consumer compilation | Both passed |
| `fast` on NVIDIA and AMD targets | Both rejected as expected |
| CPU/GPU sample output words against unmodified source at the base commit | Both matched exactly |
| Conda dependencies: osx-arm64, linux-64, linux-aarch64 | All solved |
| Clean osx-arm64 conda build and isolated installed-package mode/CPU tests | Passed, including automatic import discovery without `-I` |

These are packaging and representative consumer checks, not a new full numerical
qualification or measurement campaign. NVIDIA/AMD binaries were cross-compiled
on the Mac and were not executed on those devices. Linux CPU execution is left
to the platform build jobs; no result here claims it ran locally.

The first precompile failed because the tokenizer's generated table had not been
included, a shared module contained a standalone `main`, and a dormant
normalization kernel passed `Bool` across a boundary requiring `DevicePassable`.
The package now runs the existing pinned Unicode generator, preserves embedded
checks under private names, and passes that kernel predicate as `Int32` before
converting it back to a predicate. Numerical formulas and experiment defaults
are unchanged. The first GPU consumer also needed explicit pointer-origin casts;
the corrected example retains its array owners through synchronization.
The first conda build exposed the separation between build tools and host MAX
libraries. Adding the host prefix to the precompile import path fixed it. That
failed build was stopped after the missing-MAX diagnostics were established;
its full log and work directory were retained.

Full logs, original failure records, native binaries, and source hashes are
retained at:

`/Users/andrewhendel/mojolearn-evidence/native-mojo-community-20261014/`

Key records:

- `build-3.log`: successful precompile.
- `stage-tests-final.log`: packaging test exit/coverage.
- `consumer-2/results.tsv`: every selected consumer check, including expected failures.
- `validation-2.log` and `consumer-2/*.log`: full compiler and execution logs.
- `reference/main-sha.txt`, `reference-comparison.log`, `reference/*-comparison.log`: source comparison.
- `install/share/mojolearn/native-sources.json`: original and relocated source hashes.
- `environment.json`: hardware, compiler and built package hash.
- `solve2-*.log`: platform dependency resolutions.
- `conda-build-2.log`: clean conda build, isolated installation and passing tests.
- `conda-output/osx-arm64/mojolearn-0.8.44-h60d57d3_0.conda`: built distribution artifact.
