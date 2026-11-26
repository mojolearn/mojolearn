# CI/CD speed-up: handoff (2026-11-26)

Branch `claude/keen-noether-iaze9n`. It is main (9bdb1a9e8c) plus `docs/release-only-work` (today's rules),
`lane/verify-profiles`, the integration branch's test fix 3423ae2 (cherry-picked), and the work below. Nothing is
merged to main. Andrew asked for this work to be merged to main when it is done; it is not done.

Goal (Andrew): move faster with appropriate checks. No check is added. Things are removed or made cheaper.
Read AGENTS.md part 1 "RELEASE-ONLY WORK" first: a round runs the compile, the tests and the GPU smoke only.

## What was found

1. Every workflow is manual dispatch. A round is `tools/release.py candidate` (GitHub build, wheels only) plus box
   jobs the operator submits: `cpu_box_checks_job.sh`, `native_checks_gpu_job.sh`, `proving_job.sh`,
   `gpu_smoke_job.sh` (all in `tools/ops/`).
2. The wheel build already reuses unchanged groups across runners through `tools/bincache.py` and R2. The
   native-check programs had no reuse: built on GitHub (CPU column), again on the CPU box, and the GPU column on the
   L40S, in full every time.
3. Reuse would hit. Over main's last 45 commits (`tools/ops/group_reuse_count.py`): 96 percent of the 90 development
   binding group keys unchanged commit to commit; 35 of 44 commits changed no group. Native-check group programs
   (14 a column): 98 percent (CPU) and 99 percent (GPU) unchanged commit to commit; 7 of 14 (CPU) and 11 of 14 (GPU)
   unchanged 44 commits apart. No unresolved import in any check file's closure (3,735 files parsed).
4. A check program runs from another checkout only with the current toolchain's library directory on
   `LD_LIBRARY_PATH` (its runpath is the build checkout's). Proved with Mojo 1.2.0: built, build directory deleted,
   ran and passed elsewhere; rc 127 without the path. It also embeds the build checkout's source path (failure
   messages only). The check programs already use the wheel's x86-64-v3 baseline (`native_testsuite.sh:65`).
5. The development group key holds absolute paths on purpose (`tools/ops/README.md`: "Repository and cache paths are
   part of the key because they can reach compiler/linker output"). Finding 4 confirms the outputs do hold paths.
6. Release-only items still on the round path (file:line as of this branch):
   - `tools/ops/wheel_verify_smoke.sh:71,74`: `selftest` in the default STEPS on both sides (`verify --self-test`
     compares `ols/base` with the shipped table). Runs after every build (verify-smoke.yml, `gpu_smoke_job.sh`).
   - `tools/ops/feature_proof_job.sh:274-296`, run by `proving_job.sh:53`: `verify --self-test`,
     `--cross-check quick`, `--check`, `--fit-both`, ten `--recipe ... --self-test --record-out`.
   - `tools/ops/box_job_lib.sh:244`: the fault gate is on by default (`PC_FAULTS:-1`): it builds and runs sabotage
     arms and the self-test in every `paper_compare_job.sh` run (6 to 9 pod minutes the first time per commit).
   - `wheel_verify_smoke.py:629-639` runs every lane through `mojolearn._identity_break --repeats 1`: this is the
     smoke's "every lane once"; it compares nothing. Left as it is.
7. Workflows nothing dispatches and no doc names: `wheel-ci.yml` (its own header says deleted rather than repaired)
   and `gpu-validation.yml` (targets self-hosted runners that do not exist). `forest-host-gate.yml`,
   `byte-lm-cpu-gate.yml` have no tool dispatcher but docs name them; `python-tests.yml` is the Apple column.
8. `gpu_smoke_job.sh:59-65` compares the wheel's commit with `git rev-parse HEAD`. After `dev_pod.sh sync` HEAD is
   the merge base with main, so an integration commit not on main would be refused. `native_checks_gpu_job.sh:53-66`
   already reads `.git/devpod_synced`. Read from code, not seen in a run.
9. `proving_job.sh` never checks that its wheels match the tree; `native_checks_gpu_job.sh` checks the commit only
   when `NC_EXPECT_COMMIT` is set.

## Done on this branch

- `tools/ops/native_testsuite_cache.py` (new): `keys`, `get`, `put`. Key contents are in its docstring.
- `tools/ops/native_testsuite.sh`: keys after the plan when `NC_TS_CACHE` is set; `compile_unit` copies a hit and
  stores a successful compile; the build's last line counts reused units; `run` sets `LD_LIBRARY_PATH`.
- `tools/ops/cpu_box_checks_job.sh` and `native_checks_gpu_job.sh`: `NC_TS_CACHE` defaults to a directory on the box
  (`$CPU_CHECKS_HOME/ts-cache`, `${PC_CACHE_ROOT:-/root/pc-cache}/ts-cache`); empty turns it off.
- `tools/ops/group_reuse_count.py`: the count of finding 3 for the binding groups.

State of that work: `bash -n` passes on the three scripts. NOT run end to end and no tool test exists. The scratch
run stopped before any compile, in `prepare` (generated sources), a fault of the test setup, not of the change.

## Left to do, in order

A. Prove the cache (needs the compiler; `pip install mojo==1.2.0` works where conda.modular.com is blocked).
   Scratch recipe: two copies of the tree (A, B); in each, `.pixi/envs/default/bin/mojo` and `.pixi/envs/default/lib`
   linked to the compiler and its `modular/lib`; a `pixi` shim on PATH (`run` = exec the command with the compiler on
   PATH; `exec --spec python=3.14 ...` = exec with a Python whose unicodedata is 16.0.0). Then
   `NC_TS_CACHE=<dir> NC_TS_GROUPS=2 IN_FILTER=hostptr_check,histograms_helper_check,gbdt_partition_route_check
   bash tools/ops/native_testsuite.sh build cpu <bin> <out>` and show, with logs:
   1. first build in A compiles and stores; the same build in B reuses every unit and `run` passes each check in B;
   2. editing a source file one group imports recompiles that group only;
   3. changing a define (`NC_DEFINES`) or the compiler bytes changes every key;
   4. a flag or environment naming SABOTAGE or FAULT_INJECT prints no keys;
   5. a truncated `bin` in the cache is a miss, not a run.
   The first scratch attempt failed in `tools/generated_sources.py prepare` (`generator exited 127`, then again after
   the shim learned `exec`): read `build/generated-sources/*/generate-unicode.log.attempt1` first.
B. Tool tests `tools/tests/test_native_testsuite_cache.py` for A.2 to A.5 at the key level (no compiler: a temp tree
   with a fake `.pixi/envs/default/bin/mojo`), a `get`/`put` round trip, and one test that `unit_keys` runs on the
   real tree (it uses `SourceGraph.nodes`, `.hashes`, `.hash_files`, `.parse_all`). Update
   `tools/test_native_checks_ci.py` where it pins lines of the three scripts. Run `bash tools/run_tool_tests.sh`.
C. Release-only defaults (finding 6), each one line plus its doc comment and tests:
   - `wheel_verify_smoke.sh`: drop `selftest` from both default STEPS; the release passes STEPS with it.
   - `feature_proof_job.sh`: section 3 (verify options) only when `FP_RELEASE=1`.
   - `box_job_lib.sh`: `PC_FAULTS` default 0. `tools/test_gpu_fault_gate.py` does not pin the default.
   Each touches verify tooling: say so and say how verify was exercised (AGENTS.md "The verify command is protected").
D. Delete `wheel-ci.yml` and `gpu-validation.yml` with what exists only for them (`packaging/wheel_ci.py`, the
   `pixi.toml` task, the mentions in `CONTRIBUTING.md`, `bindings/build.sh`, `native_checks_steps.sh`,
   `apple-fast-check.yml`, `python-tests.yml`, `byte-lm-cpu-gate.yml`: read each mention first).
E. Commit checks (findings 8, 9): `gpu_smoke_job.sh` and `proving_job.sh` read `.git/devpod_synced` the way
   `native_checks_gpu_job.sh:53-66` does and refuse a wheel of another commit. No fallback to compiling anywhere.
F. Draft PR to main, then merge when A to E are green (Andrew asked for the merge).

## Not started, and why

- Sharing the native-check cache between boxes through R2: the R2 credentials never leave the Mac
  (`tools/box_persist.py`), so a box needs presigned URLs minted at submit, as `tools/bincache.py` does. The directory
  layout (`<key[:2]>/<key>/{bin,meta.json}`) is ready for it. Worth doing after A shows the box-local hit rate.
- GitHub `native-checks.yml` reuse (`actions/cache` over the same directory): the size of 14 group programs a column
  is not measured; the repository cache limit is 10 GB.
- A path-free development group key: finding 5 says the path is in the key for a reason. It needs a build that shows
  a relocated binding `.so` loads and gives the same bytes; do not change the key before that.
- Sabotage seeded from the clean build: sabotage is release only now, so it is off the round path.
- Host groups compiled by more than one leg of `release-linux-build.yml` on a cache miss
  (`tools/release_github_build.py:119-128`): inferred from code, not measured.

## For Andrew to rule on

1. `selftest` in the round smoke: the hard rule says no verify against references in a round; the verify rule says
   the smoke is how a change shows verify still works. C removes it from the round default. Say if it stays.
2. The Apple column early: allowed only after the freeze (CPU on GitHub, then Apple, then the rented GPUs).

## Rules for whoever continues

Lanes write code and push; they do not rent, dispatch or run box jobs. Keep logs out of context. Bits can change;
they must match across vendors within one version. No Python fallbacks. Never `git push --no-verify`. No model name
in commits. Push only to `claude/keen-noether-iaze9n` until the merge.
