# Mojolearn handoff — weekly limit, 2026-11-11

## READ FIRST: status update and plan for the next agent (2026-11-11 15:25 ET)

Added by the Claude orchestrator session after Codex reached its weekly limit. The original Codex handoff follows
below, unchanged; where the two differ, the live status files win over both.

### Where things stand

- **main** = product code of `c6dbe7ced4` (Mojo 1.0.0 / MAX 26.5.0 pins) plus documentation commits only. None of
  the 2026-11-11 product work is on main.
- **All new product work** is on `preinteg/native-consolidated-20261111` (`33b043de8b`): the Mojo 1.2.0 / MAX 26.7.0
  upgrade, 54 more migrated roles (GP, SVM incl. OneClassSVM, Ridge/OLS), the first native entry table
  (Ridge/OLS only), the IPCA workaround removal, the OPTICS capture fix, `docs/MODULAR_ASKS.md` updates, and the
  small-tooling merge (`%2B` wheel-install fix, istella_rank staging, lane_select pandas-gate fixes).
- **The r2 build under Mojo 1.2.0 is still running** (controller PIDs 96835/96837; status
  `~/mojolearn-evidence/python-mojo-migration-20261110/consolidated-120-r2/build/result.json`). At 15:22 ET:
  **8 of 25 units compiled, 10 failed, 7 unfinished.** The controller merges only if all 25 compile, so it will
  stop without merging. Let it finish; do not start a second build.
- Failure classes in the r2 logs (count of error lines across the failed logs):
  - 1.2.0 API changes the source has not caught up with: `module 'intrinsics' does not contain 'ldg'` (14),
    `top_k` `k` argument no longer accepts `None` (`knn_brute_force.mojo:1648`, units `identical-_mojolearn_native`
    and `_m1`), `matmul` `elementwise_lambda_fn` type change (`fast_trsm.mojo:344`, unit `identical-..._trees`),
    indentation errors in two `except` blocks (orchestration and tokenizer host units). Ordinary source fixes.
  - `_cpython.mojo:2208:24: error: operand #1 does not dominate this use` (12, units byte_trainer, classical_trees,
    classical_family, classical_prep, classical_decomp hosts). Raised inside Mojo's own CPython layer around a
    `kgen.call_indirect`: possibly a new Mojo 1.2.0 compiler bug. Not reduced yet.

### Plan, in order (each step names the test that decides it)

1. **Finish r2, then fix the compile errors in one batch** on a new frozen SHA; rerun as attempt `r3` through the
   same controller and builder (reuse compiled units). Done when: 25/25 Mac units compile.
   - For "does not dominate": reduce to the smallest standalone file. If it reproduces on Mojo 1.2.0 with only
     stdlib/MAX imports, search `modular/modular` issues and file it (format of #7543). Do not add a Python
     fallback, lower optimization or patch the toolchain (AGENTS.md "Wait for Mojo and Modular").
2. **Merge to main** after 25/25 compile, the packed wheel imports, and the light smoke passes on the Mac
   (AGENTS.md "Compile-green is not a gate"). Rebase the controller's base check onto the current main head.
   Done when: `import mojolearn` from the main wheel succeeds.
3. **Run the prepared native batch** (`~/mojolearn-evidence/python-mojo-migration-20261110/native-upgrade-120/combined-native/`,
   179 CPU / 127 GPU cases; GPU groups ordered because of modular/modular#7544). Then the entry-table proofs
   (`docs/plans/native_entry_table/native-proof.json`). Done when: the suite manifest's counts all pass.
4. **Measure** current main on NVIDIA (L40S) and AMD (MI300X) against 0.8.55 and 0.8.66 with
   `docs/plans/PRE_0867_MEASUREMENT.md`. The last measurement (AMD only, Nov 9 main `cd02a3f6fe`) had 50 of 102
   recipes timed, geomean 0.92x vs 0.8.55, 13 still slower (iforest fit 1.94x, GBDT pairlogit 1.83x, Samba forward
   1.69x, byte-LM forward/train ~1.2x); 52 recipes did not run because 78 of 174 smoke cells failed (mostly trees),
   and NVIDIA produced nothing (every smoke cell "not run"). Fix the smoke failures first. Done when: both vendors
   have every recipe timed, the toolchain change included.
5. **Native call table across modules** (the largest Python removal per hour): brief
   `~/mojolearn-evidence/briefs-20261111/codex-native-call-table.md`. Main still has about 7,226
   `py_call`/`py_call_typed` sites (Mojo calling Mojo through CPython) and 206 of 207 `bindings/*_native.mojo`
   doors touching Python. Ridge/OLS has the first entry table on the branch; extend family by family, worst first
   (trees meta/boost/gbdt fit, causal LM, byte LM trainer, prep encode, x_linear), deleting each `py_call` site
   and its packing. Done when: the runtime crossing observer shows only UNPACK/PACK crossings per family.
6. **Continue the boundary migration** (original handoff step 7): 26 of 170 estimators reviewed, 144 not reviewed.
7. **Branch and worktree cleanup.** Audit: `~/mojolearn-evidence/branch-audit-20261111.tsv` (341 unmerged refs,
   about 210 branch names older than 2026-11-05; 91 worktrees). Before any deletion, record each losing experiment
   with branch@sha in `docs/TOMBSTONES.md` / `docs/apple-fast/EXPERIMENTS.md` per AGENTS.md "Deleting an
   experiment" (code that never reached main gets branch@sha, no patch). Then delete only refs that are merged or
   recorded. Worktree removal needs Andrew's OK. `lane/nvidia-one-wheel*` look superseded (main ships PTX at
   `cuda/sm_80`); confirm before recording them.

### Modular status (verified 2026-11-11 on Mojo 1.2.0 / MAX 26.7; `~/mojolearn-evidence/modular-check-20261111/SUMMARY.md`)

- Fixed upstream in 1.2.0: IPCA nested-capture segfault, DBSCAN field-capture diagnostic, DeadArgumentElimination
  (#7164). OPTICS is a rule violation in our code, fixed on the branch.
- Filed: modular/modular#7543 (PythonObject runtime default evaluated at compile time), #7544 (Metal: concurrent
  submission through one shared DeviceContext aborts; `num_streams()` is 1 on Apple).
- Not filed (no small reproducer): the compiler deadlock (mutual recursion + default) and "compile offload result
  missing kernelId". The 11 older asks in `docs/MODULAR_ASKS.md` need NVIDIA/AMD/Linux to recheck.

### Standing constraints the next agent must keep

No release, tag, PyPI publish or identity run (Andrew decides releases). Laptop is shared with mambik: compile
through `tools/compile_slot.sh`, at most 4 test workers, never a test during a build. Hand-tuned kernels are not
touched. No Python fallbacks. Do not restore the retired source-count guards. Never delete a worktree without
Andrew's OK.

---

Andrew requested that all worktrees be saved to Git and that running work remain
in flight. Do not interrupt or restart the active build. No release was requested.

This handoff and its backup snapshots were subsequently cherry-picked onto main
as documentation only. The product integration remains unmerged and the running
build remains frozen at the source below. Its controller still expects main's
older `c6dbe7ced4` base: if compilation eventually succeeds, that safety check
will preserve the build and stop instead of overwriting this documentation
commit. After the current job finishes, incorporate the documentation advance
into the integration branch and use the actual current main head for the next
merge attempt; reuse valid compiled units through the existing builder.

## Current source and main

- Combined, pushed branch: `preinteg/native-consolidated-20261111` at
  `33b043de8b94b06d79f750ba6ae4c34533cc594d`.
- Build worktree: `~/mojolearn-wt/integ-python-followon-20261111`.
- Main worktree: `~/mojolearn-wt/main-integration-20261110`.
- Main's product baseline is `c6dbe7ced4cc47c64e170b3e0fa30659f5f857d1`, followed
  by this documentation checkpoint. The new product batch is **not merged**.
  Never infer otherwise from a pushed integration branch.
- This handoff branch inherits the combined source but adds documentation and
  backup snapshots only. Do not replace the running build's source with it.

Read `AGENTS.md` in full. All logs stay on disk; inspect structured status and
bounded diagnostics. Root alone compiles/tests. Do not restore source-count
guards, add Python fallbacks, change tuned algorithms, or run release identity.

## In flight: leave running

The finite controller is already building all 25 Mac identical/host binding
units under **Mojo 1.2.0 / MAX 26.7.0** using the standard development builder.
It uses the normal compile semaphore and compiler-worker policy.

- Controller PID at handoff: `96837`; build controller PID `96853`.
- Tool session, if still available: `16677`.
- Controller: `~/mojolearn-evidence/python-mojo-migration-20261110/integrate-consolidated-120.py`.
- Integration status: `~/mojolearn-evidence/python-mojo-migration-20261110/consolidated-120-r2/integration.json`.
- Build status: `~/mojolearn-evidence/python-mojo-migration-20261110/consolidated-120-r2/build/result.json`.
- Per-unit full logs are beside that build result. The outer controller log is
  `~/mojolearn-evidence/python-mojo-migration-20261110/consolidated-120-r2-controller.log`.

At the saved 20:03 UTC snapshot, **4 units had compiled, 2 had failed, and 19
were unfinished**. Read the live result for the final outcome; this is a snapshot,
not a claim about later completion. The two known failures were:

- `host-_mojolearn_native_byte_trainer_host.log`
- `host-_mojolearn_native_orchestration_host.log`

Both reported `Mojo/stdlib/std/python/_cpython.mojo:2208:24: error: operand #1
does not dominate this use`, involving a `kgen.call_indirect`. The byte trainer
trace includes `byte_lm_host_trainer_native` / `linalg_ops_native`; the
orchestration trace includes `runtime_model_selection_entry` /
`model_selection_native`. This is a compiler diagnostic, not yet an isolated
upstream bug. Preserve the full logs and make the smallest reproducer if it
remains after the complete diagnostics are understood. Do not invent a Python
fallback or lower optimization to get a pass.

The controller merges and normally pushes the **exact frozen source** only if
all 25 units compile successfully and main is unchanged/clean apart from the
owned hold marker. Any retained failure prevents that merge. There are no
installed-wheel smoke or release-identity gates in this development merge.

The main worktree's untracked `PREINTEGRATION_MERGE_HOLD_20261111` is intentional.
It prevented the older controller from merging a partial batch. Its bytes are
backed up below. Do not commit it to main or remove it while a controller expects
it; the current controller handles it only after a successful combined build.

## What is already saved and implemented

The combined branch includes the small-tooling merge and native migration work,
then the toolchain upgrade and simplification:

- Wheel-install provenance accepts pip's encoded `%2B`; lane-selection repairs
  and istella-rank staging instructions are included. Dataset staging itself and
  NVIDIA/AMD remeasurement are still owed.
- Native GP and SVM state/transport changes, including OneClassSVM. Audited total:
  **332 distinct migrated roles in source, 278 on earlier main, 54 awaiting this
  integration**. These are roles, not allocation counts or a completion percentage.
- Ridge/OLS native entry tables, native state handles/leases, lifetime admission,
  and metrics tier/vendor checks. Existing tuned provider operations remain.
- Exact Mojo/MAX pins and lockfile, matching package/adapter requirements.
- Current GPU imports, array/origin names, memory/atomic APIs, compile-time
  control syntax, sort API, GP `Sized` declaration and QN request ownership.
- Original nested IPCA test storage restored; six-buffer compiler exception
  removed. OPTICS captures one outer result owner without copying its buffers.
- One inline `core/native_string.mojo` helper replaces 27 repeated retired
  constructors. One `core/accelerator_target.mojo` module provides compile-time
  vendor constants while preserving accelerator-free CPU behavior.
- `docs/MODULAR_ASKS.md` records the upstream fixes and filed #7543/#7544.

Independent source audits checked numerical bodies, route/default preservation,
copy expressions, pointer lifetimes and API substitutions. They are not runtime
or speed evidence. OneClass has 12 CPU / 22 default GPU native cases and 35
Python boundary cases authored, not executed in this batch.

## Accepted evidence versus pending work

- Older source `183c32da489eaa413715617bfe9caf1d8bbfb6a2` compiled 25/25 Mac
  binding units on Mojo 1.0. Its merge was intentionally held. Those binaries do
  not qualify Mojo 1.2.
- First 1.2 attempt, source `6ebb6c98ca89f72e236050b0dd62af65b904c695`, failed
  all 25 units on shared API diagnostics. Those repairs are in the current r2
  source. Its full evidence is `consolidated-120-r1/`.
- Tooling's frozen full run: 183 passed, 3 expected failures, 1 failure. The
  repaired dependency-membership case subsequently passed its focused rerun.
- Packaging/adapter pin controls: 40 passed. Python crossing observer controls:
  5 passed. These do not prove product runtime or GPU behavior.
- No new NVIDIA/AMD compilation, performance measurement, cross-vendor identity,
  installed-wheel qualification, release, tag or PyPI publication was performed.

## Resume in this order

1. Check the live r2 status and logs. Leave unfinished compilation alone. Preserve
   every failure and accepted unit. Do not start a duplicate build.
2. After the controller finishes, repair the remaining compiler errors in a
   coherent batch. Reuse compiled units only through the existing builder's
   source/compiler/flags/target/dependency checks. The root controller accepts a
   new frozen SHA, unchanged main base, and a fresh attempt name (`r3`, etc.).
   Its saved copy is under `20261111-weekly-limit/resume-evidence/`; do not run
   that copy blindly while r2 is active.
3. Complete the authorized combined development merge/push after successful
   compilation. Preserve intervening main changes if another session advanced it.
   Compilation is the development gate, not a claim of runtime success.
4. Execute the prepared native batch separately from the merge claim:
   `~/mojolearn-evidence/python-mojo-migration-20261110/native-upgrade-120/combined-native/`.
   It contains CPU/GPU wrappers and an exact suite manifest: **179 CPU / 127 GPU**
   at default flags, or 117 GPU with OneClass resident disabled. GPU groups are
   ordered because shared Metal-context submission remains broken; kernels stay
   parallel. Record the final source SHA and actual skips/counts before running.
5. Run the 20 entry-table contracts as included in that batch. Run the separate
   production handle→table→predict proof in a fresh process: 2 CPU graphs and
   1 missing-publication precondition. Build provider and consumer as separate
   shared images for the cross-image proof; use `run-entry-table-images.py`.
   Never combine those images or run the fresh-publication proof in an already
   populated process. Recipes are in `docs/plans/native_entry_table/native-proof.json`.
6. Execute remaining authored Python boundary seams; remove superseded Python
   numerical tests only after the corresponding native cases actually pass.
   Boundary tests and independent Python oracles remain legitimate testing tools.
7. Continue the Python-at-the-boundary migration family by family. The whole
   library is **not Python-free**. Owed examples: OneClass coefficient weak-cache
   tuple, SVC's eight inventoried roles, SVR/LinearSVR scores, shared weighted/
   precomputed scratch, general NumPy/pandas conversion, and tree/meta/GBDT native
   call-table consumers. Retain necessary public exporter/image pins at UNPACK.
8. The crossing observer cannot see every direct C-API call; zero observed events
   are not proof of a Python-free operation. Finish runtime crossing evidence and
   NVIDIA/AMD measurements before making speed claims. Identity happens only once
   at the eventual release freeze, when Andrew requests release work.

## Git safety checkpoint

All 23 originally checked-out named branch heads were verified against origin;
the six previously missing/behind remote refs were pushed normally. The 11
detached worktrees have named `checkpoint/weekly-limit-20261111/...` backup refs.
The accompanying manifests record their exact paths, heads and remote evidence.

All 13 dirty files across four worktrees were copied byte-for-byte into
`20261111-weekly-limit/worktree-snapshots/` and indexed with SHA-256 hashes in
`worktree-snapshots.json`. They are generated measurement/verification JSON and
the owned main hold marker. Their original working copies, indexes and heads
were left untouched so in-flight jobs and historical references are not altered.
These are backup snapshots, **not accepted measurement or identity records**.

The current product code is on the pushed integration and producer branches.
Ignored build binaries/caches and full raw logs remain on disk. No worktree or
old branch was deleted during this checkpoint; the earlier branch-audit losers
still require permanent recovery records before any pruning.

Selected resume scripts, native wrappers, audits and status snapshots are saved
under `20261111-weekly-limit/resume-evidence/`. Original evidence roots remain:

- `~/mojolearn-evidence/python-mojo-migration-20261110/`
- `~/mojolearn-evidence/modular-check-20261111/`
- `~/mojolearn-evidence/weekly-limit-20261111/`

Use the live controller result rather than this static handoff to decide what
actually completed after the weekly limit.
