# TEMPORARY. Subagent workstreams handed from the mojolearn-ea session to the harness owner (mojolearn-99), 2026-09-14 ~14:30 ET. Delete when every item below is merged or moved into a lane brief.

Andrew asked that management of these workstreams move to the peer session. The
mojolearn-ea session keeps ONE item: the 136-lane record (section 4) and its
gate-column switch, whose last MI300X leg is on a box. Everything else here is
yours to drive, with your own subagents; each brief below is the exact prompt the
killed agent received, so a fresh agent can start from it. Rules that bound every
agent: isolated worktree off origin/main, code only, ONE core on the Mac (nice 19,
MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1 OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1),
no gate or harness runs beyond one-lane base-fixture simulations, pixi env borrowed
by symlink, never `pixi install`, never touch another worktree, explicit-path
commits only, no push; the parent runs the boxes and merges when a gate is green.

## 1. lane/expose-d (workstream D), pushed at 2b2f568b0, NOT mergeable yet

What it holds: Cholesky (in the GP binding, mojolearn.linalg.Cholesky), KernelRidge,
Nystroem, RBFSampler (new _mojolearn_kernel_methods), GaussianMixture (new
_mojolearn_mixture), HDBSCAN (new _mojolearn_hdbscan), bootstrap, permutation_test,
monte_carlo_integrate (new _mojolearn_resample), the six training primitives in
mojolearn.training, KMeans metric and oversampling_factor arms, IVF-FLAT and the
embedding sabotage arm prepared but not exposed, and the harness lane bodies
docs/lanes/LANE_BODY_{cholesky,hdbscan,ivf,kernel_methods,kmeans,mixture,resample,training_primitives}.py
(fifteen lanes; you integrated them on lane/claim-surface-lanes already). The
_NOT_YET rows for HDBSCAN, GaussianMixture, KernelRidge, RandomFourierFeatures,
Bootstrap and cholesky are stale on that lane (register owner's edit).

First GPU run, Hot Aisle MI300X gfx942 at 2b2f568b0 (leg directory named in the
mojolearn-ea scratchpad file expose_d_leg_dir.txt; its remote/ has build_*.log,
test_*_surface.log, check-*.log, extra.log; the useful lines are reproduced here):
- builds: gp, kernel_methods, mixture, resample, base OK; **build_hdbscan.sh CRASHES
  THE MOJO COMPILER** (exit 139, "Running pass 'AMDGPU DAG->DAG Pattern Instruction
  Selection' on function '@hdbscan_impl_detail_stabiliti...'"); the same binding
  compiles for Apple.
- surface tests: cholesky GREEN (23), resample GREEN (34), kmeans_metric GREEN (12);
  **mixture RED** "FIT init_params='random' also separates the blobs";
  **kernel_methods RED x2** "NYS with every row a component, phi phi^T reproduces the
  linear kernel at 1e-2 -- 2.851021765361576" and "REFUSE RBFSampler n_components=0,
  refused on the Mojo host by name -- message: mojolearn: null float32 buffer address";
  hdbscan not built; training_primitives did not run because the leg body forgot
  bindings/build_training.sh (parent's omission).
- checks: check-cholesky "[FAST] ALL PASSED"; check-kernel-methods was still compiling
  when the leg's 60-minute bound hit, so check-mixture, check-hdbscan, check-resample
  have no result. Bound the checks (small sizes) or split them across two legs.

The brief the killed D-fixes agent received (verbatim, adjust the branch name):

> Fix what the first GPU run of WORKSTREAM D found. Setup: `git fetch origin && git
> checkout -b lane/expose-d-fixes origin/lane/expose-d && git rebase origin/main`.
> 1. build_hdbscan.sh crashes the Mojo compiler on gfx942 in AMDGPU instruction
> selection on the HDBSCAN stability kernel (hdbscan/impl/detail/, mangled name
> hdbscan_impl_detail_stabiliti...). Restructure it so the AMDGPU backend can select
> it; the usual culprits in this repo's history are a large SIMD or struct value live
> across a loop, a Float64 atomic or 64-bit integer division on the device, a recursive
> or very large inlined function, or a dynamic-size stack array (`git log --all
> --oneline -S'AMDGPU' | head`, `git log --oneline --grep=gfx942 | head` show earlier
> workarounds). Keep the bits: a spelling change of the same arithmetic in the same
> order, with a comment naming this crash. Compile-check the Apple target once.
> 2. test_mixture_surface FAIL "FIT init_params='random' also separates the blobs":
> decide whether the init is wrong (fix) or the test's expectation is too strong for a
> random init on that fixture (make the test measure what the code promises; say which).
> 3. test_kernel_methods_surface FAIL on the square Nystroem case (n_components equal to
> the row count should reproduce the linear kernel Gram; error 2.85 means the
> normalization or the eigendecomposition path is wrong for the square case): fix the
> code, not the tolerance, unless the tolerance is shown to be the defect.
> 4. test_kernel_methods_surface FAIL RBFSampler n_components=0 refuses from the buffer
> layer: add the by-name refusal in the Python class or binding before any buffer.
> 5. Leave the leg ready: bindings/build_training.sh in the body; each check task a
> bounded run; report per-check durations on the Mac so the next leg can be sized.

Then the MI300X leg again (body docs/lanes/handoff_subagents_2026-09-14/expose_d_body.sh
plus `sh bindings/build_training.sh` in its build list; through tools/hotaisle_leg.sh
from a CLEAN detached worktree, MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_COMMIT=<sha>",
spec 8core or 13core), and an H100 leg of the same body; when builds, seven test
modules and five checks are green on both, rebase and fast-forward D to main, then
your 151-lane push.

## 2. lane/packaging-f (workstream F), pushed at 7621a3eae, unbuilt

Ships all ten host bindings in both wheels through the manifest (host_surface.py
ships_in_wheel=True on every family, `--wheel-families`/`--wheel-bindings`/
`--training-gpu-column-record` flags read by build_sets.sh and
build_release_wheel.sh), mojolearn/reference_cards/, the identity harness and the
three TRAINING_GPU_COLUMNS inside the wheel, and `python -m mojolearn identity`
(python/mojolearn/_identity.py; exit 0 on IDENTICAL x4 over the record's lanes, 1 on a
mismatch, `--check` runs nothing). Found and fixed on the way: stage_libs.py never
reached host/, so the 0.8.5 Linux byte LM host binding shipped with the build box's
RUNPATH outside the closure check. CHANGELOG has a 0.8.6 section; version NOT bumped;
nothing built, no leg run, no test run. The freeze goes through docs/RELEASE_CHECKLIST.md
with `pixi run check-mamba-poison` added, three Linux sets plus macOS, and per vendor:
ten host rows reading cpu in readback.txt, identical host binding digests across the
three legs (pack_wheel refuses otherwise), `python -m mojolearn identity --check` then
`identity` from the installed wheel on each vendor's box and on the Mac. Rule from the
outsider runs (docs/VERIFY_EXTERNALLY.md): the shipped record and the wheel must come
from the same commit, so take a record at the release commit and switch
TRAINING_GPU_COLUMNS to it in the freeze commit.

## 3. lane/cpu-training-e3 (workstream E batch 3), NEVER STARTED

The last NO_CPU_PATH training families besides UMAP, GP, ARIMA and the neural blocks:
the random forests (rf-clf, rf-reg) and gradient boosting (gbdt-rmse, gbdt-symmetric,
gbdt-depthwise, gbdt-lossguide, then the adapters and variants the lanes reach). The
pattern is batches 1 and 2 on main (`git log --oneline main | grep "CPU training for"`),
their oracles under */host/*.mojo, and their CPU-only simulation before each commit
(host bindings into a scratch MOJOLEARN_HOST_OUTDIR, MOJOLEARN_HOST_DIR pointing at it,
the lane body from tools/identity_break.py on the base fixture against the three
TRAINING_GPU_COLUMNS; the batch 2 agent left simulate_cpu_only*.py in the mojolearn-ea
scratchpad). The brief the killed agent received: a host oracle per lane spelled as a
second implementation importing only checks/numerics.mojo leaves and the existing
core/forest_host_predict.mojo and core/gbdt_host_predict.mojo, exported from the forest
host binding with the read-back trio and MOJOLEARN_FOREST_HOST_SABOTAGE that must move
the bits; Python routing so the estimator trains on the host binding with no GPU
present; training_lanes and NO_CPU_PATH in host_surface.py and the docs spans
(`tools/docs_facts.py --write` then `--check`); tests as modules; if a lane reads
DIVERGENT, MOJOLEARN_IDENTITY_TRACE against an existing Apple GPU binding names the
first differing stage (batch 2 did this for kmeans and found DEVIATION 2714). Each lane
passes only when the seven-runner CPU identity gate reads require-columns 4 IDENTICAL on
every cell (push the lane; the workflow runs on push).

Also from batch 2, small: kmeans-random, kmeans-array, kmeans-weighted,
spectral-precomputed and the scaler twins share the host entries but were not declared
covered because the 47-lane record lacks them; the 136-lane record carries them, so
after section 4 lands they can join training_lanes with one gate run.

## 4. The 136-lane record and the gate switch: ALSO YOURS (Andrew, 14:50 ET: "keep nothing")

Branch lane/136-lane-record (595c58691, pushed, off main) holds everything staged:
bench/results/identity_break/2026-09-14_136-lanes/ with apple-m4.json (1224 stable),
nvidia-h100-sm_90a.json (1224 stable; diff.apple-h100.txt IDENTICAL=1224, infer/model
IDENTICAL=1665 N/A=783), nvidia-2xh100-sm_90a.par-devices-0-1.json (144/144 IDENTICAL x3
with Apple and the one-device H100 over the sixteen par lanes, diff.par-two-devices.txt,
same-vendor two-device result), amd-mi300x-gfx942.json INCOMPLETE (1161 of 1224, its
process cut by the 60-minute cap before iforest, iforest-tuned, par-arima, par-byte-lm,
par-iforest, par-mlp, par-samba; complete=false), the README with the rows so far,
python/mojolearn/host_surface.py TRAINING_GPU_COLUMNS switched to this record, and
.github/workflows/cpu-identity-gate.yml with the paths and sparse checkout switched and
the comment updated BUT the two `grep -F 'summary: ...'` lines of the "--diff over the
three committed GPU columns" step still saying IDENTICAL=423 and IDENTICAL=585, N/A=261.

Two legs are running on this machine at 4048e1b51 and finish on their own (never cancel
them); their runners' logs and output directories:
- Hot Aisle MI300X 8core, the four fast missing lanes (iforest, iforest-tuned, par-arima,
  par-iforest): log /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/efc56304-19ac-4d4f-b875-6a7805b7bb05/scratchpad/identity136_tail4_mi300x.log,
  output under /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/efc56304-19ac-4d4f-b875-6a7805b7bb05/scratchpad/wt-leg-9ade/bench/results/e1g/*-amd-mi300x-hotaisle-identity-136-tail4/remote/identity_break.tail4.json
  (done when the log says "balance at the end"). The three neural par lanes
  (par-byte-lm, par-mlp, par-samba) do not fit the 8core VM's cap even alone with the
  builds; leave them out of the MI300X column and say so.
- DigitalOcean MI325X (gfx942, 24.04 image), a FULL 136-lane column: log
  /private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/efc56304-19ac-4d4f-b875-6a7805b7bb05/scratchpad/identity136_leg_mi325x.log,
  output under .../scratchpad/wt-leg-9ade/bench/results/e1g/*-amd-mi325x-do-identity-136-lanes/remote/identity/identity_break.amd-gfx942.json
  (done when the log shows the post-destroy 404).

To finish: (a) merge the tail4 cells into amd-mi300x-gfx942.json ONLY if its
package.bindings sha256 list equals the first run's (both built at 4048e1b51; state
"two processes, identical binding digests" in the README; keep complete=false and name
the three absent lanes); (b) add the MI325X column as amd-mi325x-gfx942.json, the AMD
column that carries every lane including the three neural par lanes, and diff it
against apple-m4 and nvidia-h100-sm_90a; (c) run
`PYTHONPATH=python python3 tools/identity_break.py --diff <apple> <h100> <mi300x> --require-columns 3`
and put its two `summary` lines verbatim into the workflow's two grep lines (the CPU
gate's covered lanes are all in the MI300X column, so require-columns holds); (d) one
commit: record, README, host_surface.py, workflow; rebase on main; push; the gate runs
on the push and must be green on seven runners before anything else moves. Then the
AMD two-device column diffs against apple-m4 and the one-device AMD columns here.

## 5. Leg-body traps learned today (every body in handoff_subagents_2026-09-14/ already avoids them)

The shipped source archive carries no bench/results (curl the record columns from
raw.githubusercontent.com; the Hot Aisle container has no git). The Mamba lanes import
all_finite_f32 from the BASE binding, built by bindings/build.sh, which the build_*.sh
glob does not match. The family builders refuse an existing output file. Both runners
cap a lease at 60 minutes and refuse more; split the work, never extend. identity_break
needs PYTHONPATH=<tree>/python. Hot Aisle refuses a dirty tree; launch from a detached
worktree. RunPod passes no environment to the body; bake commit.txt and labels into a
wrapper. tools/gemm_remote_leg.sh takes MOJOLEARN_GEMM_LEG_GPU_COUNT=2 for a two-GPU pod.
