# 0.8.7 final release work

Updated 2026-09-17 evening. Publication of both wheels is explicitly authorized
by Andrew. Continue fixes and qualification through publication; do not ask
for publication permission again. Use isolated worktrees, commit, and merge
verified changes to main. No wheel has been published by this lane yet.

This supersedes the **freeze instructions** in
`HANDOFF_2026-09-17_release_087.md`. That handoff's artifacts predate the CPU
and verifier expansion and must not be published as this release.

Native build source is `c9541a011402b8c1a3d625075754aec14834fdeb`.
Later tools/workflow changes preserve that native inventory. The clean Linux
worktree is `~/mojolearn-wt/release-087-linux-final`; the Mac worktree is
`~/mojolearn-wt/release-087-final`, branch `release/087-final`.
Mac builds generate Python helper copies: do not use its resulting untracked
files as the Linux native inventory. Evidence and live process state are in
`~/mojolearn-evidence/release-087-final/release-state.json` and `logs/`.

Completed:

- Full Python tests: 1847 passed, 98 environment/dependency skips.
- Applicable Metal Python release gates: 36/36 passed.
- A fresh diagnostic Mac wheel passed installed fits on Python 3.10–3.14
  in all three modes, and the installed verifier gate passed. Its Mamba host
  source was subsequently regenerated; the final release must build afresh.
- Regenerated Mamba CPU code: 54 cells, 162 train/infer/model comparisons
  unchanged against the previous binding, nine fixtures, two repeats.
- Apple Mamba clean capture: four lanes, nine fixtures, two repeats;
  36 stable cells with stable inference, batch and RL-pair parts.
- Installed Linux qualifications now include the wheel's own Mamba harness
  and verifier CLI. Optional properties lacking references remain OWED;
  mismatches, refusals and inconsistent exit/verdict combinations fail.
- Release tooling tests: 25 passed and 33 subtests passed.

Unresolved release work:

- Finish fresh Linux builds for CUDA sm_89, CUDA sm_90a and HIP gfx942,
  assemble/audit the combined wheel, and qualify those exact bytes on each
  actual device. The first AMD build predates the regenerated host source
  and is superseded. No old build may be silently substituted.
- The Mamba poison capture is stable and agrees with the available clean
  cells, but the gate FAILED because its old reference record lacks current
  lane revisions. Bank current NVIDIA and AMD Mamba columns during installed
  qualification, then resolve this gate with real three-vendor evidence.
  Do not reduce its required columns or call the current failure a pass.
- Run the final installed CPU replay and standard release-provenance workflow
  with its CPU certification. Stage the qualified combined Linux wheel before
  starting the ephemeral real-Metal runner, so both wheels publish in one run.
- Verify actual PyPI bytes and hashes, then record publication evidence.

Limits remain one local numerical/compiler worker and two cloud CPU workers;
GPU rentals are sequential and must have watchdogs, external deadmen and
verified teardown. Consult the external state before renting anything.
Do not touch other owners' processes or rentals.

Coverage scope remains 162 default CPU lanes, 17 withheld and 50 parallel
default exclusions. Opt-in pending routes and 17 logical-shard drivers are
exposed without falsely qualifying physical multi-GPU execution. Portable
HostForest/HostGBDT saved models are included in the installed model gate.
These counts describe source scope, not a claim that 0.8.7 is already on PyPI.
