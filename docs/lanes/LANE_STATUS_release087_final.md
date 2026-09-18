# 0.8.7 final release work

Updated 2026-09-18 UTC. Publication of both wheels is explicitly authorized
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

Main subsequently received native Python hot-path work at `3aefe1127`, outside
this release freeze. **Do not pull those native changes into the active release
branch.** Release-tool changes are merged into main while the release branch
keeps its recorded native inputs. Dispatch/tag the release branch's final
commit explicitly; main is no longer a substitute for it.

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
- After integrating the parallel-scaler numerical-mismatch reporting fix:
  103 verifier/reference-gate tests passed.
- Additional targeted Apple columns: 20 property lanes and four ordinary
  neural lanes, all nine fixtures twice. Together with Mamba: 252/252 stable
  cells, applicable inference/model/batch/RL-pair parts stable. These captures
  do not enable the separate step/full flag. The existing current CPU core
  capture matches Apple on all 162 compared train/infer/model parts.
- Fresh AMD gfx942 build at the native freeze: all 61 binding hashes and the
  source inventory verified; rental deleted (HTTP 404). R2 retention confirmed.

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

The CPU workflow's historical-reference preflight now accounts explicitly for
18 x 9 stale-revision cells per vendor; all three actual historical-reference
steps, including negative controls, passed. Retained CPU records also exposed
optional model/batch properties whose native prediction faults do not alter
their hashes. Fresh independent GPU property columns are being collected for
the certification diff; its OWED-property rule is not being weakened.

The H100 attempt `zaq7fr4osqlq7c` lost its artifacts while the Mac slept and its
lease expired; absence was verified with HTTP 404. Subsequent controllers run
under a bounded awake assertion, with an additional object-scoped R2 upload
armed on each rented box. The external `build_nvidia_remaining.py` driver owns
two sequential attempts (L40S then H100), stops on failure, verifies all 61
fetched binding hashes, and verifies deletion before renting the next device.
Consult the external state and logs before starting any additional rental.

Limits remain one local numerical/compiler worker and two cloud CPU workers;
GPU rentals are sequential and must have watchdogs, external deadmen and
verified teardown. Consult the external state before renting anything.
Do not touch other owners' processes or rentals.

Coverage scope remains 162 default CPU lanes, 17 withheld and 50 parallel
default exclusions. Opt-in pending routes and 17 logical-shard drivers are
exposed without falsely qualifying physical multi-GPU execution. Portable
HostForest/HostGBDT saved models are included in the installed model gate.
These counts describe source scope, not a claim that 0.8.7 is already on PyPI.
