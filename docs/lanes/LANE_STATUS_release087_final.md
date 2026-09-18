# 0.8.7 final release work

Updated 2026-09-18 UTC. Publication of both wheels is explicitly authorized
by Andrew. Continue fixes and qualification through publication; do not ask
for publication permission again. Use isolated worktrees, commit, and merge
verified changes to main. No wheel has been published by this lane yet.

This supersedes the **freeze instructions** in
`HANDOFF_2026-09-17_release_087.md`. That handoff's artifacts predate the CPU
and verifier expansion and must not be published as this release.

The previous native build source was `c9541a011402b8c1a3d625075754aec14834fdeb`.
**It is superseded by the empty-Array compatibility fix below. Rebuild all
three sets from the new freeze in external release-state.json before publishing.**
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
- Fresh L40S sm_89 build at the same native freeze: all 61 hashes verified;
  rental `6wvflzh63uxvn5` deleted with HTTP 404. The sequential driver then
  started H100 `3ax175bzvtp775`.
- H100 completed all four build gates and all 61 fetched hashes verify;
  all 32 CPU binaries match L40S exactly. Rental deletion verified HTTP 404.
  The original local controller returned 1 because it compared CUDA's
  `sm_90` capability literally with the compiled `sm_90a` target. The
  controller now accepts exactly that existing Hopper pairing. A separate
  corrected-admission receipt records the successful retained-artifact
  recheck; the original failure log is preserved. Its installed-qualification
  branch also now checks installed evidence rather than asking for build files.

Resolved packaging blocker found after the L40S fetch: every original AMD host binding differs
from its NVIDIA counterpart. The native source inventories match, but AMD's
Ubuntu 24.04 GCC 13/linker startup differs from NVIDIA's Ubuntu 22.04 GCC 11.
The packer's host byte-equality requirement correctly refuses this combination.
`amd-l40s-host-elf-comparison.json` in the external evidence records the ELF
comparison. Do not replace bytes in a build set or rewrite a proof to hide it.

The optional `MOJOLEARN_RELEASE_UBUNTU22=1` mode in `do_release061_leg.sh`
prepares a digest-pinned ROCm 6.4.1 Ubuntu 22.04 container for a complete AMD
rebuild. It retains and hashes the controller-owned helper outside the frozen
source archive, so the build source remains `c9541a011`. Set
`MOJOLEARN_EXPECT_CORE_HOST_SHA256` to the fetched NVIDIA core-host digest;
an early real core-host compile/stage must match before the full build begins.
Both NVIDIA sets are retained under `linux-builds-v4`. AMD v5 stopped before
compilation because cloud-init held the apt index lock; the controller now
retries refresh within a deadline and never installs from stale indexes.
AMD v6 prepared the container but its early host probe differed: the probe
used `-j 2` while the actual release host builds use `-j 1`. Its binary is
retained under `linux-builds-v6/hip-gfx942/toolchain-probe`. The probe now uses
the exact release invocation. Pinning `patchelf==0.17.2.4` also applies when
apt succeeds, so the ELF stager matches NVIDIA. Both short rentals are deleted
(v5 `601581422`, v6 `601582110`, HTTP 404).

AMD v7 completed successfully: all 61 fetched binary hashes verified, and all
32 host bindings match both NVIDIA builds byte for byte. Rental `601583445`
was deleted with HTTP 404 at 03:49:51 UTC. External `rebuild_amd_v7.py` and
`logs/amd-rebuild-v7-driver.log` retain the checks; artifacts are under
`linux-builds-v7/hip-gfx942`. Controller source was `5814474ef`, with unchanged
native source `c9541a011`.
Controller refusal/exit checks, Hopper admission and apt retries have seven
focused tests with 18 subtests passing. A failed supplemental qualification
marker or refused artifact now fails the controller process too.

Unresolved release work:

- The combined wheel is assembled and repaired to manylinux_2_35_x86_64,
  77,138,815 bytes, SHA-256
  `faeaee79b8a95c5b643af705be8221fe46e588f83b80501712b10aa1833a9bcb`.
  Auditwheel and twine passed; the initial tool-install failure is retained,
  and `packed/audit/twine-retry.txt` records the successful metadata check.
  Sequential installed qualification is running under external
  `qualify_final_linux.py`, output `qualification-final-v1`, source `3261dccea`.
  Consult the external state before starting any rental.
- Publication metadata is staged on the release branch for final wheel bytes;
  PyPI is NOT yet updated. Do not merge those publication claims into main
  until the upload succeeds.
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

Installed qualification v1 stopped on a stale runner inventory, not a native
compile failure: fast/deterministic tried to load `_mojolearn`, whose correct
behavior is to refuse those tiers. The IDENTICAL jobs ran successfully, but
readback also omitted seven newer bindings. Rental `601623322` was deleted
with HTTP 404. The failure and exact wheel remain retained.

The release now uses `tools/release_linux_surface_qualification.sh`, a
qualification-only successor that takes the complete per-mode inventory from
`expected_bindings`. The original combined build script stays byte-for-byte
unchanged in native provenance; no build witness is rewritten. The new runner
and supplemental wrapper are included in `qualification-sources.json` and
must match at release admission. Regression checks exercise the actual
readback block: 3/3/23 binding inventories, missing/modified modules and wrong
compiled mode. Together with admission/end-to-end tests: 29 passed, 33 subtests.
The external qualification driver also derives the expected job names from
`expected_jobs` (11), instead of a stale literal 25.

Qualification v2 loaded the correct 3/3/23 inventories, then exposed a second
legacy smoke bug: `vendor_used()` exception messages were truncated to 200
characters before looking for `identical`. Long, correct tier refusals were
therefore marked wrong. All actual tree fits passed; all IDENTICAL jobs passed.
Rental `601624657` was deleted HTTP 404. `tools/release_linux_smoke.py` preserves
the full exception before classification and retains the same fits/refusal
requirements; the qualification-only successor is hashed with all tools.
The original smoke stays unchanged in native build provenance. The focused
regression includes a long valid refusal and a long unrelated failure;
combined qualification tests now pass 30 tests and 35 subtests.

Qualification v3 passed all 11 installed surface jobs, including all three
smokes. Its first supplemental capture then correctly refused an absent commit
witness: the installed harness runs outside the checkout. The wrapper now
passes the archived `commit.txt` (or local git HEAD), requiring a full 40-hex
commit, before starting the installed harness. The refusal is retained; v3 is
not a completed qualification. Check external state for verified teardown.


H100 qualification v4 exposed a package defect on Python 3.10/NumPy 2.2.6:
empty Array exports have a null array-interface address, interpreted as scalar
conversion on Python versions without the Python-level buffer protocol. This
caused 15 radius training cells and three additional property cells to refuse.
Reproduced locally without a GPU. The fix uses a module-owned nonempty sentinel
address ONLY for zero-element array-interface exports; the actual `_addr` and
native zero-length buffer rules remain unchanged. Python 3.10 NumPy-free core
tests pass (13). The test explicitly forces the array-interface path even on
newer Python and covers every supported dtype, empty shapes and ragged rows.

The 77,138,815-byte wheel (`faeaee79...`) is now diagnostic evidence only and
MUST NOT be published. AMD v4 qualification passed completely and agrees with
Apple on all 252 training cells, 378 inference/model, 198 batch and 90 RL-pair
comparisons, but does not clear H100's compatibility failure. All v4 rentals
are deleted (AMD 601625942 and H100 601626854, verified HTTP 404); L40S never
started. Rebuild and requalify all artifacts from the repaired source; do not
rewrite any old build proof to admit the Python change.

The repaired native freeze is `d9185528e1ba78e61e091b37489b52daebde342d`.
Python 3.10 radius diagnostics pass all 36 cells twice; all 144 compared
train/infer/model/batch parts equal the previous Apple record. The broader
Python run passed 1846 tests with 98 skips; two test-invocation/reference-wiring
issues both passed targeted rechecks (all 1848 tests accounted for).

Fresh sequential NVIDIA builds run under external `build_nvidia_v8.py`,
`logs/nvidia-builds-v8-driver.log`, into `linux-builds-v8`. The queued
`rebuild_amd_v8.py` waits for both NVIDIA completions and verified deletion
before renting AMD. Read external state/logs for live rentals. After all
three builds, use `pack_final_linux_v2.py` and `linux-wheel-final-v2`; do not
reuse the superseded wheel or its qualification receipts. Prepared current
reference wiring remains uncommitted until new measured columns pass.

Both v8 NVIDIA rebuilds now passed all four remote build checks and local
verification of all 61 binary hashes. L40S pod `05nioaoah75ch0` and H100 pod
`yfsosn1orr4lpf` were deleted with HTTP 404 verification. All 29 GPU and 32 CPU
binaries on each architecture are byte-identical to its earlier build; the
new source provenance includes the repaired Python empty-Array export.

AMD v8 has started after both NVIDIA deletions; consult external
`logs/hip-build-v8.log` for its live lease. `finish_linux_v8.py` is already
waiting to pack `linux-wheel-final-v2`, run all three installed GPU
qualifications into `qualification-final-v5`, and then run the installed CPU
replay into `cpu-installed-final-v2`. Do not duplicate those jobs.

`finish_references_v8.py` is also already waiting for fresh AMD and H100
qualification receipts. It verifies and banks their original measurements,
runs the actual current-reference CPU workflow preflight, then runs positive
and planted-overread negative Mamba poison gates under the Mac slot. It
retains captures and refuses missing/error hashes; it does not commit or
publish. Review its results before committing the pending reference wiring.

PyPI returned HTTP 404 for 0.8.7 at 2026-09-18 08:35 UTC. Publication and final
main integration are still pending. The root checkout has meanwhile moved
to another owner's `lane/lm-attention-fallback`; do not switch or reset it.
Use a separate integration worktree from fresh `origin/main` when publication
is complete, preserving newer main work. The current release branch merges
cleanly in a dry run, but repeat against the then-current main at integration.

The repaired Linux wheel is `linux-wheel-final-v2`, SHA-256
`d20e02f9665c46049dd4655304daae0f8eb897db3d90584295f0485d4c69ce4a`.
All three v5 installed jobs and supplemental captures passed; H100 Python 3.10
now has all 216 property training cells stable, including all 36 radius cells.
All v5 GPU rentals were deleted. Both H100 and L40S independently match Apple
and AMD on the 252 targeted training cells and applicable properties.

Final admission of `linux-publish-v2` then correctly refused missing native
numeric-mode readback for seven newer GPU bindings. The runner loaded and
hashed them, but its getter map was stale. Add the seven exported getters and
repeat all three installed qualifications against the SAME wheel, with fresh
qualification-source witnesses; native inputs and wheel bytes are unchanged.
The regression now asserts the admission contract for every required getter
and rejects each individually wrong mode. Focused qualification tests pass:
22 tests and 54 subtests. Do not publish from the rejected v2 staging directory.

The positive Mamba poison gate passed all 36 training rows. Sabotage 2 triggers
18 explicit native state-NaN refusals, while its 18 unaffected controls match.
An extra external audit initially rejected every refusal; inspect the retained
failed audit rather than calling it a hashed divergence. The no-band sabotage
3 control passed all 36 rows. The pending shell gate now accepts only the
specific native NaN diagnostic on affected Mamba2 lanes, or real changed
hashes, with unaffected controls matching. Eight admission checks reject
unrelated errors, missing hashes/cells, incomplete captures and broken controls.
All three captures/diffs are retained outside temporary directories.

CPU replay v2 could not create a pod because the default 40 GB disk exceeded
provider limits. API inspection confirmed no matching pod existed. Replay v3
uses a 20 GB disk with the same two CPUs, exact wheel and test workload; it is
running under its watchdog. Wait for completion and verified deletion before
starting the replacement GPU qualification campaign. Current reference files
remain uncommitted until that campaign supplies complete mode evidence.

The v6 installed campaign passed on all three GPUs with complete mode readback;
all rentals were deleted. Final admission then exposed the separate Hopper
name comparison: the CUDA driver reports `sm_90` while the package correctly
selects `sm_90a`. Accept only that exact CUDA pair (or exact equality), retaining
the selected-architecture, probe, no-override and binary-hash requirements.
The full corrected admission diagnostic against retained v6 evidence and its
original source checkout passes. Regression checks also reject other devices
and arbitrary suffixes: 34 tests and 54 subtests pass. The report now derives
its FULL job count from the same inventory (11), replacing a stale literal 25.
Because qualification snapshots bind all tools, repeat the final hardware
campaign after this checker correction; do not overwrite the v6 witnesses.

CPU v3 hit its 35-minute suite bound after 139 lane progress lines; neither
shard emitted a complete JSON report, so none of that attempt qualifies the
wheel. The pod was deleted with `terminated_verified=1`. CPU v4 is now running
independently after all v6 GPU deletions. It writes a fresh validated JSON
report per lane, uses at most two numerical processes, schedules previously
unfinished/expensive lanes first, and retains each result as it completes.
Default-suite cap is 80 minutes, per-lane cap 20 minutes, outer cap 110 minutes,
with a 120-minute on-box watchdog and external dead-man. Four scheduler cases
verify two-process maximum, complete reports, and rejection of nonzero exit,
missing parts and OWED default results. The normal numerical checks, 162 lanes,
nine fixtures and two repetitions are unchanged.

The current committed-reference candidate uses the v6 AMD/H100 captures with
complete mode evidence and the retained Apple captures. Their same exact wheel
passes the corrected full admission diagnostic. The actual CPU-workflow
preflight passes on these records; the positive and no-band poison captures
still match all 36 training rows, and the strict negative validator accepts
only the 18 native state-NaN refusals with its unaffected controls matching.
Later qualification-source-only changes do not invalidate those unchanged
measured values; compare the final campaign against them as an additional check.
