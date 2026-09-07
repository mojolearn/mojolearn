# Combined Linux 0.6.1 preparation — September 7, 2026

Root alone ran bounded host/file checks with two-thread limits, 60-second
deadlines and a 1 GiB sampled RSS cap. No native binaries or GPU models were
loaded. `root-checks.json` links exact retained receipts and their hashes.

- Packer: 3 tests passed.
- Packer plus complete three-architecture admission fixture: 4 tests and
  6 subtests passed; includes rejection of stale wheel/source and missing or
  mislabeled architecture evidence.
- Existing release/qualification regression: 20 tests and 29 subtests passed.
  The first collection stopped because PyYAML was absent; retained separately.
  Root installed binary PyYAML 6.0.3 into the disposable test environment.
- Alpha publication/overlay verification: 16 tests passed.

Source version metadata is now 0.6.1. The packer, installed qualification and
publication path require one exact final wheel and real sm_89, sm_90, gfx942
records. All 135 standard extensions require explicit build provenance. The
separate byte-LM extension is absent from this profile. Published 0.6.0 is
unchanged. No fresh GPU build, installed qualification or 0.6.1 publication
has happened; the execution checklist remains open.

Root froze the candidate in `/tmp/mojolearn-release061-20260907`; current
snapshot commit is `bffb64ee5cb36f3a7dbbd608499e03a3e4c4f7d0`. The initial
source-snapshot.json preserves the first revision. Later commits fix the
release-specific rehearsal and retain the repository's ignore policy.
The helper now requires preinstalled patchelf and pins vendor kernel column
and x86-64-v3. Both shell syntax checks passed. Profile7 rehearsal r3 passed
all 50 checks without renting; r1's wrong Mamba generator prerequisite and
r2's missing snapshot ignore policy remain retained failures/blockers.
Actual native builds remain unrun while Apple execution is prioritized.
