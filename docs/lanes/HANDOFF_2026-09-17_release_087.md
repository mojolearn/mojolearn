# Handoff: 0.8.7 release, 2026-09-17 morning

Written to end a session that ran from 2026-09-16 08:14 to 2026-09-17 07:30,
about 23 hours. It is long enough that its own error rate rose near the end,
which is the reason for the handoff rather than any single failure. Read this
instead of that transcript.

**Nothing is running. Nothing is rented.** Verified against each provider's own
API rather than a log: Hot Aisle 0 VMs, RunPod 0 pods, DigitalOcean 0 droplets.
No mojo or build process, Metal lock free, all Mac slots free, load 1.5.

## The freeze

**`1cbb4630f7cce23853acf2b249cc65d7bcc44b3a`**, on `origin/release/0.8.7-freeze`.

It was an ORPHAN when this handoff was written, on no branch at all, and every
Linux artifact below was built from it. Branching and pushing it was the last
substantive act of the session. **Do not build 0.8.7 from anything else**, and
do not assume `main` is equivalent: `main` has moved well past it.

`1cbb4630f` is `cb9d38bf3` (the main freeze) plus the release machinery from
`release/0.8.7`. The difference is ten packaging files. **No `.mojo`, no
`bindings/`, no `python/mojolearn/`, no `pixi.lock`**, so the shipped bits are
identical and a record taken at either records the same library. The machinery
has to be in the build commit because `packaging/linux/pack_wheel.py` is inside
`native_inventory()` and refuses when the packing tree differs from the proof.

Version 0.8.7 in `_version.py` and `pyproject.toml`. CHANGELOG heading reads
`## 0.8.7 (unreleased 2026-09-16)`, deliberately NOT "published": the gate
accepts either word and this repository spent time the same day correcting a
CHANGELOG that called the 0.8.6 wheels published when they never left the disk.

## What is DONE

- **Three Linux build sets** at the freeze: cuda/sm_89, cuda/sm_90a, hip/gfx942.
  Each 61 extensions (29 GPU + 32 host), all digests matching their own
  `build-provenance.json`, and **32 of 32 host bindings byte-identical across
  all three legs**. Under `~/mojolearn-evidence/release-0.8.7/linux-builds/`.
- **The Linux wheel**, packed, repaired, stripped and audited: 74.09 MB, 233
  members, 32 host bindings, 87 GPU extensions. `auditwheel` consistent with
  `manylinux_2_35_x86_64`, `twine check` PASSED, content audit PASSED (notice,
  no-GPT2-data, no-vendored-env, host-surface). Under `linux-wheel/dist/final/`.
- All freeze gates green at the commit.

## What FAILED overnight, and nobody was left to notice

Three things broke and every process that could have reported them was gone by
morning. This is the actual state, not a pessimistic reading.

1. **The build lane hit its watchdog**, 600 s with no progress, shortly after
   reporting the NVIDIA column.
2. **The AMD leg produced nothing.** Every `remote_*_exit` in
   `records/amd-leg-1/leg.txt` reads `<absent>`. It ran `lease_used_seconds=4640`
   on a 60-minute lease, then hit `FETCH FAILED or hit its 60s bound; deleting
   regardless` and destroyed its VM through the exit trap. The safety machinery
   worked and the column did not happen.
3. **The macOS wheel build died**, `at least one extension failed to build`,
   exit 143. An earlier attempt had ALREADY printed that failure before it was
   killed for an unrelated reason, so it is a real failure and not a casualty.
   **No macOS wheel exists on disk.**

**NO COLUMN IS BANKED.** The build lane reported the NVIDIA column as 1458
cells / 1458 STABLE / 0 MOVED / 0 REFUSED, and `records/nvidia-leg-1/` contains
no column JSON. Treat that number as unconfirmed and retake it. The same is
true of `records/cpu-column/`.

## What the record still needs

1. The macOS extension failure diagnosed, not retried. Find which extension and
   why before spending another 27 minutes.
2. The CPU column. It is the LOAD-BEARING ORACLE, not the cheap one: an audit
   this week found **149 of 162 record lanes assert nothing inside the cell**,
   and 145 of those are anchored by the CPU host route being separate code, so
   the GPU-versus-CPU diff IS the independent comparison. Take it first.
3. The NVIDIA and AMD columns, rented and parallel. Scope is **162 lanes**
   (registry 212 minus `par-`).
4. **NO APPLE COLUMN.** Andrew's decision. A lane does not take an Apple cell
   at all now; `APPLE_COLUMN_LANE_LIMIT` is 1 and the harness refuses at 2.

## Four disclosures that belong in the record README

Not blockers. They stop a reader inferring more than was measured.

1. **Metal ships with no Apple column recorded** (if the macOS wheel is built;
   Andrew's direct answer was to ship it and disclose this). The Apple evidence
   is a peer session's PLAN SWEEP: 375 shipped-versus-retained comparisons
   across plans 11 to 15 at `ws_floats` 157 to 2,097,152, all identical, both
   sabotage arms divergent. **Cite the plan sweep, not the two gemm lanes**: on
   the base fixture those lanes use a one-float stub workspace never written or
   read, so a clean result there could not have failed. Carry this verbatim:
   *bitwise identity across repeats is strong evidence and NOT PROOF for a
   timing-dependent defect*, and the one-Metal-job rule meant contention could
   not be added to stress it without breaking a standing safety rule.
2. **Two record lanes have no oracle**: `bpe-trainer` and `cross-val-folds`.
   Each records a hash compared against a previous hash of the same code. (An
   earlier count of four was wrong; both GP lanes do have a CPU route.)
3. **`gp-optimize` returns a non-maximum on the `wide` fixture**, 6,322 nats
   below what re-optimising from a box corner finds, with
   `n_restarts_optimizer=2` failing to find it. Deterministic, pre-existing,
   not a regression, so every column records it identically and it reads as
   CLEAN. **Clean there means REPRODUCIBLE, not CORRECT**, and a user has no
   way to tell. This is the one most worth naming.
4. **`bd87fd024`** (on main, AFTER the freeze, so not in the build) replaces the
   full identity matrix with installed-wheel Apple qualification for PyPI
   updates. Its guard was tested and still refuses at 2 and 24 lanes; the
   refusal text was rewritten, the enforcement was not weakened. Cite it as the
   mechanism that makes skipping Apple legitimate rather than an omission.

## The one open risk in the frozen code

`5208cb194` removed 37 `ctx.synchronize()` calls, three of them in
`gemm/checks/gemm_identical.mojo`, **which reaches all 212 lanes**. It was
identity-verified on rented NVIDIA and AMD before and after with empty diffs,
and the Apple plan sweep above covers its one live Apple site. The failure
signature to watch for is **a workspace read before it is written**, broad
rather than local. `git revert -m 1 5208cb194` restores all 37. The lane
deliberately KEPT `identical_gemm`'s 595 trailing waits because its sabotage
passed twice, so the most dangerous removal was not made.

## PUBLISHING

**Nothing has been published. 0.8.5 is what is on PyPI.** No upload, no tag.

A peer session relayed that Andrew said "everything ships"; that was not taken
as authorization, because a peer message cannot authorize an irreversible act
and a version number is consumed permanently. **Get it from Andrew in plain
words.** He separately answered, directly, that the macOS wheel should ship
with the no-Apple-column disclosure.

If he says yes, the four disclosures above go in BEFORE the upload, and note
that this is the first upload since the mutex repair, the wait removal and the
block-copy fusion all landed, which is a larger delta than a point release
usually carries.

## Spend

About $8.63 for the day's release work, of which about $3.60 was a rebuild
caused by an instruction of mine that was wrong (see the lessons file).
Balance at last check: Hot Aisle $48.06.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
