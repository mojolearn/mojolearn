# Crash recovery, 2026-09-16 morning

The Mac's Claude sessions died at about 08:14 EDT. **The machine did not reboot**
(`uptime` reads 10 days), so this was a session crash, not a kernel panic. Two
sessions were lost: one Claude session running many lane subagents
(`e52730fc-…`, 601 scratchpad entries) and one Fable session on the random
forest and Metal work (`67533bb6-…`, started 07:52).

This file is what a session with no context needs to resume. Read it first.

## Nothing was lost

Every uncommitted lane change was committed to its **own branch** (never main)
before writing this. Three of the dead worktrees lived under
`/private/tmp/claude-501/…`, which the OS reaps, so the work is now in git
rather than in a directory that may not survive:

| branch | what was preserved |
|---|---|
| `lane/lane-selector` | `tools/lane_select.py`, `tools/verify_lanes.py`, `tools/test_lane_select.py`, `docs/lanes/VERIFICATION_TIERS.md` — all five files were UNTRACKED |
| `lane/rf-mutex-claim-acquire` | `LANE_STATUS_lane-rf-mutex-claim-acquire.md` |
| `lane/metal-launch-overhead` | `training/checks/metal_launch_probe.mojo` |
| `lane/data-ordering-determinism` | `model_selection.py`, `identity_break.py` in progress |
| `fix/amd-merge-ordering` | the review doc's additions |
| `lane/ship-cpu-host-families` | ten files, the whole "all thirty-two families ship" change |

## The rented box, and what was rescued from it

A Hot Aisle MI300X (`enc1-gpuvm012`, `1d348750-3bca-4442-9520-806f775f4b45`)
was rented by the `codex-device-mutex` leg. **Its parent process died in the
crash**, so nothing was left to pull its results or delete it; only the Mac
dead-man survived, set to fire at 12:31:33Z.

The run had already FINISHED on the box (`body_exit=0`, `extra_exit=0`). Its
output was pulled by hand before the dead-man fired, into
`~/mojolearn-evidence/codex-device-mutex-gfx942-2026-09-16b/box_out/`.

A second leg (`rf-claimfix-ab`, the A/B for `lane/rf-mutex-claim-acquire`)
survived the crash as a live process, was still retrying for 13-core stock,
and carries its own dead-man and its own result pull. It was left alone.

## The result that changes the RF lanes

`device_mutex.log` from that box, on **gfx942**, is the mechanism evidence
that `lane/rf-score-weighted-nondeterminism` recorded as MISSING:

```
stock:    4/6 schedules lack payload ordering
witness:  T0.test -> T1.test -> T0.cas -> T0.payload -> T0.release ->
          T1.cas -> T1.payload -> T1.release
acqcas:   0/6 schedules lack payload ordering
postload: 0/6 schedules lack payload ordering
mutex 0 -> 1   … count 256 handoff-error 0 accepted True
mutex -2 -> -1 … count 256 handoff-error 0 accepted True
mutex 0 -> 1   skip-write True count 0 handoff-error 0 accepted False
PASS device_mutex: both state pairs, sabotage rejected
```

Read it carefully, because it is a model check plus a device run and the two
carry different weight:

* The schedule enumeration is the **model**: the shipped spelling admits 4 of
  6 schedules with no payload ordering, and prints a concrete witness. Both
  candidate repairs (acquire-CAS, and the post-claim acquire load) admit 0.
* The mutex lines are the **device**: both the forest's `0 -> 1` protocol and
  kNN's `-2 -> -1` consumer claim ran on the MI300X with zero handoff errors,
  and the sabotage arm (`skip-write True`) was REJECTED, so the check can fail.

This does not by itself replace the A/B leg. What it settles is the question
`lane/rf-score-weighted-nondeterminism` left open at leg 14: that lane's probe
measured the shipped spelling as EXACT (512/512) and called it a NULL rather
than a clearance, because one acquisition per block almost never makes the
spin loop. The schedule enumeration reaches the window the probe could not.

## Three lanes are on the same defect and must be reconciled

Andrew's instruction on `lane/rf-mutex-claim-acquire` is explicit: reconcile
with the other work on this defect **before anything reaches main**.

| branch | author | what it holds |
|---|---|---|
| `lane/rf-score-weighted-nondeterminism` | Opus | 17 commits: the diagnosis, twelve MI300X legs, the traces, the A/B EFFECT replicated (control 7/300 then 6/300 against 0/300 twice, p ≈ 1e-3 each) |
| `lane/rf-mutex-claim-acquire` | Fable | the repair applied at all four claim sites, the memory-model argument, `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` as the A/B control |
| `fix/amd-merge-ordering` | Codex | portable `core/device_mutex.mojo`, the model checker, the counterexample, an UNAPPLIED integration patch |

They agree on the repair: a **post-success acquire load** of the mutex, which
is one spelling for every column and legalizes on Apple, where an acquire
compare-exchange is rejected by name. The four claim sites are
`ensemble/…/split.mojo` `_publish_to_global`, `extratrees/…/split.mojo`,
and `neighbors/impl/detail/fused_l2_knn.mojo` (consumer and producer).

The reconciliation is the next piece of work, not a new lane.

## What was in flight and stopped mid-run

* **`lane/ship-cpu-host-families`** was running `verify --all` on a CPU-only
  install to count how many lanes become checkable. The log ends `Terminated:
  15` — the crash killed it. The CHANGELOG and LANE_STATUS drafts carry a
  literal `<!--TBD-->` where that count goes. **The run must be retaken.**
  Drafts are in the dead session's scratchpad; the branch has the code.
* **`lane/lane-selector`** is complete but was entirely untracked until this
  recovery. It is the verification-time answer: one command, three tiers,
  199 lanes read from the registry by import rather than by grep.

## Lanes explicitly dropped for the session, not merely idle

`lane/gbdt-rest` and `lane/neighbors-rest` both recorded themselves dropped
on 2026-09-16 with their owed evidence named. Do not restart them as though
they had stalled.

## Why the machine fell over, and the two changes made

`mac_slot.sh` already carried the diagnosis from the day before: "the Mac hit
load 38 on 10 cores with ~15 agents each running one core". This is a 10-core
M4 with **16 GB** of memory, and that is the binding constraint, not cores.

Two things were changed:

1. **Spotlight was indexing every worktree.** At recovery time the load
   average was 14 / 20 / 23 with no mojo process running at all: it was
   `mds_stores` and a dozen `mdworker_shared` processes indexing 29 worktrees
   of 3 to 5 GB each. `.metadata_never_index` was placed in
   `~/mojolearn-wt`, `~/CascadeProjects/mojolearn-wt`, `~/mojolearn-evidence`
   and `/private/tmp/claude-501`.
2. **Every lane runs through the slot helper at one core.** The helper was
   preserved out of the dead session's scratchpad to
   `~/mojolearn-evidence/tools/mac_slot.sh`. Andrew's standing rule is one
   core per lane; the slot cap is what keeps the SUM of lanes bounded, and it
   is the part that was missing when the machine went down.

## The standing rules that bound all of this

* One Metal job at a time, through the slot helper. A Metal-only cell taken
  under contention is not evidence.
* A partial kill leaves an orphan on the GPU: kill the parent loop first,
  sweep, then verify zero survivors before removing a lock.
* Never cancel an owed run. "Finish up" is not authorization.
* GPU records are taken once per PyPI release: one Apple, one NVIDIA, one AMD.
* Merge on local one-core evidence; do not wait on CI.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
