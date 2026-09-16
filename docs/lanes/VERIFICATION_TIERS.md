# Verification tiers: what to run, and what not to run

Read this before running anything. It exists because checking a two-file
change had come to mean running a sweep of identity lanes, usually on the
Mac's Metal GPU, which is the scarcest resource the project owns: one
machine, one GPU, one job at a time. A full Apple column measured 10.77
hours. That cost was being paid for changes that touch two files.

There are three tiers. Use the smallest one that covers your change.

## The one command

All three tiers run through the same tool, `tools/verify_lanes.py`, which
gets its lane set from `tools/lane_select.py`. There is deliberately NO
separate full-sweep script: a second code path grows its own bugs and its own
idea of what the lane set is, and that is how one afternoon produced four
lane totals (176, 192, 199, 210) that each claimed to be the count. The
registry holds **199** lanes, read by importing `identity_break.LANES`.
`grep -c '@lane('` answers 176 and is an artifact, because 23 lanes register
by call rather than by decorator.

```sh
python3 tools/verify_lanes.py --lane logistic                # one lane
python3 tools/verify_lanes.py --changed-since origin/main    # this change
python3 tools/verify_lanes.py --all --shards 16              # everything
```

`--plan` prints what would run and stops. `--runner pods` prints one
`tools/runpod_cpu_leg.sh` command per shard and rents nothing.

## Tier 1, ROUTINE: on a change, before merging

Only the lanes the change can affect, on the CPU host route, small fixtures.
This is what a lane runs before it merges.

```sh
python3 tools/verify_lanes.py --changed-since origin/main --fixtures base
```

The selector prints the lanes it chose and WHY it chose them. Read that
output rather than the exit code alone. Two of its lines matter most:

* `FALLING BACK TO EVERY LANE` means it could not determine the blast radius
  of some changed path, so the selection was widened to all 199. That is a
  full sweep and must not be reported as a narrow run.
* `0 of 199 lanes selected` happens only for prose and evidence paths. The
  runner REFUSES an empty selection rather than exiting 0, because an empty
  run that reads as a pass is the failure this whole mechanism exists to
  prevent.

## Tier 2, OCCASIONAL: the broad sweep, on rented CPU

The full lane set, sharded across rented CPU pods. Parallel, cheap, and
bitwise equal to Metal, which is what makes it a substitute for the GPU
columns rather than a weaker check. NOT on the Mac: a full sweep on one Mac
is the 10.77 hour problem in a different shirt.

```sh
python3 tools/verify_lanes.py --all --shards 16 --runner pods --plan
```

That prints one leg command per shard and a cost line. A pod is about
$0.24/h at 8 vCPU (`docs/RUNPOD_CPU_LEG.md`) and about $0.48/h at 16 vCPU,
so the sweep is bounded by how many pods are worth paying for, not by wall
clock on one box. Ask before renting.

The merge is where a sharded run can lie, so it is checked three ways:

1. **A failed shard is never dropped.** A shard that exits non-zero, or whose
   part file never appeared, makes the run INCOMPLETE, writes
   `<out>/column.incomplete.json` instead of `<out>/column.json`, and exits 1.
   This is the same failure as `verify --all` printing VERIFIED with most
   parts refused, and it is refused by construction.
2. **The shards are the lane set.** The union is asserted when the split is
   made and again at merge time, against the registry count, never a grep. A
   lane that carries no cell is named and fails the run.
3. **The split is deterministic.** The same lane set always produces the same
   shards, so a rerun is comparable to the run before it.

## Tier 3, PER RELEASE ONLY: the three GPU vendor columns

One AMD, one NVIDIA, one Apple column, once, at a PyPI release. Never
routinely. Between releases, write what needs a GPU as OWED to the next
release record. This rule is not new and is not this lane's to relax.

## What Metal is for

A lane proving its OWN new cells, one job at a time, through
`mac_slot.sh metal`. Never a sweep. Concurrent Metal jobs on the shared M4
have returned NaN, constant and zero outputs, so a Metal cell taken under
contention is not evidence.

The CPU host route is what makes all of this safe: it reproduces the GPU
columns bit for bit on the covered lanes, which is what the CPU identity gate
diffs (`--require-columns 4` against the three committed GPU columns). A lane
that reads IDENTICAL on the CPU route has not been checked less carefully
than one that ran on Metal; it has been checked on the column that is cheap
to run and easy to shard.

## How the selector decides, and where it gives up

`tools/lane_select.py` derives lane -> source files from declarations that
something else already enforces: the registry itself, each lane function's
own code object, the Python doors indexed by what they define, the
`_mojolearn_*` bindings those doors name, each binding's own import lines
resolved into the Mojo tree, and `python/mojolearn/host_surface.py` for which
lanes a CPU family serves. Nothing here is a list kept by hand, because a
hand-kept list rots silently and then answers "nothing is affected" long
after that stopped being true.

It is conservative in three specific places, and says so each time:

* a changed path it cannot attribute selects EVERY lane;
* `python/mojolearn/_backend.py`, `python/mojolearn/host_surface.py` and the
  other files that enumerate the whole binding surface select EVERY lane,
  because the map deliberately drops their per-lane edges (otherwise every
  change selected all 199);
* `tools/identity_break.py` selects every lane unless the diff touches only
  lane function bodies, in which case it selects exactly those lanes.

Only prose and evidence paths select nothing.

```sh
python3 tools/lane_select.py --lanes-for-paths <path>     # what does this touch
python3 tools/lane_select.py --lane logistic --why        # what does this rest on
python3 tools/lane_select.py --selfcheck                  # every lane maps to real files
python3 tools/test_lane_select.py                         # the properties above, as tests
```
