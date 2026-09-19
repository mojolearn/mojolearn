# Verification tiers: what to run, and what not to run

Read this before running anything. It exists because checking a two-file
change had come to mean running a sweep of identity lanes, usually on the
Mac's Metal GPU, which is the scarcest resource the project owns: one
machine, one GPU, one job at a time. A broad Apple run exceeded seven hours and was stopped before completion. That cost was being paid for changes that touch two files.

There are three tiers. Use the smallest one that covers your change.

## The one command

For routine rounds use `pixi run -e test test-algo --lane NAME --out DIR`
with prebuilt CPU host bindings, or `test-identity-changed --base REF` for
changed-code selection. These use `tools/identity_iterate.py` and enforce the
budgets and summaries in [TEST_RUNTIME.md](../TEST_RUNTIME.md).

`tools/verify_lanes.py` remains the explicit broader qualification/sharding
runner. It shares `tools/lane_select.py` and now enforces backend readback, per-shard deadlines and a total budget.
Use it for explicit CPU parallelism; inspect its plan before broader work.

The count is not written down here either, for the same reason. Ask the
registry:

```sh
python3 tools/lane_select.py --count      # imports identity_break.LANES
```

Measured on this tree 2026-09-16 it answers **211**, while `grep -c '@lane('`
answers 188. The 23-lane gap is the kde, knn, radius, gp and gmm families,
which register by call rather than by decorator, so a tool built on the grep
would silently skip them. The gap is a property of how those families
register, not a number to maintain: four commits earlier the same two
questions answered 199 and 176, with the same 23 lanes between them.
`tools/test_lane_select.py` asserts the property, never the totals.

```sh
python3 tools/verify_lanes.py --lane logistic --plan         # inspect one lane
python3 tools/verify_lanes.py --changed-since origin/main --plan
python3 tools/verify_lanes.py --all --shards 16 --plan       # explicit broad plan
```

`--plan` prints what would run and stops. `--runner pods` prints one
`tools/runpod_cpu_leg.sh` command per shard and rents nothing.

`--changed-since REF` compares REF against YOUR WORKING TREE, which is what a
lane wants: the change is what you have, not what is committed. It follows
that pointing it at an old base from a much later tip measures the whole span
between them and not the commit you had in mind. To ask what ONE landed commit
affects, compare that commit against its own parent.

Some paths still select every lane and should. `pixi.toml` pins the toolchain,
and a toolchain change can move bits on every lane.

## Tier 1, OPTIONAL: checking a change

**Nothing here is required before a merge (2026-09-19).** Verification happens
once per release: the CPU column and the Apple pass (Tier 3). A change merges
without a check. Use this tier only when you WANT to know whether a kernel
edit moved a bit, and then only on the lanes that edit can reach.


The bounded routine entry point is now `pixi run -e test test-algo --lane NAME`.
It defaults to the CPU oracle, the base fixture and core checks, with separate
60-second queue and execution limits. Batch and decode contracts are selected
with `--probe-group`; `all` creates separate jobs. See
[TEST_RUNTIME.md](../TEST_RUNTIME.md) and [TEST_ORACLE_SCOPE.md](../TEST_ORACLE_SCOPE.md).
The full-column runners below remain explicit qualification tools.

Only the lanes the change can affect, on the CPU host route, small fixtures.

```sh
python3 tools/verify_lanes.py --changed-since origin/main --fixtures base
```

The selector prints the lanes it chose and WHY it chose them. Read that
output rather than the exit code alone. Two of its lines matter most:

* `FALLING BACK TO EVERY LANE` means it could not determine the blast radius
  of some changed path, so the selection was widened to every registered
  lane. That is a full sweep and must not be reported as a narrow run.
* `0 of N lanes selected` happens only for prose and evidence paths.
  `tools/verify_lanes.py` then REFUSES and exits 2 rather than exiting 0,
  because an empty run that reads as a pass is the failure this whole
  mechanism exists to prevent.

## Tier 2, OCCASIONAL: the broad sweep, on rented CPU

The full lane set, sharded across rented CPU pods. Parallel, cheap, and
bitwise equal to Metal, which is what makes it a substitute for the GPU
columns rather than a weaker check. NOT on the Mac: a full sweep on one Mac
recreates the oversized-suite problem.

```sh
python3 tools/verify_lanes.py --all --shards 16 --runner pods --plan
```

That prints one leg command per shard and a cost line. A pod is about
$0.24/h at 8 vCPU (`docs/RUNPOD_CPU_LEG.md`) and about $0.48/h at 16 vCPU,
so the sweep is bounded by how many pods are worth paying for, not by wall
clock on one box. Ask before renting.

The merge is where a sharded run can lie, so it is checked three ways:

0. **A refused lane is never a checked lane.** A shard can exit 0, write its
   part and merge cleanly with every cell reading REFUSED, which is what a
   stale host binding set produced on 2026-09-16: `verdict COMPLETE` in three
   seconds for a column that checked nothing. COMPLETE means every selected
   lane carries a cell that RAN, and the run prints how many lanes were
   actually checked next to how many were selected.
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

## Tier 3: release qualification and explicit GPU diagnostics

**A release is verified by the CPU column and the Apple pass, and by nothing
else (2026-09-19).** NVIDIA and AMD runs are optional diagnostics, not release
requirements, and the release workflow admits a Linux wheel without them.

### The Apple pass

```sh
pixi run -e test apple-pass                 # every lane that runs on Metal
python3 tools/verify_lanes.py --changed-since origin/main --apple-pass
```

This is the whole Apple check (2026-09-19). Metal answers ONE question: is its
end model the reference's. So each cell is fitted once and hashes train, infer
and save/reload; the batch, decode and gradient probes check batching logic,
which the CPU column carries on rented pods and one light NVIDIA pass confirms
on `base`. Fixtures are `base` (the ordinary path), `denormal` (where Apple is
known to differ, R2) and `odd` (every tile tail); the other six vary properties
the CPU column already covers on all nine. Lanes whose arithmetic never reaches
Metal are left out and named. The budget is 600 seconds total, and exhausting
it is a failure, never reduced coverage. `--apple-pass` refuses anything that
widens it (more repeats, probe groups, `--exhaustive`, another backend).

Between releases, routine iteration uses CPU. An Apple-specific failure may
need a real-Metal check: valid compiled AIR does not guarantee successful
pipeline creation, and CPU results cannot certify Metal buffer lifetimes.

Use `test-algo --lane NAME --mode metal --out DIR` for that investigation. The
default is one base/core job with a **60-second total budget**, including the
queue. Metal runs whatever is selected; no release marker or diagnostic flag is
needed, and several jobs share the same total budget unless `--budget` changes it. Fixture floors remain intact. Never silently count skipped or timed-out work as coverage.

CPU checks cover their own implementation and comparisons with named reference
records. Agreement with an older GPU record is not proof of a changed GPU
implementation. Keep reference provenance and numerical claims explicit.

See [TEST_RUNTIME.md](../TEST_RUNTIME.md) and
[the release checklist](../RELEASE_CHECKLIST.md#5b-apple-qualification-once-per-pypi-update).

## What stops the map narrowing WRONGLY

Every rule above widens what returns a narrow answer. The map itself can fail
the other way, and that failure is silent: a lane whose map is missing a file
gets a green run for a change that moved its bits. Two inversions are checked
over the WHOLE TREE, not against a case list, because a case list is what
missed `python/mojolearn/neural_inference.py` and the six mamba and samba
lanes it serves through subclasses.

* if F imports B, a lane reaching F executes B, so lanes(F) is a subset of
  lanes(B);
* if F subclasses or patches a class in B, a lane reaching B can run F's
  override, so lanes(B) is a subset of lanes(F);
* the same question on the Mojo side, where a miss costs more because a Mojo
  file IS the arithmetic: if F defines a struct conforming to a trait declared
  in B, lanes(B) is a subset of lanes(F). It holds with no backward edge
  needed, because Mojo conformance is not Python subclassing: a conforming
  struct is reached only when something parametrises on the trait and is
  handed that struct BY NAME, and naming a symbol from another file requires
  importing it. The inversion stays in the tree so that remains true.

`python3 tools/lane_select.py --census N` lists the files attributed to N
lanes or fewer with what each defines. A missing edge hides in a file credited
with too few.

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
  change selected every lane). Those files are found by COUNTING the bindings
  each one names, not by a list kept here, so a new registry file is caught
  the day it lands. On this tree they are `_backend.py` (42 bindings),
  `host_surface.py` (55), `_classical_host.py` (26), `__init__.py` (8) and
  `_parallel_worker.py` (4);
* `tools/identity_break.py` selects every lane unless the diff touches only
  lane function bodies, in which case it selects exactly those lanes.

Two more kinds of path select nothing, and each says which it is:

* a Python file whose code is IDENTICAL to the ref once docstrings are
  stripped. Comments never reach an AST and a dump without attributes has no
  line numbers, so the comparison is of code alone. Only the first statement
  of a module, class or function is dropped, so an edited error message is a
  code change and is attributed normally. There is no parser for Mojo here, so
  a Mojo comment edit still falls back. The exception is a file whose BYTES are
  hashed while the library runs, such as the six sources
  `python/mojolearn/_byte_lm_impl.py` publishes under `source_sha256`; those
  are found by looking for the modules that hash a file, and are never exempt.
* `tools/identity_break.py` when the diff only ADDS lanes. Additive is checked,
  not assumed: every existing top-level statement must be present unchanged and
  in order, and each addition must be an undecorated `def` with a new name,
  constant defaults and no mention in any existing lane's code.

ALL OF THESE WIDEN what returns a narrow answer, which is the dangerous
direction: a subtly wrong rule turns a real change into "nothing affected",
and that costs a defect where an over-broad sweep only costs time. So each is
tested from the failing side first. Seven code-change pairs must compare
different before six docstring pairs may compare equal. Eight harness edits
must answer every lane. Seven registry edits must answer every lane, among
them an entry removed, a reorder, an edited body, a new bare statement and a
decorated class. And a probe file that reads unreachable must go back to
falling back the moment a file every lane reaches names it, by path or only
through a glob over its directory. See `tools/test_lane_select.py`.

* a path NOTHING REACHES: no lane's derived source set contains it, and no
  file that any lane DOES reach names it, its stem or any directory above it.
  The corpus is the map itself, because for a lane to reach a file something
  in that lane's closure has to name it; `pixi.toml`, a CI workflow and a
  contribution gate all name `umap/checks` and none of them is in any lane's
  closure. Corpus files are searched with docstrings and comments stripped, a
  directory counts only when what follows it is not another path component
  (the glob shape), and a one-word top-level directory is not searched at all
  because it is a word, not a path. The rule under-fires on a short or common
  file name, which is the safe direction.
* a test module under `python/mojolearn/tests/`, when nothing outside that
  directory imports it. `_python_files()` already leaves the directory out of
  the map because a test cannot change what a lane computes; the import check
  is what keeps that true, and it runs over the whole tracked tree, because an
  import from a tool would still be an import.
* a whole-surface registry whose diff only ADDS to it. Every old statement
  must be present and in order, either byte for byte or as the same assignment
  whose container grew or whose value changed under an unchanged key, and each
  new statement must be an import of a module a lane already reaches, an
  undecorated def with a new name, or an undecorated class with a new name
  whose body is a docstring, defs and assignments. The addition is attributed
  through the files that define the names it mentions.

Only prose and evidence paths select nothing unconditionally.

```sh
python3 tools/lane_select.py --lanes-for-paths <path>     # what does this touch
python3 tools/lane_select.py --lane logistic --why        # what does this rest on
python3 tools/lane_select.py --selfcheck                  # every lane maps to real files
python3 tools/test_lane_select.py                         # the properties above, as tests
```
