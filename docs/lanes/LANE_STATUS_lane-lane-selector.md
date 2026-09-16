# lane/lane-selector: run only the lanes a change can move

Branch `lane/lane-selector`, from `origin/main` at 8cc2002b0.
Worktree `/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/wt-laneselect`.

## The problem this answers

Checking a two-file change had come to mean running a sweep of identity
lanes, usually on the Mac's Metal GPU: one machine, one GPU, one job at a
time, and a full Apple column measured 10.77 hours. Nothing in the tree could
answer "run only the lanes this change can possibly affect", so the answer
was always "run everything".

## What is on the branch

| file | what it is |
|---|---|
| `tools/lane_select.py` | the map and the selector: lane -> source files, derived; file -> lanes, inverted; and the selection for a set of changed paths |
| `tools/verify_lanes.py` | the ONE command that runs lanes: one, the selected set, or all of them, sharded, with a merge that cannot silently pass |
| `tools/test_lane_select.py` | the properties that must not regress, each written to fail loudly rather than quietly |
| `docs/lanes/VERIFICATION_TIERS.md` | the three tiers, to read before running anything |

## How to run it

```sh
cd <worktree>
python3 tools/lane_select.py --changed-since origin/main      # what would run
python3 tools/lane_select.py --lanes-for-paths <path>         # what does this path touch
python3 tools/lane_select.py --lane logistic --why            # what does one lane rest on
python3 tools/lane_select.py --selfcheck                      # every lane maps to real files
python3 tools/test_lane_select.py                             # the properties, as tests

python3 tools/verify_lanes.py --changed-since origin/main --fixtures base   # tier 1
python3 tools/verify_lanes.py --all --shards 16 --runner pods --plan        # tier 2, plan only
```

The CPU host route needs a host binding set. This lane used a COPY of the
shared checkout's `python/mojolearn/host/*.so` in its own directory, never
the shared inodes, because a binding rewritten under a running process kills
it with exit 137 and no output:

```sh
mkdir -p $SP/hostdir-laneselect && cp /Users/andrewhendel/CascadeProjects/mojolearn/python/mojolearn/host/*.so $SP/hostdir-laneselect/
env PYTHONPATH=<worktree>/python MOJOLEARN_HOST_DIR=$SP/hostdir-laneselect \
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/verify_lanes.py --lane kmeans --fixtures base
```

## How the map is derived

Nothing here is a list kept by hand. A hand-kept map rots silently and then
answers "nothing is affected" long after that stopped being true.

* the registry: `identity_break.LANES`, read by IMPORT. `--count` prints the
  size and nothing writes it down: on the merged tree 2026-09-16 it is 211,
  against the 188 `grep -c '@lane('` finds. The 23-lane gap is the kde, knn,
  radius, gp and gmm families, which register by call. Four commits earlier
  the same two questions answered 199 and 176, with the SAME 23 lanes in the
  gap, which is why the test now pins the gap and not the totals.
* the lane body: the lane function's own code object, followed through the
  module's helpers.
* the Python door: `python/mojolearn/*.py` indexed by what each file defines,
  closed over the package's own imports.
* the binding: resolved from the door's SYNTAX (`_backend.binding(...)`,
  `self._bind(...)`, `_BINDING = ...`, `from . import _mojolearn_*`).
* the Mojo tree: each binding's own import lines, resolved per EXPORT rather
  than wholesale, then followed transitively.
* the CPU surface: `python/mojolearn/host_surface.py` declares which lanes a
  host family serves.

## Where it deliberately gives up, and says so

* a changed path it cannot attribute selects EVERY lane;
* the files that enumerate the whole binding surface
  (`python/mojolearn/_backend.py`, `python/mojolearn/host_surface.py`,
  `python/mojolearn/_classical_host.py`,
  `python/mojolearn/_parallel_worker.py`, `python/mojolearn/__init__.py`)
  select EVERY lane, because the map drops their per-lane edges on purpose;
* `tools/identity_break.py` selects every lane unless the diff touches only
  lane function bodies;
* only prose and evidence paths select nothing.

## Two defects found in this tool, by this tool's own tests

1. **The silent zero.** `--lanes-for-paths $files` under zsh arrives as ONE
   argument (zsh does not word-split an unquoted variable). That string began
   with `CHANGELOG.md`, matched the inert-prefix rule, and a twelve-file
   commit selected ZERO lanes without complaint. An argument containing
   whitespace is now never inert: it falls back to every lane and says why.
   `test_several_paths_in_one_argument_are_never_inert` pins it.
2. **Binding edges from prose.** Harvesting `_mojolearn_*` names by text
   search matched them in docstrings and comments, which this codebase is
   full of. Every lane picked up the forest and byte LM host bindings, hit
   their whole-closure fallback, and `core/gbdt_host_predict.mojo` selected
   every lane at once. Edges now come from syntax only.
   `test_binding_edges_do_not_come_from_prose` pins it.

A third correction, to a claim made mid-lane: the first single-lane CPU
timing (2 s) was a REFUSAL, not a measurement. The copied host binding set
dates from Sep 14 and `_mojolearn_estimators_host` exports no `qn_fit`, so
`logistic/base` read `REFUSED`. A refusal that returns in 2 s is not evidence
that the tier is fast.

## Measured

OWED in this file until the numbers are in hand. Nothing here may be quoted
as a saving until it names the run that produced it.

## Owed / not done

* The three push gates (`docs_facts --check`, `packaging/wheel_ci.py pins .`,
  `packaging/wheel_ci.py inventory python/mojolearn`) run before any push.
* No pod was rented. Tier 2 is implemented and planned, never executed.
* No fixture and no lane definition was edited: another lane is shrinking
  fixtures for ten expensive lanes, and this branch only adds selection
  machinery around them.
* The Metal lock was never taken.
