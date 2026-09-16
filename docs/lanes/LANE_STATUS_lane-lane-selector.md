# lane/lane-selector: run only the lanes a change can move

Branch `lane/lane-selector`, from `origin/main` at 8cc2002b0, merged with
`origin/main` at 2dbeedec6 and verified there.
Worktree `/Users/andrewhendel/mojolearn-wt/lane-selector`. The original
worktree was under `/private/tmp/claude-501/` and did not survive the crash.

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

## Verified 2026-09-16, on the merged tree, one core

Branch merged with `origin/main` at 2dbeedec6 and re-run. Everything below is
from a run in `/Users/andrewhendel/mojolearn-wt/lane-selector`, through
`mac_slot.sh` at one core. Logs are in
`/Users/andrewhendel/mojolearn-evidence/lane-selector-2026-09-16/`.

| check | result |
|---|---|
| `--count` (registry, by import) | 211; `grep -c '@lane('` 188; the 23 in the gap are the kde, knn, radius, gp and gmm families |
| `--selfcheck` | OK, 0 lanes with an empty side, 723 files mapped |
| `tools/test_lane_select.py` | 16 tests, 0 failures, 2.4 s |
| one lane end to end (`--lane knn`) | COMPLETE, exit 0, 2 s |

### What was made to fail, to show each guard is the thing that holds

* **The fallback.** `pixi.toml` printed `FALLING BACK TO EVERY LANE` and
  selected 211 of 211. With the `fallback = True` line deleted from the
  unattributable branch, the same path printed `0 of 211 lanes selected` and
  exited 0, silently. `cluster/host/kmeans_oracle.mojo` printed no fallback
  line at all and selected 20 lanes, so the message is conditional.
* **The empty refusal.** `verify_lanes --lanes-for-paths CHANGELOG.md --plan`
  printed REFUSING and exited 2. With the refusal block deleted, the same call
  exited 0 and planned `identity_break --lanes` with an empty lane list.
* **The merge.** `--lane knn` is COMPLETE at exit 0, so the check can pass.
  `--lane kmeans` (REFUSED on this stale host binding set) is INCOMPLETE at
  exit 1; `--lanes knn,kmeans` is INCOMPLETE too, so one good lane does not
  cover for a refused one; and `--lanes knn,knn-clf --shards 2` with
  `MOJOLEARN_NUMERIC_MODE` unset had both shards exit 1 and named both rather
  than dropping either.

The two sabotage copies were scratch files under `tools/_sab_*`, never
committed. What they proved is now pinned by
`test_an_unattributable_path_falls_back_to_every_lane_and_says_so` and
`test_the_runner_refuses_an_empty_selection`, which calls `verify_lanes.main`
and requires exit 2 for an empty selection and exit 0 for a narrow one.

### Three defects this verification found

1. **The lane total had already rotted.** 199 and 176 were right when written
   and wrong four commits later. No total is written down now.
2. **The second claimed fix was inert.** Binding edges from SYNTAX stopped
   prose creating them; the IMPORT CLOSURE still created the same ones, since
   `_bufcheck.py` -> `_buffer.py` is in every lane's closure and reaches
   `_forest_host.py`, `_byte_lm_impl.py` and `_byte_lm_host.py`.
   `core/gbdt_host_predict.mojo`, the file the docstring named as fixed,
   still selected all 211 lanes. A binding every lane reaches now gives an
   undeclared lane its source only.

   | | before | after |
   |---|---|---|
   | files selecting every lane | 73 | 41 |
   | median lanes per file | 35 | 32 |
   | `core/forest_host_predict.mojo` | 211 | 7 |
   | `core/gbdt_host_predict.mojo` | 211 | 23 |
   | `mamba/impl/modeling/modeling_mamba.mojo` | 211 | 26 |
   | files in the map | 718 | 723 |

   6066 file edges dropped, 468 added, and every lane keeps its own family
   tree.
3. **`verdict COMPLETE` for a column of refusals.** See the commit; COMPLETE
   now means every selected lane carries a cell that ran.

Two more, smaller: a family case named a DIRECTORY and so passed vacuously,
and the selector read its own three files as NOT ATTRIBUTABLE, so this lane's
first real use of its own tool fell back to all 211 lanes.

### 2026-09-16 afternoon: the first lane to use it reported a false sweep

`lane/data-ordering-determinism` ran `--changed-since origin/main` and got
FALLING BACK TO EVERY LANE, 212 of 212, for a diff whose blast radius was
three lanes. Two causes, both fixed.

1. **A comment counted as a change.** A Python path whose code is identical to
   the ref once docstrings are stripped is now inert. Only the first statement
   of a module, class or function is dropped; a string literal anywhere else
   is a value. Files whose bytes are hashed at run time are excluded, derived
   from the modules that hash a file.
2. **Inserting a lane renumbered everything below it.** `harness_lanes` keyed
   a bare top-level statement by its LINE NUMBER, so an additive hunk changed
   every key and returned every lane. Segments are AST dumps now.

Proved on the real tree, both directions, at `/Users/andrewhendel/mojolearn-evidence/lane-selector-2026-09-16/real_tree_arms_2026-09-16.md`:
a comment, a docstring and an inserted lane gave **1 of 213 lanes selected**,
naming the added lane; the same three files with a real `tol` change and a
real helper edit fell back to 213 of 213; and an f-string ERROR MESSAGE
changed alone in the same file moved it from inert to 3 lanes.

### 2026-09-16 late: four more false sweeps, from four lanes

Andrew's complaint that afternoon was that we are testing too much, and the
selector's fallback had become the mechanism of it. Each of these returned
212 of 212 and each was caught only because the lane read the reason line and
refused to run the sweep.

| lane | cause | now |
|---|---|---|
| `lane/kmeans-save` | an additive entry in a whole-surface registry | 36 of 212 |
| `lane/umap-batch-determinism` | three new files under `umap/checks/` that no lane references | 0, and the runner refuses |
| `lane/discarded-atomic-audit` | an untracked `.metadata_never_index` in the worktree | 0, and the runner refuses |

The second and third are one rule: a path is unreachable when no lane's map
contains it AND no file that any lane DOES reach names it, its stem or any
directory above it. Four things had to be got right and each was found by
running it: prose in a corpus file is not a reference, a directory token must
name the directory rather than any file under it, a one-word top-level
directory is a word and not a path, and there must be no skip set for the rest
of the same change, which would have hidden a new source added together with
the import that pulls it in.

The first is the registry rule: additive is verified against the old
statements, and the addition is attributed through the files that define the
names it mentions. Registries are excluded from that resolution, because
`_HostBound` is defined in `_classical_host.py` and every lane reaches that
file, which attributed the addition to all 212.

Also folded in, rather than opening a lane for it:
`tools/source_hygiene_check.py` keeps `_ = Atomic.load(...)` and
`_ = Atomic.compare_exchange(...)` at zero, in light-checks and as
`pixi run check-source-hygiene`. THE GREP AS THE AUDIT WROTE IT DID NOT WORK:
`git grep -E` is POSIX ERE and has no `\s`, so
`'^\s*_ = Atomic\.(load|compare_exchange)'` matches nothing, exits 1 and reads
exactly like a clean tree. Its self-test now runs every pattern through the
real `git grep` and requires the probe lines to match and the near misses not
to.

### Re-checked against the commits as they LANDED, not against branches

The registry rule was reported as not narrowing on main. It does. The report
measured `--changed-since a3a9c891b^` from a much later tip, and
`--changed-since` compares a ref against the WORKING TREE, so that span was
many lanes' commits rather than the one commit in question. Two OTHER paths in
that span were forcing the fallback, and the whole-selection FALLING BACK line
sat under all of them.

The kmeans-save merge alone, `a3a9c891b^..a3a9c891b`, eight paths:

    python/mojolearn/_classical_host.py   additive registry: 33 lane(s)
    python/mojolearn/host_surface.py      additive registry: 9 lane(s)
    python/mojolearn/cluster.py           33 lane(s)
    tests/test_host_model_kmeans.py       a test module
    tools/kmeans_save_boundary_check.py   nothing reaches it
    tools/kmeans_save_nvidia_leg.sh       nothing reaches it
    CHANGELOG.md, one LANE_STATUS         inert
    === 36 of 212 lanes selected; paths forcing the fallback: NONE

The other three rules, also against landed commits:

* **docstrings and comments.** Four landed commits on main change a package
  `.py` file in docstrings alone. One of them, `b8a2bbe2a`, is a three-way
  control by itself: `ensemble.py` is inert, while `_byte_lm_impl.py` and
  `_mlp_impl.py` are equally code-identical and correctly still fall back,
  because their bytes are hashed at run time under `source_sha256`.
* **an added lane.** Of the last 40 landed commits touching
  `tools/identity_break.py`, five narrow to exactly one lane each
  (`bpe-trainer`, `cross-val-folds`, `tokenizer`) and 35 answer every lane.
* **nothing reaches it.** Of 120 files added by recent landed commits, 26 read
  unreachable and 94 still fall back, among them `core/device_mutex.mojo` and
  `python/mojolearn/_bpe_trainer.py`, which are product code.

None of the three moved.

### The one path left, and why it stays

`pixi.toml` pins the toolchain. A toolchain change can move bits on every
lane, so it selects every lane and that is not a gap to close.

### Measured cost

Deriving the map took 146 s per call before memoization, and the property
tests took 11 minutes. With per-file results memoized, `lane_sources()` is
0.9 s and the suite is 2.4 s. The tier 2 plan for all 211 lanes over 16 pods
prints a longest shard of 120 s against a serial equivalent of 1883 s, about
$0.26 at $0.48/h per pod.

NOT MEASURED: any saving against the Metal sweep this lane exists to replace.
No Apple column was run and none should be, so the 10.77 hour figure is the
old record, not a before-and-after.

## Owed / not done

* The three push gates (`docs_facts --check`, `packaging/wheel_ci.py pins .`,
  `packaging/wheel_ci.py inventory python/mojolearn`) run before any push.
* No pod was rented. Tier 2 is implemented and planned, never executed.
* The copied host binding set dates from Sep 14 and serves few lanes:
  `kmeans`, `ols`, `pca`, `dbscan` and `tsvd` all read REFUSED against it.
  A tier 1 run that is worth anything needs the host families rebuilt first,
  and the new verdict says so instead of printing COMPLETE.
* No fixture and no lane definition was edited: another lane is shrinking
  fixtures for ten expensive lanes, and this branch only adds selection
  machinery around them.
* The Metal lock was never taken.
