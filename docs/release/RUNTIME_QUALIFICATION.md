# Installed-runtime qualification: bounded, resumable, per unit

The `runtime-<vendor>` ledger rows (`runtime-nvidia`, `runtime-amd`, `runtime-apple`, `runtime-cpu`
in `tools/release_requirements.json`) rest on `tools/release_runtime_installed.py run`: the 73
registered public test files (`tools/release_runtime_coverage.json`) against the installed candidate
wheels. At 0.8.65 that was one pytest run under one 900 s budget; it stopped at 12% (NVIDIA) and 23%
(AMD) and the full run takes about 125 minutes on an L40S. This page is the contract that replaces it.
The code is `tools/runtime_qualifier_units.py`.

## Units and receipts

A **unit** is one registered test file. The qualifier (`tools/qualify_native_runtime_installed.py
--unit-plan`) runs each unit as its own bounded worker (`qualify/run/runs/<stem>/`) and, the moment the
unit ends, writes one receipt:

- `qualify/units/<file>.json` and its raw junit `qualify/units/<stem>.tests.xml` in the run's output;
- the same pair appended to the unit store (`--unit-store`): `<store>/<stem>/<finished_at>-<sha12>.json`.

The units run side by side, heaviest budget first: one per core of the host (affinity and cgroup
quota), at most one per 1.5 GiB of free memory, and a unit starts only while the free memory holds the
largest peak RSS measured so far. `--jobs` or `MOJOLEARN_QUALIFY_JOBS` sets the count. Each unit keeps
its own budget and stall bound. The log opens with `QUALIFY units=<n> workers=<w> of <cores> core(s)`
and prints `QUALIFY busy cores=<b> of <total>` every 60 s. On the Mac, `runtime-apple` holds the Metal
lock once and every Mac slot (`mac_slot.py --slots <n> metal`).
  The store is never overwritten or pruned.

A receipt (`mojolearn.runtime-qualifier-unit.v1`) holds: status (`passed`, `not-applicable`,
`failed`), seconds, budget, the reason a failed unit failed (always naming the unit), the candidate
wheel sha256s, vendor, backend and device class, the worker interpreter, the native modules the unit
loaded with their bytes' sha256, the source and test commits, the unit's source closure, the unit's
coverage scope re-derived from its junit, and a timestamp.

## The unit's source closure

`runtime_qualifier_units.unit_closure` derives it with the existing closure tools; nothing is
hand-listed:

- **test side**: the test file, its ancestor `conftest.py`/`__init__.py` and the test helpers it
  statically imports, walked by `tools/runtime_test_tree_reuse.py` (the resolver the runtime
  composition already proves unchanged test inputs with). An unresolved import (dynamic loading,
  reflection) makes the closure unavailable;
- **product side**: the `mojolearn` modules those files import, closed over the package's own imports
  by `tools/lane_select.py` `_python_closure` (the whole-file walk, with its backwards
  subclass/patch edge, that the identity lanes' Python doors use). The worker imports `mojolearn`, so
  `python/mojolearn/__init__.py` is always a seed. Each file's working bytes must equal the source
  commit's blob, or the closure is unavailable;
- **native side**: not derived but observed. The receipt records the native modules the unit's worker
  actually loaded, and closure reuse requires the new wheels to carry those members with the same bytes.

Hashes: `tests_sha256`, `product_sha256` and `sha256` over both, with the full file inventories kept
for audit (`CLOSURE_DEFINITION = runtime-unit-closure.v1`).

## Reuse, invalidation and resume

`run` writes `qualify/units.json` (the plan) first. For each unit, the newest receipt in the unit store
and in every `--prior-units` directory is considered; failed receipts and other device classes never
count. The unit is **reused** (its receipt copied into `qualify/units/` with a `reused` record naming
the rule, the original receipt and the current wheels and closure) when either

- `wheel-bytes`: the same candidate wheel bytes and the same test closure, or
- `closure`: the same test closure, the same product closure (or one whose only differences
  `lane_select.release_label_only` (the release `__version__` literal) or
  `lane_select.docstring_only` call inert, judged against the receipt's own source commit), the same
  native module bytes in the new wheels, and the same interpreter cache tag.

Anything else **runs**, and the plan states why (`closure changed: <files>`, `test closure changed`,
`device class ...`, `the native modules the unit loaded differ`, `no receipt for this unit`). `--redo`
runs every unit.

Resume is the same rule: a run that is interrupted keeps every receipt already written to the store,
so the next run with the same store (a new `--out`) reuses those units by wheel bytes and runs only
the missing or invalidated ones. The run prints `RUNTIME units reused=n run=m failed=k`.

## Budgets

Each unit's budget is three times its last measured seconds, at least 120 s (1,800 s for a unit with no
history). One test without progress for `--test-timeout` (clipped to the unit's budget) also stops the
unit. `--timeout` stays the outer cap over the whole run: a unit is clipped to what it leaves, and a unit
that cannot start because it is spent is a failed receipt `unit <file> not started: the whole-run budget
of N s was spent`. A unit over its budget is a failed receipt `unit <file> over its budget of N s`. No
unit is ever silently dropped: a unit with no receipt is failed in the report as `no unit receipt`.

## The report the ledger reads

`release_runtime_installed.py summarize` re-derives a per-unit run (origin `per-unit-receipts`) from the
retained files only: `coverage.json`, `request.json`, the plan and each unit receipt with its junit. Every
unit's scope is recomputed from its junit and must equal its receipt; a run unit must name this run's
wheels and closure; a reused unit must carry the plan's rule, and a closure reuse must have exactly the
plan's inert differences and matching native bytes. Mixed reused and new units make one report with the
existing coverage rows, so the `installed-runtime` ledger adapter (`tools/release_installed_evidence.py`)
admits it unchanged: the report names the current candidate wheels, and a retained file altered after the
run makes the stored report differ from its re-derivation. Earlier single-run and composed reports still
re-derive as before.

## Smoke and qualify

- **smoke**: the gate's own installed smoke, run by every gate and never reused: the eleven
  release-light-v1 installed correctness records (`tools/gate_smoke_box.py`) on NVIDIA and AMD, and
  `macos-smoke` (`tools/qualify_verifier_wheel.py --scope expanded`) on Apple. It is not a qualifier unit.
- **qualify**: the 73 units above, run only when their reuse rule refuses.

The plan records both tiers (`tiers`, `smoke`).

## Producers on the release path

| Row | Where | Device class | Unit store |
| --- | --- | --- | --- |
| `runtime-nvidia` | NVIDIA gate job (`tools/ops/release_gate_job.sh`, after the smoke) | `nvidia-sm_89` | `<results>/out/runtime-units-nvidia-sm_89/` |
| `runtime-amd` | AMD gate job | `amd-gfx942` | `<results>/out/runtime-units-amd-gfx942/` |
| `runtime-cpu` | NVIDIA gate job, after `runtime-nvidia` (`GATE_RUNTIME_CPU=1`; `--vendor cpu`, so `MOJOLEARN_VENDOR=cpu` for the qualifier only; never timed) | `cpu-linux-<machine>` | `<results>/out/runtime-units-cpu-linux-<machine>/` |
| `runtime-apple` | Mac `gate-apple` step (`tools/release.py apple_runtime_gate`, under the Metal slot) | `apple-<chip>` | `<freeze>/gate-apple/runtime-units-apple/` |

The box stores live under the job's results directory, which `tools/box_persist_run.sh` uploads every
few minutes, so an interrupted job keeps its finished units. Before a gate job, `tools/release.py
gate_runtime_prior` stages the newest passing receipt per (device class, unit) from every retained gate
output of every version and freeze, and `tools/ops/release_gate.py --runtime-prior` ships them as one
deterministic `runtime-units-prior.tgz` named by `GATE_RUNTIME_PRIOR_SHA256`. A prior set whose bytes
differ is dropped (every unit runs); it never changes the job's verdict. A source commit whose producer
predates the unit flags runs without them.

The ledger producers (`Release.retained_producers`) name `gate-nvidia/*/runtime-cpu/runtime-report.json`
and `gate-apple/runtime-apple/runtime-report.json` beside the vendor reports. `gate-apple` fails when
`runtime-apple` did not pass for the macOS wheel bytes, as the NVIDIA and AMD gates fail on theirs.
