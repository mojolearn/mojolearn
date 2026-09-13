# identity_break columns

Written by `tools/identity_break.py --json`, one file per box, diffed with
`tools/identity_break.py --diff a.json b.json ...`. The three GPU columns at
the shipped default are under `2026-09-13_three-columns/` (its README names
the boxes and the commits). The older files at this level predate the infer
and model columns and diff on the train column only.

## The fourth column, a CPU (the CPU training lane, 2026-09-13)

`.github/workflows/cpu-identity-gate.yml` runs the tool on the seven free
GitHub runners against the checkout's package with no GPU set, so
`mojolearn.vendor()` is `cpu` and every lane resolves through
`python/mojolearn/_backend.py::_HOST_MODULES`. A lane whose family has a host
binding runs; every other lane records REFUSED with the by-name sentence
`no CPU implementation of <binding>.<function> yet`. The JSON of a CPU column
carries

| key | meaning |
|---|---|
| `vendor` | `cpu-<cpu model slug>`, derived from the machine, never typed |
| `commit`, `commit_source` | the commit the run was made from (`MOJOLEARN_COMMIT`, `git rev-parse HEAD`, or a `COMMIT` / `commit.txt` witness); required, a JSON with an empty commit is never written |
| `host.cpu_model`, `host.arch`, `host.target_cpu`, `host.mojo_version`, `host.python` | the machine and the build it ran |
| `host.column` | what the host bindings read back as their kernel-matrix column (`cpu`, the `COLUMN_CPU` of `checks/kernel_matrix.mojo`, asserted at build time) |
| `host.families` | every `_mojolearn_*_host.so` built, with its `column`, `detected_column` and `sabotage` read-backs, so a REFUSED cell is attributable to an unbuilt family rather than a bug |
| `host.routed` | the `_HOST_MODULES` table the run resolved through |

The workflow's `COVERED_LANES` is the list of lanes with a CPU
implementation; it is EMPTY until phase 1 of
`docs/lanes/BRIEF_cpu_training_2026-09-13.md` lands the first (gemm-pinned).
A covered lane must read STABLE on every fixture and `IDENTICAL x4` against
the three GPU columns (`--diff ... --require-columns 4 --lanes <covered>`),
because `IDENTICAL x3` on a lane the CPU column should cover is the CPU
binding refusing, not a pass.

A CPU column enters the brief's certified table only after its uploaded
JSON is read and its `host.cpu_model` names the CPU. Nothing in this
directory is a CPU column yet.

## Provenance rules for every column since 2026-09-13

`--vendor` must be a box label (`^[a-z0-9][a-z0-9_.-]*$`, never `box-arch` or
another placeholder) and `commit` is required. The 2026-09-13 NVIDIA column
recorded `box-arch` from an unexpanded env default in the leg body and all
three GPU columns carry an empty commit; the tool refuses both now, and those
files are kept as recorded.
