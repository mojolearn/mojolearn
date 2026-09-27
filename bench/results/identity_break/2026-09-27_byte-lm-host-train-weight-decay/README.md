# byte-lm-host-train at the 0.01 weight_decay default (2026-09-27)

## What diverged and why

On main, on an NVIDIA pod, `python -m mojolearn verify --all --lanes
byte-lm-host-train` read 12 DIVERGENT parts: all nine `train` parts and the
held-out `infer` parts of ties, hashed and denormal_ftz. The CPU and GPU
arms of `tools/algos_lane_check.sh` agreed with each other.

The cause is 06073dba5 (2026-09-22, first shipped in 0.8.14). It changed
`LanguageModelHostTrainer`'s default `weight_decay` from 0.0 to 0.01 so that
the CPU trainer's first step matches `SmallByteLanguageModelTrainer`, whose
default was always 0.01. The lane trains at the library default. Its
reference was recorded at 0.0 (2026-09-14 GPU columns, 2026-09-17 CPU
column) and was never re-recorded. The `train` hash includes the updated
parameters, so all nine moved. The held-out probe is the loss of a second
step, and it moved in three of the nine fixtures.

The proof was run on the same box at 860971010 (RTX 4090, AMD EPYC 7K62):

- Unchanged tree: 12 DIVERGENT.
- The same tree with only that default set back to 0.0: the lane VERIFIED,
  0 divergent.

The change was intended, so the bits were not restored. Commit 72a64f8b9
bumps the lane's `LANE_REVISIONS` entry to `weight-decay-default-0.01-1`.
Columns recorded before it no longer count toward this lane.

## Columns (commit 72a64f8b9, one fit per cell, all nine fixtures STABLE)

| column | file | how |
|---|---|---|
| nvidia (RTX 4090 pod) | `nvidia-rtx4090-sm_89.json` | `tools/identity_break.py --lanes byte-lm-host-train --vendor nvidia-rtx4090-sm_89 --repeats 1` (`legs/nvidia.log`) |
| cpu (AMD EPYC 7K62, x86-64) | `cpu-amd-epyc-7k62-x86_64.json` | `tools/cpu_identity_gate_check.py run-column --lanes byte-lm-host-train --shards 1 -- --repeats 1 --vendor cpu-amd-epyc-7k62-x86_64` (`legs/cpu.log`) |

`diff.cpu-vs-nvidia.txt` (`--require-columns 2`): OK, every part agrees.

The table was admitted with `verify --all --batch-checks --lanes
byte-lm-host-train --reference-table python/mojolearn/verify_reference/table.json
--emit-reference`. The run reported 0 conflicts and changed only the nine
byte-lm-host-train cells. Each of those cells rests on the cpu and nvidia
columns.

## Apple and AMD

These two columns are NOT re-recorded in this table. The Apple M4 and
MI325X columns of 2026-09-14 hashed the 0.0 step, and the revision bump
drops them from this lane. Both vendors were sent to the stewards
(`tools/apple_steward.py submit`: m2pro gating, do-amd when up, m3ultra
deferred) with `legs/sabotage-cpu-weight-decay-0.patch`. That patch restores
the old 0.0 default on the CPU arm only, and the check must DISAGREE under
it. A steward PASS shows that vendor's device arm equals its CPU arm. It is
not a record column. Admitting apple and amd columns needs an identity_break
record from each.

## Verify after admission (same box)

- `legs/verify-nvidia.txt`: `verify --all --lanes byte-lm-host-train` on the
  CUDA backend.
- `legs/verify-cpu.txt`: the same with `MOJOLEARN_VENDOR=cpu`.

Both read `RESULT: VERIFIED (verified 134 of 161 cell parts (0 divergent, 0
owed, 0 refused, 27 n/a); 1 verified ...) exit 0`.

`legs/lanecheck.txt` is `tools/algos_lane_check.sh byte-lm-host-train
--sabotage legs/sabotage-cpu-weight-decay-0.patch` on the same box. Result:
PASS. The CUDA and CPU arms AGREE on 9 train and 9 infer parts. Under the
sabotage they DISAGREE on 12 parts, the same 12 that verify reported on main.
After reversal they AGREE again.
