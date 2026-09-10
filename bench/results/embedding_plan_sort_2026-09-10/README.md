# Embedding PLAN_SORT and clause 11(d), 2026-09-10

The IDENTICAL embedding backward now accepts an explicit `PLAN_SORT` and a
validated launch-thread override. The default remains `PLAN_SCAN`. Both use
the existing ascending-position FP32 gradient fold.

PLAN_SORT packs the contract's UInt64 `(id, position)` total keys, sends padding
and allocation slack to a sentinel tail, runs disjoint device bitonic passes,
and derives counts/run boundaries by integer binary search. It writes only the
used permutation; caller-owned unused tail entries are preserved. It owns a
power-of-two UInt64 temporary allocation and synchronizes before freeing it.
No tree primitive or other numeric mode was changed.

The executable source is branch `perf/embedding-plan-sort-20260910` through
`0250d63b`; production code last changed in `316f4c5f` (negative-control
registration). `tools/check_embedding_plan_sort.sh OUTPUT_DIR` reproduces the
small gates from an activated toolchain. The shipped gate is a separate remote
GPU invocation of `embedding/checks/embedding_sort_shipped_check.mojo`; it uses
about 4.3 GB of device memory and is intentionally absent from the local script.

## Apple

Root ran the final script under the shared build lock. `apple/fixture.log`
records clause (a), 17 accepted fixtures and 6,887 cells with all nine stages
bit-identical to the independent host oracle. Real clause (d) passes 102
comparisons: 17 accepted fixtures × two production plans × three launch sizes
(32, 96, 160), comparing counts, run_begin, used perm and dW exactly.

`apple/edges.log` adds 56 combinations of T=0/1/3/31/33/129/257, overwrite and
accumulate, absent padding, first/last row padding, and all-padding IDs. Each
runs both plans at all three launch sizes and checks the unused permutation
tail remains poisoned. `apple/negative.log` names the registered actual-device
sort mutation and fails at the required metadata mismatch: reversed ties move
28 of 31 permutation entries. `apple/verdict.txt` confirms the expected result.

## NVIDIA H100 and shipped shape

NVIDIA H100 80GB HBM3, driver 580.126.09, Mojo 1.0.0 (ed45d567).
`nvidia/context.txt` records the GPU UUID, UTC time, and source hashes.
The same 17-fixture/102-comparison gate, 56-edge gate and registered device
negative control pass. The Apple and NVIDIA clause (a) cards are byte-identical.

`nvidia/shipped.log` closes the required V=128256, D=4096, T=4096 leg.
Against the default scan baseline, both plans at 32/96/160 threads match
all **525,336,576** raw FP32 output cells, all 128,256 counts, all 128,257
run boundaries, and all 3,855 used permutation entries. The fixture contains
repeated IDs and 241 padding positions. Comparison runs on device with integer
per-chunk mismatch counts; every output cell is examined, with no digest or
sampling shortcut.

Single-call diagnostics captured during those required gates:

| Plan | Threads | Milliseconds |
|---|---:|---:|
| Scan, baseline | 256 | 9.807279 |
| Scan | 32 | 26.939250 |
| Scan | 96 | 13.798866 |
| Scan | 160 | 11.250845 |
| Sort | 32 | 20.073677 |
| Sort | 96 | 6.802507 |
| Sort | 160 | 4.168227 |

These are sequential single calls, with no warmup/repeated/interleaved pricing
protocol; they establish neither a crossover nor a default-dispatch change.
There is no new opponent measurement. Metadata sorting does not change the
pinned floating-point fold.

`nvidia/refusal-audit.log` separately records six correctly named host refusals
and confirms the existing device nonfinite gap: a NaN in W is not refused and
one nonfinite cell reaches gather output. The explicit acknowledgement lets
the diagnostic complete; it does not qualify clause (f).

The first wrapper invocation hit absent `rg` after all small binaries had run.
The wrapper was made portable with `grep` in `c0d34cb0`; the archived resume
script verified the expected negative-control banner/error and then ran the
shipped gate and refusal audit. No completed numerical gate was skipped or
rerun to conceal a failure. Both driver and resume evidence are retained.

## Scope

This closes the implementation gap for clause (d), not all embedding contract
debt. Production entry points already refuse invalid IDs but still omit
nonfinite W/dY refusal. The existing clause (f) audit is separate from plan
success; this change adds no host float downloads to production. Other sabotage
arms and an independent opponent performance comparison are outside this run.
No automatic scan/sort crossover is claimed. Shipped single-call diagnostics,
when present, time the production backward through synchronization; full-array
bit comparison and host readback are outside the measured interval.
