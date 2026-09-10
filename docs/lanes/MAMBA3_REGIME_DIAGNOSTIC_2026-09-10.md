# Same-binary Mamba3 latency diagnosis

Prepared from main `9bf5115a`. No builds, tests, device runs, rentals, or
production changes were performed by this lane. Trees are untouched.

## What the retained evidence actually establishes

`bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json`
records original baseline0 narrow248.337356ms/wide102.489213ms, then
identical baseline1 narrow250.333956ms/wide283.757282ms. Tiny remains
1.267016/1.271963ms. `binary-sha256.txt` confirms the retained original
baseline library, and all three complete output hashes match across arms.
Earlier `mamba-stage/timings.json` measured the same original baseline
narrow56.767391ms. Thus scratch removal cannot explain all slow executions.

The repeat script runs a fresh Python process for each library arm, with
fixed tiny/narrow/wide order. It neither repeats one shape across different
predecessors in the same process nor records continuous hardware context.
Before/after snapshots show P0 and1980MHz, but cannot exclude a clock,
contention, or memory event inside individual calls. No cause is established.
Shape-specific regime changes with a stable tiny control warrant locating
which phase changes before changing kernels or re-adopting scratch.

## Bounded diagnostic

`tools/mamba3_regime_probe.py` loads the retained seed/weight fixture helpers
from an explicit path in the SAME checkout. The helper is historical and
untracked: root must stage `bench/speed/seq_py_speed_arm.py` and its existing
fixture dependencies, as for the previous public comparison. The script
refuses a helper in another checkout to prevent its sys.path setup from
selecting an unintended binding. Binary/Python paths, library and helper
SHA256, NumPy/Python versions, CPU affinity, input hashes, and addresses are
recorded. No opponent or tree estimator executes.

A process retains each constructed block/input for repeat visits, while each
forward is fresh zero-state public prefill. This deliberately controls input
addresses across visits; it does not reproduce the old process-lifetime
allocation sequence. Each call records wall elapsed time, process user/system
time, faults/context switches, RSS high-water mark, output address/alignment,
and complete output SHA256. Hashing is outside timing. Previous output is
held through the next call by default; a separate release-before-call run
isolates that lifetime choice. Neither run is a qualified production price.

Root-only commands, activated environment and an exclusive device slot:

```
python tools/mamba3_regime_probe.py --harness "$PWD/bench/speed/seq_py_speed_arm.py" --order narrow --passes 1 --rounds 12 --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json > /fresh/narrow-only.log
python tools/mamba3_regime_probe.py --harness "$PWD/bench/speed/seq_py_speed_arm.py" --order narrow,wide,narrow,tiny,narrow,wide --passes 2 --rounds 8 --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json > /fresh/shape-order.log
```

Repeat the second command in a fresh process. If the regime changes, repeat
with `--telemetry /fresh/gpu.csv` (250ms nvidia-smi sampling) and separately
`--release-output-before-call`. Telemetry is opt-in because monitoring can
perturb timing. Its process is owned and terminated by this script; errors
remain in the CSV and the exit code is printed. Resource counters are
process-wide; RSS units are native (KiB Linux, bytes macOS), and high-water
RSS is not current resident allocation.

Build a SEPARATE IDENTICAL binding with existing
`MOJOLEARN_MAMBA3_PHASE_TIMERS=1` only after reproducing a regime. Repeat the
same order and retain interleaved `call_begin`, native `M3_PHASE`, `call_end`
records. The added synchronization changes scheduling, so this diagnostic
cannot replace the original price or rule out a regime if it disappears.
Existing labels cover surface allocations/uploads/downloads/destruction,
refusal, projections, assembly, and core stages. Compare the same library
within each diagnostic arm, not phase-on versus phase-off as a speedup.

Interpretation: wall growth accompanied by user/system CPU growth points to
host work; mostly wall growth needs native phase localization and continuous
hardware evidence. Fault/RSS/address correlations motivate an allocator or
first-touch experiment, but do not prove it. A phase-local kernel change is
justified only after a reproducible phase-specific slowdown. GPU clock/MIG/
process snapshots and optional external profiler traces should retain the
same binary/input identity rather than changing multiple variables together.

## kNN follow-up

The latest accepted512-query batching addresses selector launch occupancy
and remains bounded to400k index rows. Earlier coalesced columns, partitioned
selection, deeper scans, and fold/feature hoists lost or lacked benefit.
Do not repeat them. The next concrete memory opportunity is eliminating the
unused radix fallback scratch on an admitted small-k path, but this requires
a complete routing audit across metrics, forced selector flags, and k ranges:
two buffers currently total391MiB at400k/batch512. Their allocation may cost
host time or constrain larger batching; it does not establish a throughput
win. No speculative production edit is included until the Mamba allocation
regime is understood and a same-source phase measurement supports this work.
