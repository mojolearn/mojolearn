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
from explicit `--harness` and `--spec` paths. The retained spec supplies the
Mamba3 rows missing from the current tracked `tools/speed_torch_seq.py`.
It is loaded under `speed_torch_seq` before the helper, with `REPO` and
`DRIVER_MOJO` explicitly rooted in the current checkout. Archive-relative
sys.path additions are discarded before the public API is imported. Neither
tracked tools nor fixture files are overwritten. Original/effective roots,
helper/spec/driver hashes, loaded Python path, binary path and library
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
python tools/mamba3_regime_probe.py --harness bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py --spec bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py --order narrow --passes 1 --rounds 12 --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json > /fresh/narrow-only.log
python tools/mamba3_regime_probe.py --harness bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py --spec bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py --order narrow,wide,narrow,tiny,narrow,wide --passes 2 --rounds 8 --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json > /fresh/shape-order.log
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

Local Metal tiny smoke (root alone; no telemetry or large allocations):

```
python tools/mamba3_regime_probe.py --harness bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py --spec bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py --order tiny --passes 1 --rounds 2 --reference-json bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json > /fresh/tiny-smoke.log
```

A tiny pass only qualifies fixture loading and instrumentation. H100 large
shape regime diagnosis remains RUN OWED and is not inferred from Metal.

Root continuation: tiny Metal smoke passes both retained complete hashes; large H100 shape-order runs remain owed. Evidence: bench/results/mamba_regime_2026-09-10/.

## Retained-log review continuation

`tools/mamba3_regime_summary.py LOG` validates the complete scheduled call
sequence, one binary digest, per-shape output digests, and paired begin/end
records before emitting a JSON report. It retains every sample, separates the
first call of each visit from the median of later calls (without calling those
later calls warmed up), records each visit's predecessor, and carries host CPU
and fault/switch counters alongside native phase samples. Nested phase timings
are kept separately and must not be summed. A mixed phase inventory is refused;
inspect raw logs for a shape-dependent instrumentation route or buffering before
using that capture. The report includes the original log SHA256 and fixture
provenance. It makes no opponent comparison or promotion decision.

```
python tools/mamba3_regime_summary.py /fresh/shape-order.log > /fresh/shape-order-summary.json
python -m unittest discover -s tools -p test_mamba3_regime_summary.py
```

Four host tests pass: retained tiny cold-call separation, truncated/duplicate
capture refusal, output mismatch refusal, and nested phase attribution with
inconsistent inventory refusal. The retained Metal log's first call is
385.847 ms and its second is 13.750 ms; these are instrumentation smoke samples,
not production prices. No new device measurement or kernel/default change was
made by this continuation. Large H100 diagnosis remains owed.

## Large H100 continuation (supersedes the large-run-owed status above)

The current `b69e8344` CUDA IDENTICAL binding completed two fresh-process,
96-call shape-order diagnostics on H100; all 192 complete output hashes match
the retained grid. Evidence and exact limits are in
`bench/results/mamba_regime_large_2026-09-10/README.md`. After separating the
first call of each visit, narrow visit medians span 52.939–58.420 ms and wide
91.988–94.885 ms. The old roughly 229–284 ms steady regime did not reproduce.
Cold first narrow calls were 2017.427/848.889 ms and remain in the evidence.
This is progress on large-run diagnosis, not a qualified speedup or opponent
ratio. The old binding/allocation-lifetime cause remains unresolved. No
conditional phase/lifetime arm was warranted by these results; no default
changed and no opponent ran.
