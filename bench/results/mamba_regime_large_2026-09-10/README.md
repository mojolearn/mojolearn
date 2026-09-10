# H100 large Mamba3 shape-order diagnosis

Production source: `b69e8344`; diagnostic analyzer: `fcf2793b`.
CUDA IDENTICAL binding built for `sm_90`, Mojo 1.0.0 (`ed45d567`), NVIDIA H100
80GB HBM3, driver 580.126.09. Exact binding SHA256:
`f6da73c7d378943252e7280d481351b8763657188fc9f08da2c8bdb04bce03dc`.
The exact binary is retained as `mamba-cuda-identical.so.gz` (deterministic
gzip timestamp zero); `binary-manifest.json` records packed/unpacked hashes
and sizes. This CUDA binding is not a portable wheel.

Two fresh processes each executed 96 calls: two passes through
`narrow,wide,narrow,tiny,narrow,wide`, eight calls per visit. Fixtures are the
retained seed-7 public-API grid: narrow B8/L4096/D512 and wide B8/L1024/D2048;
tiny is the B2/L4/D32 control. Every call passed the complete output SHA256
from `bench/results/staging_performance_2026-09-10/mamba-repeat/summary.json`.
Both processes used the same binding. No opponent ran, kernel changed, or
default was promoted. Transformer had released the GPU before this run.

| Process | Narrow visit median range, calls 2–8 | Wide visit median range, calls 2–8 | Tiny visit median range, calls 2–8 |
| --- | ---: | ---: | ---: |
| 0 | 52.939–54.436 ms | 91.988–92.812 ms | 1.220–1.237 ms |
| 1 | 53.936–58.420 ms | 92.607–94.885 ms | 1.214–1.219 ms |

These ranges are over visit medians, not confidence intervals. The first
narrow calls were 2017.427 and 848.889 ms; all cold calls and subsequent raw
samples remain in the logs. There is no claim that calls 2–8 satisfy a
production warmup rule. The previously observed roughly 229–284 ms large
steady regime did **not** reproduce in this experiment. Consequently the
conditional release-output-before-call, telemetry, and phase-sync arms were
not run. This does not explain the old regime: the current binding is newly
compiled, and retaining constructed blocks/inputs changes the old process's
allocation sequence. It narrows the next experiment to the historical binary
and allocation lifecycle if that slowdown recurs.

Resource counters and output hashing are outside the call timer but perturb
the surrounding schedule. Before/after `nvidia-smi -q` snapshots are retained;
there is no continuous clock or contention trace. The compiler and version
query emitted a container `mbind` warning; it is retained and is not evidence
that it caused any timing regime. Diagnostic timings are not qualified
production prices or a new opponent ratio. `run.sh` is the exact controller;
raw logs, validated summaries, build output, and SHA256 manifest are retained.
