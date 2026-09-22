# Holt-Winters with the estimated initialization as the default (2026-09-22)

`ExponentialSmoothing(initialization_method="estimated")` is the default on
branch `feat/holtwinters-estimated-init-sep22`. It estimates the initial level,
trend and every seasonal state jointly with alpha, beta and gamma over all `n`
points (`holtwinters/impl/internal/hw_estimate.mojo`). `"heuristic"` (alias
`"cuml"`) is the 0.8.13 fit, bit for bit.

The four Holt-Winters lanes use the default, so their cells move. Their
`LANE_REVISIONS` entries were bumped (`holtwinters` `obs-128-estimated-init-2`;
`holtwinters-multiplicative`, `par-holtwinters` and `par-forecast-holtwinters`
`estimated-init-1`). The committed cells now read as absent, not DIVERGENT, and
the four lanes are `stale reference` in `PUBLIC_PENDING_LANES` until the NVIDIA
and AMD columns below are recorded and the table is regenerated.

## Columns in this directory

| column | file | how |
|---|---|---|
| apple-m4 (Metal) | `apple-m4.json` | `tools/identity_break.py --lanes <one lane> --vendor apple-m4 --repeats 1`, one lane per run (the Apple one-lane rule), then `--merge`; logs `legs/apple-m4.*.log` |
| cpu, Apple M4 arm64 | `cpu-apple-m4-arm64.json` | `tools/cpu_identity_gate_check.py run-column --shards 1 -- --repeats 1 --vendor cpu-apple-m4` on a copy of the package with no GPU binding, the tsa host binding built from this branch; `column` verdict OK (`legs/cpu-apple-m4-arm64.column-check.txt`) |
| cpu sabotage, Apple M4 arm64 | `cpu-apple-m4-arm64.sabotage.json` | the same, with the tsa host binding built with `-D MOJOLEARN_HOST_SABOTAGE=1` (the estimated path's SSE step split into two roundings, plus the fitted-state bit flip) |

Every lane read STABLE (36 of 36 cells) in each column. All three were
recorded at 6eacfab28, which carries the parallel device kernel (one block
per series and start, one thread per theta column). Its cells equal those of
the one-thread-per-series build that came before it, bit for bit, on all
four lanes, and its fits are faster on the M4 (100 series, n=240, f=12:
763 ms before, 99 ms after, heuristic 122 ms; one series, n=520, f=52:
2.7 s before, 58 ms after, heuristic 41 ms).

## What agrees

`diff.apple-vs-cpu.txt`, `--require-columns 2` over the four lanes:
IDENTICAL=36 (train), infer/model IDENTICAL=72, batch IDENTICAL=36, exit 0.
Apple Metal and the arm64 CPU agree bit for bit on every changed cell.

`diff.apple-vs-cpu-sabotage.txt`: DIVERGENT=36, infer/model DIVERGENT=72,
batch DIVERGENT=36, exit 1. The negative control moves every cell.

`diff.vs-166-record.txt`: against the three `TRAINING_GPU_COLUMNS`, the old
cells of all four lanes are held at an older lane revision and are not
compared; the two new columns agree (IDENTICAL=36).

Mojo gates on the M4 under IDENTICAL: `legs/apple-m4.hw_estimate_check.txt`
(device == host oracle, bit for bit, on 17 fits: seven fixtures at two
block widths through the parallel arm, its f = 59 edge, the serial arm at
f = 60, and the heuristic default unchanged) and `legs/apple-m4.hw_check.txt` (the
existing heuristic gate, ALL OK).

## NVIDIA, AMD and x86-64 CPU columns (admitted)

All at commit `3e1b87ccf` (this branch at `ed504c033`, the parallel
estimated-fit kernel, merged with main), one fit per cell, all nine fixtures,
every cell STABLE. Leg body `legs/body_hw_binary.sh` (the body_hw.sh commands
plus the gbdt-binary-columns lane); CPU command `legs/cpu_x86_cmd.sh` under
`tools/runpod_cpu_leg.sh --build core,tsa,gbdt --sabotage-build core,tsa`.

| column | file | box |
|---|---|---|
| NVIDIA H100 (sm_90a) | `nvidia-h100-sm_90a.json` | RunPod; gate `legs/nvidia-h100.gate.txt` |
| AMD MI325X (gfx942) | `amd-mi325x-gfx942.json` | DigitalOcean; gate `legs/amd-mi325x.gate.txt` |
| x86-64 CPU, EPYC 9965 | `cpu-amd-epyc-9965-x86_64.json` | RunPod CPU; column verdict OK |
| x86-64 CPU sabotage | `cpu-amd-epyc-9965-x86_64.sabotage.json` | same, `MOJOLEARN_HOST_SABOTAGE` core,tsa |

`hw_estimate_check` and `hw_check` read ALL OK under IDENTICAL on the H100 and
the MI325X.

`diff.five-columns.txt`: Apple, arm64 CPU, H100, MI325X, x86-64 CPU,
`--require-columns 5`: train IDENTICAL=36, infer/model IDENTICAL=72, batch
IDENTICAL=36, exit 0. `diff.apple-vs-cpu-x86-sabotage.txt`: DIVERGENT=36,
exit 1.

A first NVIDIA (RTX 4090) and AMD leg ran at the previous branch head
(`91f713c06`); both agree with the new-head columns on every cell, and only the
new-head columns are kept here.

The reference table was regenerated for the four lanes with
`verify --all --batch-checks --emit-reference ... --reference-table ... --lanes ...`
and the four lanes left `PUBLIC_PENDING_LANES`.
