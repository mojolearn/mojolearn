# NVIDIA GPT-3 dWeight fold-storage screen

Date: 2026-09-20

Base: `origin/main` at `39fc55256f3f0c2ec399985c2d096db292f39fc4`.
Device: one NVIDIA L40S (`sm_89`). The benchmark was built with
`MOJOLEARN_NUMERIC_IDENTICAL=1`; the timing harness alternates the shipped
FS16 arm and the opt-in specialized arm in one process.

## Result

Only the repeated GPT-3-small Q/K/V/O weight-gradient shape qualified for a
production route: `OP_TN m=768 n=768 k=2048`, weighted 48 calls per 12-layer
step. FS4 retains the six contract leaves required by this shape and does not
change their fold order.

Two independent five-sample runs gave these medians:

| run | shipped FS16 | FS4 | change | all-cell mismatches |
| --- | ---: | ---: | ---: | ---: |
| 1 | 204.814 us | 200.414 us | -2.15% | 0 |
| 2 | 202.873 us | 198.594 us | -2.11% | 0 |

At the harness weight of 48 calls, the median saving is 0.211 ms and 0.205 ms
per modeled step, respectively. Output hash was stable at `1270848159` in both
runs. The full-output comparator also reported zero mismatches against plan 10.

The 2048/3072-wide MLP dWeight shapes and the forward/dX screen were neutral
or noisy, so they retain existing routing. The production candidate build,
without the opt-in fold-specialization define, compiled and exercised the
narrow route with zero full-output mismatches; see `nvidia/prod-gate.log`.

## Physical resources

`ptxas` reported 255 registers and one 256-thread block/SM for FS16, FS8, and
FS4. Per-thread local stack fell from 4200/4232 bytes (FS16 all/group) to
2152/2184 bytes (FS8) and 1128/1160 bytes (FS4). Spill bytes remained
104/132. This supports only the measured narrow route, not a broad occupancy
claim. Hardware-counter collection was unavailable (`ERR_NVGPUCTRPERM`).

## Gates

- L40S rotated production-shape runs: stable hashes and zero all-cell
  mismatches for every measured forward, dX, and dWeight shape.
- L40S production build/run without the trial define: PASS, zero mismatches.
- Apple `gemm_production_dispatch_check.mojo` in IDENTICAL mode: PASS.
- Apple `pixi run check-gemm-identity`: PASS (existing deprecation warnings).
- The route is compile-time NVIDIA-only; Apple and AMD dispatch is unchanged.

Raw timing and compiler-resource output is under `nvidia/`.

## Lease lifecycle

Guarded pod `gkeu6eqgl5yk7u` had a 60-minute watchdog. It was explicitly
terminated at 2026-09-20 17:08:04 America/New_York: DELETE returned HTTP 204,
then inventory verification returned HTTP 404 at 17:08:05.
